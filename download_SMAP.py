import re
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import earthaccess
import h5py
import pytz

# =========================================================
# USER SETTINGS (Cold Damage)
# =========================================================
SHORT_NAME = "SPL4SMGP"   # SMAP L4 Geophysical Data
VERSION = "008"

# 이벤트 기준 기간: 30일 전 ~ 3일 후
DAYS_BEFORE = 30
DAYS_AFTER = 3

# 목표 샘플링 시간 (KST) - 하루 4회
TARGET_KST_TIMES = [
    (1, 30),
    (7, 30),
    (13, 30),
    (19, 30),
]

# 목표시간 검색 폭(UTC 기준): ±20분
SEARCH_PAD_MIN = 20

BASE_DIR = Path(r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\27th Cold Damage (Timeseries)\python")
EVENTS_CSV = BASE_DIR / "cold damage_events.csv"

OUT_DIR = BASE_DIR / f"output_smap_l4_sm_temp_{DAYS_BEFORE}d_to_{DAYS_AFTER}d_4perday"
GRANULE_DIR = OUT_DIR / "granules"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRANULE_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = OUT_DIR / "smap_l4_sm_surface_rootzone_and_temp_layers_timeseries_4perday.xlsx"
FAILED_LOG = OUT_DIR / "failed_extract.csv"

# 시간대
KST = pytz.timezone("Asia/Seoul")
UTC = pytz.utc


# =========================================================
# 파일명에서 UTC 시간 추출
# 예) ..._20201106T043000_...h5 -> 2020-11-06 04:30:00 UTC
# =========================================================
def parse_utc_time_from_filename(name: str):
    m = re.search(r"(\d{8})T(\d{6})", name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    dt = pd.to_datetime(ymd + hms, format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(dt):
        return None
    return dt.tz_localize("UTC")


def utc_stamp_str(dt_utc: pd.Timestamp) -> str:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.tz_localize("UTC")
    else:
        dt_utc = dt_utc.tz_convert("UTC")
    return dt_utc.strftime("%Y%m%dT%H%M%S")


# =========================================================
# 이벤트 시간 파서(유연)
# =========================================================
def parse_event_time(series: pd.Series) -> pd.Series:
    x = series.astype(str).copy()
    x = (
        x.str.replace("KST", "", regex=False)
         .str.replace(".", "-", regex=False)
         .str.replace("/", "-", regex=False)
         .str.strip()
    )
    return pd.to_datetime(x, errors="coerce")


# =========================================================
# HDF5 유틸
# =========================================================
def find_dataset_paths(h5: h5py.File, must_contain_keywords):
    hits = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            low = name.lower()
            if all(k in low for k in must_contain_keywords):
                hits.append(name)

    h5.visititems(visitor)
    return hits


def nearest_index(arr_1d, value):
    return int(np.argmin(np.abs(arr_1d - value)))


def get_xy_and_transformer(h5: h5py.File):
    """
    SMAP L4는 대개 x/y(EASE-Grid 2.0) 제공.
    위경도(EPSG:4326) -> EPSG:6933 변환에 pyproj 필요.
    """
    try:
        from pyproj import Transformer
    except Exception as e:
        raise RuntimeError(
            "pyproj가 필요합니다. 아래 중 하나로 설치 후 다시 실행하세요.\n"
            "  - conda install -c conda-forge pyproj\n"
            "  - pip install pyproj\n"
            f"(원인: {type(e).__name__}: {e})"
        )

    if "x" not in h5 or "y" not in h5:
        raise RuntimeError("이 파일에서 x/y 좌표변수를 찾지 못했습니다. (파일 구조 확인 필요)")

    x = h5["x"][:]  # 1D
    y = h5["y"][:]  # 1D
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    return x, y, transformer


def get_dataset(h5: h5py.File, path: str):
    """
    경로가 고정(권장)인데, 혹시 제품 구조가 다르면
    키워드 탐색으로 fallback
    """
    if path in h5:
        return h5[path]

    token = path.split("/")[-1].lower()
    cand = find_dataset_paths(h5, [token])
    if cand:
        return h5[cand[0]]

    raise RuntimeError(f"Dataset not found: {path} (fallback candidates={cand[:10]})")


def to_float(v):
    try:
        v = float(v)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


# =========================================================
# 변수 경로
# =========================================================
VAR_PATHS = {
    "sm_surface": "Geophysical_Data/sm_surface",
    "sm_rootzone": "Geophysical_Data/sm_rootzone",
    "surface_temp": "Geophysical_Data/surface_temp",
    "soil_temp_layer1": "Geophysical_Data/soil_temp_layer1",
    "soil_temp_layer2": "Geophysical_Data/soil_temp_layer2",
    "soil_temp_layer3": "Geophysical_Data/soil_temp_layer3",
    "soil_temp_layer4": "Geophysical_Data/soil_temp_layer4",
    "soil_temp_layer5": "Geophysical_Data/soil_temp_layer5",
    "soil_temp_layer6": "Geophysical_Data/soil_temp_layer6",
}


# =========================================================
# 목표 UTC 시각(하루 4개) 생성
# KST 01:30, 07:30, 13:30, 19:30
# =========================================================
def build_target_utcs(event_time_kst: pd.Timestamp, days_before: int, days_after: int):
    """
    event_time_kst: tz-aware KST
    반환: (target_utc_list, start_utc, end_utc)
    """
    if event_time_kst.tzinfo is None:
        event_time_kst = KST.localize(event_time_kst)

    start_kst = event_time_kst - timedelta(days=days_before)
    end_kst = event_time_kst + timedelta(days=days_after)

    start_utc = start_kst.astimezone(UTC)
    end_utc = end_kst.astimezone(UTC)

    day0 = start_kst.date()
    day1 = end_kst.date()
    days = pd.date_range(day0, day1, freq="D")

    targets = []
    for d in days:
        for hh, mm in TARGET_KST_TIMES:
            dt_kst = KST.localize(pd.Timestamp(d.date()).to_pydatetime()).replace(
                hour=hh, minute=mm, second=0, microsecond=0
            )
            dt_utc = dt_kst.astimezone(UTC)

            if (dt_utc >= start_utc) and (dt_utc <= end_utc):
                targets.append(pd.Timestamp(dt_utc))

    targets = sorted(set(targets))
    return targets, pd.Timestamp(start_utc), pd.Timestamp(end_utc)


# =========================================================
# earthaccess 검색 결과에서 텍스트 최대한 뽑기
# =========================================================
def granule_texts(g):
    texts = []

    if isinstance(g, dict):
        for k in ["producer_granule_id", "ProducerGranuleId", "title", "name", "GranuleUR", "granule_ur"]:
            if k in g and g[k]:
                texts.append(str(g[k]))

        umm = g.get("umm", None)
        if isinstance(umm, dict):
            for k in ["GranuleUR", "EntryTitle", "ShortName"]:
                if k in umm and umm[k]:
                    texts.append(str(umm[k]))

    for attr in ["producer_granule_id", "title", "name", "granule_ur", "granuleUR"]:
        if hasattr(g, attr):
            v = getattr(g, attr)
            if v:
                texts.append(str(v))

    try:
        texts.append(str(g))
    except Exception:
        pass

    out = []
    seen = set()
    for t in texts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def select_results_by_timestamp(results, target_stamp: str):
    """
    results 중에서 텍스트에 target_stamp(YYYYMMDDThhmmss)가 포함된 것만 반환.
    포함되는 게 없으면 원래 results 그대로 반환.
    """
    hit = []
    for r in results:
        txts = granule_texts(r)
        if any(target_stamp in t for t in txts):
            hit.append(r)
    return hit if hit else results


def list_existing_by_stamp(granule_dir: Path, target_stamp: str):
    hits = []
    for p in granule_dir.glob("*.h5"):
        if target_stamp in p.name:
            hits.append(p)
    return hits


# =========================================================
# 한 파일에서 (lat, lon) 포인트 값 추출
# =========================================================
def extract_vars_from_file(h5_path: Path, lat: float, lon: float, cached_grid=None):
    with h5py.File(h5_path, "r") as h5:
        if cached_grid is None:
            x, y, transformer = get_xy_and_transformer(h5)
            cached_grid = (x, y, transformer)
        else:
            x, y, transformer = cached_grid

        xp, yp = transformer.transform(lon, lat)
        ix = nearest_index(x, xp)
        iy = nearest_index(y, yp)

        out = {}
        for k, p in VAR_PATHS.items():
            ds = get_dataset(h5, p)
            out[k] = to_float(ds[iy, ix])

        dt_utc = parse_utc_time_from_filename(h5_path.name)
        if dt_utc is None:
            raise RuntimeError("파일명에서 UTC 시간을 추출하지 못했습니다. 파일명 패턴 확인 필요.")

        rec = {"datetime_utc": dt_utc, "h5_file": h5_path.name}
        rec.update(out)
        return rec, cached_grid


# =========================================================
# Excel 저장용: tz-aware datetime -> tz 제거
# =========================================================
def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    df_xlsx = df.copy()
    tz_cols = ["event_time_kst", "event_time_utc", "datetime_utc", "datetime_kst"]
    for c in tz_cols:
        if c in df_xlsx.columns:
            df_xlsx[c] = pd.to_datetime(df_xlsx[c]).dt.tz_localize(None)
    return df_xlsx


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    earthaccess.login()

    # 이벤트 CSV 읽기
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    events = events.loc[:, ~events.columns.astype(str).str.startswith("Unnamed")].copy()

    required = {"id", "event_time", "latitude", "longitude"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"cold damage_events.csv에 필요한 컬럼이 없습니다: {missing}\n"
            f"현재 컬럼: {list(events.columns)}"
        )

    events["event_time"] = parse_event_time(events["event_time"])
    events = events.dropna(subset=["id", "event_time", "latitude", "longitude"]).copy()

    writer = pd.ExcelWriter(OUT_XLSX, engine="openpyxl")
    failed_rows = []

    for _, ev in events.iterrows():
        try:
            event_id = int(float(ev["id"]))
        except Exception:
            event_id = str(ev["id"])

        lat = float(ev["latitude"])
        lon = float(ev["longitude"])

        # CSV의 event_time은 KST라고 가정
        event_time_kst = KST.localize(ev["event_time"].to_pydatetime())
        event_time_utc = event_time_kst.astimezone(UTC)

        target_utcs, start_dt_utc, end_dt_utc = build_target_utcs(
            event_time_kst, DAYS_BEFORE, DAYS_AFTER
        )

        print(f"\n▶ Event {event_id}")
        print(f"  event_time (KST): {event_time_kst}")
        print(f"  event_time (UTC): {event_time_utc}")
        print(f"  window UTC: {start_dt_utc} ~ {end_dt_utc}")
        print(f"  targets (4/day): {len(target_utcs)} timestamps")

        # -------------------------
        # (1) 목표 시간 4개/일만 다운로드
        # -------------------------
        downloaded_files = []
        bbox = (lon - 0.2, lat - 0.2, lon + 0.2, lat + 0.2)

        for t_utc in target_utcs:
            stamp = utc_stamp_str(t_utc)

            # 이미 있으면 skip
            existed = list_existing_by_stamp(GRANULE_DIR, stamp)
            if existed:
                downloaded_files.extend(existed)
                continue

            t0 = (t_utc - pd.Timedelta(minutes=SEARCH_PAD_MIN)).isoformat()
            t1 = (t_utc + pd.Timedelta(minutes=SEARCH_PAD_MIN)).isoformat()

            results = earthaccess.search_data(
                short_name=SHORT_NAME,
                version=VERSION,
                temporal=(t0, t1),
                bounding_box=bbox,
            )

            if not results:
                failed_rows.append({
                    "event_id": event_id,
                    "stage": "search",
                    "message": f"no_results_for_target_{stamp}",
                    "target_utc": str(t_utc),
                    "lat": lat,
                    "lon": lon,
                })
                continue

            picked = select_results_by_timestamp(results, stamp)

            try:
                files = earthaccess.download(picked, local_path=str(GRANULE_DIR))
                files = [Path(f) for f in files if f]
            except Exception as e:
                failed_rows.append({
                    "event_id": event_id,
                    "stage": "download",
                    "message": f"{type(e).__name__}: {e}",
                    "target_utc": str(t_utc),
                    "lat": lat,
                    "lon": lon,
                })
                continue

            files = [p for p in files if p.suffix.lower() == ".h5"]
            files = [p for p in files if stamp in p.name]

            if not files:
                failed_rows.append({
                    "event_id": event_id,
                    "stage": "download_filter",
                    "message": f"downloaded_but_no_stamp_match_{stamp}",
                    "target_utc": str(t_utc),
                    "lat": lat,
                    "lon": lon,
                })
                continue

            downloaded_files.extend(files)
            time.sleep(0.05)

        downloaded_files = sorted(set(downloaded_files))
        print(f"  downloaded files (after filtering): {len(downloaded_files)}")

        if not downloaded_files:
            print("  ⚠️ 다운로드/필터 후 파일이 없습니다.")
            continue

        # -------------------------
        # (2) 다운로드된 파일만 추출
        # -------------------------
        rows = []
        cached_grid = None

        for f in downloaded_files:
            try:
                rec, cached_grid = extract_vars_from_file(f, lat, lon, cached_grid=cached_grid)
                rows.append(rec)
            except Exception as e:
                print(f"  [EXTRACT FAIL] {f.name} -> {type(e).__name__}: {e}")
                failed_rows.append({
                    "event_id": event_id,
                    "stage": "extract",
                    "file": f.name,
                    "message": f"{type(e).__name__}: {e}",
                    "lat": lat,
                    "lon": lon,
                })

        if not rows:
            print("  ⚠️ 추출 결과 없음")
            continue

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)

        # window 밖 제거(안전장치)
        df = df[(df["datetime_utc"] >= start_dt_utc) & (df["datetime_utc"] <= end_dt_utc)].copy()

        # KST 시각 컬럼 추가
        df["datetime_kst"] = df["datetime_utc"].dt.tz_convert("Asia/Seoul")

        # 메타
        df.insert(0, "event_id", event_id)
        df.insert(1, "event_time_kst", pd.Timestamp(event_time_kst))
        df.insert(2, "event_time_utc", pd.Timestamp(event_time_utc))
        df.insert(3, "latitude", lat)
        df.insert(4, "longitude", lon)

        value_cols = list(VAR_PATHS.keys())

        df = df[
            [
                "event_id", "event_time_kst", "event_time_utc", "latitude", "longitude",
                "datetime_utc", "datetime_kst",
                *value_cols,
                "h5_file",
            ]
        ]

        # CSV 저장
        out_csv = OUT_DIR / (
            f"event_{event_id}_SMAP_L4_{SHORT_NAME}_v{VERSION}_"
            f"{DAYS_BEFORE}d_to_{DAYS_AFTER}d_4perday_sm_temp_layers.csv"
        )
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # Excel 저장
        df_xlsx = make_excel_safe(df)
        sheet = f"event_{event_id}"[:31]
        df_xlsx.to_excel(writer, sheet_name=sheet, index=False)

        print(f"  ✅ 저장 완료: rows={len(df)}")
        print(f"     CSV : {out_csv}")
        print(f"     XLSX: (시트 {sheet})")

    writer.close()

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_LOG, index=False, encoding="utf-8-sig")
        print(f"\n⚠️ 실패 로그 저장: {FAILED_LOG} (rows={len(failed_rows)})")

    print(f"\n🎉 완료! Excel: {OUT_XLSX}")