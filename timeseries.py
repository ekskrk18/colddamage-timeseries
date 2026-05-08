import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================
# USER SETTINGS
# =========================
SMAP_DIR = Path(
    r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\27th Cold Damage (Timeseries)\python\output_smap_l4_sm_temp_30d_to_3d_4perday"
)

SMAP_PATTERN = "event_*_SMAP_L4_SPL4SMGP*_30d_to_3d_4perday_sm_temp_layers.csv"

OUT_DIR = SMAP_DIR / "plots_timeseries_cold_damage"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# helpers
# =========================
def infer_event_id_from_filename(p: Path) -> int:
    m = re.search(r"event_(\d+)", p.name)
    if not m:
        raise ValueError(f"이벤트 id 파싱 실패: {p.name}")
    return int(m.group(1))


def to_kst_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """
    tz-aware -> KST로 맞춘 뒤 tz 제거(matplotlib 안전용)
    tz-naive면 그대로 반환
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Seoul").tz_localize(None)
    return ts


def load_smap_event_csv(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "datetime_kst",
        "event_time_kst",
        "sm_surface",
        "sm_rootzone",
        "surface_temp",
        "soil_temp_layer1",
        "soil_temp_layer2",
        "soil_temp_layer3",
        "soil_temp_layer4",
        "soil_temp_layer5",
        "soil_temp_layer6",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}에 필요한 컬럼이 없습니다. 누락: {sorted(missing)}")

    # datetime
    df["date"] = pd.to_datetime(df["datetime_kst"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # event time
    event_time = pd.to_datetime(df["event_time_kst"].iloc[0], errors="coerce")
    if pd.isna(event_time):
        raise ValueError(f"{path.name}: event_time_kst 파싱 실패")
    event_time = to_kst_naive(event_time)

    # matplotlib용
    df["date_plot"] = df["date"].apply(to_kst_naive)

    # Temperature: K -> °C
    temp_cols_k = [
        "surface_temp",
        "soil_temp_layer1",
        "soil_temp_layer2",
        "soil_temp_layer3",
        "soil_temp_layer4",
        "soil_temp_layer5",
        "soil_temp_layer6",
    ]
    for c in temp_cols_k:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c + "_C"] = df[c] - 273.15

    # Soil moisture numeric
    for c in ["sm_surface", "sm_rootzone"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    eid = int(df["event_id"].iloc[0]) if "event_id" in df.columns else infer_event_id_from_filename(path)
    return eid, event_time, df


# =========================
# plotting
# =========================
def plot_event_timeseries(eid: int, event_time: pd.Timestamp, df: pd.DataFrame, out_dir: Path):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 8.5), sharex=True, constrained_layout=True
    )

    # ---- TOP: temperatures (°C)
    temp_map = {
        "surface_temp_C": "Surface temp",
        "soil_temp_layer1_C": "Soil temp L1",
        "soil_temp_layer2_C": "Soil temp L2",
        "soil_temp_layer3_C": "Soil temp L3",
        "soil_temp_layer4_C": "Soil temp L4",
        "soil_temp_layer5_C": "Soil temp L5",
        "soil_temp_layer6_C": "Soil temp L6",
    }

    for col, label in temp_map.items():
        y = pd.to_numeric(df[col], errors="coerce").interpolate(limit_area="inside")
        ax_top.plot(df["date_plot"], y, linewidth=1.4, label=label)

    # event time
    ax_top.axvline(event_time, color="red", linestyle="--", linewidth=1.6, label="Event time")

    # 0°C line
    ax_top.axhline(0.0, color="black", linestyle=":", linewidth=1.2)

    ax_top.set_ylabel("Temperature (°C)")
    ax_top.set_title(f"Cold damage event {eid}: SMAP L4 temperature and soil moisture time series")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(ncol=4, fontsize=9, loc="upper left")

    # ---- BOTTOM: soil moisture
    ax_bot.plot(
        df["date_plot"],
        df["sm_surface"],
        linewidth=1.8,
        label="Surface soil moisture"
    )
    ax_bot.plot(
        df["date_plot"],
        df["sm_rootzone"],
        linewidth=1.8,
        label="Root-zone soil moisture"
    )

    ax_bot.axvline(event_time, color="red", linestyle="--", linewidth=1.6)

    ax_bot.set_ylabel("Soil moisture (m³/m³)")
    ax_bot.set_xlabel("Datetime (KST)")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(fontsize=9, loc="upper left")

    # x-axis formatting
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax_bot.xaxis.set_major_locator(locator)
    ax_bot.xaxis.set_major_formatter(formatter)
    plt.setp(ax_bot.get_xticklabels(), rotation=30, ha="right")

    out_png = out_dir / f"event_{eid:02d}_cold_damage_smap_timeseries.png"
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


# =========================
# main
# =========================
def main():
    files = sorted(SMAP_DIR.glob(SMAP_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"SMAP 파일을 찾지 못했습니다:\n  folder={SMAP_DIR}\n  pattern={SMAP_PATTERN}"
        )

    print(f"[INFO] found {len(files)} files")

    for p in files:
        try:
            eid, event_time, df = load_smap_event_csv(p)
            plot_event_timeseries(eid, event_time, df, OUT_DIR)
        except Exception as e:
            print(f"[FAIL] {p.name} -> {type(e).__name__}: {e}")

    print("[DONE] all events plotted.")


if __name__ == "__main__":
    main()