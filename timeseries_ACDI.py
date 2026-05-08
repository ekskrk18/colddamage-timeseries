import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================================================
# USER SETTINGS
# =========================================================
SMAP_DIR = Path(
    r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\27th Cold Damage (Timeseries)\python\output_smap_l4_sm_temp_30d_to_3d_4perday"
)
SMAP_PATTERN = "event_*_SMAP_L4_SPL4SMGP*_30d_to_3d_4perday_sm_temp_layers.csv"

OUT_DIR = SMAP_DIR / "plots_acdi_representative_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# threshold for cold intensity
TBASE_C = 5.0

# reference dryness quantile
DRY_REF_QUANTILE = 0.75

# rolling windows (short trigger-focused)
F1_WINDOW_DAYS = 2   # cold intensity
F2_WINDOW_DAYS = 1   # rapid cooling shock
F3_WINDOW_DAYS = 1   # vertical thermal gradient
F4_WINDOW_DAYS = 2   # cold-dry interaction

MIN_VALID_RATIO = 0.6

# weights
W_F1 = 1.0
W_F2 = 1.0
W_F3 = 1.0
W_F4 = 1.0


# =========================================================
# CASE OPTIONS
# =========================================================
F1_TEMP_OPTIONS = {
    "surface": "surface_temp_C",
    "layer1": "soil_temp_layer1_C",
}

F2_TEMP_OPTIONS = {
    "surface": "surface_temp_C",
    "layer1": "soil_temp_layer1_C",
}

F3_GRADIENT_OPTIONS = {
    "L1-SFC": ("soil_temp_layer1_C", "surface_temp_C"),
    "L2-SFC": ("soil_temp_layer2_C", "surface_temp_C"),
    "L2-L1": ("soil_temp_layer2_C", "soil_temp_layer1_C"),
}

F4_MOISTURE_OPTIONS = {
    "surfSM": "sm_surface",
    "rootSM": "sm_rootzone",
}

# =========================================================
# REPRESENTATIVE CASES (only plot these)
# =========================================================
REPRESENTATIVE_CASES = [
    ("surface", "layer1",  "L1-SFC", "rootSM"),
    ("surface", "surface", "L1-SFC", "rootSM"),
    ("surface", "layer1",  "L1-SFC", "surfSM"),
    ("layer1",  "layer1",  "L1-SFC", "surfSM"),
    ("surface", "surface", "L2-SFC", "surfSM"),
    ("surface", "surface", "L2-L1",  "surfSM"),
]


# =========================================================
# HELPERS
# =========================================================
def infer_event_id_from_filename(p: Path) -> int:
    m = re.search(r"event_(\d+)", p.name)
    if not m:
        raise ValueError(f"이벤트 id 파싱 실패: {p.name}")
    return int(m.group(1))


def to_kst_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Seoul").tz_localize(None)
    return ts


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mu) / sd


def rolling_min_periods(window_steps: int) -> int:
    return max(1, int(np.ceil(window_steps * MIN_VALID_RATIO)))


def estimate_steps_per_day(df: pd.DataFrame) -> int:
    dt = df["date_plot"].sort_values().diff().dropna()
    if len(dt) == 0:
        return 4
    median_hours = dt.dt.total_seconds().median() / 3600.0
    if median_hours <= 0:
        return 4
    est = int(round(24.0 / median_hours))
    return max(1, est)


def days_to_steps(df: pd.DataFrame, days: int) -> tuple[int, int]:
    steps_per_day = estimate_steps_per_day(df)
    window_steps = max(1, days * steps_per_day)
    min_periods = rolling_min_periods(window_steps)
    return window_steps, min_periods


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
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}에 필요한 컬럼이 없습니다. 누락: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["datetime_kst"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    event_time = pd.to_datetime(df["event_time_kst"].iloc[0], errors="coerce")
    if pd.isna(event_time):
        raise ValueError(f"{path.name}: event_time_kst 파싱 실패")
    event_time = to_kst_naive(event_time)

    df["date_plot"] = df["date"].apply(to_kst_naive)

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
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c + "_C"] = df[c] - 273.15

    for c in ["sm_surface", "sm_rootzone"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    eid = int(df["event_id"].iloc[0]) if "event_id" in df.columns else infer_event_id_from_filename(path)
    return eid, event_time, df


# =========================================================
# FACTOR FUNCTIONS
# =========================================================
def compute_cold_intensity(df: pd.DataFrame, temp_col: str, window_days: int) -> pd.Series:
    steps, min_periods = days_to_steps(df, window_days)
    cold_deficit = (TBASE_C - df[temp_col]).clip(lower=0)
    return cold_deficit.rolling(steps, min_periods=min_periods).mean()


def compute_rapid_cooling(df: pd.DataFrame, temp_col: str, window_days: int) -> pd.Series:
    steps, min_periods = days_to_steps(df, window_days)
    dtemp = df[temp_col].diff()
    shock = (-dtemp).clip(lower=0)
    return shock.rolling(steps, min_periods=min_periods).mean()


def compute_vertical_gradient(df: pd.DataFrame, upper_col: str, lower_col: str, window_days: int) -> pd.Series:
    steps, min_periods = days_to_steps(df, window_days)
    grad = (df[upper_col] - df[lower_col]).clip(lower=0)
    return grad.rolling(steps, min_periods=min_periods).mean()


def compute_dryness(df: pd.DataFrame, sm_col: str, window_days: int) -> pd.Series:
    steps, min_periods = days_to_steps(df, window_days)
    sm_roll = df[sm_col].rolling(steps, min_periods=min_periods).mean()
    sm_ref = sm_roll.quantile(DRY_REF_QUANTILE)
    return (sm_ref - sm_roll).clip(lower=0)


def compute_cold_dry_interaction(cold_intensity: pd.Series, dryness: pd.Series) -> pd.Series:
    cold_z = zscore_series(cold_intensity)
    dry_z = zscore_series(dryness)
    return cold_z * dry_z


# =========================================================
# ACDI CASE
# =========================================================
def compute_acdi_case(
    df: pd.DataFrame,
    f1_key: str,
    f2_key: str,
    f3_key: str,
    f4_key: str,
) -> pd.DataFrame:
    out = df.copy()

    # factor 1
    f1_temp_col = F1_TEMP_OPTIONS[f1_key]
    f1 = compute_cold_intensity(out, f1_temp_col, F1_WINDOW_DAYS)

    # factor 2
    f2_temp_col = F2_TEMP_OPTIONS[f2_key]
    f2 = compute_rapid_cooling(out, f2_temp_col, F2_WINDOW_DAYS)

    # factor 3
    upper_col, lower_col = F3_GRADIENT_OPTIONS[f3_key]
    f3 = compute_vertical_gradient(out, upper_col, lower_col, F3_WINDOW_DAYS)

    # factor 4
    f4_sm_col = F4_MOISTURE_OPTIONS[f4_key]
    dryness = compute_dryness(out, f4_sm_col, F4_WINDOW_DAYS)
    f4 = compute_cold_dry_interaction(f1, dryness)

    out["F1_cold_intensity"] = f1
    out["F2_rapid_cooling"] = f2
    out["F3_vertical_gradient"] = f3
    out["F4_cold_dry"] = f4

    out["Z_F1"] = zscore_series(out["F1_cold_intensity"])
    out["Z_F2"] = zscore_series(out["F2_rapid_cooling"])
    out["Z_F3"] = zscore_series(out["F3_vertical_gradient"])
    out["Z_F4"] = zscore_series(out["F4_cold_dry"])

    out["ACDI"] = (
        W_F1 * out["Z_F1"]
        + W_F2 * out["Z_F2"]
        + W_F3 * out["Z_F3"]
        + W_F4 * out["Z_F4"]
    )

    out["case_name"] = f"F1-{f1_key}_F2-{f2_key}_F3-{f3_key}_F4-{f4_key}"
    out["f1_key"] = f1_key
    out["f2_key"] = f2_key
    out["f3_key"] = f3_key
    out["f4_key"] = f4_key

    return out


# =========================================================
# PLOTTING
# =========================================================
def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def build_line_style(f1_key: str, f2_key: str, f3_key: str, f4_key: str):
    # f1 -> linestyle
    f1_style = {
        "surface": "-",
        "layer1": "--",
    }

    # f2 -> line width
    lw = 1.0 if f2_key == "surface" else 1.6

    # f4 -> alpha
    alpha = 0.75 if f4_key == "surfSM" else 0.95

    linestyle = f1_style.get(f1_key, "-")
    return linestyle, lw, alpha


def plot_all_cases_for_event(
    eid: int,
    event_time: pd.Timestamp,
    case_dfs: list,
    out_dir: Path,
):
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)

    for cdf in case_dfs:
        f1_key = cdf["f1_key"].iloc[0]
        f2_key = cdf["f2_key"].iloc[0]
        f3_key = cdf["f3_key"].iloc[0]
        f4_key = cdf["f4_key"].iloc[0]

        label = f"F1:{f1_key}, F2:{f2_key}, F3:{f3_key}, F4:{f4_key}"
        linestyle, lw, alpha = build_line_style(f1_key, f2_key, f3_key, f4_key)

        ax.plot(
            cdf["date_plot"],
            cdf["ACDI"],
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            label=label,
        )

    ax.axvline(event_time, color="red", linestyle="--", linewidth=2.4, label="Event time")

    ax.set_title(f"Cold damage event {eid}: Representative ACDI cases")
    ax.set_ylabel("ACDI")
    ax.set_xlabel("Datetime (KST)")
    ax.grid(True, alpha=0.3)

    format_time_axis(ax)

    # legend inside plot
    ax.legend(
        fontsize=9,
        loc="upper left",
        frameon=True
    )

    out_png = out_dir / f"event_{eid:02d}_ACDI_representative_cases.png"
    fig.savefig(out_png, dpi=250)
    plt.close(fig)

    print(f"[OK] saved: {out_png}")


# =========================================================
# MAIN
# =========================================================
def main():
    files = sorted(SMAP_DIR.glob(SMAP_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"SMAP 파일을 찾지 못했습니다:\n  folder={SMAP_DIR}\n  pattern={SMAP_PATTERN}"
        )

    print(f"[INFO] found {len(files)} event files")
    print(f"[INFO] representative cases per event = {len(REPRESENTATIVE_CASES)}")

    for p in files:
        try:
            eid, event_time, df = load_smap_event_csv(p)

            case_dfs = []

            for f1_key, f2_key, f3_key, f4_key in REPRESENTATIVE_CASES:
                cdf = compute_acdi_case(
                    df=df,
                    f1_key=f1_key,
                    f2_key=f2_key,
                    f3_key=f3_key,
                    f4_key=f4_key,
                )
                case_dfs.append(cdf)

            plot_all_cases_for_event(
                eid=eid,
                event_time=event_time,
                case_dfs=case_dfs,
                out_dir=OUT_DIR,
            )

            print(f"[DONE] event {eid} finished: total cases={len(case_dfs)}")

        except Exception as e:
            print(f"[FAIL] {p.name} -> {type(e).__name__}: {e}")

    print("[DONE] all events processed.")


if __name__ == "__main__":
    main()