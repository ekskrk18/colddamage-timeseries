import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
SMAP_DIR = Path(
    r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\27th Cold Damage (Timeseries)\python\output_smap_l4_sm_temp_30d_to_3d_4perday"
)
SMAP_PATTERN = "event_*_SMAP_L4_SPL4SMGP*_30d_to_3d_4perday_sm_temp_layers.csv"

OUT_DIR = SMAP_DIR / "acdi_metric_boxplot_heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TBASE = 5.0
EVENT_WINDOW_HOURS = 48
THRESHOLD_Q = 0.90


# =========================================================
# CASE OPTIONS
# =========================================================
F1_OPTIONS = {
    "surface": "surface_temp_C",
    "layer1": "soil_temp_layer1_C",
}

F2_OPTIONS = {
    "surface": "surface_temp_C",
    "layer1": "soil_temp_layer1_C",
}

F3_OPTIONS = {
    "L1-SFC": ("soil_temp_layer1_C", "surface_temp_C"),
    "L2-SFC": ("soil_temp_layer2_C", "surface_temp_C"),
    "L2-L1": ("soil_temp_layer2_C", "soil_temp_layer1_C"),
}

F4_OPTIONS = {
    "surfSM": "sm_surface",
    "rootSM": "sm_rootzone",
}

ROW_ORDER = [
    ("surface", "surface"),
    ("surface", "layer1"),
    ("layer1", "surface"),
    ("layer1", "layer1"),
]

COL_ORDER = [
    ("L1-SFC", "surfSM"),
    ("L1-SFC", "rootSM"),
    ("L2-SFC", "surfSM"),
    ("L2-SFC", "rootSM"),
    ("L2-L1", "surfSM"),
    ("L2-L1", "rootSM"),
]


# =========================================================
# UTILS
# =========================================================
def zscore(x: pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if np.isnan(sd) or sd == 0:
        return np.zeros(len(arr))
    return (arr - mu) / sd


def infer_event_id(path: Path) -> int:
    m = re.search(r"event_(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse event id from {path.name}")
    return int(m.group(1))


# =========================================================
# LOAD
# =========================================================
def load_event(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["date"] = pd.to_datetime(df["datetime_kst"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    event_time = pd.to_datetime(df["event_time_kst"].iloc[0], errors="coerce")
    if pd.isna(event_time):
        raise ValueError(f"event_time_kst parse failed in {path.name}")

    temp_cols = [
        "surface_temp",
        "soil_temp_layer1",
        "soil_temp_layer2",
        "soil_temp_layer3",
        "soil_temp_layer4",
        "soil_temp_layer5",
        "soil_temp_layer6",
    ]
    for c in temp_cols:
        if c in df.columns:
            df[c + "_C"] = pd.to_numeric(df[c], errors="coerce") - 273.15

    for c in ["sm_surface", "sm_rootzone"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, event_time


# =========================================================
# FACTORS (rank code와 동일)
# =========================================================
def factor_cold_intensity(temp: pd.Series) -> pd.Series:
    cold = np.maximum(0, TBASE - temp)
    return pd.Series(cold, index=temp.index).rolling(8, min_periods=1).mean()


def factor_cooling(temp: pd.Series) -> pd.Series:
    d = temp.diff()
    shock = np.maximum(0, -d)
    return shock.rolling(4, min_periods=1).mean()


def factor_gradient(t1: pd.Series, t2: pd.Series) -> pd.Series:
    g = np.maximum(0, t1 - t2)
    return pd.Series(g, index=t1.index).rolling(4, min_periods=1).mean()


def factor_dryness(sm: pd.Series) -> pd.Series:
    ref = np.nanpercentile(sm, 75)
    return pd.Series(np.maximum(0, ref - sm), index=sm.index)


# =========================================================
# ACDI (rank code와 동일)
# =========================================================
def compute_acdi(df: pd.DataFrame, f1: str, f2: str, f3: str, f4: str) -> pd.Series:
    F1 = factor_cold_intensity(df[f1])
    F2 = factor_cooling(df[f2])

    upper, lower = F3_OPTIONS[f3]
    F3 = factor_gradient(df[upper], df[lower])

    dry = factor_dryness(df[F4_OPTIONS[f4]])
    F4 = zscore(F1) * zscore(dry)

    acdi = zscore(F1) + zscore(F2) + zscore(F3) + zscore(F4)
    return pd.Series(acdi, index=df.index)


# =========================================================
# SCORES (rank code와 동일)
# =========================================================
def compute_scores(acdi: pd.Series, dates: pd.Series, event_time: pd.Timestamp):
    event_start = event_time - pd.Timedelta(hours=EVENT_WINDOW_HOURS)

    mask_event = (dates >= event_start) & (dates <= event_time)
    mask_early = dates < event_start
    mask_pre = dates < event_time

    event_vals = acdi[mask_event]
    early_vals = acdi[mask_early]
    pre_vals = acdi[mask_pre]

    sigma = np.nanstd(pre_vals)
    if np.isnan(sigma) or sigma == 0:
        sigma = 1e-6

    # 1) 사고 전 48시간 peak 우월성
    p_event = np.nanmax(event_vals) if len(event_vals) else np.nan
    p_early = np.nanmax(early_vals) if len(early_vals) else np.nan
    S1 = (p_event - p_early) / sigma

    # 2) 사고 전 48시간 집중도
    m_event = np.nanmean(event_vals) if len(event_vals) else np.nan
    m_early = np.nanmean(early_vals) if len(early_vals) else np.nan
    S2 = (m_event - m_early) / sigma

    # 3) 사고 전 48시간 threshold exceedance 지속시간
    thr = np.nanquantile(pre_vals, THRESHOLD_Q) if len(pre_vals) else np.nan
    D_event = np.sum(event_vals > thr) if len(event_vals) else np.nan
    D_early = np.sum(early_vals > thr) / max(1, len(early_vals)) if len(early_vals) else np.nan
    S3 = D_event - D_early

    return S1, S2, S3


# =========================================================
# CASE LABELS / MATRICES
# =========================================================
def make_case_name(f1: str, f2: str, f3: str, f4: str) -> str:
    return f"F1:{f1} | F2:{f2} | F3:{f3} | F4:{f4}"


def build_metric_dataframe(files):
    rows = []

    for f in files:
        df, event_time = load_event(f)
        event_id = infer_event_id(f)
        dates = df["date"]

        for f1_key, f1_col in F1_OPTIONS.items():
            for f2_key, f2_col in F2_OPTIONS.items():
                for f3_key in F3_OPTIONS.keys():
                    for f4_key in F4_OPTIONS.keys():
                        acdi = compute_acdi(df, f1_col, f2_col, f3_key, f4_key)
                        s1, s2, s3 = compute_scores(acdi, dates, event_time)

                        rows.append({
                            "event": event_id,
                            "F1": f1_key,
                            "F2": f2_key,
                            "F3": f3_key,
                            "F4": f4_key,
                            "case": make_case_name(f1_key, f2_key, f3_key, f4_key),
                            "S1_peak_dominance": s1,
                            "S2_concentration": s2,
                            "S3_exceedance_duration": s3,
                        })

    return pd.DataFrame(rows)


def make_heatmap_matrix(metric_df: pd.DataFrame, metric_col: str):
    mat = np.full((len(ROW_ORDER), len(COL_ORDER)), np.nan)

    for i, (f1, f2) in enumerate(ROW_ORDER):
        for j, (f3, f4) in enumerate(COL_ORDER):
            sel = metric_df[
                (metric_df["F1"] == f1) &
                (metric_df["F2"] == f2) &
                (metric_df["F3"] == f3) &
                (metric_df["F4"] == f4)
            ][metric_col]

            if len(sel) > 0:
                mat[i, j] = sel.mean()

    row_labels = [f"F1:{f1}\nF2:{f2}" for f1, f2 in ROW_ORDER]
    col_labels = [f"F3:{f3}\nF4:{f4}" for f3, f4 in COL_ORDER]

    return mat, row_labels, col_labels


# =========================================================
# PLOTTING
# =========================================================
def plot_boxplot(metric_df: pd.DataFrame, metric_col: str, title: str, out_png: Path):
    # case 평균 점수 순으로 정렬
    order = (
        metric_df.groupby("case")[metric_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    data = [metric_df.loc[metric_df["case"] == c, metric_col].dropna().values for c in order]

    fig, ax = plt.subplots(figsize=(22, 8), constrained_layout=True)
    ax.boxplot(data, patch_artist=False, showfliers=True)

    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


def plot_heatmap(metric_df: pd.DataFrame, metric_col: str, title: str, out_png: Path):
    mat, row_labels, col_labels = make_heatmap_matrix(metric_df, metric_col)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # 값 표기
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_col)

    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


# =========================================================
# MAIN
# =========================================================
def main():
    files = sorted(SMAP_DIR.glob(SMAP_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"SMAP files not found:\nfolder={SMAP_DIR}\npattern={SMAP_PATTERN}"
        )

    metric_df = build_metric_dataframe(files)
    metric_df.to_csv(OUT_DIR / "acdi_24cases_event_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {OUT_DIR / 'acdi_24cases_event_metrics.csv'}")

    metric_map = {
        "S1_peak_dominance": "1. 사고 전 48시간 peak 우월성",
        "S2_concentration": "2. 사고 전 48시간 집중도",
        "S3_exceedance_duration": "3. 사고 전 48시간 threshold exceedance 지속시간",
    }

    for metric_col, metric_title in metric_map.items():
        plot_boxplot(
            metric_df=metric_df,
            metric_col=metric_col,
            title=f"{metric_title} - 24 cases box plot",
            out_png=OUT_DIR / f"{metric_col}_boxplot.png",
        )

        plot_heatmap(
            metric_df=metric_df,
            metric_col=metric_col,
            title=f"{metric_title} - 24 cases heatmap",
            out_png=OUT_DIR / f"{metric_col}_heatmap.png",
        )

    # 전체 평균 테이블도 저장
    avg_df = (
        metric_df.groupby(["F1", "F2", "F3", "F4", "case"], as_index=False)[
            ["S1_peak_dominance", "S2_concentration", "S3_exceedance_duration"]
        ]
        .mean()
        .sort_values("S1_peak_dominance", ascending=False)
    )
    avg_df.to_csv(OUT_DIR / "acdi_24cases_mean_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {OUT_DIR / 'acdi_24cases_mean_metrics.csv'}")

    print("[DONE] box plots and heatmaps created.")


if __name__ == "__main__":
    main()