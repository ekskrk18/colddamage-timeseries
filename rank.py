import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# USER SETTINGS
# =========================================================
SMAP_DIR = Path(
r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\27th Cold Damage (Timeseries)\python\output_smap_l4_sm_temp_30d_to_3d_4perday"
)

SMAP_PATTERN = "event_*_SMAP_L4_SPL4SMGP*_30d_to_3d_4perday_sm_temp_layers.csv"

OUT_DIR = SMAP_DIR / "acdi_case_ranking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TBASE = 5.0
EVENT_WINDOW_HOURS = 48
THRESHOLD_Q = 0.90


# =========================================================
# CASE OPTIONS
# =========================================================

F1_OPTIONS = {
"surface":"surface_temp_C",
"layer1":"soil_temp_layer1_C"
}

F2_OPTIONS = {
"surface":"surface_temp_C",
"layer1":"soil_temp_layer1_C"
}

F3_OPTIONS = {
"L1-SFC":("soil_temp_layer1_C","surface_temp_C"),
"L2-SFC":("soil_temp_layer2_C","surface_temp_C"),
"L2-L1":("soil_temp_layer2_C","soil_temp_layer1_C")
}

F4_OPTIONS = {
"surfSM":"sm_surface",
"rootSM":"sm_rootzone"
}


# =========================================================
# UTILS
# =========================================================

def zscore(x):

    mu=np.nanmean(x)
    sd=np.nanstd(x)

    if sd==0:
        return np.zeros(len(x))

    return (x-mu)/sd


def infer_event_id(path):

    m=re.search(r"event_(\d+)",path.name)
    return int(m.group(1))


# =========================================================
# LOAD
# =========================================================

def load_event(path):

    df=pd.read_csv(path)

    df["date"]=pd.to_datetime(df["datetime_kst"])
    df=df.sort_values("date")

    event_time=pd.to_datetime(df["event_time_kst"].iloc[0])

    temp_cols=[
    "surface_temp",
    "soil_temp_layer1",
    "soil_temp_layer2",
    "soil_temp_layer3",
    "soil_temp_layer4"
    ]

    for c in temp_cols:

        if c in df.columns:
            df[c+"_C"]=df[c]-273.15

    return df,event_time


# =========================================================
# FACTORS
# =========================================================

def factor_cold_intensity(temp):

    cold=np.maximum(0,TBASE-temp)

    return pd.Series(cold).rolling(8,min_periods=1).mean()


def factor_cooling(temp):

    d=temp.diff()

    shock=np.maximum(0,-d)

    return shock.rolling(4,min_periods=1).mean()


def factor_gradient(t1,t2):

    g=np.maximum(0,t1-t2)

    return pd.Series(g).rolling(4,min_periods=1).mean()


def factor_dryness(sm):

    ref=np.nanpercentile(sm,75)

    return np.maximum(0,ref-sm)


# =========================================================
# ACDI
# =========================================================

def compute_acdi(df,f1,f2,f3,f4):

    F1=factor_cold_intensity(df[f1])

    F2=factor_cooling(df[f2])

    upper,lower=F3_OPTIONS[f3]

    F3=factor_gradient(df[upper],df[lower])

    dry=factor_dryness(df[F4_OPTIONS[f4]])

    F4=zscore(F1)*zscore(dry)

    acdi=zscore(F1)+zscore(F2)+zscore(F3)+zscore(F4)

    return pd.Series(acdi,index=df.index)


# =========================================================
# SCORES
# =========================================================

def compute_scores(acdi,dates,event_time):

    event_start=event_time-pd.Timedelta(hours=EVENT_WINDOW_HOURS)

    mask_event=(dates>=event_start)&(dates<=event_time)

    mask_early=(dates<event_start)

    event_vals=acdi[mask_event]
    early_vals=acdi[mask_early]

    pre_vals=acdi[dates<event_time]

    sigma=np.nanstd(pre_vals)

    if sigma==0:
        sigma=1e-6

    # S1 dominance
    p_event=np.nanmax(event_vals)
    p_early=np.nanmax(early_vals)

    S1=(p_event-p_early)/sigma

    # S2 concentration
    m_event=np.nanmean(event_vals)
    m_early=np.nanmean(early_vals)

    S2=(m_event-m_early)/sigma

    # S3 exceedance
    thr=np.nanquantile(pre_vals,THRESHOLD_Q)

    D_event=np.sum(event_vals>thr)

    D_early=np.sum(early_vals>thr)/max(1,len(early_vals))

    S3=D_event-D_early

    score=(S1+S2+S3)/3

    return S1,S2,S3,score


# =========================================================
# MAIN
# =========================================================

all_results=[]

files=sorted(SMAP_DIR.glob(SMAP_PATTERN))

for f in files:

    df,event_time=load_event(f)

    eid=infer_event_id(f)

    dates=df["date"]

    for f1 in F1_OPTIONS:
        for f2 in F2_OPTIONS:
            for f3 in F3_OPTIONS:
                for f4 in F4_OPTIONS:

                    acdi=compute_acdi(df,
                                      F1_OPTIONS[f1],
                                      F2_OPTIONS[f2],
                                      f3,
                                      f4)

                    S1,S2,S3,score=compute_scores(acdi,dates,event_time)

                    case_name=f"F1:{f1}_F2:{f2}_F3:{f3}_F4:{f4}"

                    all_results.append({
                    "event":eid,
                    "case":case_name,
                    "S1_dominance":S1,
                    "S2_concentration":S2,
                    "S3_exceedance":S3,
                    "score":score
                    })


results=pd.DataFrame(all_results)

# =========================================================
# SAVE EVENT RANKING
# =========================================================

event_rank=results.sort_values(["event","score"],ascending=[True,False])

event_rank.to_csv(OUT_DIR/"event_case_ranking.csv",index=False)

# =========================================================
# GLOBAL RANKING
# =========================================================

global_rank=results.groupby("case").mean(numeric_only=True)

global_rank=global_rank.sort_values("score",ascending=False)

global_rank.to_csv(OUT_DIR/"global_case_ranking.csv")

print("ranking finished")