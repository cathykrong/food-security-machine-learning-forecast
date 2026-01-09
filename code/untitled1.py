import pandas as pd
import numpy as np
from pathlib import Path
import re

BASE = Path(r"E:\1Cathy\hsbdc\AI2026\data2\quickstats_downloads")

import pandas as pd


def to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in {"(D)","(NA)","(N/A)","(S)","(X)",""}: return np.nan
    if s == "(Z)": return 0.0
    s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

def yearly_sum_value(csv_path, chunksize=200_000):
    total = {}
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        # normalize column names
        chunk.columns = [c.upper() for c in chunk.columns]

        # keep only annual/county/survey when present
        if "SOURCE_DESC" in chunk.columns:
            chunk = chunk[chunk["SOURCE_DESC"].str.upper().eq("SURVEY")]
        if "AGG_LEVEL_DESC" in chunk.columns:
            chunk = chunk[chunk["AGG_LEVEL_DESC"].str.upper().eq("COUNTY")]
        if "FREQ_DESC" in chunk.columns:
            chunk = chunk[chunk["FREQ_DESC"].str.upper().eq("ANNUAL")]

        # year + value
        y = pd.to_numeric(chunk["YEAR"], errors="coerce")
        v = chunk["VALUE"].map(to_num)

        # sum by year for this chunk
        g = pd.DataFrame({"YEAR": y, "VAL": v}).dropna(subset=["YEAR"])
        s = g.groupby("YEAR")["VAL"].sum(min_count=1)

        # accumulate
        for yr, val in s.items():
            total[int(yr)] = total.get(int(yr), 0.0) + (0.0 if pd.isna(val) else float(val))

    out = pd.DataFrame({"year": sorted(total.keys()), "sum": [total[k] for k in sorted(total.keys())]})
    return out

def build_crop_us_annual(crop_prefix):
    prod = yearly_sum_value(BASE / f"{crop_prefix}_production_county.csv").rename(columns={"sum": f"{crop_prefix}_production"})
    ah   = yearly_sum_value(BASE / f"{crop_prefix}_area_harvested_county.csv").rename(columns={"sum": f"{crop_prefix}_area_harvested"})
    ap   = yearly_sum_value(BASE / f"{crop_prefix}_area_planted_county.csv").rename(columns={"sum": f"{crop_prefix}_area_planted"})

    df = prod.merge(ah, on="year", how="outer").merge(ap, on="year", how="outer").sort_values("year")

    # National yield = total production / total harvested area
    df[f"{crop_prefix}_yield_us"] = df[f"{crop_prefix}_production"] / df[f"{crop_prefix}_area_harvested"]
    return df

corn = build_crop_us_annual("corn")
wheat = build_crop_us_annual("wheat")
soy = build_crop_us_annual("soybeans")

qs_us = corn.merge(wheat, on="year", how="outer").merge(soy, on="year", how="outer").sort_values("year")
qs_us.to_csv(BASE / "quickstats_us_annual_3crops.csv", index=False)
print("Saved quickstats_us_annual_3crops.csv with rows:", len(qs_us))
