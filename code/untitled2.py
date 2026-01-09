import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# CONFIG: point this to your folder with the downloaded CSVs
# ------------------------------------------------------------
BASE = Path(r"E:\1Cathy\hsbdc\AI2026\data2\quickstats_downloads")  # <-- change if needed
OUT_PATH = BASE / "quickstats_us_annual_3crops.csv"

NA_STRINGS = {"(D)", "(NA)", "(N/A)", "(S)", "(X)", ""}

def to_num(x):
    """QuickStats VALUE -> float (handles commas + suppression codes)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in NA_STRINGS:
        return np.nan
    if s == "(Z)":
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def pick_mode(peek: pd.DataFrame):
    """
    Some of your '..._county.csv' files are actually NATIONAL + 'US TOTAL'.
    We auto-detect whether the file is NATIONAL or COUNTY so we don't filter away everything.
    """
    cols = set(peek.columns)
    if "AGG_LEVEL_DESC" in cols:
        a = peek["AGG_LEVEL_DESC"].astype(str).str.upper()
        if (a == "NATIONAL").any():
            return "NATIONAL"
        if (a == "COUNTY").any():
            return "COUNTY"
    if "LOCATION_DESC" in cols:
        if peek["LOCATION_DESC"].astype(str).str.upper().eq("US TOTAL").any():
            return "NATIONAL"
    return None

def apply_filters(df: pd.DataFrame, filters: dict | None):
    """Case-insensitive equality filters like SHORT_DESC, DOMAIN_DESC, etc."""
    if not filters:
        return df
    for col, val in filters.items():
        col = col.upper()
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.upper()
        if isinstance(val, (list, tuple, set)):
            df = df[s.isin([str(v).upper() for v in val])]
        else:
            df = df[s.eq(str(val).upper())]
    return df

def yearly_sum_value(csv_path: Path, filters=None, prefer_survey=True, chunksize=200_000):
    """
    Returns a DataFrame: year, sum
    - Works for NATIONAL (US TOTAL) files OR COUNTY files.
    - If COUNTY: sums across counties by year.
    - If NATIONAL: uses US TOTAL by year.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame({"year": [], "sum": []})

    # Peek to decide NATIONAL vs COUNTY and whether SURVEY exists
    peek = pd.read_csv(csv_path, nrows=5000, dtype=str)
    peek.columns = [c.upper() for c in peek.columns]
    mode = pick_mode(peek)

    has_survey = False
    if "SOURCE_DESC" in peek.columns:
        has_survey = peek["SOURCE_DESC"].astype(str).str.upper().eq("SURVEY").any()

    total = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        chunk.columns = [c.upper() for c in chunk.columns]

        # baseline filters (only apply if the column exists)
        if "FREQ_DESC" in chunk.columns:
            chunk = chunk[chunk["FREQ_DESC"].astype(str).str.upper().eq("ANNUAL")]
        if "COUNTRY_NAME" in chunk.columns:
            chunk = chunk[chunk["COUNTRY_NAME"].astype(str).str.upper().eq("UNITED STATES")]
        if prefer_survey and has_survey and "SOURCE_DESC" in chunk.columns:
            chunk = chunk[chunk["SOURCE_DESC"].astype(str).str.upper().eq("SURVEY")]

        # mode filters
        if mode == "NATIONAL":
            if "AGG_LEVEL_DESC" in chunk.columns:
                chunk = chunk[chunk["AGG_LEVEL_DESC"].astype(str).str.upper().eq("NATIONAL")]
            if "LOCATION_DESC" in chunk.columns:
                chunk = chunk[chunk["LOCATION_DESC"].astype(str).str.upper().eq("US TOTAL")]
        elif mode == "COUNTY":
            if "AGG_LEVEL_DESC" in chunk.columns:
                chunk = chunk[chunk["AGG_LEVEL_DESC"].astype(str).str.upper().eq("COUNTY")]

        # your specific series filters (SHORT_DESC, DOMAIN_DESC, etc.)
        chunk = apply_filters(chunk, filters)

        if len(chunk) == 0:
            continue

        y = pd.to_numeric(chunk.get("YEAR"), errors="coerce")
        v = chunk.get("VALUE", pd.Series([np.nan] * len(chunk))).map(to_num)

        g = pd.DataFrame({"YEAR": y, "VAL": v}).dropna(subset=["YEAR"])
        s = g.groupby("YEAR")["VAL"].sum(min_count=1)

        for yr, val in s.items():
            if pd.isna(yr):
                continue
            yr_i = int(yr)
            total[yr_i] = total.get(yr_i, 0.0) + (0.0 if pd.isna(val) else float(val))

    return pd.DataFrame({"year": sorted(total.keys()),
                         "sum": [total[k] for k in sorted(total.keys())]})

def build_crop_us_annual(
    crop_prefix: str,
    production_path: Path,
    harvested_path: Path,
    planted_path: Path,
    production_filters: dict,
    harvested_filters: dict,
    planted_filters: dict,
    year_min: int = 1970,
    year_max: int = 2023,
):
    prod = yearly_sum_value(production_path, filters=production_filters).rename(
        columns={"sum": f"{crop_prefix}_production"}
    )
    ah = yearly_sum_value(harvested_path, filters=harvested_filters).rename(
        columns={"sum": f"{crop_prefix}_area_harvested"}
    )
    ap = yearly_sum_value(planted_path, filters=planted_filters).rename(
        columns={"sum": f"{crop_prefix}_area_planted"}
    )

    df = prod.merge(ah, on="year", how="outer").merge(ap, on="year", how="outer")
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].sort_values("year")

    # National yield = total production / total harvested area (weighted, consistent)
    df[f"{crop_prefix}_yield_us"] = df[f"{crop_prefix}_production"] / df[f"{crop_prefix}_area_harvested"]
    return df

# ------------------------------------------------------------
# CROP SETUP (matches your uploaded example files)
# ------------------------------------------------------------
corn = build_crop_us_annual(
    "corn",
    BASE / "corn_production_county.csv",
    BASE / "corn_area_harvested_county.csv",
    BASE / "corn_area_planted_county.csv",
    production_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "CORN, GRAIN - PRODUCTION, MEASURED IN BU",
    },
    harvested_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "CORN, GRAIN - ACRES HARVESTED",
    },
    planted_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "CORN - ACRES PLANTED",
    },
)

soy = build_crop_us_annual(
    "soybeans",
    BASE / "soybeans_production_county.csv",
    BASE / "soybeans_area_harvested_county.csv",
    BASE / "soybeans_area_planted_county.csv",
    production_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "SOYBEANS - PRODUCTION, MEASURED IN BU",
    },
    harvested_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "SOYBEANS - ACRES HARVESTED",
    },
    planted_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "SOYBEANS - ACRES PLANTED",
    },
)

wheat = build_crop_us_annual(
    "wheat",
    BASE / "wheat_production_county.csv",
    BASE / "wheat_area_harvested_county.csv",
    BASE / "wheat_area_planted_county.csv",
    production_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "WHEAT - PRODUCTION, MEASURED IN BU",
    },
    harvested_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "WHEAT - ACRES HARVESTED",
    },
    planted_filters={
        "DOMAIN_DESC": "TOTAL",
        "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
        "SHORT_DESC": "WHEAT - ACRES PLANTED",
    },
)

# ------------------------------------------------------------
# MERGE + SAVE
# ------------------------------------------------------------
qs_us = corn.merge(wheat, on="year", how="outer").merge(soy, on="year", how="outer").sort_values("year")
qs_us.to_csv(OUT_PATH, index=False)

print(f"Saved: {OUT_PATH}")
print("Rows:", len(qs_us), "| Years:", int(qs_us['year'].min()), "-", int(qs_us['year'].max()))
