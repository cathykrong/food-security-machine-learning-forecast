import pandas as pd
import numpy as np
from pathlib import Path
import re

# ============================================================
# STEP #2 — Build modeling table (FSI target + crop features)
# from a QuickStats DOWNLOAD FOLDER (many CSVs)
#
# What it does:
# 1) Scans your folder for corn/soybeans/wheat annual CSVs
# 2) Keeps only: production, area harvested, yield (if present)
# 3) Aggregates across all rows per year (county/state/etc) -> USA annual totals
# 4) Builds derived yield = production / area_harvested
# 5) Adds YoY% shocks + 1–2 year lags
# 6) Merges with your normalized FSI file
# ============================================================

# ----------------------------
# USER SETTINGS (edit these)
# ----------------------------
QUICKSTATS_DIR = Path(r"E:\1Cathy\hsbdc\AI2026\data2\quickstats_downloads")

FSI_CSV = Path(r"E:\1Cathy\hsbdc\AI2026\data3\usa_4vars_normalized_geomeanFSI_1961_2023.csv")  # <-- update if different

OUT_CSV = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")

KEEP_YEARS = (1970, 2023)
TRAIN_END_YEAR = 2012      # for later modeling, not used here, but kept for consistency
LAGS = [1, 2]
EPS_DIV = 1e-9

# Crops + measures we want
CROPS = ["CORN", "SOYBEANS", "WHEAT"]
MEASURES = {
    "PRODUCTION": "production",
    "AREA HARVESTED": "area_harvested",
    "YIELD": "yield"
}

# ----------------------------
# Helpers
# ----------------------------
def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"(D)", "(NA)", "(N/A)", "(S)", "(X)", ""}:
        return np.nan
    if s == "(Z)":
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def colmap_upper(df):
    m = {c.upper(): c for c in df.columns}
    return m

def getcol(df, name_upper):
    m = colmap_upper(df)
    return m.get(name_upper)

def pct_change_safe(x: pd.Series) -> pd.Series:
    prev = x.shift(1)
    denom = prev.replace(0, np.nan) + EPS_DIV
    return (x - prev) / denom

def detect_crop_measure_from_filename(fname: str):
    """
    Try to infer crop + measure from filename.
    Works with names like 'corn_production_county.csv', etc.
    Returns (crop_upper, measure_key_upper) or (None, None)
    """
    f = fname.lower()
    crop = None
    if "corn" in f: crop = "CORN"
    elif "soy" in f: crop = "SOYBEANS"
    elif "wheat" in f: crop = "WHEAT"

    meas = None
    if "production" in f: meas = "PRODUCTION"
    elif "area_harvested" in f or "area harvested" in f: meas = "AREA HARVESTED"
    elif "yield" in f: meas = "YIELD"

    return crop, meas

def yearly_sum_value(csv_path: Path, crop_upper=None, measure_upper=None):
    """
    Reads one QuickStats CSV and returns annual sum(Value) after filtering:
      commodity_desc == crop_upper (if provided)
      statisticcat_desc == measure_upper (if provided)
      freq_desc == ANNUAL (if present)
      (optionally source_desc == SURVEY if present)
    """
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.upper() for c in df.columns]

    # Column names used in QuickStats
    YEAR = "YEAR"
    VALUE = "VALUE"

    if YEAR not in df.columns or VALUE not in df.columns:
        raise ValueError(f"Missing YEAR/VALUE in {csv_path.name}")

    # Filters if those columns exist
    if "FREQ_DESC" in df.columns:
        df = df[df["FREQ_DESC"].str.upper().eq("ANNUAL")]

    if "SOURCE_DESC" in df.columns:
        # Survey is usually what you want, but if your file is already filtered, this is safe.
        df = df[df["SOURCE_DESC"].str.upper().eq("SURVEY")]

    if crop_upper and "COMMODITY_DESC" in df.columns:
        df = df[df["COMMODITY_DESC"].str.upper().eq(crop_upper)]

    if measure_upper and "STATISTICCAT_DESC" in df.columns:
        df = df[df["STATISTICCAT_DESC"].str.upper().eq(measure_upper)]

    out = df[[YEAR, VALUE]].copy()
    out["year"] = pd.to_numeric(out[YEAR], errors="coerce").astype("Int64")
    out["value"] = out[VALUE].map(to_num)

    out = out.dropna(subset=["year"])
    out = out.groupby("year", as_index=False)["value"].sum(min_count=1)
    out["year"] = out["year"].astype(int)
    return out.sort_values("year")

# ----------------------------
# 1) Scan folder & collect relevant series
# ----------------------------
found = []  # (crop, measure, path)
for p in QUICKSTATS_DIR.rglob("*.csv"):
    crop_u, meas_u = detect_crop_measure_from_filename(p.name)
    if crop_u in CROPS and meas_u in MEASURES:
        found.append((crop_u, meas_u, p))

if not found:
    raise RuntimeError(f"No crop/measure CSVs matched in {QUICKSTATS_DIR}.\n"
                       f"Expected filenames containing corn/soy/wheat + production/area_harvested/yield")

# Prefer "national" over "county" if you have both; else just use what exists.
# We'll group by (crop, measure) and pick the best by filename heuristic.
picked = {}
for crop_u, meas_u, p in found:
    key = (crop_u, meas_u)
    score = 0
    name = p.name.lower()
    if "national" in name: score += 3
    if "state" in name: score += 2
    if "county" in name: score += 1
    # Higher score = more aggregated (national best). If tied, newest modified file wins.
    meta = (score, p.stat().st_mtime)
    if key not in picked or meta > picked[key][0]:
        picked[key] = (meta, p)

# ----------------------------
# 2) Build crop table (production, area harvested, derived yield)
# ----------------------------
crop_tables = []
for crop_u in CROPS:
    # production + area harvested are required for derived yield
    prod_path = picked.get((crop_u, "PRODUCTION"), (None, None))[1]
    area_path = picked.get((crop_u, "AREA HARVESTED"), (None, None))[1]
    yld_path  = picked.get((crop_u, "YIELD"), (None, None))[1]

    if prod_path is None or area_path is None:
        raise RuntimeError(f"Missing production or area_harvested file for {crop_u}. "
                           f"Found keys: {list(picked.keys())}")

    prod = yearly_sum_value(prod_path, crop_upper=crop_u, measure_upper="PRODUCTION") \
        .rename(columns={"value": f"{crop_u.lower()}_production"})
    area = yearly_sum_value(area_path, crop_upper=crop_u, measure_upper="AREA HARVESTED") \
        .rename(columns={"value": f"{crop_u.lower()}_area_harvested"})

    t = prod.merge(area, on="year", how="outer").sort_values("year")
    t[f"{crop_u.lower()}_yield_derived"] = t[f"{crop_u.lower()}_production"] / (
        t[f"{crop_u.lower()}_area_harvested"].replace(0, np.nan) + EPS_DIV
    )

    # optional: keep provided yield (for diagnostics only)
    if yld_path is not None:
        yld = yearly_sum_value(yld_path, crop_upper=crop_u, measure_upper="YIELD") \
            .rename(columns={"value": f"{crop_u.lower()}_yield_sum_not_used"})
        t = t.merge(yld, on="year", how="left")

    crop_tables.append(t)

crops = crop_tables[0]
for t in crop_tables[1:]:
    crops = crops.merge(t, on="year", how="outer")
crops = crops.sort_values("year")

# ----------------------------
# 3) Load FSI target + merge
# ----------------------------
fsi = pd.read_csv(FSI_CSV)
fsi.columns = [c.lower() for c in fsi.columns]
if "year" not in fsi.columns or "fsi_geomean" not in fsi.columns:
    raise ValueError("FSI file must contain columns: year, FSI_geomean")

fsi = fsi.rename(columns={"fsi_geomean": "FSI_geomean"})
fsi["year"] = pd.to_numeric(fsi["year"], errors="coerce").astype("Int64")
fsi = fsi.dropna(subset=["year"]).copy()
fsi["year"] = fsi["year"].astype(int)
fsi = fsi.sort_values("year")

# targets
fsi["FSI_lag1"] = fsi["FSI_geomean"].shift(1)
fsi["dFSI"] = fsi["FSI_geomean"] - fsi["FSI_lag1"]

df = fsi.merge(crops, on="year", how="inner").sort_values("year")

y0, y1 = KEEP_YEARS
df = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()

# ----------------------------
# 4) Shocks + lags
# ----------------------------
base_features = []
for crop_u in CROPS:
    c = crop_u.lower()
    base_features += [f"{c}_production", f"{c}_area_harvested", f"{c}_yield_derived"]

# YoY shocks
for col in base_features:
    df[f"{col}_yoy"] = pct_change_safe(df[col])

# Lags for levels and shocks
for L in LAGS:
    for col in base_features:
        df[f"{col}_lag{L}"] = df[col].shift(L)
        df[f"{col}_yoy_lag{L}"] = df[f"{col}_yoy"].shift(L)
    df[f"FSI_lag{L}"] = df["FSI_geomean"].shift(L)
    df[f"dFSI_lag{L}"] = df["dFSI"].shift(L)

# ----------------------------
# 5) Save
# ----------------------------
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("\nPicked files:")
for (crop_u, meas_u), (meta, p) in sorted(picked.items()):
    print(f"  {crop_u:8s} {meas_u:14s} -> {p}")

print("\nRows:", len(df), "Years:", df["year"].min(), "to", df["year"].max())
print("\nMissing counts (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\nPreview:")
cols_preview = ["year", "FSI_geomean", "FSI_lag1", "dFSI"] + base_features
print(df[cols_preview].head(12))
