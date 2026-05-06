import pandas as pd
from pathlib import Path

# ----------------------------
# INPUTS (edit NEW_DATA_PATH)
# ----------------------------
MACRO_SOURCE_PATH = Path(r"E:\1Cathy\hsbdc\AI2026\data2\us_quickstats_fsi_plus_macro_with_shocks_lags.csv")

# <-- set this to your "new data" csv you want to enrich
NEW_DATA_PATH = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")

OUT_PATH = NEW_DATA_PATH.with_name(NEW_DATA_PATH.stem + "_plus_macro_cpi_pinksheet.csv")

# ----------------------------
# Load
# ----------------------------
macro = pd.read_csv(MACRO_SOURCE_PATH)
newdf = pd.read_csv(NEW_DATA_PATH)

# Make sure year is clean ints
macro["year"] = pd.to_numeric(macro["year"], errors="coerce").astype("Int64")
newdf["year"] = pd.to_numeric(newdf["year"], errors="coerce").astype("Int64")

# ----------------------------
# Choose macro columns to bring over
# (auto-detect CPI + PinkSheet-ish commodity series + their lags/yoy)
# ----------------------------
base_series = [
    "cpi_food_index",
    "cpi_food_yoy_pct",
    "CRUDE_PETRO",
    "MAIZE",
    "SOYBEANS",
    "WHEAT_US_HRW",
    "WHEAT_US_SRW",
    "us_population",
]

def is_series_related(col: str) -> bool:
    # bring base series + engineered variants (lags/yoy_pct)
    for b in base_series:
        if col == b:
            return True
        # match patterns like "MAIZE_lag1", "MAIZE_yoy_pct", "cpi_food_index_lag2", etc.
        if col.startswith(b + "_"):
            return True
    return False

macro_keep_cols = ["year"] + [c for c in macro.columns if is_series_related(c)]

# If macro has duplicates by year, collapse (shouldn’t, but just in case)
macro_keep = (
    macro[macro_keep_cols]
    .dropna(subset=["year"])
    .groupby("year", as_index=False)
    .mean(numeric_only=True)
)

# ----------------------------
# Merge into new data (LEFT merge keeps all your new rows)
# ----------------------------
merged = newdf.merge(macro_keep, on="year", how="left")

# Optional: sanity printouts
print("New data rows:", len(newdf), "Merged rows:", len(merged))
print("Added macro cols:", [c for c in macro_keep.columns if c != "year"])

# ----------------------------
# Save
# ----------------------------
merged.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
