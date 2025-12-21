# Stitch (merge) all your cleaned datasets into ONE master panel table
# Output: final_master_dataset.csv
#
# Inputs (already cleaned, long format):
#  - temp_c_cleaned.csv              -> iso3, year, temp_c
#  - precip_mm_cleaned.csv           -> iso3, year, precip_mm
#  - natural_disasters_cleaned.csv   -> iso3, year, disaster_count (and maybe total_affected_sum)
#  - gdp_per_capita_cleaned.csv      -> iso3, year, gdp_per_capita
#  - master_faostat_cleaned.csv      -> iso3, year, (food/ag variables)
#  - cleaned_undernourishment_firstyear.csv -> iso3, year, undernourishment_pct  (NEW)
#
# Requirements:
#   pip install pandas

import pandas as pd
from pathlib import Path

# ---- INPUT PATHS ----
TEMP_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\temp_c_cleaned.csv")
PRECIP_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\precip_mm_cleaned.csv")
DISASTER_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\natural_disasters_cleaned.csv")
GDP_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\gdp_per_capita_cleaned.csv")
FAOSTAT_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\master_faostat_cleaned.csv")

# NEW: undernourishment (first-year version)
UNDERNOURISH_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\cleaned_undernourishment_firstyear.csv")

OUT_PATH = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\final_master_dataset.csv")

# ---- SETTINGS ----
KEEP_ISO3 = {"BRA", "FRA", "IND", "NGA", "USA"}  # set to None to keep all
YEAR_MIN, YEAR_MAX = 2000, 2022                 # undernourishment coverage window (typical)

def read_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize merge keys
    if "iso3" not in df.columns or "year" not in df.columns:
        raise ValueError(f"{path.name} must contain 'iso3' and 'year' columns. Found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Drop rows missing keys
    df = df.dropna(subset=["iso3", "year"])

    # Deduplicate (average numeric columns if duplicates exist)
    non_key_cols = [c for c in df.columns if c not in ("iso3", "year")]
    if df.duplicated(subset=["iso3", "year"]).any():
        agg = {}
        for c in non_key_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                agg[c] = "mean"
            else:
                agg[c] = "first"
        df = df.groupby(["iso3", "year"], as_index=False).agg(agg)

    return df

def build_spine(iso3_set: set[str], year_min: int, year_max: int) -> pd.DataFrame:
    years = list(range(year_min, year_max + 1))
    spine = pd.DataFrame([(c, y) for c in sorted(iso3_set) for y in years], columns=["iso3", "year"])
    spine["year"] = spine["year"].astype("Int64")
    return spine

def main():
    temp = read_clean(TEMP_PATH)
    precip = read_clean(PRECIP_PATH)
    disasters = read_clean(DISASTER_PATH)
    gdp = read_clean(GDP_PATH)
    fao = read_clean(FAOSTAT_PATH)
    under = read_clean(UNDERNOURISH_PATH)  # NEW

    # Decide which countries to keep
    if KEEP_ISO3:
        iso3_set = KEEP_ISO3
    else:
        iso3_set = set(pd.concat([
            temp["iso3"], precip["iso3"], disasters["iso3"], gdp["iso3"], fao["iso3"], under["iso3"]
        ]).unique())

    # Create a complete country-year spine
    master = build_spine(iso3_set, YEAR_MIN, YEAR_MAX)

    # Left-join everything onto the spine
    master = master.merge(temp, on=["iso3", "year"], how="left")
    master = master.merge(precip, on=["iso3", "year"], how="left")
    master = master.merge(disasters, on=["iso3", "year"], how="left")
    master = master.merge(gdp, on=["iso3", "year"], how="left")
    master = master.merge(fao, on=["iso3", "year"], how="left")
    master = master.merge(under, on=["iso3", "year"], how="left")  # NEW

    # OPTIONAL: For disasters, blank = 0 (if you interpret missing as no events that year)
    if "disaster_count" in master.columns:
        master["disaster_count"] = master["disaster_count"].fillna(0)

    # Sort and save
    master = master.sort_values(["iso3", "year"]).reset_index(drop=True)
    master.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH.resolve()}")
    print("Columns:", list(master.columns))
    print(master.head(10))

    # Quick completeness check
    print("\nMissingness (%):")
    miss = (master.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(miss)

    # Quick undernourishment check
    if "undernourishment_pct" in master.columns:
        print("\nUndernourishment non-null count:", master["undernourishment_pct"].notna().sum())
        print(master[["iso3", "year", "undernourishment_pct"]].dropna().head(10))

if __name__ == "__main__":
    main()
