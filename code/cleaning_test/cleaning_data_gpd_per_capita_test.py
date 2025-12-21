# Clean World Bank GDP per capita (constant USD) to long format
# Input: API_NY.GDP.PCAP.KD_DS2_en_csv_v2_141.csv
# Output: gdp_per_capita_long_clean.csv
#
# Requirements:
#   pip install pandas

import pandas as pd
from pathlib import Path

# ---- FILE PATHS ----
INPUT_CSV = Path(r"E:\1Cathy\hsbdc 202526\data\gpdpercapita_1960_2023\API_NY.GDP.PCAP.KD_DS2_en_csv_v2_141.csv")
OUT_CSV = r"E:\1Cathy\hsbdc 202526\data\cleaned_test\gdp_per_capita_cleaned.csv"

# ---- SETTINGS ----
KEEP_ISO3 = {"USA", "FRA", "IND", "NGA", "BRA"}
YEAR_MIN, YEAR_MAX = 1960, 2023   # you can later subset to 2000–2022

def clean_gdp_per_capita(csv_path: Path) -> pd.DataFrame:
    # World Bank CSVs have 4 metadata rows at the top
    df = pd.read_csv(csv_path, skiprows=4)

    # Rename country code
    df = df.rename(columns={"Country Code": "iso3"})

    # Keep only the 5 countries
    df = df[df["iso3"].isin(KEEP_ISO3)].copy()

    # Identify year columns (1960, 1961, ...)
    year_cols = [c for c in df.columns if c.isdigit()]

    # Wide → long
    long_df = df.melt(
        id_vars=["iso3"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_per_capita",
    )

    # Clean types
    long_df["year"] = long_df["year"].astype(int)
    long_df["gdp_per_capita"] = pd.to_numeric(
        long_df["gdp_per_capita"], errors="coerce"
    )

    # Filter year range
    long_df = long_df[
        (long_df["year"] >= YEAR_MIN) & (long_df["year"] <= YEAR_MAX)
    ]

    # Drop missing values
    long_df = long_df.dropna(subset=["gdp_per_capita"])

    # Sort & keep final columns
    long_df = (
        long_df[["iso3", "year", "gdp_per_capita"]]
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )

    return long_df

if __name__ == "__main__":
    cleaned = clean_gdp_per_capita(INPUT_CSV)
    cleaned.to_csv(OUT_CSV, index=False)

    print(f"Saved to: {OUT_CSV}")
    print(cleaned.head(15))
