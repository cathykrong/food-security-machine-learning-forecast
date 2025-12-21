# Clean CRU Climate Portal "timeseries" exports (wide -> long) into a merge-ready table:
# Output columns: iso3 | year | temp_c
#
# Works for files shaped like:
# code | name | 1901-07 | 1902-07 | ...
#
# Usage:
#   python clean_climate.py
#
# Requirements:
#   pip install pandas openpyxl

import re
import pandas as pd
from pathlib import Path

INPUT_XLSX = "E:/1Cathy/hsbdc 202526/data/tas_1901_2023.xlsx"     # <-- change if needed
SHEET_NAME = 0                        # 0 = first sheet, or put the sheet name string
OUT_CSV = "E:/1Cathy/hsbdc 202526/data/cleaned_test/temp_c_cleaned.csv"

# Optional filters (recommended for your project)
KEEP_ISO3 = {"BRA", "FRA", "IND", "NGA", "USA"}
YEAR_MIN, YEAR_MAX = 1901, 2023       # you can later subset to 1970–2022 or 2000–2022

def extract_year(col_name: str) -> int | None:
    """
    Extracts the year from column headers like '1901-07' or '1901'.
    Returns None if no year found.
    """
    m = re.match(r"^(\d{4})(?:[-_/].*)?$", str(col_name).strip())
    return int(m.group(1)) if m else None

def clean_climate_wide_to_long(
    xlsx_path: str | Path,
    sheet_name=0,
    value_col_name="temp_c",
    keep_iso3: set[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Normalize expected columns
    # Some exports use 'code'/'name', others might use 'Code'/'Name'
    cols_lower = {c.lower(): c for c in df.columns}
    code_col = cols_lower.get("code") or cols_lower.get("iso3") or cols_lower.get("country code")
    name_col = cols_lower.get("name") or cols_lower.get("country") or cols_lower.get("country name")

    if not code_col:
        raise ValueError(f"Couldn't find a country code column in: {list(df.columns)}")

    # Identify year columns (e.g., 1901-07)
    year_cols = []
    year_map = {}  # original col -> int year
    for c in df.columns:
        y = extract_year(c)
        if y is not None:
            year_cols.append(c)
            year_map[c] = y

    if not year_cols:
        raise ValueError("Couldn't find any year columns like '1901-07' in the sheet.")

    # Keep only necessary columns
    keep_cols = [code_col] + ([name_col] if name_col else []) + year_cols
    df = df[keep_cols].copy()

    # Rename code to iso3
    df = df.rename(columns={code_col: "iso3"})
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()

    # Optional: filter ISO3
    if keep_iso3:
        df = df[df["iso3"].isin(keep_iso3)].copy()

    # Melt wide -> long
    long_df = df.melt(
        id_vars=["iso3"] + (["name"] if name_col else []),
        value_vars=year_cols,
        var_name="year_raw",
        value_name=value_col_name,
    )

    # Convert year
    long_df["year"] = long_df["year_raw"].map(year_map).astype("Int64")
    long_df = long_df.drop(columns=["year_raw"])

    # Coerce values to numeric
    long_df[value_col_name] = pd.to_numeric(long_df[value_col_name], errors="coerce")

    # Optional year filter
    if year_min is not None:
        long_df = long_df[long_df["year"] >= year_min]
    if year_max is not None:
        long_df = long_df[long_df["year"] <= year_max]

    # Drop missing values
    long_df = long_df.dropna(subset=[value_col_name, "year"])

    # Keep only merge keys + value (drop name if present)
    long_df = long_df[["iso3", "year", value_col_name]].sort_values(["iso3", "year"]).reset_index(drop=True)

    return long_df

if __name__ == "__main__":
    out = clean_climate_wide_to_long(
        xlsx_path=INPUT_XLSX,
        sheet_name=SHEET_NAME,
        value_col_name="temp_c",      # change to "precip_mm" for precipitation file
        keep_iso3=KEEP_ISO3,          # set to None to keep all
        year_min=YEAR_MIN,
        year_max=YEAR_MAX,
    )

    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(out.head(10))
