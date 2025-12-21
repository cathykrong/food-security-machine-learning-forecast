# Generalized FAOSTAT cleaner for your 4 files (and similar FAOSTAT exports)
# Produces clean long tables: iso3 | year | <value_col>
# Also writes one merged file (outer join) for convenience.
#
# Handles:
# - Files WITH ISO3 column (Area Code (ISO3), ISO, etc.)
# - Files WITHOUT ISO3 but WITH M49 (Area Code (M49)) using a fallback mapping
#
# Requirements:
#   pip install pandas

import pandas as pd
from pathlib import Path

# --------- CONFIG: set your file paths here (Windows: use r"..." raw strings) ----------
FILES = [
    (Path(r"E:\1Cathy\hsbdc 202526\data\cerealtotal_1961_2023.csv"), "cereal_total_yield"),
    (Path(r"E:\1Cathy\hsbdc 202526\data\fpi_1961_2023.csv"), "food_prod_index"),
    (Path(r"E:\1Cathy\hsbdc 202526\data\%undernourishment_2000_2023.csv"), "undernourishment_pct"),
]

OUT_DIR = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep only these countries (set to None to keep all)
KEEP_ISO3 = {"BRA", "FRA", "IND", "NGA", "USA"}

# Optional year bounds (leave wide; subset later)
YEAR_MIN, YEAR_MAX = 1900, 2100

# Fallback mapping for when ISO3 is missing (your undernourishment file uses M49)
M49_TO_ISO3 = {
    76:  "BRA",   # Brazil (often appears as 076)
    250: "FRA",   # France
    356: "IND",   # India
    566: "NGA",   # Nigeria
    840: "USA",   # United States of America
}

# Optional additional fallback by country name (in case M49 is missing/odd)
AREA_TO_ISO3 = {
    "Brazil": "BRA",
    "France": "FRA",
    "India": "IND",
    "Nigeria": "NGA",
    "United States of America": "USA",
    "United States": "USA",
}

# --------- Helpers ----------
def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def standardize_iso3(df: pd.DataFrame) -> pd.Series:
    """
    Return an ISO3 series. Priority:
    1) ISO3 column if present
    2) M49 numeric code mapping if present
    3) Area name mapping if present
    """
    iso3_col = find_col(df, [
        "Area Code (ISO3)", "area code (iso3)", "ISO3", "iso3",
        "ISO", "iso", "Country Code", "country code", "Code", "code"
    ])

    if iso3_col:
        s = df[iso3_col].astype(str).str.strip().str.upper()
        # FAOSTAT ISO sometimes includes blanks; keep only 3-letter where possible
        return s

    # Try M49
    m49_col = find_col(df, ["Area Code (M49)", "area code (m49)", "M49", "m49"])
    if m49_col:
        m49 = pd.to_numeric(df[m49_col], errors="coerce")
        iso3 = m49.map(M49_TO_ISO3)
        # fallback to Area name if still missing
        area_col = find_col(df, ["Area", "area", "Country", "country"])
        if area_col:
            iso3 = iso3.fillna(df[area_col].map(AREA_TO_ISO3))
        return iso3.astype(str).str.strip().str.upper()

    # Try Area name only
    area_col = find_col(df, ["Area", "area", "Country", "country"])
    if area_col:
        iso3 = df[area_col].map(AREA_TO_ISO3)
        return iso3.astype(str).str.strip().str.upper()

    raise ValueError("Could not infer ISO3: no ISO3 column, no M49, and no Area/Country name column.")

def clean_faostat(csv_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    year_col = find_col(df, ["Year", "year", "Time", "time", "Year Code", "year code"])
    value_col = find_col(df, ["Value", "value", "Value (numeric)", "value (numeric)"])

    if not year_col or not value_col:
        raise ValueError(
            f"[{csv_path.name}] Missing required columns. "
            f"Need year + value. Found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["iso3"] = standardize_iso3(df)
    out["year"] = pd.to_numeric(df[year_col], errors="coerce")
    out[value_name] = pd.to_numeric(df[value_col], errors="coerce")

    # Filter: valid iso3 (3 letters) and optional country list
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    out = out[out["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()

    if KEEP_ISO3:
        out = out[out["iso3"].isin(KEEP_ISO3)].copy()

    # Year filters
    out = out[(out["year"] >= YEAR_MIN) & (out["year"] <= YEAR_MAX)].copy()
    out["year"] = out["year"].astype("Int64")

    # Drop missing
    out = out.dropna(subset=["iso3", "year", value_name])

    # Deduplicate if needed (sometimes FAOSTAT has repeated rows with same keys)
    out = (
        out.groupby(["iso3", "year"], as_index=False)[value_name]
           .mean()
           .sort_values(["iso3", "year"])
           .reset_index(drop=True)
    )

    return out

def main():
    cleaned = {}
    for path, value_name in FILES:
        df = clean_faostat(path, value_name=value_name)
        cleaned[value_name] = df

        out_path = OUT_DIR / f"cleaned_{value_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path} | rows={len(df)}")
        print(df.head(3), "\n")

    # Merge all (outer join keeps all country-years across variables)
    master = None
    for value_name, df in cleaned.items():
        master = df if master is None else master.merge(df, on=["iso3", "year"], how="outer")

    master = master.sort_values(["iso3", "year"]).reset_index(drop=True)
    master_path = OUT_DIR / "master_faostat_cleaned.csv"
    master.to_csv(master_path, index=False)
    print(f"Saved -> {master_path} | rows={len(master)}")

if __name__ == "__main__":
    main()
