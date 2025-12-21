import pandas as pd
from pathlib import Path

# ---- YOUR PATHS (Windows-safe) ----
INPUT_XLSX = Path(r"E:\1Cathy\hsbdc 202526\data\naturaldisasters_1901_2023.xlsx")
SHEET_NAME = 0
OUT_CSV = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\natural_disasters_cleaned.csv")

# ---- SETTINGS ----
KEEP_ISO3 = {"BRA", "FRA", "IND", "NGA", "USA"}   # set to None to keep all
YEAR_MIN, YEAR_MAX = 1901, 2023

# EM-DAT "Disaster Type" values to keep (edit if your file uses slightly different labels)
CLIMATE_DISASTER_TYPES = {
    "Drought",
    "Flood",
    "Storm",
    "Extreme temperature",
    "Wildfire",
    # optional:
    # "Landslide",
}

def clean_emdat_event_to_country_year(xlsx_path: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # required columns in your file (from your error message)
    for col in ["ISO", "Start Year", "Disaster Type"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}. Found: {list(df.columns)}")

    df["iso3"] = df["ISO"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["Start Year"], errors="coerce")

    # filter to your 5 countries
    if KEEP_ISO3:
        df = df[df["iso3"].isin(KEEP_ISO3)].copy()

    # filter to your year window
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # climate-related disasters only
    df["Disaster Type"] = df["Disaster Type"].astype(str).str.strip()
    df = df[df["Disaster Type"].isin(CLIMATE_DISASTER_TYPES)].copy()

    # choose affected column if you want it (optional but useful)
    affected_col = "Total Affected" if "Total Affected" in df.columns else None
    if affected_col:
        df[affected_col] = pd.to_numeric(df[affected_col], errors="coerce").fillna(0)
    else:
        df["__affected__"] = 0
        affected_col = "__affected__"

    # event count column (DisNo. is unique per event)
    event_id_col = "DisNo." if "DisNo." in df.columns else None
    if event_id_col:
        # Count distinct events in each country-year
        out = (
            df.groupby(["iso3", "year"], as_index=False)
              .agg(
                  disaster_count=(event_id_col, "nunique"),
                  total_affected_sum=(affected_col, "sum"),
              )
        )
    else:
        # Fallback: count rows
        out = (
            df.groupby(["iso3", "year"], as_index=False)
              .agg(
                  disaster_count=("iso3", "count"),
                  total_affected_sum=(affected_col, "sum"),
              )
        )

    out["year"] = out["year"].astype("Int64")
    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    return out

if __name__ == "__main__":
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cleaned = clean_emdat_event_to_country_year(INPUT_XLSX, SHEET_NAME)
    cleaned.to_csv(OUT_CSV, index=False)

    print(f"Saved to: {OUT_CSV}")
    print(cleaned.head(15))
