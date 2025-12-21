# Clean FAO undernourishment CSV where Year is a 3-year interval (e.g., "2003-2005")
# and convert to FIRST YEAR (e.g., 2003), then output merge-ready:
# iso3 | year | undernourishment_pct
#
# Works with FAOSTAT columns like:
#   Area Code (M49), Area, Year, Value
#
# Requirements:
#   pip install pandas

import re
import pandas as pd
from pathlib import Path

INPUT = Path(r"E:\1Cathy\hsbdc 202526\data\%undernourishment_2000_2023.csv")
OUTPUT = Path(r"E:\1Cathy\hsbdc 202526\data\cleaned_test\cleaned_undernourishment_firstyear.csv")

# Only keep these 5 countries
M49_TO_ISO3 = {
    76:  "BRA",
    250: "FRA",
    356: "IND",
    566: "NGA",
    840: "USA",
}

AREA_TO_ISO3 = {
    "Brazil": "BRA",
    "France": "FRA",
    "India": "IND",
    "Nigeria": "NGA",
    "United States of America": "USA",
    "United States": "USA",
}

def year_to_first_year(x) -> int | None:
    """Convert '2003-2005' (or '2003–2005') -> 2003; '2003' -> 2003."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.match(r"^(\d{4})\s*[–-]\s*(\d{4})$", s)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\d{4}", s):
        return int(s)
    return None

def clean_undernourishment_firstyear(inp: Path) -> pd.DataFrame:
    df = pd.read_csv(inp, encoding="utf-8-sig")

    required = ["Area Code (M49)", "Area", "Year", "Value"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

    # Build iso3 from M49 (preferred) then Area name (fallback)
    m49 = pd.to_numeric(df["Area Code (M49)"], errors="coerce")
    df["iso3"] = m49.map(M49_TO_ISO3)
    missing = df["iso3"].isna()
    df.loc[missing, "iso3"] = df.loc[missing, "Area"].map(AREA_TO_ISO3)

    # Keep only your 5 countries
    df = df[df["iso3"].isin(set(M49_TO_ISO3.values()))].copy()

    # Convert Year interval -> first year
    df["year"] = df["Year"].apply(year_to_first_year)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Value -> undernourishment %
    df["undernourishment_pct"] = pd.to_numeric(df["Value"], errors="coerce")

    out = (
        df[["iso3", "year", "undernourishment_pct"]]
        .dropna(subset=["iso3", "year", "undernourishment_pct"])
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )

    # If FAO provides duplicate rows per iso3-year (rare), average them
    out = out.groupby(["iso3", "year"], as_index=False)["undernourishment_pct"].mean()

    return out

if __name__ == "__main__":
    cleaned = clean_undernourishment_firstyear(INPUT)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT, index=False)

    print(f"Saved -> {OUTPUT}")
    print(cleaned.head(12))
    print("\nCoverage by country:")
    print(cleaned.groupby("iso3")["year"].agg(["min", "max", "count"]))
