import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# INPUTS (your uploaded files)
# ----------------------------
FBS_OLD = Path("E:/1Cathy/hsbdc/AI2026/data2/fbsh1.csv")                       # kcal/cap/d (old methodology)
FBS_NEW = Path("E:/1Cathy/hsbdc/AI2026/data2/fbsh2.csv")                       # kcal/cap/d (2010-)
GDP_IN  = Path("E:/1Cathy/hsbdc/AI2026/data2/gdp.csv")                         # GDP per capita (USD)
TEMP_IN = Path("E:/1Cathy/hsbdc/AI2026/data2/FAOSTAT_data_en_1-8-2026.csv")    # Temperature change on land (°C)
FPI_IN  = Path("E:/1Cathy/hsbdc/AI2026/data2/FAOSTAT_data_en_1-4-2026 (1).csv")# Food production index

OUT     = Path("E:/1Cathy/hsbdc/AI2026/data3/usa_4vars_merged_1961_2023.csv")

# ----------------------------
# Helpers
# ----------------------------
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def pick_series(df, year_col="Year", value_col="Value"):
    out = df[[year_col, value_col]].copy()
    out["year"] = out[year_col].astype(int)
    out["value"] = to_num(out[value_col])
    return out[["year", "value"]]

# ----------------------------
# 1) K(t): Dietary energy supply (kcal/cap/day) from Food Balances
#     - fbsh1 covers earlier years, fbsh2 covers 2010+ (overlap handled)
#     - prefer fbsh2 when years overlap
# ----------------------------
fbs1 = pd.read_csv(FBS_OLD)
fbs2 = pd.read_csv(FBS_NEW)

# These files are already only the needed series, but keep filters for robustness
fbs1 = fbs1[(fbs1["Element"] == "Food supply (kcal/capita/day)") & (fbs1["Item"] == "Grand Total")].copy()
fbs2 = fbs2[(fbs2["Element"] == "Food supply (kcal/capita/day)") & (fbs2["Item"] == "Grand Total")].copy()

k1 = pick_series(fbs1).rename(columns={"value": "kcal_supply_kcal_cap_day"})
k2 = pick_series(fbs2).rename(columns={"value": "kcal_supply_kcal_cap_day"})

k1["src_rank"] = 0   # old
k2["src_rank"] = 1   # new (preferred)

kcal = (
    pd.concat([k1, k2], ignore_index=True)
      .sort_values(["year", "src_rank"])
      .drop_duplicates(subset=["year"], keep="last")
      .drop(columns=["src_rank"])
      .sort_values("year")
)

# ----------------------------
# 2) G(t): GDP per capita (USD)
# ----------------------------
gdp = pd.read_csv(GDP_IN)
gdp = gdp[(gdp["Element"] == "Value US$ per capita") & (gdp["Item"] == "Gross Domestic Product")].copy()

gdp = pick_series(gdp).rename(columns={"value": "gdp_pc_usd"})

# ----------------------------
# 3) F(t): Food production index
# ----------------------------
fpi = pd.read_csv(FPI_IN)
fpi = fpi[(fpi["Item"] == "Food") &
          (fpi["Element"] == "Gross Production Index Number (2014-2016 = 100)")].copy()

fpi = pick_series(fpi).rename(columns={"value": "food_prod_index"})

# ----------------------------
# 4) T(t): Temperature change on land (°C), meteorological year
# ----------------------------
temp = pd.read_csv(TEMP_IN)
temp = temp[(temp["Element"] == "Temperature change") &
            (temp["Months"] == "Meteorological year")].copy()

temp = pick_series(temp).rename(columns={"value": "temp_change_c"})

# ----------------------------
# Merge all 4 on year (outer join to keep full coverage)
# ----------------------------
merged = kcal.merge(gdp, on="year", how="outer") \
             .merge(fpi, on="year", how="outer") \
             .merge(temp, on="year", how="outer") \
             .sort_values("year")

# Optional: keep common range (1961–2023) since FPI ends 2023 in your file
merged = merged[(merged["year"] >= 1961) & (merged["year"] <= 2023)].copy()

# Save
merged.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(merged), "Years:", merged["year"].min(), "to", merged["year"].max())
print("Missing counts:\n", merged.isna().sum())
print("\nHead:\n", merged.head(10))
print("\nTail:\n", merged.tail(10))
