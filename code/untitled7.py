import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# USER SETTINGS
# ---------------------------
DATA_PATH = r"E:/1Cathy/hsbdc/AI2026/data2/us_quickstats_fsi_plus_macro_with_shocks_lags.csv"
TRAIN_END = 2012
TOP_K = 15

# ---------------------------
# 1) LOAD DATA
# ---------------------------
df = pd.read_csv(DATA_PATH).sort_values("year").copy()
df.columns = df.columns.map(str)  # safety

# ---------------------------
# 2) CHOOSE FEATURES (avoid leakage)
# ---------------------------
exclude_prefixes = ("A_", "X_", "U_", "S_")  # if your FSI pillar columns exist

def is_excluded(col: str) -> bool:
    if col in {"year", "FSI_geomean", "dFSI"}:
        return True
    if col.startswith(exclude_prefixes):
        return True
    # exclude any other FSI columns except FSI_lag1 (ok to include for ΔFSI model)
    if ("FSI" in col) and (col != "FSI_lag1"):
        return True
    return False

feature_cols = [c for c in df.columns if not is_excluded(c)]

# drop rows missing anything we need
need_cols = ["year", "dFSI"] + feature_cols
data = df.dropna(subset=need_cols).copy()

train = data[data["year"] <= TRAIN_END]
X_train = train[feature_cols]
y_train = train["dFSI"]

# ---------------------------
# 3) TRAIN RANDOM FOREST (ΔFSI)
# ---------------------------
rf = RandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    max_features="sqrt"
)
rf.fit(X_train, y_train)

# ---------------------------
# 4) TOP-K IMPORTANCES
# ---------------------------
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
top = imp.head(TOP_K).copy()

# ---------------------------
# 5) PRETTY LABELS (common-language y-axis)
# ---------------------------
import re

# Base names ONLY (no YoY/Lag text here)
SPECIAL = {
    "CRUDE_PETRO": "Crude oil price",
    "MAIZE": "Maize price",
    "SOYBEANS": "Soybean price",
    "WHEAT_US_HRW": "U.S. wheat price (HRW)",
    "WHEAT_US_SRW": "U.S. wheat price (SRW)",
    "iENERGY": "Energy price index",
    "iAGRICULTURE": "Agriculture price index",
    "iFOOD": "Food price index",
    "iGRAINS": "Grains price index",
    "iFERTILIZERS": "Fertilizers price index",
    "iMETMIN": "Metals & minerals price index",
    "cpi_food_index": "Food CPI (index)",
    "cpi_food_yoy_pct": "Food CPI inflation",  # base; suffix will add (YoY %)
}

CROP_PREFIX = {"corn": "Corn", "soybeans": "Soybeans", "wheat": "Wheat"}
CROP_METRIC = {
    "production": "production",
    "yield": "yield",
    "area_harvested": "harvested area",
    "area_planted": "planted area",
}

def pretty_feature(raw: str) -> str:
    name = str(raw)

    # ---- suffix rules you requested ----
    yoy_suffix = ""
    lag_suffix = ""

    # add "(YoY %)" if ends with _yoy_pct
    if name.endswith("_yoy_pct"):
        name = name[:-8]
        yoy_suffix = " (YoY %)"

    # add "(Lag)" if ends with _lag2
    if name.endswith("_lag2"):
        name = name[:-5]
        lag_suffix = " (Lag)"

    # (Optional) handle lag1 if you ever want it:
    # if name.endswith("_lag1"):
    #     name = name[:-5]
    #     lag_suffix = " (Lag)"

    # special mapping
    if name in SPECIAL:
        return f"{SPECIAL[name]}{yoy_suffix}{lag_suffix}"

    # crop mapping: corn_production, soybeans_yield, etc.
    m = re.match(r"^(corn|soybeans|wheat)_(production|yield|area_harvested|area_planted)$", name)
    if m:
        crop = CROP_PREFIX[m.group(1)]
        metric = CROP_METRIC[m.group(2)]
        return f"{crop} {metric}{yoy_suffix}{lag_suffix}"

    # fallback readable
    return f"{name.replace('_',' ').title()}{yoy_suffix}{lag_suffix}"


CROP_PREFIX = {"corn": "Corn", "soybeans": "Soybeans", "wheat": "Wheat"}
CROP_METRIC = {
    "production": "production",
    "yield": "yield",
    "area_harvested": "harvested area",
    "area_planted": "planted area",
}

def pretty_feature(raw: str) -> str:
    name = str(raw)

    # detect lag
    lag = ""
    if name.endswith("_lag1"):
        name = name[:-5]
        lag = " (t−1)"
    elif name.endswith("_lag2"):
        name = name[:-5]
        lag = " (t−2)"

    # detect shock (YoY %)
    shock = ""
    if name.endswith("_yoy_pct"):
        name = name[:-8]
        shock = " (YoY % change)"

    # direct special mapping
    if name in SPECIAL:
        return f"{SPECIAL[name]}{shock}{lag}"

    # crop engineered like corn_production, soybeans_yield, etc.
    m = re.match(r"^(corn|soybeans|wheat)_(production|yield|area_harvested|area_planted)$", name)
    if m:
        crop = CROP_PREFIX[m.group(1)]
        metric = CROP_METRIC[m.group(2)]
        return f"{crop} {metric}{shock}{lag}"

    # fallback: make readable
    return name.replace("_", " ").title() + shock + lag

pretty_names = [pretty_feature(f) for f in top.index]

# ---------------------------
# 6) PLOT (with y-axis title)
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(pretty_names[::-1], top.values[::-1])  # reverse so biggest is on top visually
ax.set_xlabel("Feature importance (Random Forest)")
ax.set_ylabel("Predictor")  # <-- y-axis title you asked for
ax.set_title("Top 15 Feature Importances (Random Forest ΔFSI)")
plt.tight_layout()

# save
OUT_PATH = "fig_feature_importance_top15_pretty.png"
plt.savefig(OUT_PATH, dpi=300)
plt.show()

print(f"Saved: {OUT_PATH}")
