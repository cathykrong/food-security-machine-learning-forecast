# FSI time-series plot: Actual vs RandomForest (ΔFSI) vs Persistence (2013–2023)
# - Uses your merged dataset: us_quickstats_fsi_plus_macro_1970_2023.csv
# - Rebuilds shocks + lags exactly like your benchmark
# - Trains RandomForest on ΔFSI (train < 2013), reconstructs FSI_pred = FSI_lag1 + dFSI_pred
# - Adds customizable legend (labels, location, font size, frame, etc.)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# ----------------------------
# USER SETTINGS (edit these)
# ----------------------------
DATA_PATH = Path("E:/1Cathy/hsbdc/AI2026/data2/us_quickstats_fsi_plus_macro_1970_2023.csv")  # change if needed
TARGET = "FSI_geomean"
PILLAR_COLS = ["A_availability", "X_access", "U_utilization", "S_stability"]

CUTOFF_YEAR = 2013  # train < 2013, test >= 2013

# RandomForest hyperparams (match your benchmark)
RF_PARAMS = dict(
    random_state=42,
    n_estimators=400,
    min_samples_leaf=2,
    max_features="sqrt",
)

# Legend customization
LEGEND_CFG = dict(
    labels={
        "actual": "Actual FSI",
        "rf": "RandomForest (ΔFSI)",
        "persist": "Persistence (FSI(t−1))",
    },
    loc="best",          # e.g., "upper left", "upper right", "lower left", "best"
    fontsize=11,
    frameon=True,
    title=None,          # e.g., "Legend"
    ncol=1,
)

# Plot cosmetics (no explicit colors required; matplotlib will cycle defaults)
TITLE = "Observed vs Predicted FSI Over Time: Random Forest (ΔFSI) vs Persistence (2013–2023)"
XLABEL = "Year"
YLABEL = "FSI (0–1)"
MARKER = "o"
ROTATE_XTICKS = 45

# Optional save
SAVE_FIG = True
OUT_PNG = Path("fig_fsi_timeseries_2013_2023.png")
DPI = 250

# ----------------------------
# Helpers
# ----------------------------
def add_features(df_in: pd.DataFrame, cols, lags=(1, 2), add_yoy=True) -> pd.DataFrame:
    """Add lag1/lag2 and YoY% features for each column in cols."""
    out = df_in.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
        if add_yoy:
            out[f"{c}_yoy_pct"] = out[c].pct_change() * 100.0
    return out

# ----------------------------
# Load and engineer features
# ----------------------------
df = pd.read_csv(DATA_PATH).sort_values("year").reset_index(drop=True)

# Ensure numeric
for c in df.columns:
    if c != "year":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Identify predictors (exclude pillars + target)
base_pred_cols = [c for c in df.columns if c not in ["year", TARGET] + PILLAR_COLS]
crop_prefixes = ("corn_", "wheat_", "soybeans_")
crop_cols = [c for c in base_pred_cols if c.startswith(crop_prefixes)]
macro_cols = [c for c in base_pred_cols if c not in crop_cols]

# Create lag target and delta target
df["FSI_lag1"] = df[TARGET].shift(1)
df["dFSI"] = df[TARGET] - df["FSI_lag1"]

# Add shocks + lags
df_fe = add_features(df, crop_cols, lags=(1, 2), add_yoy=True)
df_fe = add_features(df_fe, macro_cols, lags=(1, 2), add_yoy=True)

# Drop rows that cannot form dFSI
df_fe = df_fe.dropna(subset=["FSI_lag1", "dFSI"]).reset_index(drop=True)

# Build the exact feature list: original + lag1 + lag2 + yoy_pct (for each original col)
crop_fe_cols = []
for c in crop_cols:
    for feat in [c, f"{c}_lag1", f"{c}_lag2", f"{c}_yoy_pct"]:
        if feat in df_fe.columns:
            crop_fe_cols.append(feat)

macro_fe_cols = []
for c in macro_cols:
    for feat in [c, f"{c}_lag1", f"{c}_lag2", f"{c}_yoy_pct"]:
        if feat in df_fe.columns:
            macro_fe_cols.append(feat)

X_cols = ["year"] + crop_fe_cols + macro_fe_cols

# ----------------------------
# Train/test split
# ----------------------------
train = df_fe[df_fe["year"] < CUTOFF_YEAR].copy()
test = df_fe[df_fe["year"] >= CUTOFF_YEAR].copy()

X_train = train[X_cols]
y_train_delta = train["dFSI"].values

X_test = test[X_cols]
years = test["year"].values
fsi_actual = test[TARGET].values
fsi_lag1 = test["FSI_lag1"].values

# ----------------------------
# Fit RF on ΔFSI and reconstruct level FSI
# ----------------------------
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i = imp.transform(X_test)

rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(X_train_i, y_train_delta)

dFSI_pred = rf.predict(X_test_i)
fsi_pred = fsi_lag1 + dFSI_pred

# Persistence baseline in level
fsi_persist = fsi_lag1

# ----------------------------
# Plot
# ----------------------------
YEAR_TICK_STEP = 1   # set to 1 for every year, 2 for every 2 years, etc.

plt.figure(figsize=(9, 5.5))

plt.plot(years, fsi_actual, marker=MARKER, label=LEGEND_CFG["labels"]["actual"])
plt.plot(years, fsi_pred, marker=MARKER, label=LEGEND_CFG["labels"]["rf"])
plt.plot(years, fsi_persist, marker=MARKER, label=LEGEND_CFG["labels"]["persist"])

plt.title(TITLE)
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)

# ---- axis controls ----
plt.ylim(bottom=0)          # y-axis starts at 0
plt.xlim(years.min(), years.max())

# ---- nicer x ticks ----
xticks = np.arange(int(years.min()), int(years.max()) + 1, YEAR_TICK_STEP)
plt.xticks(xticks, rotation=ROTATE_XTICKS)

plt.tight_layout()

plt.legend(
    loc=LEGEND_CFG["loc"],
    fontsize=LEGEND_CFG["fontsize"],
    frameon=LEGEND_CFG["frameon"],
    title=LEGEND_CFG["title"],
    ncol=LEGEND_CFG["ncol"],
)

if SAVE_FIG:
    plt.savefig(OUT_PNG, dpi=DPI)

plt.show()

