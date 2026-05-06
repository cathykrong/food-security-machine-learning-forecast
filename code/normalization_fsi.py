import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
IN_CSV  = Path("E:/1Cathy/hsbdc/AI2026/data3/usa_4vars_merged_1961_2023.csv")
OUT_CSV = Path("E:/1Cathy/hsbdc/AI2026/data3/usa_4vars_normalized_geomeanFSI_1961_2023.csv")

# Columns in merged file
COL_K = "kcal_supply_kcal_cap_day"   # higher = better
COL_G = "gdp_pc_usd"                 # higher = better
COL_F = "food_prod_index"            # higher = better
COL_T = "temp_change_c"              # higher = worse -> invert after scaling risk

# Scale using training years only (avoids peeking)
TRAIN_END_YEAR = 2012

# Temperature risk definition:
#   False -> warming-only risk max(0, anomaly)
#   True  -> absolute anomaly risk abs(anomaly)
USE_ABS_TEMP = False

# Small epsilon so geometric mean never sees 0
EPS = 1e-6

# ----------------------------
# Helpers
# ----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def minmax_fit(s: pd.Series):
    s = to_num(s)
    mn = np.nanmin(s.values)
    mx = np.nanmax(s.values)
    return mn, mx

def minmax_apply(s: pd.Series, mn: float, mx: float):
    s = to_num(s)
    if not (np.isfinite(mn) and np.isfinite(mx)) or mx <= mn:
        return pd.Series(np.nan, index=s.index)
    return (s - mn) / (mx - mn)

def geomean_rows(df: pd.DataFrame, cols, eps=1e-6):
    X = df[cols].astype(float).to_numpy()
    # clip to avoid log(0) and negative
    X = np.clip(X, eps, None)
    ok = np.isfinite(X).all(axis=1)
    out = np.full(len(df), np.nan, dtype=float)
    out[ok] = np.exp(np.mean(np.log(X[ok]), axis=1))
    return out

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(IN_CSV)
df["year"] = df["year"].astype(int)

train_mask = df["year"] <= TRAIN_END_YEAR

# ----------------------------
# Build temp risk series
# ----------------------------
temp = to_num(df[COL_T])
if USE_ABS_TEMP:
    temp_risk = temp.abs()
else:
    temp_risk = temp.clip(lower=0)

# ----------------------------
# Fit min/max on TRAINING YEARS only
# ----------------------------
k_min, k_max = minmax_fit(df.loc[train_mask, COL_K])
g_min, g_max = minmax_fit(df.loc[train_mask, COL_G])
f_min, f_max = minmax_fit(df.loc[train_mask, COL_F])
t_min, t_max = minmax_fit(temp_risk.loc[train_mask])

# ----------------------------
# Apply scaling to ALL YEARS using training min/max
# ----------------------------
df["K_norm"] = minmax_apply(df[COL_K], k_min, k_max)  # higher better
df["G_norm"] = minmax_apply(df[COL_G], g_min, g_max)  # higher better
df["F_norm"] = minmax_apply(df[COL_F], f_min, f_max)  # higher better

risk_scaled  = minmax_apply(temp_risk, t_min, t_max)  # higher worse
df["T_norm"] = 1 - risk_scaled                        # higher better

# Keep norms in [0,1]
for c in ["K_norm", "G_norm", "F_norm", "T_norm"]:
    df[c] = df[c].clip(0, 1)

# ----------------------------
# Final FSI: geometric mean of 4 normalized pillars (no weights)
# ----------------------------
pillars = ["K_norm", "G_norm", "F_norm", "T_norm"]

# Avoid exact zeros for geomean
df[pillars] = df[pillars].clip(lower=EPS)

df["FSI_geomean"] = geomean_rows(df, pillars, eps=EPS)

# ----------------------------
# Save
# ----------------------------
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("\nTraining min/max used:")
print("  K:", (k_min, k_max))
print("  G:", (g_min, g_max))
print("  F:", (f_min, f_max))
print("  Temp_risk:", (t_min, t_max), "| USE_ABS_TEMP =", USE_ABS_TEMP)

print("\nPreview:")
print(df[["year", COL_K, "K_norm", COL_G, "G_norm", COL_F, "F_norm", COL_T, "T_norm", "FSI_geomean"]].head(15))
