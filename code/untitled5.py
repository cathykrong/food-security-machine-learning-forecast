# Shock + lag sensitivity analysis (RandomForest on ΔFSI, reconstruct FSI)
# Uses your CSV: /mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv
# Train: 1970–2012, Test: 2013–2023

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

# ---------------------------
# 1) LOAD
# ---------------------------
import pandas as pd

PATH = r"E:/1Cathy/hsbdc/AI2026/data2/us_quickstats_fsi_plus_macro_with_shocks_lags.csv"
df = pd.read_csv(PATH).sort_values("year").copy()


# ---------------------------
# 2) SPLIT YEARS
# ---------------------------
TRAIN_END = 2012
TEST_START = 2013
TEST_END = 2023

# ---------------------------
# 3) COLUMN GROUPS (auto-detected)
#    We exclude any internal FSI component columns if present, and any FSI columns
#    other than FSI_lag1 (needed for reconstruction / persistence baseline).
# ---------------------------
exclude_prefixes = ("A_", "X_", "U_", "S_")  # if you have internal FSI component cols
pop_cols = [c for c in df.columns if "population" in c.lower()]  # drop if missingness hurts

def exclude_col(c: str) -> bool:
    if c in {"year", "FSI_geomean", "dFSI"}:
        return True
    if c.startswith(exclude_prefixes):
        return True
    if ("FSI" in c) and (c != "FSI_lag1"):
        return True
    if c in pop_cols:  # optional: remove if your population cols have NAs in 2013–2023
        return True
    return False

usable = [c for c in df.columns if not exclude_col(c)]

# base levels = not lag, not shock
base_levels = [
    c for c in usable
    if (not c.endswith("_lag1"))
    and (not c.endswith("_lag2"))
    and (not c.endswith("_yoy_pct"))
    and ("_yoy_pct_" not in c)
]
# ensure FSI_lag1 is included and first
base_levels = ["FSI_lag1"] + [c for c in base_levels if c != "FSI_lag1"]

# shocks and lags
shock_cols = [c for c in usable if c.endswith("_yoy_pct") and "_lag" not in c]
shock_lag1 = [c for c in usable if c.endswith("_yoy_pct_lag1")]
shock_lag2 = [c for c in usable if c.endswith("_yoy_pct_lag2")]
lag1_cols  = [c for c in usable if c.endswith("_lag1") and not c.endswith("_yoy_pct_lag1") and c != "FSI_lag1"]
lag2_cols  = [c for c in usable if c.endswith("_lag2") and not c.endswith("_yoy_pct_lag2")]

def build_features(shocks_on: bool, lag_depth: int, lagged_shocks_on: bool) -> list[str]:
    feats = list(base_levels)
    if shocks_on:
        feats += shock_cols
    if lag_depth >= 1:
        feats += lag1_cols
        if shocks_on and lagged_shocks_on:
            feats += shock_lag1
    if lag_depth >= 2:
        feats += lag2_cols
        if shocks_on and lagged_shocks_on:
            feats += shock_lag2

    # de-duplicate while keeping order
    seen = set()
    feats = [f for f in feats if not (f in seen or seen.add(f))]
    return feats

# ---------------------------
# 4) MODEL (match your main RF style; tweak if desired)
# ---------------------------
rf_params = dict(
    n_estimators=1000,
    random_state=42,
    max_features="sqrt",
    max_depth=3,
    min_samples_leaf=2,
    min_samples_split=4,
)

def eval_config(name: str, shocks_on: bool, lag_depth: int, lagged_shocks_on: bool) -> dict:
    feats = build_features(shocks_on, lag_depth, lagged_shocks_on)

    required = ["year", "dFSI", "FSI_geomean", "FSI_lag1"] + feats
    dsub = df.dropna(subset=required).sort_values("year")

    train = dsub[dsub["year"] <= TRAIN_END]
    test  = dsub[(dsub["year"] >= TEST_START) & (dsub["year"] <= TEST_END)]

    Xtr, ytr = train[feats], train["dFSI"]
    Xte, yte = test[feats], test["dFSI"]

    rf = RandomForestRegressor(**rf_params)
    rf.fit(Xtr, ytr)

    # predict ΔFSI then reconstruct FSI
    d_pred = rf.predict(Xte)
    fsi_pred = test["FSI_lag1"].to_numpy() + d_pred
    fsi_true = test["FSI_geomean"].to_numpy()

    rmse = np.sqrt(mean_squared_error(fsi_true, fsi_pred))
    mae  = mean_absolute_error(fsi_true, fsi_pred)
    r2   = r2_score(fsi_true, fsi_pred)

    # optional directional stats: ΔFSI > 0
    y_true = (yte.to_numpy() > 0).astype(int)
    y_hat  = (d_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    bal = (tpr + tnr) / 2 if np.isfinite(tpr) and np.isfinite(tnr) else np.nan

    return dict(
        Configuration=name,
        Shocks="On" if shocks_on else "Off",
        Lag_depth=lag_depth,
        Lagged_shocks="On" if lagged_shocks_on else "Off",
        n_features=len(feats),
        RMSE=rmse,
        MAE=mae,
        R2=r2,
        Balanced_accuracy=bal,
        Test_N=len(test),
    )

# ---------------------------
# 5) RUN SENSITIVITY GRID
# ---------------------------
configs = [
    ("No shocks + no lags", False, 0, False),
    ("Shocks only", True, 0, False),
    ("Lag1 only", False, 1, False),
    ("Shocks + lag1", True, 1, True),
    ("Shocks + lag1+lag2", True, 2, True),
    ("Shocks + lag1+lag2 (no lagged shocks)", True, 2, False),
]

rows = [eval_config(*cfg) for cfg in configs]

# persistence baseline: FSI_hat(t) = FSI(t-1)
test = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END)].dropna(subset=["FSI_geomean", "FSI_lag1"])
rows.append(dict(
    Configuration="Persistence baseline (FSI(t)=FSI(t-1))",
    Shocks="—",
    Lag_depth="—",
    Lagged_shocks="—",
    n_features=0,
    RMSE=np.sqrt(mean_squared_error(test["FSI_geomean"], test["FSI_lag1"])),
    MAE=mean_absolute_error(test["FSI_geomean"], test["FSI_lag1"]),
    R2=r2_score(test["FSI_geomean"], test["FSI_lag1"]),
    Balanced_accuracy=np.nan,
    Test_N=len(test),
))

out = pd.DataFrame(rows).sort_values("RMSE")
print(out.to_string(index=False))

# If you want to save:
out.to_csv(r"E:/1Cathy/hsbdc/AI2026/data2/shock_lag_sensitivity_results.csv", index=False)
