"""
Change-based formulation (ΔFSI) with 3 models:
- RidgeCV
- RandomForestRegressor
- GradientBoostingRegressor

Goal:
1) Predict annual change: dFSI(t) = FSI_geomean(t) - FSI_geomean(t-1)
2) Reconstruct level:     FSI_pred(t) = FSI_geomean(t-1) + dFSI_pred(t)
3) Compute residuals:     residual(t) = FSI_geomean(t) - FSI_pred(t)   # Observed - Predicted

Split (time-respective):
Train: 1970–2012
Test : 2013–2023

Baseline (persistence level):
FSI_persist(t) = FSI_geomean(t-1)
"""

# from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ----------------------------
# CONFIG (edit)
# ----------------------------
DATA_PATH = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags_plus_macro_cpi_pinksheet.csv")
YEAR_COL = "year"
TARGET_COL = "FSI_geomean"

TRAIN_END = 2012
TEST_START = 2013
TEST_END = 2023

OUT_DIR = Path(r"E:\1Cathy\hsbdc\AI2026\results3\outputs_deltaFSI")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Metrics
# ----------------------------
def reg_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ----------------------------
# Feature selection (avoid leakage)
# ----------------------------
def make_features(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """
    Uses numeric-only features.
    Drops YEAR and any column that contains 'FSI' to avoid leaking targets/lagged targets.
    (We use FSI(t-1) ONLY for reconstruction and persistence baseline, not as a feature.)
    """
    X = df.copy()

    if year_col in X.columns:
        X = X.drop(columns=[year_col])

    # Drop any FSI-related columns (including lags/pillars) to prevent leakage
    drop_cols = [c for c in X.columns if "FSI" in c.upper()]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # numeric only
    X = X.select_dtypes(include=[np.number])
    return X


# ----------------------------
# Main
# ----------------------------
df = pd.read_csv(DATA_PATH).sort_values(YEAR_COL).reset_index(drop=True)

# Build ΔFSI target
df["FSI_prev"] = df[TARGET_COL].shift(1)
df["dFSI_true"] = df[TARGET_COL] - df["FSI_prev"]

# Drop first row (no previous year)
df_model = df[df["dFSI_true"].notna()].copy()

# Train/test by year (dFSI for year t belongs to year t)
train = df_model[df_model[YEAR_COL] <= TRAIN_END].copy()
test  = df_model[(df_model[YEAR_COL] >= TEST_START) & (df_model[YEAR_COL] <= TEST_END)].copy()

# X/y
Xtr = make_features(train, YEAR_COL)
ytr = train["dFSI_true"].astype(float).to_numpy()

Xte = make_features(test, YEAR_COL)
yte = test["dFSI_true"].astype(float).to_numpy()

# Reconstruction requires actual FSI(t-1) for each test year t
FSI_prev_test = test["FSI_prev"].astype(float).to_numpy()
FSI_true_test = test[TARGET_COL].astype(float).to_numpy()

# Persistence baseline (level)
FSI_persist = FSI_prev_test.copy()

models = {
    "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 60)),
    "RandomForest": RandomForestRegressor(
        n_estimators=1200, random_state=42, min_samples_leaf=2
    ),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

rows = []
all_preds = []

for name, model in models.items():
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    pipe.fit(Xtr, ytr)

    # Predict ΔFSI
    dFSI_pred = pipe.predict(Xte)

    # Reconstruct FSI levels
    FSI_pred = FSI_prev_test + dFSI_pred

    # Residuals in level space (Observed - Predicted)
    resid_level = FSI_true_test - FSI_pred

    # Metrics on ΔFSI
    m_delta = reg_metrics(yte, dFSI_pred)

    # Metrics on reconstructed FSI levels
    m_level = reg_metrics(FSI_true_test, FSI_pred)

    # Baseline metrics on level
    m_base = reg_metrics(FSI_true_test, FSI_persist)

    rows.append({
        "model": name,
        **{f"delta_test_{k}": v for k, v in m_delta.items()},
        **{f"level_test_{k}": v for k, v in m_level.items()},
        **{f"level_base_{k}": v for k, v in m_base.items()},
    })

    pred_df = pd.DataFrame({
        "year": test[YEAR_COL].to_numpy(),
        "FSI_true": FSI_true_test,
        "FSI_prev_actual": FSI_prev_test,
        "FSI_pred": FSI_pred,
        "FSI_persist": FSI_persist,
        "dFSI_true": yte,
        "dFSI_pred": dFSI_pred,
        "residual_level_obs_minus_pred": resid_level,
    })
    pred_df["model"] = name
    all_preds.append(pred_df)

    # save per-model preds
    pred_df.to_csv(OUT_DIR / f"preds_deltaFSI_{name.lower()}.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT_DIR / "metrics_deltaFSI_and_reconstructed_level.csv", index=False)

# Save combined predictions (stacked)
preds_all = pd.concat(all_preds, ignore_index=True)
preds_all.to_csv(OUT_DIR / "preds_deltaFSI_all_models_stacked.csv", index=False)

print("Saved:", OUT_DIR / "metrics_deltaFSI_and_reconstructed_level.csv")
print(metrics_df)
print("\nSaved stacked preds:", OUT_DIR / "preds_deltaFSI_all_models_stacked.csv")
