"""
RF ΔFSI pipeline with diagnostics + walk-forward tuning to improve LEVEL R².

Edit the USER SETTINGS and run:
python improve_rf_deltafsi_levelr2.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# USER SETTINGS (edit these)
# ============================================================
DATA_PATH = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table__MEDIAN_IMPUTED.csv")
YEAR_COL = "year"
TARGET_COL = "FSI_geomean"

TRAIN_END = 2012
TEST_START = 2013
TEST_END = 2023

OUT_DIR = Path(r"E:\1Cathy\hsbdc\AI2026\results3\outputs_deltaFSI_tuned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Directional label definition (optional; not used in tuning)
EPS = 0.0

# ---------- model-side stabilizers (turn on/off) ----------
CLIP_FSI_TO_01 = True             # clip FSI into [0,1] (harmless if already valid)
WINSORIZE_DFSI = True             # cap extreme ΔFSI outliers (helps if FSI has spuriously huge jumps)
DFSI_CAP_ABS = 0.30               # cap ΔFSI to [-0.30, +0.30]  (edit; try 0.20/0.25/0.30)

# ---------- safe autoregressive features (recommended) ----------
ADD_DFSI_LAGS = [1, 2]            # add dFSI_lag1, dFSI_lag2 (uses only past)
INCLUDE_YEAR_AS_FEATURE = True    # allow model to learn long-term trend

# ---------- which predictors to use ----------
# Keep numeric predictors, drop anything with "FSI" in the name to avoid leakage.
# Then sensitivity logic controls shocks/lags terms.
SHOCKS_ON = True
LAG_DEPTH = 2
LAGGED_SHOCKS_ON = False
# ============================================================


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
# Feature selection
# ----------------------------
def base_feature_cols(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in num_cols if c != YEAR_COL]
    cols = [c for c in cols if "FSI" not in c.upper()]  # no leakage from target/pillars/lagged FSI
    return cols


def select_cols_shock_lag(cols: list[str], shocks_on: bool, lag_depth: int, lagged_shocks_on: bool) -> list[str]:
    def is_shock(c: str) -> bool:
        return "yoy" in c.lower()

    def lag_k(c: str) -> int:
        c2 = c.lower()
        if "_lag2" in c2: return 2
        if "_lag1" in c2: return 1
        return 0

    kept = []
    for c in cols:
        lk = lag_k(c)
        shock = is_shock(c)

        if lk > lag_depth:
            continue
        if (not shocks_on) and shock:
            continue
        if shock and lk > 0 and (not lagged_shocks_on):
            continue

        kept.append(c)
    return kept


# ----------------------------
# Diagnostics
# ----------------------------
def print_target_diagnostics(df: pd.DataFrame):
    s = pd.to_numeric(df[TARGET_COL], errors="coerce")
    yr = pd.to_numeric(df[YEAR_COL], errors="coerce")
    tmp = pd.DataFrame({YEAR_COL: yr, TARGET_COL: s}).dropna().sort_values(YEAR_COL)

    fsi_prev = tmp[TARGET_COL].shift(1)
    dfsi = tmp[TARGET_COL] - fsi_prev
    tmp["FSI_prev"] = fsi_prev
    tmp["dFSI"] = dfsi
    tmp["abs_dFSI"] = dfsi.abs()

    print("\n=== TARGET SANITY CHECK ===")
    print("FSI min/max:", float(np.nanmin(s)), float(np.nanmax(s)))
    print("FSI percentiles (1,5,50,95,99):", [float(np.nanpercentile(s, p)) for p in [1,5,50,95,99]])

    topj = tmp.dropna().sort_values("abs_dFSI", ascending=False).head(8)
    print("\nBiggest year-to-year jumps (should usually be small):")
    print(topj[[YEAR_COL, "FSI_prev", TARGET_COL, "dFSI", "abs_dFSI"]].to_string(index=False))


# ----------------------------
# Walk-forward tuning on LEVEL R²
# ----------------------------
def walkforward_level_r2(train_df: pd.DataFrame, feat_cols: list[str], rf_params: dict, n_splits: int = 5) -> float:
    train_df = train_df.sort_values(YEAR_COL).reset_index(drop=True)

    X = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_delta = train_df["dFSI_true"].astype(float).to_numpy()
    fsi_prev = train_df["FSI_prev"].astype(float).to_numpy()
    fsi_true = train_df[TARGET_COL].astype(float).to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y_delta[tr_idx]

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(**rf_params)),
        ])
        pipe.fit(Xtr, ytr)

        d_pred = pipe.predict(Xva)
        f_pred = fsi_prev[va_idx] + d_pred  # reconstruct level
        m = reg_metrics(fsi_true[va_idx], f_pred)
        scores.append(m["r2"])

    return float(np.nanmean(scores)) if len(scores) else float("nan")


def main():
    df = pd.read_csv(DATA_PATH)

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.sort_values(YEAR_COL).reset_index(drop=True)

    # Target prep
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if CLIP_FSI_TO_01:
        df[TARGET_COL] = df[TARGET_COL].clip(0.0, 1.0)

    df["FSI_prev"] = df[TARGET_COL].shift(1)
    df["dFSI_true"] = df[TARGET_COL] - df["FSI_prev"]

    # Optional: stabilize insane ΔFSI spikes (model-side band-aid)
    if WINSORIZE_DFSI:
        df["dFSI_true"] = df["dFSI_true"].clip(-DFSI_CAP_ABS, DFSI_CAP_ABS)

    # Add safe autoregressive ΔFSI lags (uses only past)
    for k in ADD_DFSI_LAGS:
        df[f"dFSI_lag{k}"] = df["dFSI_true"].shift(k)

    # Print diagnostics
    print_target_diagnostics(df)

    # Modeling frame (needs dFSI and FSI_prev to exist)
    df_model = df.dropna(subset=["dFSI_true", "FSI_prev"]).copy()

    train = df_model[df_model[YEAR_COL] <= TRAIN_END].copy()
    test  = df_model[(df_model[YEAR_COL] >= TEST_START) & (df_model[YEAR_COL] <= TEST_END)].copy()

    # Base feature pool from numeric predictors (no FSI leakage)
    cols_pool = base_feature_cols(df_model)
    feat_cols = select_cols_shock_lag(cols_pool, SHOCKS_ON, LAG_DEPTH, LAGGED_SHOCKS_ON)

    # Add optional features
    if INCLUDE_YEAR_AS_FEATURE:
        feat_cols = [YEAR_COL] + feat_cols
    for k in ADD_DFSI_LAGS:
        col = f"dFSI_lag{k}"
        if col in df_model.columns:
            feat_cols = feat_cols + [col]

    # Drop any missing feature columns quietly
    feat_cols = [c for c in feat_cols if c in df_model.columns]

    print("\nUsing n_features =", len(feat_cols))

    # ----- hyperparameter search (small, targeted) -----
    grid = []
    for max_depth in [None, 3, 5, 8]:
        for min_samples_leaf in [1, 2, 4]:
            for max_features in ["sqrt", 0.5, 1.0]:
                grid.append({
                    "n_estimators": 1500,
                    "random_state": 42,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "n_jobs": -1,
                })

    best = {"cv_level_r2": -np.inf, "params": None}
    for params in grid:
        r2_cv = walkforward_level_r2(train, feat_cols, params, n_splits=5)
        if r2_cv > best["cv_level_r2"]:
            best = {"cv_level_r2": r2_cv, "params": params}

    print("\n=== BEST (walk-forward CV on LEVEL R²) ===")
    print("CV level R²:", round(best["cv_level_r2"], 4))
    print("Params:", best["params"])

    # ----- fit best model on full train, evaluate on test -----
    Xtr = train[feat_cols].apply(pd.to_numeric, errors="coerce")
    ytr = train["dFSI_true"].astype(float).to_numpy()

    Xte = test[feat_cols].apply(pd.to_numeric, errors="coerce")
    yte = test["dFSI_true"].astype(float).to_numpy()

    fsi_prev_test = test["FSI_prev"].astype(float).to_numpy()
    fsi_true_test = test[TARGET_COL].astype(float).to_numpy()

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(**best["params"])),
    ])
    pipe.fit(Xtr, ytr)

    d_pred = pipe.predict(Xte)
    f_pred = fsi_prev_test + d_pred
    f_persist = fsi_prev_test.copy()

    m_level = reg_metrics(fsi_true_test, f_pred)
    m_pers  = reg_metrics(fsi_true_test, f_persist)
    m_delta = reg_metrics(yte, d_pred)

    metrics = pd.DataFrame([{
        "model": "RF_dFSI_tuned",
        "level_test_rmse": m_level["rmse"],
        "level_test_mae": m_level["mae"],
        "level_test_r2": m_level["r2"],
        "persistence_test_r2": m_pers["r2"],
        "delta_test_r2": m_delta["r2"],
        "shocks_on": SHOCKS_ON,
        "lag_depth": LAG_DEPTH,
        "lagged_shocks_on": LAGGED_SHOCKS_ON,
        "dfsi_cap_abs": (DFSI_CAP_ABS if WINSORIZE_DFSI else np.nan),
        "features": len(feat_cols),
        "cv_level_r2": best["cv_level_r2"],
    }])
    metrics.to_csv(OUT_DIR / "metrics_rf_tuned.csv", index=False)
    print("\nSaved:", OUT_DIR / "metrics_rf_tuned.csv")
    print(metrics.to_string(index=False))

    preds = pd.DataFrame({
        "year": test[YEAR_COL].to_numpy(),
        "FSI_true": fsi_true_test,
        "FSI_prev_actual": fsi_prev_test,
        "FSI_pred": f_pred,
        "FSI_persist": f_persist,
        "dFSI_true": yte,
        "dFSI_pred": d_pred,
        "residual": fsi_true_test - f_pred,
    })
    preds.to_csv(OUT_DIR / "preds_rf_tuned.csv", index=False)
    print("Saved:", OUT_DIR / "preds_rf_tuned.csv")

    # Quick time-series plot
    fig = plt.figure(figsize=(9, 4), dpi=200)
    ax = plt.gca()
    ax.plot(preds["year"], preds["FSI_true"], marker="o", linewidth=2, label="Actual FSI")
    ax.plot(preds["year"], preds["FSI_pred"], marker="o", linewidth=2, label="RF tuned (ΔFSI)")
    ax.plot(preds["year"], preds["FSI_persist"], marker="o", linewidth=2, label="Persistence")
    ax.set_xlabel("Year")
    ax.set_ylabel("FSI (0–1)")
    ax.set_title("Observed vs Predicted FSI (Test)")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "FIG_timeseries_tuned.png", bbox_inches="tight")
    plt.show()

    # If still bad, print worst residual years
    worst = preds.assign(abs_res=np.abs(preds["residual"])).sort_values("abs_res", ascending=False).head(6)
    print("\nWorst residual years:")
    print(worst[["year", "FSI_true", "FSI_prev_actual", "FSI_pred", "residual"]].to_string(index=False))


if __name__ == "__main__":
    main()
