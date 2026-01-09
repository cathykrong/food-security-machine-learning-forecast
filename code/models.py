"""
Manuscript-style pipeline:
- Predict ΔFSI(t) = FSI(t) - FSI(t-1)
- Reconstruct FSI_pred(t) = FSI(t-1) + ΔFSI_pred(t)
- Baseline: persistence FSI_persist(t) = FSI(t-1)
- Train: 1970–2012, Test: 2013–2023
- Outputs: metrics + Figure 2/3/4/5 + ROC + Confusion Matrix + sensitivity table

How to run (no CLI args needed):
python manuscript_deltafsi_rf.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, balanced_accuracy_score
)

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

# Optional RF hyperparams
RF_N_ESTIMATORS = 1200
RF_RANDOM_STATE = 42
RF_MIN_SAMPLES_LEAF = 2

# Plot control
FIG2_YMIN = 0.60

# Directional label definition for ROC/CM
EPS = 0.0  # Positive = ΔFSI > EPS
# ============================================================


# ----------------------------
# Metrics (regression)
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
# Feature selection (manuscript style)
# Keep numeric predictors; drop year + any FSI-related columns to avoid leakage
# Then the sensitivity function controls shocks/lags inclusion.
# ----------------------------
def base_feature_cols(df: pd.DataFrame, year_col: str) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in num_cols if c != year_col]
    # drop any FSI-like columns (FSI itself, pillars, lagged FSI, etc.)
    cols = [c for c in cols if "FSI" not in c.upper()]
    return cols


def select_cols_shock_lag(cols: list[str], shocks_on: bool, lag_depth: int, lagged_shocks_on: bool) -> list[str]:
    """
    Assumes naming conventions:
      - shocks: contain "yoy" (e.g., _yoy_pct)
      - lags: contain "_lag1" / "_lag2"
      - lagged shocks: shock columns that ALSO have lag suffixes, e.g., _yoy_pct_lag1

    Rule:
      - lag_depth controls inclusion of *_lag1 and *_lag2 (for both raw and shock variables),
        but lagged_shocks_on decides whether shock-lag terms are allowed.
    """
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

        # lag depth filter
        if lk > lag_depth:
            continue

        # shocks on/off
        if (not shocks_on) and shock:
            continue

        # lagged shocks on/off (only affects shock variables with lag suffix)
        if shock and lk > 0 and (not lagged_shocks_on):
            continue

        kept.append(c)
    return kept


# ----------------------------
# Plots
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ----------------------------
# Helper: shift + extrapolate (same logic as Fig 2)
# ----------------------------
def _shift_and_extrapolate(df: pd.DataFrame):
    """
    Returns:
      x_shifted (len n): years shifted back by 1 (drop last x, add extrapolated final x)
      y_true_shifted (len n): drop first y_true, add extrapolated final y_true
      y_pred_shifted (len n): drop first y_pred, add extrapolated final y_pred
    """
    df = df.sort_values("year").reset_index(drop=True).copy()

    year = df["year"].to_numpy(dtype=float)
    y_true = df["FSI_true"].to_numpy(dtype=float)

    # prefer explicit reconstructed level if available
    if {"dFSI_pred", "FSI_prev_actual"}.issubset(df.columns):
        y_pred = (df["dFSI_pred"] + df["FSI_prev_actual"]).to_numpy(dtype=float)
    else:
        y_pred = df["FSI_pred"].to_numpy(dtype=float)

    if len(df) < 3:
        raise ValueError("Need at least 3 rows to shift and extrapolate safely.")

    # x shifted: years[0..n-2] + (last year + step)
    year_step = year[-1] - year[-2] if np.isfinite(year[-1] - year[-2]) else 1.0
    x_shifted = np.concatenate([year[:-1], [year[-1] + year_step]])

    # extrapolate final y from last two original points
    y_true_extrap = y_true[-1] + (y_true[-1] - y_true[-2])
    y_pred_extrap = y_pred[-1] + (y_pred[-1] - y_pred[-2])

    # y shifted: drop first + add extrapolated final
    y_true_shifted = np.concatenate([y_true[1:], [y_true_extrap]])
    y_pred_shifted = np.concatenate([y_pred[1:], [y_pred_extrap]])

    return x_shifted, y_true_shifted, y_pred_shifted


# ----------------------------
# FIG 2 (unchanged except uses helper)
# ----------------------------
def plot_fig2_timeseries(out_df: pd.DataFrame, out_path: Path, y_min: float = -0.5):
    x_shifted, y_true_shifted, y_pred_shifted = _shift_and_extrapolate(out_df)

    fig = plt.figure(figsize=(9, 4), dpi=200)
    ax = plt.gca()

    ax.plot(x_shifted, y_true_shifted, marker="o", linewidth=2, label="Actual FSI")
    ax.plot(x_shifted, y_pred_shifted, marker="o", linewidth=2, label="Random Forest (ΔFSI + FSI(t−1))")

    ax.set_xlabel("Year")
    ax.set_ylabel("FSI (0–1)")
    ax.set_title("Observed vs Predicted FSI Over Time")
    ax.set_ylim(bottom=y_min)
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


# ----------------------------
# FIG 3 (shifted)
# ----------------------------
def plot_fig3_scatter(out_df: pd.DataFrame, out_path: Path):
    """
    Scatter using the same shift logic as Fig 2:
      x-axis: shifted Actual FSI
      y-axis: shifted Predicted FSI (reconstructed from ΔFSI + FSI(t−1) if available)
    Includes y=x reference line.
    """
    _, y_true_shifted, y_pred_shifted = _shift_and_extrapolate(out_df)

    x = np.asarray(y_true_shifted, dtype=float)
    y = np.asarray(y_pred_shifted, dtype=float)

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = plt.gca()
    ax.scatter(x, y)

    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    ax.plot([lo, hi], [lo, hi], linewidth=2)

    ax.set_xlabel("Actual FSI")
    ax.set_ylabel("Predicted ΔFSI + FSI(t-1)")
    ax.set_title("Predicted vs Observed FSI")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


# ----------------------------
# FIG 4 (shifted residuals)
# ----------------------------
def plot_fig4_residuals(out_df: pd.DataFrame, out_path: Path):
    """
    Residuals over shifted years:
      residual_shifted = (shifted actual) - (shifted predicted)
      plotted against x_shifted.
    """
    x_shifted, y_true_shifted, y_pred_shifted = _shift_and_extrapolate(out_df)
    residual_shifted = np.asarray(y_true_shifted, dtype=float) - np.asarray(y_pred_shifted, dtype=float)

    fig = plt.figure(figsize=(7, 4), dpi=200)
    ax = plt.gca()
    ax.axhline(0, linewidth=1)
    ax.plot(x_shifted, residual_shifted, marker="o", linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (Actual − Pred)")
    ax.set_title("Residuals Over Time (ΔFSI + FSI(t-1))")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()



def plot_fig5_feature_importance(feature_names: list[str], importances: np.ndarray, out_path: Path, topk: int = 15):
    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(topk)[::-1]
    fig = plt.figure(figsize=(8, 5), dpi=200)
    ax = plt.gca()
    ax.barh(s.index, s.values)
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_ylabel("Predictor")
    ax.set_title("Top 15 Feature Importances: Random Forest ΔFSI")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_roc_and_confmat(y_true_bin: np.ndarray, score: np.ndarray, y_pred_bin: np.ndarray, out_roc: Path, out_cm: Path):
    # ROC (using ΔFSI_pred as a continuous score)
    fpr, tpr, _ = roc_curve(y_true_bin, score)
    auc = roc_auc_score(y_true_bin, score)

    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = plt.gca()
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve (Positive = ΔFSI > {EPS:g})")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(out_roc, bbox_inches="tight")
    plt.show()

    # Confusion matrix column-normalized (percent within predicted class)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).astype(float)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_pct = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0) * 100.0

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = plt.gca()
    im = ax.imshow(cm_pct, vmin=0, vmax=100)
    for (i, j), v in np.ndenumerate(cm_pct):
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Negative", "Pred Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Binary Confusion Matrix (Percent within Predicted Class)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_cm, bbox_inches="tight")
    plt.show()


# ----------------------------
# Main training (RF ΔFSI)
# ----------------------------
def main():
    df = pd.read_csv(DATA_PATH).sort_values(YEAR_COL).reset_index(drop=True)

    # ΔFSI target
    df["FSI_prev"] = pd.to_numeric(df[TARGET_COL], errors="coerce").shift(1)
    df["dFSI_true"] = pd.to_numeric(df[TARGET_COL], errors="coerce") - df["FSI_prev"]

    df_model = df[df["dFSI_true"].notna()].copy()

    train = df_model[df_model[YEAR_COL] <= TRAIN_END].copy()
    test  = df_model[(df_model[YEAR_COL] >= TEST_START) & (df_model[YEAR_COL] <= TEST_END)].copy()

    # base columns pool
    cols_pool = base_feature_cols(df_model, YEAR_COL)

    # manuscript best configuration (default):
    shocks_on = True
    lag_depth = 2
    lagged_shocks_on = False
    feat_cols = select_cols_shock_lag(cols_pool, shocks_on, lag_depth, lagged_shocks_on)

    Xtr = train[feat_cols].apply(pd.to_numeric, errors="coerce")
    ytr = train["dFSI_true"].astype(float).to_numpy()

    Xte = test[feat_cols].apply(pd.to_numeric, errors="coerce")
    yte = test["dFSI_true"].astype(float).to_numpy()

    FSI_prev_test = test["FSI_prev"].astype(float).to_numpy()
    FSI_true_test = pd.to_numeric(test[TARGET_COL], errors="coerce").astype(float).to_numpy()

    # RF (manuscript-style)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF
    )

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", rf),
    ])
    pipe.fit(Xtr, ytr)

    dFSI_pred = pipe.predict(Xte)
    FSI_pred = FSI_prev_test + dFSI_pred
    FSI_persist = FSI_prev_test.copy()
    residual = FSI_true_test - FSI_pred

    # Metrics
    m_delta = reg_metrics(yte, dFSI_pred)
    m_level = reg_metrics(FSI_true_test, FSI_pred)
    m_base  = reg_metrics(FSI_true_test, FSI_persist)

    metrics_out = pd.DataFrame([{
        "model": "RandomForest_dFSI",
        **{f"delta_test_{k}": v for k, v in m_delta.items()},
        **{f"level_test_{k}": v for k, v in m_level.items()},
        **{f"level_base_{k}": v for k, v in m_base.items()},
        "shocks_on": shocks_on,
        "lag_depth": lag_depth,
        "lagged_shocks_on": lagged_shocks_on,
        "n_features": len(feat_cols),
    }])
    metrics_out.to_csv(OUT_DIR / "metrics_rf_deltafsi_manuscript_style.csv", index=False)

    # Predictions table
    out = pd.DataFrame({
        "year": test[YEAR_COL].to_numpy(),
        "FSI_true": FSI_true_test,
        "FSI_prev_actual": FSI_prev_test,
        "FSI_pred": FSI_pred,
        "FSI_persist": FSI_persist,
        "dFSI_true": yte,
        "dFSI_pred": dFSI_pred,
        "residual": residual,
    })
    out.to_csv(OUT_DIR / "preds_rf_deltafsi_manuscript_style.csv", index=False)

    print("Saved:", OUT_DIR / "metrics_rf_deltafsi_manuscript_style.csv")
    print(metrics_out)

    # ----------------------------
    # Figures 2–5
    # ----------------------------
    plot_fig2_timeseries(out, OUT_DIR / "FIG2_timeseries_rf_dFSI_vs_persistence.png", y_min=FIG2_YMIN)
    plot_fig3_scatter(out, OUT_DIR / "FIG3_scatter_pred_vs_actual.png")
    plot_fig4_residuals(out, OUT_DIR / "FIG4_residuals_over_time.png")

    # Feature importance
    rf_fit = pipe.named_steps["model"]
    importances = rf_fit.feature_importances_
    plot_fig5_feature_importance(feat_cols, importances, OUT_DIR / "FIG5_top15_feature_importance.png", topk=15)

    # ----------------------------
    # Directional evaluation (ΔFSI > EPS)
    # ----------------------------
    y_true_bin = (yte > EPS).astype(int)
    y_pred_bin = (dFSI_pred > EPS).astype(int)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    print("Directional balanced accuracy (ΔFSI>EPS):", round(float(bal_acc), 4))

    plot_roc_and_confmat(
        y_true_bin=y_true_bin,
        score=dFSI_pred,
        y_pred_bin=y_pred_bin,
        out_roc=OUT_DIR / "FIG6_ROC_curve.png",
        out_cm=OUT_DIR / "FIG6_confusion_matrix_pct_within_pred.png",
    )

    # ----------------------------
    # Shock–lag sensitivity table (Table 1 style)
    # ----------------------------
    grid = [
        ("Persistence baseline (FSI(t)=FSI(t−1))", None),
        ("Shocks + lag1+lag2 (no lagged shocks)", (True, 2, False)),
        ("Shocks + lag1+lag2", (True, 2, True)),
        ("Shocks + lag1", (True, 1, True)),
        ("No shocks + no lags", (False, 0, False)),
        ("Shocks only", (True, 0, False)),
        ("Lag1 only", (False, 1, False)),
    ]

    sens_rows = []
    for label, cfg in grid:
        if cfg is None:
            pred_level = FSI_persist
            pred_delta = np.zeros_like(yte)
        else:
            s_on, ld, ls_on = cfg
            cols_cfg = select_cols_shock_lag(cols_pool, s_on, ld, ls_on)
            Xtr_cfg = train[cols_cfg].apply(pd.to_numeric, errors="coerce")
            Xte_cfg = test[cols_cfg].apply(pd.to_numeric, errors="coerce")

            pipe_cfg = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=RF_N_ESTIMATORS,
                    random_state=RF_RANDOM_STATE,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF
                )),
            ])
            pipe_cfg.fit(Xtr_cfg, ytr)
            pred_delta = pipe_cfg.predict(Xte_cfg)
            pred_level = FSI_prev_test + pred_delta

        m = reg_metrics(FSI_true_test, pred_level)

        y_pred_bin_cfg = (pred_delta > EPS).astype(int)
        bal = balanced_accuracy_score(y_true_bin, y_pred_bin_cfg) if cfg is not None else np.nan

        sens_rows.append({
            "Configuration": label,
            "Shocks": ("On" if (cfg and cfg[0]) else "Off") if cfg is not None else "—",
            "Lag_depth": (cfg[1] if cfg is not None else "—"),
            "Lagged_shocks": ("On" if (cfg and cfg[2]) else "Off") if cfg is not None else "—",
            "RMSE": m["rmse"],
            "MAE": m["mae"],
            "R2": m["r2"],
            "Balanced_accuracy": bal,
        })

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_DIR / "TABLE1_shock_lag_sensitivity.csv", index=False)
    print("Saved:", OUT_DIR / "TABLE1_shock_lag_sensitivity.csv")
    print(sens_df)


if __name__ == "__main__":
    main()
