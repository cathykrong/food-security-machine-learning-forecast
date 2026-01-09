import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)

# ============================================================
# LOGREG CLASSIFICATION FIGURES + SENSITIVITY ANALYSIS
# (No regression anywhere)
# ============================================================

# ----------------------------
# PATHS (edit)
# ----------------------------
IN_TABLE = Path(r"E:\1Cathy\hsbdc\AI2026\data2\us_model_table_fsi_plus_crops_shocks_lags.csv")
OUT_DIR  = Path(r"E:\1Cathy\hsbdc\AI2026\data2\figs_cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# SPLIT
# ----------------------------
TRAIN_END  = 2012
TEST_START = 2013
TEST_END   = 2023

# ----------------------------
# WARNING LABEL (robust) options
# ----------------------------
LABEL_METHOD = "log_change"   # "log_change" recommended

# For log_change
EPS = 1e-6

# Main run (your default)
DROP_PCT_MAIN = 0.10

# Sensitivity analysis (warning definition)
DROP_PCTS_SENS = [0.05, 0.10, 0.15, 0.20]

# ----------------------------
# THRESHOLDING (for predictions)
# ----------------------------
TUNE_THRESHOLD_ON_TRAIN = True
THRESH_METRIC = "f1"  # "f1" or "bal_acc"

# ----------------------------
# Helpers
# ----------------------------
def best_threshold_from_train(y_true, prob, metric="f1"):
    """
    Tune probability threshold on TRAIN probabilities.
    """
    prec, rec, thr = precision_recall_curve(y_true, prob)
    # thr length is (len(prec)-1)
    best_t, best_val = 0.5, -np.inf
    for t in thr:
        yhat = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()

        if metric == "f1":
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            val = (2*p*r/(p+r)) if (p+r) else 0.0
        elif metric == "bal_acc":
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            val = 0.5 * (tpr + tnr)
        else:
            raise ValueError("metric must be 'f1' or 'bal_acc'")

        if val > best_val:
            best_val, best_t = val, float(t)

    return best_t, best_val

def get_feature_cols(df):
    """
    Crop-only numeric engineered features; remove anything FSI/label-derived (leakage).
    """
    leak = {
        "FSI_geomean","FSI_prev","y_warning",
        "FSI_pct_change","FSI_pct_change_robust","FSI_log_change",
        "FSI_lag1","FSI_lag2","dFSI","dFSI_lag1","dFSI_lag2"
    }
    cols = []
    for c in df.columns:
        if c == "year" or c in leak:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def build_warning_label(df, drop_pct):
    df = df.sort_values("year").copy()
    df["FSI_prev"] = df["FSI_geomean"].shift(1)

    if LABEL_METHOD == "log_change":
        # warning if log(FSI_t) - log(FSI_{t-1}) <= log(1 - drop_pct)
        df["FSI_log_change"] = np.log(df["FSI_geomean"] + EPS) - np.log(df["FSI_prev"] + EPS)
        df["y_warning"] = (df["FSI_log_change"] <= np.log(1 - drop_pct)).astype("Int64")
    else:
        raise ValueError("Only 'log_change' implemented here (recommended).")

    df.loc[df["FSI_prev"].isna(), "y_warning"] = pd.NA
    return df

def train_eval_logreg(df, drop_pct):
    df2 = build_warning_label(df, drop_pct)

    train = df2[(df2["year"] <= TRAIN_END) & (df2["y_warning"].notna())].copy()
    test  = df2[(df2["year"] >= TEST_START) & (df2["year"] <= TEST_END) & (df2["y_warning"].notna())].copy()

    y_train = train["y_warning"].astype(int).to_numpy()
    y_test  = test["y_warning"].astype(int).to_numpy()

    feature_cols = get_feature_cols(df2)
    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c]  = pd.to_numeric(test[c], errors="coerce")

    X_train = train[feature_cols].to_numpy()
    X_test  = test[feature_cols].to_numpy()

    imp = SimpleImputer(strategy="median")
    X_train_i = imp.fit_transform(X_train)
    X_test_i  = imp.transform(X_test)

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train_i, y_train)

    p_train = model.predict_proba(X_train_i)[:, 1]
    p_test  = model.predict_proba(X_test_i)[:, 1]

    if TUNE_THRESHOLD_ON_TRAIN:
        thr, thr_score = best_threshold_from_train(y_train, p_train, metric=THRESH_METRIC)
    else:
        thr, thr_score = 0.5, np.nan

    yhat_test = (p_test >= thr).astype(int)

    cm = confusion_matrix(y_test, yhat_test, labels=[0,1])
    bal = balanced_accuracy_score(y_test, yhat_test)
    roc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan
    ap  = average_precision_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan

    out = test[["year","FSI_geomean","FSI_prev","FSI_log_change"]].copy()
    out["y_warning"] = y_test
    out["prob_LogReg"] = p_test
    out["pred_LogReg"] = yhat_test
    out["threshold_used"] = thr
    out["drop_pct_label"] = drop_pct

    metrics = {
        "drop_pct": drop_pct,
        "threshold": thr,
        "thr_train_score": thr_score,
        "balanced_acc": bal,
        "roc_auc": roc,
        "avg_precision": ap,
        "TN": int(cm[0,0]), "FP": int(cm[0,1]),
        "FN": int(cm[1,0]), "TP": int(cm[1,1]),
        "n_test": int(len(y_test)),
        "pos_test": int(y_test.sum())
    }
    return out, metrics

# ============================================================
# Load modeling table
# ============================================================
df = pd.read_csv(IN_TABLE).sort_values("year").reset_index(drop=True)

# ============================================================
# A) MAIN FIGURES for DROP_PCT_MAIN
# ============================================================
pred_main, met_main = train_eval_logreg(df, DROP_PCT_MAIN)

pred_csv = OUT_DIR / f"pred_logreg_warning_drop{int(DROP_PCT_MAIN*100)}pct_{TEST_START}_{TEST_END}.csv"
pred_main.to_csv(pred_csv, index=False)
print("Saved predictions:", pred_csv)
print("Main metrics:", met_main)

# ---- Figure: Probability over time + true labels
fig1 = plt.figure(figsize=(10,4), dpi=200)
ax = plt.gca()
ax.plot(pred_main["year"], pred_main["prob_LogReg"], linewidth=2, label="Predicted warning probability")
ax.scatter(pred_main["year"], pred_main["y_warning"], label="Actual warning (0/1)")
ax.set_xlabel("Year")
ax.set_ylabel("Warning probability / label")
ax.set_title(f"LogReg: Warning probability vs actual (drop ≥ {int(DROP_PCT_MAIN*100)}%)")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=True)
fig1_path = OUT_DIR / f"FIG_prob_over_time_logreg_drop{int(DROP_PCT_MAIN*100)}.png"
plt.tight_layout()
plt.savefig(fig1_path, bbox_inches="tight")
plt.show()
print("Saved:", fig1_path)

# ---- Figure: "Prediction vs Actual" (probability vs label)
fig2 = plt.figure(figsize=(6,4), dpi=200)
ax = plt.gca()
# jitter x a bit so 0 and 1 labels don't overlap too much
x = pred_main["y_warning"].astype(float).to_numpy()
x_jitter = x + np.random.uniform(-0.03, 0.03, size=len(x))
ax.scatter(x_jitter, pred_main["prob_LogReg"].to_numpy())
ax.set_xlabel("Actual label (0/1)")
ax.set_ylabel("Predicted probability")
ax.set_title("Predicted probability vs actual label")
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.05, 1.05)
fig2_path = OUT_DIR / f"FIG_pred_vs_actual_logreg_drop{int(DROP_PCT_MAIN*100)}.png"
plt.tight_layout()
plt.savefig(fig2_path, bbox_inches="tight")
plt.show()
print("Saved:", fig2_path)

# ---- Figure: "Residuals" over time (classification residual = y - p)
pred_main["residual_y_minus_p"] = pred_main["y_warning"].astype(float) - pred_main["prob_LogReg"].astype(float)

fig3 = plt.figure(figsize=(10,4), dpi=200)
ax = plt.gca()
ax.axhline(0, linewidth=1)
ax.plot(pred_main["year"], pred_main["residual_y_minus_p"], linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Residual (y - p)")
ax.set_title("Classification residuals over time (y − predicted probability)")
fig3_path = OUT_DIR / f"FIG_residuals_over_time_logreg_drop{int(DROP_PCT_MAIN*100)}.png"
plt.tight_layout()
plt.savefig(fig3_path, bbox_inches="tight")
plt.show()
print("Saved:", fig3_path)

# ============================================================
# B) SENSITIVITY ANALYSIS: vary drop threshold definition
# ============================================================
sens_rows = []
for dp in DROP_PCTS_SENS:
    _, m = train_eval_logreg(df, dp)
    sens_rows.append(m)

sens = pd.DataFrame(sens_rows).sort_values("drop_pct")
sens_csv = OUT_DIR / "SENS_logreg_warning_drop_thresholds.csv"
sens.to_csv(sens_csv, index=False)
print("Saved sensitivity table:", sens_csv)
print(sens)

# ---- Sensitivity plot: balanced accuracy vs drop threshold
fig4 = plt.figure(figsize=(7,4), dpi=200)
ax = plt.gca()
ax.plot(sens["drop_pct"]*100, sens["balanced_acc"], marker="o")
ax.set_xlabel("Warning definition: drop threshold (%)")
ax.set_ylabel("Balanced accuracy (test)")
ax.set_title("Sensitivity analysis: performance vs warning threshold")
ax.set_ylim(0, 1)
fig4_path = OUT_DIR / "FIG_sensitivity_balanced_acc.png"
plt.tight_layout()
plt.savefig(fig4_path, bbox_inches="tight")
plt.show()
print("Saved:", fig4_path)

# ---- Sensitivity plot: PR AUC vs drop threshold (optional but useful for rare positives)
fig5 = plt.figure(figsize=(7,4), dpi=200)
ax = plt.gca()
ax.plot(sens["drop_pct"]*100, sens["avg_precision"], marker="o")
ax.set_xlabel("Warning definition: drop threshold (%)")
ax.set_ylabel("Average Precision (PR AUC, test)")
ax.set_title("Sensitivity analysis: PR AUC vs warning threshold")
ax.set_ylim(0, 1)
fig5_path = OUT_DIR / "FIG_sensitivity_pr_auc.png"
plt.tight_layout()
plt.savefig(fig5_path, bbox_inches="tight")
plt.show()
print("Saved:", fig5_path)
