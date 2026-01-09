import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)
from sklearn.calibration import calibration_curve

# ============================================================
# LOGREG CLASSIFICATION FIGURES (analog of Fig 2/3/4) + SENSITIVITY
# Target: y_warning = 1 if FSI drops by >= X% from previous year
# Robust label via log-change:
#   y=1 if log(FSI_t) - log(FSI_{t-1}) <= log(1 - drop_pct)
# ============================================================

# ----------------------------
# INPUTS (edit paths)
# ----------------------------
IN_TABLE = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")

OUT_DIR = Path(r"E:\1Cathy\hsbdc\AI2026\results3\figs_logreg_warning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# SPLIT (your standard)
# ----------------------------
TRAIN_END  = 2012
TEST_START = 2013
TEST_END   = 2023

# ----------------------------
# MAIN WARNING THRESHOLD (for the figures)
# ----------------------------
DROP_PCT_MAIN = 0.10     # 10% drop definition

# ----------------------------
# SENSITIVITY THRESHOLDS (table + plots)
# ----------------------------
DROP_PCTS = [0.05, 0.10, 0.15, 0.20, 0.25]  # edit as you like

# ----------------------------
# LABEL STABILITY
# ----------------------------
EPS = 1e-6  # prevents log(0)

# ----------------------------
# FEATURE SELECTION: avoid leakage
# ----------------------------
LEAK_COLS = {
    "FSI_geomean","FSI_prev","y_warning","FSI_log_change",
    "FSI_pct_change","FSI_pct_change_robust",
    "FSI_lag1","FSI_lag2","dFSI","dFSI_lag1","dFSI_lag2",
}

# ============================================================
# Helpers
# ============================================================
def build_warning_label(df, drop_pct):
    df = df.sort_values("year").copy()
    df["FSI_prev"] = df["FSI_geomean"].shift(1)
    df["FSI_log_change"] = np.log(df["FSI_geomean"] + EPS) - np.log(df["FSI_prev"] + EPS)
    df["y_warning"] = (df["FSI_log_change"] <= np.log(1 - drop_pct)).astype("Int64")
    df.loc[df["FSI_prev"].isna(), "y_warning"] = pd.NA
    return df

def get_numeric_feature_cols(df):
    cols = []
    for c in df.columns:
        if c == "year" or c in LEAK_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def fit_logreg_and_predict(df_labeled):
    train = df_labeled[(df_labeled["year"] <= TRAIN_END) & (df_labeled["y_warning"].notna())].copy()
    test  = df_labeled[(df_labeled["year"] >= TEST_START) & (df_labeled["year"] <= TEST_END) & (df_labeled["y_warning"].notna())].copy()

    y_train = train["y_warning"].astype(int).to_numpy()
    y_test  = test["y_warning"].astype(int).to_numpy()

    feature_cols = get_numeric_feature_cols(df_labeled)

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

    prob_test = model.predict_proba(X_test_i)[:, 1]
    return test, y_test, prob_test

def compute_metrics(y_true, prob, threshold=0.5):
    yhat = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, yhat, labels=[0,1])
    bal = balanced_accuracy_score(y_true, yhat)
    roc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    ap  = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold_used": threshold,
        "balanced_acc": bal,
        "roc_auc": roc,
        "avg_precision": ap,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "pos_test": int(np.sum(y_true)), "n_test": int(len(y_true))
    }

# ============================================================
# Load base table
# ============================================================
df = pd.read_csv(IN_TABLE).sort_values("year").reset_index(drop=True)

# ============================================================
# A) MAIN RUN (DROP_PCT_MAIN) -> make "Fig 2/3/4 analogs"
# ============================================================
df_main = build_warning_label(df, DROP_PCT_MAIN)
test_df, y_true, prob = fit_logreg_and_predict(df_main)

# Use a fixed threshold for “pred class” (edit if you used a tuned one)
THRESH = 0.50
y_pred = (prob >= THRESH).astype(int)

out_pred = test_df[["year","FSI_geomean","FSI_prev","FSI_log_change"]].copy()
out_pred["y_warning"] = y_true
out_pred["prob_LogReg"] = prob
out_pred["pred_LogReg"] = y_pred
out_pred["drop_pct_label"] = DROP_PCT_MAIN
out_pred["threshold_used"] = THRESH

pred_path = OUT_DIR / f"pred_logreg_warning_drop{int(DROP_PCT_MAIN*100)}pct_{TEST_START}_{TEST_END}.csv"
out_pred.to_csv(pred_path, index=False)

m = compute_metrics(y_true, prob, threshold=THRESH)
print("Saved predictions:", pred_path)
print("Main metrics:", m)

# ----------------------------
# FIGURE 2 ANALOG (time-series story)
# - Shows: actual warning (0/1), predicted probability, and threshold line
# ----------------------------
fig = plt.figure(figsize=(9,4), dpi=200)
ax = plt.gca()
ax.plot(out_pred["year"], out_pred["prob_LogReg"], marker="o", linewidth=2, label="LogReg predicted warning probability")
ax.scatter(out_pred["year"], out_pred["y_warning"], label="Observed warning (0/1)")
ax.axhline(THRESH, linestyle="--", linewidth=1, label=f"Decision threshold = {THRESH:.2f}")

ax.set_xlabel("Year")
ax.set_ylabel("Probability / Label")
ax.set_title(f"Observed vs Predicted Warning Over Time (LogReg, drop ≥ {int(DROP_PCT_MAIN*100)}%, {TEST_START}–{TEST_END})")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=True, loc="lower right")
plt.tight_layout()

fig2_path = OUT_DIR / f"FIG2_like_time_series_logreg_drop{int(DROP_PCT_MAIN*100)}.png"
plt.savefig(fig2_path, bbox_inches="tight")
plt.show()
print("Saved:", fig2_path)

# ----------------------------
# FIGURE 3 ANALOG (predicted vs observed)
# For classification, the clean version is CALIBRATION:
# x = predicted probability, y = observed positive rate (binned)
# ----------------------------
frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=5, strategy="uniform")

fig = plt.figure(figsize=(6,5), dpi=200)
ax = plt.gca()
ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="LogReg calibration")
ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Perfect calibration (y=x)")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed warning rate")
ax.set_title(f"Predicted vs Observed (Calibration)\nLogReg, drop ≥ {int(DROP_PCT_MAIN*100)}%, {TEST_START}–{TEST_END}")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.legend(frameon=True, loc="upper left")
plt.tight_layout()

fig3_path = OUT_DIR / f"FIG3_like_calibration_logreg_drop{int(DROP_PCT_MAIN*100)}.png"
plt.savefig(fig3_path, bbox_inches="tight")
plt.show()
print("Saved:", fig3_path)

# ----------------------------
# FIGURE 4 ANALOG (residuals over time)
# Classification residual: r(t) = y(t) - p(t)
# (same “around zero” interpretation as your residuals figure)
# ----------------------------
resid = out_pred["y_warning"].astype(float).to_numpy() - out_pred["prob_LogReg"].astype(float).to_numpy()

fig = plt.figure(figsize=(9,4), dpi=200)
ax = plt.gca()
ax.axhline(0, linewidth=1)
ax.plot(out_pred["year"], resid, marker="o", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Residual (y - p)")
ax.set_title(f"Residuals Over Time (Observed − Predicted prob)\nLogReg, drop ≥ {int(DROP_PCT_MAIN*100)}%, {TEST_START}–{TEST_END}")
plt.tight_layout()

fig4_path = OUT_DIR / f"FIG4_like_residuals_logreg_drop{int(DROP_PCT_MAIN*100)}.png"
plt.savefig(fig4_path, bbox_inches="tight")
plt.show()
print("Saved:", fig4_path)

# ============================================================
# B) SENSITIVITY ANALYSIS (vary the WARNING definition itself)
# Table: drop threshold vs BalancedAcc / ROC / PR + confusion counts
# ============================================================
rows = []
for dp in DROP_PCTS:
    df_dp = build_warning_label(df, dp)
    test_df_dp, y_true_dp, prob_dp = fit_logreg_and_predict(df_dp)

    # Keep threshold fixed (0.5) so results reflect label definition changes only
    met = compute_metrics(y_true_dp, prob_dp, threshold=0.50)
    met["drop_pct_label"] = dp
    rows.append(met)

sens = pd.DataFrame(rows).sort_values("drop_pct_label").reset_index(drop=True)

sens_path = OUT_DIR / "SENS_logreg_warning_drop_thresholds.csv"
sens.to_csv(sens_path, index=False)
print("Saved sensitivity table:", sens_path)
print(sens)

# Optional sensitivity plot (balanced accuracy vs drop threshold)
fig = plt.figure(figsize=(7,4), dpi=200)
ax = plt.gca()
ax.plot(sens["drop_pct_label"]*100, sens["balanced_acc"], marker="o", linewidth=2)
ax.set_xlabel("Warning definition: drop threshold (%)")
ax.set_ylabel("Balanced accuracy (test)")
ax.set_title("Sensitivity: LogReg performance vs warning threshold")
ax.set_ylim(0, 1)
plt.tight_layout()

sens_fig_path = OUT_DIR / "FIG_sensitivity_balanced_accuracy.png"
plt.savefig(sens_fig_path, bbox_inches="tight")
plt.show()
print("Saved:", sens_fig_path)
