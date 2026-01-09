import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# ============================================================
# RIDGE CV + GRADIENT BOOSTING (classification ONLY)
# y_warning = 1 if FSI drops >= 10% YoY (robust log-change)
# - Trains on <=2012, tests 2013–2023
# - Uses crop-only features (no FSI leakage)
# - Saves predictions + scores
# ============================================================

# ----------------------------
# PATHS (edit if needed)
# ----------------------------
IN_TABLE = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")
OUT_CSV  = Path(r"E:\1Cathy\hsbdc\AI2026\data3\models_cls_predictions_warning_10pctdrop_2013_2023_ridge_gb.csv")

# ----------------------------
# SPLIT
# ----------------------------
TRAIN_END  = 2012
TEST_START = 2013
TEST_END   = 2023

# ----------------------------
# WARNING DEFINITION (robust)
# ----------------------------
DROP_PCT = 0.10
EPS = 1e-6  # log stability

# ----------------------------
# THRESHOLDING
# ----------------------------
# Because positives are rare, do NOT force 0.5. We tune threshold on TRAIN.
TUNE_THRESHOLD_ON_TRAIN = True
THRESH_METRIC = "f1"  # "f1" or "bal_acc"

# ============================================================
# Helpers
# ============================================================
def best_threshold_from_train(y_true, score, metric="f1"):
    """
    Finds best threshold using TRAIN scores (decision scores or probabilities).
    """
    # sweep thresholds across score range
    lo, hi = np.nanmin(score), np.nanmax(score)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return 0.0, 0.0

    best_t, best_val = 0.0, -np.inf
    for t in np.linspace(lo, hi, 201):
        yhat = (score >= t).astype(int)
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

def evaluate(name, y_true, yhat, score):
    cm = confusion_matrix(y_true, yhat, labels=[0,1])
    bal = balanced_accuracy_score(y_true, yhat)

    roc = roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan
    ap  = average_precision_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan

    print("\n====================")
    print("MODEL:", name)
    print("====================")
    print("Balanced accuracy:", round(bal, 4))
    print("ROC AUC:", round(roc, 4))
    print("Avg Precision (PR AUC):", round(ap, 4))
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)
    print("\nReport:\n", classification_report(y_true, yhat, digits=4))

def get_feature_cols(df):
    # remove leakage columns
    leak = {
        "FSI_geomean","FSI_prev","y_warning","FSI_log_change",
        "FSI_pct_change","FSI_pct_change_robust",
        "FSI_lag1","FSI_lag2","dFSI","dFSI_lag1","dFSI_lag2"
    }
    cols = []
    for c in df.columns:
        if c in leak or c == "year":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

# ============================================================
# Load + build label
# ============================================================
df = pd.read_csv(IN_TABLE).sort_values("year").reset_index(drop=True)

df["FSI_prev"] = df["FSI_geomean"].shift(1)
df["FSI_log_change"] = np.log(df["FSI_geomean"] + EPS) - np.log(df["FSI_prev"] + EPS)
df["y_warning"] = (df["FSI_log_change"] <= np.log(1 - DROP_PCT)).astype("Int64")
df.loc[df["FSI_prev"].isna(), "y_warning"] = pd.NA

train = df[(df["year"] <= TRAIN_END) & (df["y_warning"].notna())].copy()
test  = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END) & (df["y_warning"].notna())].copy()

y_train = train["y_warning"].astype(int).to_numpy()
y_test  = test["y_warning"].astype(int).to_numpy()

print("Label counts (train):", np.bincount(y_train))
print("Label counts (test): ", np.bincount(y_test))

# ============================================================
# Features + imputation
# ============================================================
feature_cols = get_feature_cols(df)

for c in feature_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c]  = pd.to_numeric(test[c], errors="coerce")

X_train = train[feature_cols].to_numpy()
X_test  = test[feature_cols].to_numpy()

imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i  = imp.transform(X_test)

# ============================================================
# 1) RidgeClassifierCV (your “Ridge CV” classifier)
# ============================================================
alphas = np.logspace(-4, 4, 200)
try:
    ridge = RidgeClassifierCV(alphas=alphas, cv=5, class_weight="balanced")
except TypeError:
    # older sklearn may not support class_weight here
    ridge = RidgeClassifierCV(alphas=alphas, cv=5)

ridge.fit(X_train_i, y_train)

ridge_score_train = ridge.decision_function(X_train_i)   # continuous score
ridge_score_test  = ridge.decision_function(X_test_i)

if TUNE_THRESHOLD_ON_TRAIN:
    ridge_thr, ridge_best = best_threshold_from_train(y_train, ridge_score_train, metric=THRESH_METRIC)
    print(f"\n[RidgeCV] tuned threshold on TRAIN ({THRESH_METRIC}) = {ridge_thr:.4f} (score={ridge_best:.4f})")
else:
    ridge_thr = 0.0

ridge_pred_test = (ridge_score_test >= ridge_thr).astype(int)
evaluate(f"RidgeClassifierCV (thr={ridge_thr:.4f})", y_test, ridge_pred_test, ridge_score_test)

# ============================================================
# 2) GradientBoostingClassifier
# ============================================================
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_i, y_train)

gb_score_train = gb.predict_proba(X_train_i)[:, 1]
gb_score_test  = gb.predict_proba(X_test_i)[:, 1]

if TUNE_THRESHOLD_ON_TRAIN:
    gb_thr, gb_best = best_threshold_from_train(y_train, gb_score_train, metric=THRESH_METRIC)
    print(f"\n[GB] tuned threshold on TRAIN ({THRESH_METRIC}) = {gb_thr:.4f} (score={gb_best:.4f})")
else:
    gb_thr = 0.5

gb_pred_test = (gb_score_test >= gb_thr).astype(int)
evaluate(f"GradientBoosting (thr={gb_thr:.4f})", y_test, gb_pred_test, gb_score_test)

# ============================================================
# Save output
# ============================================================
out = test[["year","FSI_geomean","FSI_prev","FSI_log_change"]].copy()
out["y_warning"] = y_test

out["score_RidgeCV"] = ridge_score_test
out["pred_RidgeCV"] = ridge_pred_test

out["score_GB"] = gb_score_test
out["pred_GB"] = gb_pred_test

out.to_csv(OUT_CSV, index=False)
print("\nSaved:", OUT_CSV)
print(out.head(12))
