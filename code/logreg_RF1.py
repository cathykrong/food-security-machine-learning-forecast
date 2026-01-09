import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CLASSIFICATION ONLY (robust warning label)
# y_warning = 1 if "FSI dropped by 10%" using one of:
#   (A) denominator floor (% change with floor)
#   (B) log-change (recommended)
# Then train classifiers on crop-only features (no FSI leakage)
# ============================================================

# ----------------------------
# PATHS (edit if needed)
# ----------------------------
IN_TABLE = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")
OUT_PRED = Path(r"E:\1Cathy\hsbdc\AI2026\data3\models_cls_predictions_warning_10pctdrop_2013_2023_robust.csv")

# ----------------------------
# SPLIT
# ----------------------------
TRAIN_END  = 2012
TEST_START = 2013
TEST_END   = 2023

# ----------------------------
# WARNING DEFINITION
# ----------------------------
DROP_PCT = 0.10  # 10% drop threshold

# choose one:
WARNING_METHOD = "log_change"      # "log_change" or "denom_floor"

DEN_FLOOR = 0.20                   # only used for denom_floor; try 0.10–0.30
EPS = 1e-6                         # for log-change stability

# ----------------------------
# MODELS
# ----------------------------
USE_LOGREG = True
USE_RF = True

# optional: tune decision threshold on TRAIN to improve rare-event detection
TUNE_THRESHOLD_ON_TRAIN = True     # if False, uses 0.5
THRESH_METRIC = "f1"               # "f1" (recommended) or "bal_acc"

# ============================================================
# Load
# ============================================================
df = pd.read_csv(IN_TABLE).sort_values("year").reset_index(drop=True)

# ============================================================
# Build robust warning label
# ============================================================
df["FSI_prev"] = df["FSI_geomean"].shift(1)

if WARNING_METHOD == "denom_floor":
    denom = df["FSI_prev"].clip(lower=DEN_FLOOR)
    df["FSI_pct_change_robust"] = (df["FSI_geomean"] - df["FSI_prev"]) / denom
    df["y_warning"] = (df["FSI_pct_change_robust"] <= -DROP_PCT).astype("Int64")

elif WARNING_METHOD == "log_change":
    # log(FSI_t) - log(FSI_{t-1}) <= log(0.9) means >=10% drop
    df["FSI_log_change"] = np.log(df["FSI_geomean"] + EPS) - np.log(df["FSI_prev"] + EPS)
    df["y_warning"] = (df["FSI_log_change"] <= np.log(1 - DROP_PCT)).astype("Int64")

else:
    raise ValueError("WARNING_METHOD must be 'denom_floor' or 'log_change'")

df.loc[df["FSI_prev"].isna(), "y_warning"] = pd.NA  # first year no label

# ============================================================
# Train/test split
# ============================================================
train = df[(df["year"] <= TRAIN_END) & (df["y_warning"].notna())].copy()
test  = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END) & (df["y_warning"].notna())].copy()

y_train = train["y_warning"].astype(int).to_numpy()
y_test  = test["y_warning"].astype(int).to_numpy()

print("Label counts (train):", np.bincount(y_train))
print("Label counts (test): ", np.bincount(y_test))

# ============================================================
# Features: crop-only engineered features (avoid FSI leakage)
# ============================================================
LEAK_COLS = {
    "FSI_geomean","FSI_prev","y_warning",
    "FSI_pct_change","FSI_pct_change_robust","FSI_log_change",
    "FSI_lag1","FSI_lag2","dFSI","dFSI_lag1","dFSI_lag2",
}

feature_cols = []
for c in df.columns:
    if c in LEAK_COLS or c == "year":
        continue
    # keep numeric columns
    if pd.api.types.is_numeric_dtype(df[c]):
        feature_cols.append(c)

# force numeric (in case some came in as object)
for c in feature_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c]  = pd.to_numeric(test[c], errors="coerce")

X_train = train[feature_cols].to_numpy()
X_test  = test[feature_cols].to_numpy()

imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i  = imp.transform(X_test)

# ============================================================
# Threshold tuning helper (on TRAIN only)
# ============================================================
def best_threshold_from_train(y_true, prob, metric="f1"):
    # Use PR curve thresholds (works well for rare positives)
    prec, rec, thr = precision_recall_curve(y_true, prob)
    # thr has length n-1 compared to prec/rec
    # compute F1 or balanced accuracy over thresholds
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

# ============================================================
# Train + evaluate models
# ============================================================
def evaluate(name, y_true, yhat, prob):
    cm = confusion_matrix(y_true, yhat, labels=[0,1])
    bal = balanced_accuracy_score(y_true, yhat)
    roc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    ap  = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan

    print("\n====================")
    print("MODEL:", name)
    print("====================")
    print("Balanced accuracy:", round(bal, 4))
    print("ROC AUC:", round(roc, 4))
    print("Avg Precision (PR AUC):", round(ap, 4))
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)
    print("\nReport:\n", classification_report(y_true, yhat, digits=4))

out = test[["year","FSI_geomean","FSI_prev"]].copy()
if "FSI_pct_change_robust" in df.columns:
    out["FSI_pct_change_robust"] = test.get("FSI_pct_change_robust")
if "FSI_log_change" in df.columns:
    out["FSI_log_change"] = test.get("FSI_log_change")
out["y_warning"] = y_test

# ---- Logistic Regression
if USE_LOGREG:
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train_i, y_train)

    p_train = lr.predict_proba(X_train_i)[:, 1]
    p_test  = lr.predict_proba(X_test_i)[:, 1]

    if TUNE_THRESHOLD_ON_TRAIN:
        thr, score = best_threshold_from_train(y_train, p_train, metric=THRESH_METRIC)
        print(f"\n[LogReg] tuned threshold on TRAIN ({THRESH_METRIC}) = {thr:.3f} (score={score:.4f})")
    else:
        thr = 0.5

    yhat = (p_test >= thr).astype(int)
    evaluate(f"LogReg (thr={thr:.3f})", y_test, yhat, p_test)

    out["pred_LogReg"] = yhat
    out["prob_LogReg"] = p_test

# ---- Random Forest
if USE_RF:
    rf = RandomForestClassifier(
        n_estimators=1200,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(X_train_i, y_train)

    p_train = rf.predict_proba(X_train_i)[:, 1]
    p_test  = rf.predict_proba(X_test_i)[:, 1]

    if TUNE_THRESHOLD_ON_TRAIN:
        thr, score = best_threshold_from_train(y_train, p_train, metric=THRESH_METRIC)
        print(f"\n[RF] tuned threshold on TRAIN ({THRESH_METRIC}) = {thr:.3f} (score={score:.4f})")
    else:
        thr = 0.5

    yhat = (p_test >= thr).astype(int)
    evaluate(f"RF (thr={thr:.3f})", y_test, yhat, p_test)

    out["pred_RF"] = yhat
    out["prob_RF"] = p_test

# ============================================================
# Save predictions
# ============================================================
out.to_csv(OUT_PRED, index=False)
print("\nSaved:", OUT_PRED)
print(out.head(12))
