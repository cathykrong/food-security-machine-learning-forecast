import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================
# STEP #3 — CLASSIFICATION ONLY
# Target: warning year if %ΔFSI <= -10% (drop >= 10% from last year)
# - Uses your modeling table from Step #2 (FSI + crops + shocks + lags)
# - Trains classifiers on 1970–2012, tests on 2013–2023
# - Outputs predictions + metrics + confusion matrix
# ============================================================

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
IN_TABLE = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")
OUT_PRED = Path(r"E:\1Cathy\hsbdc\AI2026\data3\cls_predictions_warning_10pctdrop_2013_2023.csv")

TRAIN_END = 2012
TEST_START = 2013
TEST_END = 2023

DROP_THRESHOLD = -0.10   # warning if %ΔFSI <= -10%
EPS_DIV = 1e-9

# Choose model(s)
USE_LOGREG = True
USE_RF = True

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(IN_TABLE)
df = df.sort_values("year").reset_index(drop=True)

# ----------------------------
# Build classification target: %ΔFSI
# ----------------------------
df["FSI_prev"] = df["FSI_geomean"].shift(1)
df["FSI_pct_change"] = (df["FSI_geomean"] - df["FSI_prev"]) / (df["FSI_prev"].replace(0, np.nan) + EPS_DIV)

# Binary label: 1 = warning (drop >=10%), 0 = not warning
df["y_warning"] = (df["FSI_pct_change"] <= DROP_THRESHOLD).astype("Int64")
df.loc[df["FSI_prev"].isna(), "y_warning"] = pd.NA  # first year no label

# ----------------------------
# Split train/test
# ----------------------------
train = df[(df["year"] <= TRAIN_END) & (df["y_warning"].notna())].copy()
test  = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END) & (df["y_warning"].notna())].copy()

# ----------------------------
# Feature set (use engineered crop features; exclude any target-leaking cols)
# ----------------------------
DROP_COLS = {
    "FSI_geomean", "FSI_prev", "FSI_pct_change", "y_warning",
    "FSI_lag1", "dFSI", "dFSI_lag1", "dFSI_lag2",  # remove any FSI-derived features (avoid leakage)
}
# Keep numeric columns except dropped
feature_cols = []
for c in df.columns:
    if c in DROP_COLS or c == "year":
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        feature_cols.append(c)

# Convert to numeric (some columns might still be object)
for c in feature_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c]  = pd.to_numeric(test[c], errors="coerce")

X_train = train[feature_cols].to_numpy()
y_train = train["y_warning"].astype(int).to_numpy()

X_test  = test[feature_cols].to_numpy()
y_test  = test["y_warning"].astype(int).to_numpy()

# Impute missing
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i  = imp.transform(X_test)

# ----------------------------
# Models
# ----------------------------
results = {}

if USE_LOGREG:
    # class_weight helps if warnings are rare
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train_i, y_train)

    p_lr = lr.predict_proba(X_test_i)[:, 1]
    yhat_lr = (p_lr >= 0.5).astype(int)

    results["LogReg"] = (yhat_lr, p_lr)

if USE_RF:
    rf = RandomForestClassifier(
        n_estimators=800,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(X_train_i, y_train)

    p_rf = rf.predict_proba(X_test_i)[:, 1]
    yhat_rf = (p_rf >= 0.5).astype(int)

    results["RF"] = (yhat_rf, p_rf)

# ----------------------------
# Evaluation helper
# ----------------------------
def eval_model(name, yhat, prob):
    cm = confusion_matrix(y_test, yhat, labels=[0,1])
    bal = balanced_accuracy_score(y_test, yhat)
    try:
        roc = roc_auc_score(y_test, prob)
    except Exception:
        roc = np.nan
    try:
        ap = average_precision_score(y_test, prob)
    except Exception:
        ap = np.nan

    print("\n====================")
    print("MODEL:", name)
    print("====================")
    print("Threshold: 0.5")
    print("Balanced accuracy:", round(bal, 4))
    print("ROC AUC:", round(roc, 4))
    print("Avg Precision (PR AUC):", round(ap, 4))
    print("\nConfusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, yhat, digits=4))

for name, (yhat, prob) in results.items():
    eval_model(name, yhat, prob)

# ----------------------------
# Save predictions (include both models if used)
# ----------------------------
out = test[["year", "FSI_geomean", "FSI_prev", "FSI_pct_change", "y_warning"]].copy()

for name, (yhat, prob) in results.items():
    out[f"pred_{name}"] = yhat
    out[f"prob_{name}"] = prob

out.to_csv(OUT_PRED, index=False)
print("\nSaved predictions:", OUT_PRED)
print("\nPreview saved file:")
print(out.head(12))
