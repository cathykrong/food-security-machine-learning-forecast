import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# ============================================================
# OPTION 2: %ΔFSI relative to last year (classic percent change)
# %ΔFSI = 100*(FSI_t - FSI_{t-1}) / FSI_{t-1}
#
# CLASSIFICATION MODEL: LogReg predicts warning (0/1)
# PLOTS: use %ΔFSI values (not probabilities)
# ============================================================

# ----------------------------
# PATHS (edit)
# ----------------------------
IN_TABLE = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags.csv")
OUT_DIR  = Path(r"E:\1Cathy\hsbdc\AI2026\results3\figs_logreg_pctdeltaFSI")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# SPLIT
# ----------------------------
TRAIN_END  = 2012
TEST_START = 2013
TEST_END   = 2023

# ----------------------------
# WARNING DEFINITION (percent drop)
# e.g., warning if %ΔFSI <= -10%
# ----------------------------
DROP_PCT = 10.0     # percent
WARN_IF_PCT_LEQ = -DROP_PCT

# ----------------------------
# IMPORTANT: denominator floor to avoid insane % when FSI_prev is tiny
# Set to 0.0 to disable (pure option 2).
# Typical safe values: 0.05, 0.10, 0.20
# ----------------------------
DEN_FLOOR = 0.10

# ----------------------------
# Prediction threshold for class (only for marking predicted warnings)
# ----------------------------
PROB_THRESH = 0.50

# ----------------------------
# Avoid leakage in features
# ----------------------------
LEAK_COLS = {
    "FSI_geomean","FSI_prev","dFSI","pct_dFSI","y_warning",
    "FSI_pct_change","FSI_pct_change_robust","FSI_log_change",
    "FSI_lag1","FSI_lag2","dFSI_lag1","dFSI_lag2",
}

def get_feature_cols(df):
    cols = []
    for c in df.columns:
        if c == "year" or c in LEAK_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

# ----------------------------
# Load + build %ΔFSI
# ----------------------------
df = pd.read_csv(IN_TABLE).sort_values("year").reset_index(drop=True)

df["FSI_prev"] = df["FSI_geomean"].shift(1)
df["dFSI"] = df["FSI_geomean"] - df["FSI_prev"]

den = df["FSI_prev"].copy()
if DEN_FLOOR and DEN_FLOOR > 0:
    den = den.clip(lower=DEN_FLOOR)

df["pct_dFSI"] = 100.0 * (df["dFSI"] / den)

# Label: warning if %ΔFSI <= -DROP_PCT
df["y_warning"] = (df["pct_dFSI"] <= WARN_IF_PCT_LEQ).astype("Int64")
df.loc[df["FSI_prev"].isna(), "y_warning"] = pd.NA  # first year undefined

train = df[(df["year"] <= TRAIN_END) & (df["y_warning"].notna())].copy()
test  = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END) & (df["y_warning"].notna())].copy()

y_train = train["y_warning"].astype(int).to_numpy()
y_test  = test["y_warning"].astype(int).to_numpy()

print("Train label counts:", np.bincount(y_train) if len(y_train) else "empty")
print("Test  label counts:", np.bincount(y_test) if len(y_test) else "empty")

# ----------------------------
# Train LogReg classifier (classification only)
# ----------------------------
feature_cols = get_feature_cols(df)

for c in feature_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c]  = pd.to_numeric(test[c], errors="coerce")

X_train = train[feature_cols].to_numpy()
X_test  = test[feature_cols].to_numpy()

imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i  = imp.transform(X_test)

logreg = LogisticRegression(max_iter=5000, class_weight="balanced")
logreg.fit(X_train_i, y_train)

prob_test = logreg.predict_proba(X_test_i)[:, 1]
pred_test = (prob_test >= PROB_THRESH).astype(int)

cm = confusion_matrix(y_test, pred_test, labels=[0,1])
bal = balanced_accuracy_score(y_test, pred_test)
print("\nConfusion matrix:\n", cm)
print("Balanced accuracy:", round(bal, 4))

out = test[["year","FSI_geomean","FSI_prev","dFSI","pct_dFSI","y_warning"]].copy()
out["prob_LogReg"] = prob_test
out["pred_LogReg"] = pred_test

out_path = OUT_DIR / f"pred_logreg_pctdeltaFSI_drop{int(DROP_PCT)}pct_{TEST_START}_{TEST_END}.csv"
out.to_csv(out_path, index=False)
print("Saved:", out_path)

# ============================================================
# PLOTS BASED ON %ΔFSI (NOT probabilities)
# ============================================================

# 1) FSI level with actual vs predicted warning markers
fig = plt.figure(figsize=(10,4), dpi=200)
ax = plt.gca()
ax.plot(out["year"], out["FSI_geomean"], marker="o", linewidth=2, label="FSI (level)")

ax.scatter(out.loc[out["y_warning"]==1, "year"],
           out.loc[out["y_warning"]==1, "FSI_geomean"],
           s=70, label=f"Actual warning (%ΔFSI ≤ {WARN_IF_PCT_LEQ:.0f}%)")

ax.scatter(out.loc[out["pred_LogReg"]==1, "year"],
           out.loc[out["pred_LogReg"]==1, "FSI_geomean"],
           marker="x", s=90, label=f"Predicted warning (p≥{PROB_THRESH:.2f})")

ax.set_xlabel("Year")
ax.set_ylabel("FSI (0–1)")
ax.set_title("FSI Level with Warning Years (actual vs predicted)")
ax.legend(frameon=True, loc="best")
plt.tight_layout()
p1 = OUT_DIR / "FIG_FSI_level_with_warnings_pctdelta.png"
plt.savefig(p1, bbox_inches="tight")
plt.show()
print("Saved:", p1)

# 2) %ΔFSI over time (this is the main plot you asked for)
fig = plt.figure(figsize=(10,4), dpi=200)
ax = plt.gca()
ax.axhline(0, linewidth=1)
ax.axhline(WARN_IF_PCT_LEQ, linestyle="--", linewidth=1,
           label=f"Warning threshold = {WARN_IF_PCT_LEQ:.0f}%")
ax.plot(out["year"], out["pct_dFSI"], marker="o", linewidth=2,
        label="%ΔFSI = 100*(FSI_t - FSI_{t-1})/FSI_{t-1}")

# mark predicted warnings on the %Δ plot
ax.scatter(out.loc[out["pred_LogReg"]==1, "year"],
           out.loc[out["pred_LogReg"]==1, "pct_dFSI"],
           marker="x", s=90, label="Predicted warning")

ax.set_xlabel("Year")
ax.set_ylabel("%ΔFSI")
ax.set_title(f"Year-to-year %ΔFSI (relative to last year) — drop threshold {WARN_IF_PCT_LEQ:.0f}%")
ax.legend(frameon=True, loc="best")
plt.tight_layout()
p2 = OUT_DIR / "FIG_pctdeltaFSI_over_time.png"
plt.savefig(p2, bbox_inches="tight")
plt.show()
print("Saved:", p2)

# 3) Actual %ΔFSI vs predicted class (0/1)
fig = plt.figure(figsize=(7,4), dpi=200)
ax = plt.gca()
ax.scatter(out["pct_dFSI"], out["pred_LogReg"])
ax.axvline(WARN_IF_PCT_LEQ, linestyle="--", linewidth=1, label="Warning boundary")
ax.set_xlabel("Actual %ΔFSI")
ax.set_ylabel("Predicted warning class")
ax.set_title("Predicted warning vs actual %ΔFSI")
ax.set_yticks([0,1])
ax.set_yticklabels(["No warning", "Warning"])
ax.legend(frameon=True, loc="best")
plt.tight_layout()
p3 = OUT_DIR / "FIG_predclass_vs_pctdeltaFSI.png"
plt.savefig(p3, bbox_inches="tight")
plt.show()
print("Saved:", p3)
