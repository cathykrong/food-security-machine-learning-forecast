import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# ----------------------------
# INPUT
# ----------------------------
PRED_PATH = Path(r"E:/1Cathy/hsbdc/AI2026/data2/final_predictions_2013_2023_rf_deltaFSI_from_original_engineered.csv")
EPS = 0.0
N_GRID = 500
CLIP_TO_01 = True

LEGEND_LOC = "upper left"
SHOW_RAW_POINTS = True

# ----------------------------
# LOAD + SHOW DATA USED
# ----------------------------
pred = pd.read_csv(PRED_PATH)
used_cols = ["year", "dFSI_actual", "dFSI_pred"]
print("\nColumns used for ROC:", used_cols)
print("\nPreview of data used (first 12 rows):")
print(pred[used_cols].head(12).to_string(index=False))

y_true = (pred["dFSI_actual"].to_numpy() > EPS).astype(int)
scores = pred["dFSI_pred"].to_numpy()

print("\nTest years:", pred["year"].min(), "to", pred["year"].max(), f"(N={len(pred)})")
print("Actual positives:", int(y_true.sum()), "| Actual negatives:", int((1-y_true).sum()))

# ----------------------------
# RAW ROC + AUC
# ----------------------------
fpr, tpr, _ = roc_curve(y_true, scores)
auc = roc_auc_score(y_true, scores)

# Force endpoints into raw ROC arrays (for display)
if not (np.isclose(fpr[0], 0.0) and np.isclose(tpr[0], 0.0)):
    fpr = np.insert(fpr, 0, 0.0)
    tpr = np.insert(tpr, 0, 0.0)
if not (np.isclose(fpr[-1], 1.0) and np.isclose(tpr[-1], 1.0)):
    fpr = np.append(fpr, 1.0)
    tpr = np.append(tpr, 1.0)

# Compress duplicate FPR values (keep max TPR)
order = np.argsort(fpr)
fpr_s = fpr[order]
tpr_s = tpr[order]
fpr_u = np.unique(fpr_s)
tpr_u = np.array([tpr_s[fpr_s == x].max() for x in fpr_u])

# ----------------------------
# QUADRATIC FIT WITH ENDPOINT CONSTRAINTS
# Model: TPR(x) = a*x*(x-1) + x
# => TPR(0)=0 and TPR(1)=1 automatically
# Solve for 'a' via least squares:
# minimize sum_i [a*x_i*(x_i-1) + x_i - tpr_i]^2
# ----------------------------
x = fpr_u
y = tpr_u
phi = x * (x - 1.0)               # basis term
rhs = y - x                       # move "+ x" to the other side

# Least squares for a: rhs ≈ a*phi
den = np.dot(phi, phi)
a = float(np.dot(phi, rhs) / den) if den != 0 else 0.0

print("\nSingle-equation quadratic (forced through (0,0) and (1,1)):")
print(f"TPR(FPR) = ({a:.6f})*FPR*(FPR - 1) + FPR")
print(f"Raw ROC AUC (official, from scores): {auc:.3f}")

# Generate smooth curve
fpr_grid = np.linspace(0.0, 1.0, N_GRID)
tpr_quad = a * fpr_grid * (fpr_grid - 1.0) + fpr_grid

# Ensure exact endpoints
tpr_quad[0] = 0.0
tpr_quad[-1] = 1.0

if CLIP_TO_01:
    tpr_quad = np.clip(tpr_quad, 0.0, 1.0)

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_grid, tpr_quad, linewidth=2, label=f"ROC Curve (AUC = {auc:.2f})")


plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Chance")
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve (Positive = ΔFSI > 0)")
plt.legend(loc=LEGEND_LOC, frameon=True)
plt.tight_layout()
plt.show()
