# Confusion matrix with DIFFERENT SHADES OF BLUE (matplotlib "Blues" colormap)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

PRED_CSV = Path(r"E:\1Cathy\hsbdc\AI2026\data3\models_cls_predictions_warning_10pctdrop_2013_2023_robust.csv")
OUT_PNG  = Path(r"E:\1Cathy\hsbdc\AI2026\data3\FIG2_LogReg_warning_ROC_confmat_BLUE.png")

PROB_COL = "prob_LogReg"
THRESH = 0.50

df = pd.read_csv(PRED_CSV).sort_values("year").reset_index(drop=True)
y_true = df["y_warning"].astype(int).to_numpy()
y_score = df[PROB_COL].astype(float).to_numpy()
y_pred = (y_score >= THRESH).astype(int)

# ROC
fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

# Confusion matrix -> column-normalized %
cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
col_sums = cm.sum(axis=0, keepdims=True)
cm_pct = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0) * 100.0

fig = plt.figure(figsize=(12, 5), dpi=200)

# ROC
ax1 = plt.subplot(1, 2, 1)
ax1.plot(fpr, tpr, linewidth=2, label=f"LogReg ROC (AUC = {auc:.3f})")
ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve (2013–2023)")
ax1.legend(loc="lower right", frameon=True)

# Confusion matrix (blue shades)
ax2 = plt.subplot(1, 2, 2)
im = ax2.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")  # <-- blue shades

ax2.set_title(f"Confusion Matrix (%), column-normalized\nThreshold = {THRESH:.2f}")
ax2.set_xlabel("Predicted class")
ax2.set_ylabel("True class")
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Negative (0)", "Positive (1)"])
ax2.set_yticklabels(["Negative (0)", "Positive (1)"])

for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center")

# Colorbar to show blue intensity scale
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Percent (%)")

plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PNG)
