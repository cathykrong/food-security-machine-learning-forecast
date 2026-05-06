"""
Directional evaluation for ΔFSI:
- ROC curve (Positive = ΔFSI > 0)
- Confusion matrix (normalized by predicted class / column-normalized)

Usage:
python scripts/10_directional_eval.py --pred_csv outputs/preds_randomforest_deltafsi_test.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def col_normalized_cm(y_true, y_pred) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    colsum = cm.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = np.divide(cm, colsum, where=colsum != 0)
    return cmn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/figs"))
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv).sort_values("year")

    # True / score for ROC: use predicted ΔFSI as score
    y_true = (df["dFSI_true"].values > args.eps).astype(int)
    scores = df["dFSI_pred"].values.astype(float)

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ROC plot (force start 0,0 and end 1,1)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_roc.png", dpi=200)

    # Confusion matrix at default threshold 0 (predict positive if score>0)
    y_pred = (scores > args.eps).astype(int)
    cmn = col_normalized_cm(y_true, y_pred) * 100.0  # percent

    plt.figure()
    plt.imshow(cmn, aspect="auto")
    for (i, j), v in np.ndenumerate(cmn):
        plt.text(j, i, f"{v:.1f}%", ha="center", va="center")
    plt.xticks([0, 1], ["Pred Neg", "Pred Pos"])
    plt.yticks([0, 1], ["Actual Neg", "Actual Pos"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_confusion_matrix_colnorm.png", dpi=200)

    print(f"Saved ROC + confusion matrix to: {args.out_dir}")

if __name__ == "__main__":
    main()
