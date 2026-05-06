import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_quadratic_roc_from_auc(
    auc: float = 0.71,
    out_roc: Path = Path("ROC_quadratic_smooth_AUC_0p71.png"),
    show_empirical: bool = False
):
    """
    Draw a *purely illustrative* smooth ROC curve from a single AUC value using:
        y = x^k,  (0<=x<=1)
    This curve:
      - starts at (0,0) and ends at (1,1)
      - is smooth and monotone
      - has exact AUC = 1/(k+1)

    Given AUC, we set:
        k = (1/AUC) - 1

    For AUC=0.71, k≈0.40845 (concave curve above chance).
    """

    auc = float(auc)
    if not (0.5 < auc < 1.0):
        raise ValueError("AUC must be between 0.5 and 1.0 for a meaningful ROC above chance.")

    k = (1.0 / auc) - 1.0

    x = np.linspace(0.0, 1.0, 500)
    y = np.power(x, k)

    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = plt.gca()

    # Smooth ROC from AUC
    ax.plot(x, y, linewidth=2, label=f"Smooth ROC (AUC = {auc:.2f})")

    # Chance line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve")
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    plt.savefig(out_roc, bbox_inches="tight")
    plt.show()


# Example usage:
plot_quadratic_roc_from_auc(
    auc=0.71,
    out_roc=Path("ROC_quadratic_smooth_AUC_0p71.png")
)
