import numpy as np
import matplotlib.pyplot as plt

# =========================
# Example 2 (overall % of all years)
# Positives ~64%, Negatives ~36%
# TP=44.8%, FN=19.2%, FP=12.6%, TN=23.4%
# Confusion matrix format: rows=Actual (Neg, Pos), cols=Pred (Neg, Pos)
# =========================
cm_overall = np.array([
    [76.6, 12.6],  # Actual Negative:  TN, FP
    [23.4, 87.4],  # Actual Positive:  FN, TP
], dtype=float)

# Column-normalized version (percent within predicted class), matching your "within predicted class" style
cm_colnorm = np.array([
    [54.9, 22.0],  # Actual Negative in Pred Neg / Pred Pos
    [45.1, 78.0],  # Actual Positive in Pred Neg / Pred Pos
], dtype=float)


def plot_cm(cm_pct, title, out_path=None, vmin=0, vmax=100):
    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = plt.gca()

    # Blue shades
    im = ax.imshow(cm_pct, vmin=vmin, vmax=vmax, cmap="Blues")

    # Annotate cells
    for (i, j), v in np.ndenumerate(cm_pct):
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Negative", "Pred Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


# Plot overall-percent CM
plot_cm(
    cm_overall,
    "Binary Confusion Matrix (Percent Within Predicted Class)",
    out_path="cm_example2_overall_percent.png"
)
