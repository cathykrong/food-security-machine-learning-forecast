"""
Make the core figures:
- Time series: Actual vs Pred vs Persistence
- Pred vs Actual scatter (+ y=x line)
- Residuals over time

Usage:
python scripts/08_plots_core.py --pred_csv outputs/preds_randomforest_deltafsi_test.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/figs"))
    ap.add_argument("--ymin", type=float, default=None)  # e.g., 0.60
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv).sort_values("year")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Time series
    plt.figure()
    plt.plot(df["year"], df["FSI_true"], marker="o", label="Actual FSI")
    plt.plot(df["year"], df["FSI_pred"], marker="o", label="Model (ΔFSI-reconstructed)")
    plt.plot(df["year"], df["FSI_persist"], marker="o", label="Persistence (FSI(t-1))")
    plt.xlabel("Year")
    plt.ylabel("FSI (0–1)")
    if args.ymin is not None:
        plt.ylim(bottom=args.ymin)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_timeseries.png", dpi=200)

    # 2) Scatter
    plt.figure()
    x = df["FSI_true"].values
    y = df["FSI_pred"].values
    plt.scatter(x, y)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual FSI")
    plt.ylabel("Predicted FSI")
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_scatter.png", dpi=200)

    # 3) Residuals
    plt.figure()
    resid = df["FSI_true"] - df["FSI_pred"]
    plt.plot(df["year"], resid, marker="o")
    plt.axhline(0)
    plt.xlabel("Year")
    plt.ylabel("Residual (Actual − Pred)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_residuals.png", dpi=200)

    print(f"Saved figures to: {args.out_dir}")

if __name__ == "__main__":
    main()
