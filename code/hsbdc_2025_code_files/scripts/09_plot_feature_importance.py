"""
Train a RandomForest on ΔFSI and plot top-N feature importances (barh).

Usage:
python scripts/09_plot_feature_importance.py --data /mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv --target FSI
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--year_col", type=str, default="year")
    ap.add_argument("--target", type=str, default="FSI")
    ap.add_argument("--train_end", type=int, default=2012)
    ap.add_argument("--top_n", type=int, default=15)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/figs"))
    args = ap.parse_args()

    df = pd.read_csv(args.data).sort_values(args.year_col).reset_index(drop=True)
    df["FSI_lag1"] = df[args.target].shift(1)
    df["dFSI"] = df[args.target] - df["FSI_lag1"]
    df = df.dropna(subset=["FSI_lag1", "dFSI"]).copy()

    train = df[df[args.year_col] <= args.train_end].copy()

    X = train.drop(columns=[args.target, "dFSI"], errors="ignore")
    if args.year_col in X.columns:
        X = X.drop(columns=[args.year_col])
    X = X.select_dtypes(include=[np.number])
    y = train["dFSI"].astype(float)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=800, random_state=42)),
    ])
    pipe.fit(X, y)

    rf = pipe.named_steps["rf"]
    importances = rf.feature_importances_
    feat = np.array(X.columns)

    idx = np.argsort(importances)[::-1][: args.top_n]
    feat_top = feat[idx][::-1]
    imp_top = importances[idx][::-1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.barh(feat_top, imp_top)
    plt.xlabel("Feature importance (RandomForest)")
    plt.ylabel("Predictor")
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_feature_importance_top15.png", dpi=200)
    print(f"Saved: {args.out_dir / 'fig_feature_importance_top15.png'}")

if __name__ == "__main__":
    main()
