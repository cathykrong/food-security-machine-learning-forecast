"""
Train models to predict ΔFSI and reconstruct FSI_pred(t)=FSI(t-1)+ΔFSI_pred(t).

Usage (your main benchmark):
python scripts/07_train_models_delta.py --data /mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv --target FSI
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from scripts.utils_metrics import reg_metrics

def make_numeric_X(df: pd.DataFrame, year_col: str, drop_cols: list[str]):
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if year_col in X.columns:
        X = X.drop(columns=[year_col])
    X = X.select_dtypes(include=[np.number])
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--year_col", type=str, default="year")
    ap.add_argument("--target", type=str, default="FSI")
    ap.add_argument("--train_end", type=int, default=2012)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs"))
    ap.add_argument("--rf_trees", type=int, default=800)
    args = ap.parse_args()

    df = pd.read_csv(args.data).sort_values(args.year_col).reset_index(drop=True)

    # Build ΔFSI
    df["FSI_lag1"] = df[args.target].shift(1)
    df["dFSI"] = df[args.target] - df["FSI_lag1"]

    # Drop first row (no lag1)
    df2 = df.dropna(subset=["FSI_lag1", "dFSI"]).copy()

    train = df2[df2[args.year_col] <= args.train_end].copy()
    test  = df2[df2[args.year_col] >  args.train_end].copy()

    Xtr = make_numeric_X(train, args.year_col, drop_cols=[args.target])
    ytr = train["dFSI"].astype(float)

    Xte = make_numeric_X(test, args.year_col, drop_cols=[args.target])
    yte = test["dFSI"].astype(float)

    models = {
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 60)),
        "RandomForest": RandomForestRegressor(
            n_estimators=args.rf_trees, random_state=42, min_samples_leaf=1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    # Persistence baseline for level prediction: FSI(t)=FSI(t-1)
    # In reconstructed delta terms, that’s Δ=0 ⇒ FSI_pred(t)=FSI_lag1
    y_level_true = test[args.target].astype(float).values
    y_level_persist = test["FSI_lag1"].astype(float).values

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, model in models.items():
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", model),
        ])
        pipe.fit(Xtr, ytr)
        d_pred = pipe.predict(Xte)

        # reconstruct level
        fsi_pred = test["FSI_lag1"].values + d_pred

        m = reg_metrics(y_level_true, fsi_pred)
        mb = reg_metrics(y_level_true, y_level_persist)

        rows.append({
            "model": name,
            **{f"test_{k}": v for k, v in m.items()},
            **{f"base_{k}": v for k, v in mb.items()},
        })

        out_pred = args.out_dir / f"preds_{name.lower()}_deltafsi_test.csv"
        pd.DataFrame({
            "year": test[args.year_col].values,
            "FSI_true": y_level_true,
            "FSI_pred": fsi_pred,
            "FSI_persist": y_level_persist,
            "dFSI_true": yte.values,
            "dFSI_pred": d_pred,
        }).to_csv(out_pred, index=False)

    out_metrics = args.out_dir / "metrics_deltafsi_reconstructed.csv"
    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    print(f"Saved metrics: {out_metrics}")
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
