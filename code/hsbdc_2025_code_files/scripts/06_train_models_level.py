"""
Train models to predict FSI(t) directly (level formulation).

Usage:
python scripts/06_train_models_level.py --data /mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv --target FSI
"""

import pandas as pd
from train_models_level import train_models_level

df = pd.read_csv(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags_plus_macro_cpi_pinksheets.csv")
res = train_models_level(df=df, target="FSI_geomean", out_dir=r"E:\1Cathy\hsbdc\AI2026\data2\outputs_level")

print(res.metrics)
print(res.preds["RidgeCV"].head())

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from scripts.utils_metrics import reg_metrics

def make_Xy(df: pd.DataFrame, year_col: str, target: str):
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    # drop year from features (keeps time-respecting split without leaking)
    if year_col in X.columns:
        X = X.drop(columns=[year_col])
    # keep numeric only
    X = X.select_dtypes(include=[np.number])
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--year_col", type=str, default="year")
    ap.add_argument("--target", type=str, default="FSI")
    ap.add_argument("--train_end", type=int, default=2012)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    df = pd.read_csv(args.data).sort_values(args.year_col).reset_index(drop=True)

    train = df[df[args.year_col] <= args.train_end].copy()
    test  = df[df[args.year_col] >  args.train_end].copy()

    Xtr, ytr = make_Xy(train, args.year_col, args.target)
    Xte, yte = make_Xy(test,  args.year_col, args.target)

    models = {
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 60)),
        "RandomForest": RandomForestRegressor(
            n_estimators=800, random_state=42, min_samples_leaf=1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, model in models.items():
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", model),
        ])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

        # persistence baseline (level): FSI(t)=FSI(t-1)
        # aligned on test years: need lagged actual from full df
        full = df[[args.year_col, args.target]].copy()
        full["pers"] = full[args.target].shift(1)
        base = full.loc[full[args.year_col].isin(test[args.year_col]), "pers"].values

        m = reg_metrics(yte, pred)
        mb = reg_metrics(yte, base)

        rows.append({
            "model": name,
            **{f"test_{k}": v for k, v in m.items()},
            **{f"base_{k}": v for k, v in mb.items()},
        })

        # save preds
        out_pred = args.out_dir / f"preds_level_{name.lower()}.csv"
        pd.DataFrame({
            "year": test[args.year_col].values,
            "y_true": yte.values,
            "y_pred": pred,
            "y_persist": base,
        }).to_csv(out_pred, index=False)

    out_metrics = args.out_dir / "metrics_level.csv"
    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    print(f"Saved metrics: {out_metrics}")
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
