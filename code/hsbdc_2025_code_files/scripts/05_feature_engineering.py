"""
Create shocks (YoY %) and lags (t-1, t-2) for all numeric predictors.

Output includes original columns + *_yoy_pct + *_lag1 + *_lag2 + (optional) lagged shocks.

Usage:
python scripts/05_feature_engineering.py --in_csv /mnt/data/us_quickstats_fsi_plus_macro_1970_2023.csv --out_csv outputs/us_with_shocks_lags.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def yoy_pct(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    prev = s.shift(1)
    out = (s - prev) / prev
    # avoid blow-ups where prev is 0 or NaN
    out = out.where(prev.replace(0, np.nan).notna(), np.nan)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--year_col", type=str, default="year")
    ap.add_argument("--target_col", type=str, default="FSI")
    ap.add_argument("--lag_depth", type=int, default=2)
    ap.add_argument("--make_lagged_shocks", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df.sort_values(args.year_col).reset_index(drop=True)

    # Numeric predictors = all numeric-ish columns except year and target
    exclude = {args.year_col, args.target_col}
    cols = [c for c in df.columns if c not in exclude]

    # Keep only columns that can be numeric
    num_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            df[c] = s
            num_cols.append(c)

    # shocks
    for c in num_cols:
        df[f"{c}_yoy_pct"] = yoy_pct(df[c])

    # lags
    for lag in range(1, args.lag_depth + 1):
        for c in num_cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
        for c in [f"{c}_yoy_pct" for c in num_cols]:
            if args.make_lagged_shocks:
                df[f"{c}_lag{lag}"] = df[c].shift(lag)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}  shape={df.shape}")

if __name__ == "__main__":
    main()
