"""
Build FSI from 4 FAOSTAT variables using min-max normalization then equal-weight mean.

Expected columns in input:
- year
- kcal_supply_kcal_cap_day
- protein_supply_g_cap_day
- gdp_pc_usd
- food_prod_index

Output: fsi_annual.csv with columns: year, FSI, plus normalized components.

Usage:
python scripts/03_build_fsi_from_faostat.py --faostat data/faostat_4vars.csv --out data/fsi_annual.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COMPONENTS = [
    "kcal_supply_kcal_cap_day",
    "protein_supply_g_cap_day",
    "gdp_pc_usd",
    "food_prod_index",
]

def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo = np.nanmin(s.values)
    hi = np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faostat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.faostat)
    if "year" not in df.columns:
        raise SystemExit("Input must contain column: year")
    for c in COMPONENTS:
        if c not in df.columns:
            raise SystemExit(f"Missing FAOSTAT component: {c}")

    out = df[["year"]].copy()
    for c in COMPONENTS:
        out[c + "_norm"] = minmax(df[c])

    out["FSI"] = out[[c + "_norm" for c in COMPONENTS]].mean(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("year").to_csv(args.out, index=False)
    print(f"Saved: {args.out}  shape={out.shape}")

if __name__ == "__main__":
    main()
