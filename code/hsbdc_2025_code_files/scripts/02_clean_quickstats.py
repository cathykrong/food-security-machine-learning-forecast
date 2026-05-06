"""
Clean/reshape QuickStats raw into annual wide format: one row per year.

Input: quickstats_raw.csv (from 01) OR any QuickStats export with VALUE, YEAR, COMMODITY_DESC, STATISTICCAT_DESC
Output: crops_annual_wide.csv

Usage:
python scripts/02_clean_quickstats.py --raw data/raw_quickstats/quickstats_raw.csv --out data/crops_annual_wide.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NA_TOKENS = {"(D)", "(NA)", "(N/A)", "(S)", "(X)", ""}

def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in NA_TOKENS:
        return np.nan
    if s == "(Z)":
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def norm_name(commodity: str, statcat: str) -> str:
    c = commodity.strip().lower().replace(" ", "_")
    s = statcat.strip().lower().replace(" ", "_")
    return f"{c}_{s}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.raw, dtype=str)
    # Harmonize likely column names
    df.columns = [c.upper() for c in df.columns]

    # Required columns
    need = {"YEAR", "COMMODITY_DESC", "STATISTICCAT_DESC", "VALUE"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["VALUE_NUM"] = df["VALUE"].map(to_num)

    # Keep national annual numeric values
    df = df.dropna(subset=["YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)

    df["VAR"] = [
        norm_name(c, s) for c, s in zip(df["COMMODITY_DESC"], df["STATISTICCAT_DESC"])
    ]

    wide = (
        df.pivot_table(index="YEAR", columns="VAR", values="VALUE_NUM", aggfunc="mean")
        .reset_index()
        .rename(columns={"YEAR": "year"})
        .sort_values("year")
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  shape={wide.shape}")

if __name__ == "__main__":
    main()
