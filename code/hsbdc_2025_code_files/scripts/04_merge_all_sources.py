"""
Merge crops wide + FSI + macro series into a single year table.

If you already have: /mnt/data/us_quickstats_fsi_plus_macro_1970_2023.csv
you can skip this script.

Usage:
python scripts/04_merge_all_sources.py --crops data/crops_annual_wide.csv --fsi data/fsi_annual.csv --macro data/macro.csv --out data/us_plus_macro.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def read_year_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" not in df.columns:
        # try common variants
        for alt in ["YEAR", "Year"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "year"})
                break
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.dropna(subset=["year"]).assign(year=lambda d: d["year"].astype(int))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--fsi", type=Path, required=True)
    ap.add_argument("--macro", type=Path, required=False)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    crops = read_year_csv(args.crops)
    fsi = read_year_csv(args.fsi)

    df = crops.merge(fsi, on="year", how="inner")

    if args.macro:
        macro = read_year_csv(args.macro)
        df = df.merge(macro, on="year", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("year").to_csv(args.out, index=False)
    print(f"Saved: {args.out}  shape={df.shape}")

if __name__ == "__main__":
    main()
