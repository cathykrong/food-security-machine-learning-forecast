"""
Download USDA NASS QuickStats for USA annual crop metrics.

Usage example:
export QUICKSTATS_KEY="YOUR_KEY"
python scripts/01_quickstats_download.py --out_dir data/raw_quickstats

Notes:
- This script is optional if you already have cleaned data.
- You can expand metrics/commodities as needed.
"""
from __future__ import annotations
import os
import time
import random
import argparse
from pathlib import Path
import requests
import pandas as pd

BASE = "https://quickstats.nass.usda.gov/api"

DEFAULT_COMMODITIES = ["CORN", "SOYBEANS", "WHEAT"]
# QuickStats often uses these statistic categories; you can adjust to match your pulls.
DEFAULT_ITEMS = [
    # (statisticcat_desc, unit_desc)
    ("YIELD", None),
    ("PRODUCTION", None),
    ("AREA HARVESTED", None),
]

def fetch_quickstats(params: dict, key: str, max_retries: int = 5) -> list[dict]:
    url = f"{BASE}/api_GET/"
    params = dict(params)
    params["key"] = key

    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            js = r.json()
            return js.get("data", [])
        # backoff on rate limit / transient issues
        sleep_s = min(30, 1.5 ** attempt) + random.random()
        time.sleep(sleep_s)
    r.raise_for_status()
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--start_year", type=int, default=1970)
    ap.add_argument("--end_year", type=int, default=2023)
    ap.add_argument("--commodities", nargs="+", default=DEFAULT_COMMODITIES)
    args = ap.parse_args()

    key = os.getenv("QUICKSTATS_KEY", "").strip()
    if not key:
        raise SystemExit("Set QUICKSTATS_KEY env var first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for commodity in args.commodities:
        for (statcat, unit) in DEFAULT_ITEMS:
            params = {
                "sector_desc": "CROPS",
                "group_desc": "FIELD CROPS",
                "commodity_desc": commodity,
                "freq_desc": "ANNUAL",
                "agg_level_desc": "NATIONAL",
                "year__GE": str(args.start_year),
                "year__LE": str(args.end_year),
                "source_desc": "SURVEY",
                "statisticcat_desc": statcat,
            }
            if unit:
                params["unit_desc"] = unit

            data = fetch_quickstats(params, key=key)
            all_rows.extend(data)
            time.sleep(0.25)  # gentle pacing

    df = pd.DataFrame(all_rows)
    out = args.out_dir / "quickstats_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}  rows={len(df):,}")

if __name__ == "__main__":
    main()
