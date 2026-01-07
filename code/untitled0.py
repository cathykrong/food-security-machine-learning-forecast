
# -*- coding: utf-8 -*-
import time
import random
import requests
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
BASE = "https://quickstats.nass.usda.gov/api"
KEY = "6F514F01-32FC-397B-B0CE-78B7857D8A9E"   # <-- paste your key here as a string

OUT_DIR = Path("E:/1Cathy/hsbdc/AI2026/data2/quickstats_downloads")
OUT_DIR.mkdir(exist_ok=True)

YEARS = range(1960, 2024)

STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
    "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
    "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

SOURCES = ["SURVEY", "CENSUS"]

CROPS = ["CORN", "WHEAT", "SOYBEANS"]

# Use component fields instead of short_desc guessing.
# (short_desc must match exactly and is a concatenation of multiple columns) [1](https://github.com/jackheinemann/nasspython/blob/main/README.md)
DATA_ITEMS = [
    {"name": "yield",          "statisticcat_desc": "YIELD",          "unit_desc": "BU / ACRE"},
    {"name": "production",     "statisticcat_desc": "PRODUCTION",     "unit_desc": "BU"},
    {"name": "area_planted",   "statisticcat_desc": "AREA PLANTED",   "unit_desc": "ACRES"},
    {"name": "area_harvested", "statisticcat_desc": "AREA HARVESTED", "unit_desc": "ACRES"},
]

BASE_PARAMS = {
    "sector_desc": "CROPS",
    "group_desc": "FIELD CROPS",
    "agg_level_desc": "NATIONAL",
    "freq_desc": "ANNUAL",
    # "domain_desc": "TOTAL",  # <-- remove this
}

# ----------------------------
# HTTP: session + headers + pacing + retry/backoff
# ----------------------------
SESSION = requests.Session()
SESSION.headers.update({
    # Many gateways/WAFs block “bot-like” clients; a browser-like UA helps. [3](https://stackoverflow.com/questions/38489386/how-to-fix-403-forbidden-errors-when-calling-apis-using-python-requests)
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
})

_MIN_GAP_SECONDS = 0.35  # ~3 req/sec; conservative to avoid throttling
_last_call = 0.0

def _paced():
    global _last_call
    now = time.monotonic()
    wait = _MIN_GAP_SECONDS - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()

def _get(endpoint, params, timeout=60, max_retries=8):
    """
    Robust GET with pacing + retries/backoff on 403/429/5xx.
    403 can be caused by access restriction policies at gateways. [2](https://learn.microsoft.com/en-us/troubleshoot/azure/api-mgmt/availability/request-throttling-http-403)
    """
    url = f"{BASE}/{endpoint.lstrip('/')}"
    params = {**params, "key": KEY}

    last = None
    for attempt in range(max_retries):
        _paced()
        r = SESSION.get(url, params=params, timeout=timeout)
        last = r

        if r.ok:
            return r

        # Retry these common transient/policy outcomes
        if r.status_code in (403, 429, 500, 502, 503, 504):
            backoff = min(60, (2 ** attempt) + random.random())
            time.sleep(backoff)
            continue

        raise requests.HTTPError(
            f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text[:500]}"
        )

    raise requests.HTTPError(
        f"Failed after {max_retries} retries\nURL: {last.url}\n"
        f"Last status: {last.status_code}\nBody: {last.text[:500]}"
    )

def qs_count(params):
    r = _get("get_counts/", params, timeout=60)
    return int(r.json()["count"])

def qs_get(params):
    r = _get("api_GET/", params, timeout=120)
    return pd.DataFrame(r.json().get("data", []))

def fetch_safe(params, years):
    frames = []
    for year in years:
        p_year = {**params, "year": str(year)}
        n = qs_count(p_year)
        if n == 0:
            continue
        frames.append(qs_get(p_year))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    

# ----------------------------
# MAIN: download
# ----------------------------
def main():
    if not KEY or "Y" in KEY:
        raise RuntimeError("You must set KEY = 'YOUR_API_KEY' at the top of the script.")

    for crop in CROPS:
        for item in DATA_ITEMS:
            all_frames = []
            for src in SOURCES:
                params = {
                    **BASE_PARAMS,
                    "commodity_desc": crop,
                    "source_desc": src,
                    "statisticcat_desc": item["statisticcat_desc"],
                    "unit_desc": item["unit_desc"],
                }

                print(f"Downloading: {crop} | {item['name']} | {src}")
                df_part = fetch_safe(params, YEARS)
                if not df_part.empty:
                    all_frames.append(df_part)

            df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

            out_path = OUT_DIR / f"{crop.lower()}_{item['name']}_county.csv"
            df.to_csv(out_path, index=False)
            print(f"  ✔ Saved {out_path} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
