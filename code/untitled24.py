"""
Median-impute missing values (column-wise) and save a cleaned CSV.

- Treats common missing tokens like "(D)", "(NA)", "(S)", "(X)", "", "NA", "N/A" as missing
- For columns that are numeric (or can be coerced to numeric), fills NaNs with the column median
- Non-numeric columns are left unchanged (median doesn't apply); script prints which were skipped

Run:
python median_impute_csv.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================
DATA_PATH = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table_fsi_plus_crops_shocks_lags_plus_macro_cpi_pinksheet.csv")
OUT_PATH  = Path(r"E:\1Cathy\hsbdc\AI2026\data3\us_model_table__MEDIAN_IMPUTED.csv")

# Add/remove tokens that should be treated as missing
MISSING_TOKENS = {
    "(D)", "(NA)", "(N/A)", "(S)", "(X)", "(Z)",  # NOTE: if you want (Z)=0, remove it here
    "", " ", "NA", "N/A", "nan", "NaN", "NULL", "null"
}

# If True, tries to coerce object columns to numeric (recommended for your dataset)
COERCE_OBJECT_TO_NUMERIC = True

# If True, any column that becomes fully missing after coercion is left as-is (no median exists)
SKIP_ALL_MISSING_COLUMNS = True
# =========================


def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    # Replace specified missing tokens anywhere in the df with actual NaN
    return df.replace(list(MISSING_TOKENS), np.nan)


def median_impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_out = df.copy()

    # Track what happened
    report = {
        "imputed_numeric_cols": [],
        "skipped_non_numeric_cols": [],
        "skipped_all_missing_cols": [],
        "medians_used": {}
    }

    # Optionally coerce object columns to numeric if possible
    if COERCE_OBJECT_TO_NUMERIC:
        for c in df_out.columns:
            if df_out[c].dtype == "object":
                df_out[c] = pd.to_numeric(df_out[c], errors="ignore")

    for c in df_out.columns:
        s = df_out[c]

        # Only median-impute numeric columns
        if not pd.api.types.is_numeric_dtype(s):
            report["skipped_non_numeric_cols"].append(c)
            continue

        # If all missing, no median exists
        if s.isna().all():
            if SKIP_ALL_MISSING_COLUMNS:
                report["skipped_all_missing_cols"].append(c)
                continue
            else:
                # If you ever want to force-fill all-missing numeric columns, pick 0 or something here.
                continue

        med = float(np.nanmedian(s.to_numpy(dtype=float)))
        report["medians_used"][c] = med

        if s.isna().any():
            df_out[c] = s.fillna(med)
            report["imputed_numeric_cols"].append(c)

    return df_out, report


def main():
    df = pd.read_csv(DATA_PATH, dtype=str)  # read as str first so tokens are preserved
    df = normalize_missing_tokens(df)

    # Try to convert everything to numeric where possible (after token normalization)
    if COERCE_OBJECT_TO_NUMERIC:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    df_imputed, report = median_impute(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_imputed.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print(f"Imputed numeric columns: {len(report['imputed_numeric_cols'])}")
    print(f"Skipped non-numeric columns: {len(report['skipped_non_numeric_cols'])}")
    print(f"Skipped all-missing numeric columns: {len(report['skipped_all_missing_cols'])}")

    # Optional: show a few medians used
    if report["medians_used"]:
        sample = list(report["medians_used"].items())[:15]
        print("\nSample medians used (first 15):")
        for k, v in sample:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
