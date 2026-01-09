from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Default data paths (you can override via CLI flags)
    DATA_PLUS_MACRO: Path = Path("/mnt/data/us_quickstats_fsi_plus_macro_1970_2023.csv")
    DATA_SHOCKS_LAGS: Path = Path("/mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv")

    OUT_DIR: Path = Path("outputs")

    # Target names (adjust if your column name differs)
    TARGET_FSI: str = "FSI"          # or "FSI_geomean" if that's what you kept
    YEAR_COL: str = "year"

    # Train/test split
    TRAIN_END_YEAR: int = 2012
    TEST_START_YEAR: int = 2013
