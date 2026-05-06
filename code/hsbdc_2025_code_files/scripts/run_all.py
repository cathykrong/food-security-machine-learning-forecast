"""
One-command runner if you already have the shocks+lags dataset.

Usage:
python scripts/run_all.py --data /mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv --target FSI
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--target", type=str, default="FSI")
    args = ap.parse_args()

    run([sys.executable, "scripts/07_train_models_delta.py", "--data", str(args.data), "--target", args.target])
    run([sys.executable, "scripts/06_train_models_level.py", "--data", str(args.data), "--target", args.target])

    # Pick the RF delta output by convention
    pred = Path("outputs/preds_randomforest_deltafsi_test.csv")
    if pred.exists():
        run([sys.executable, "scripts/08_plots_core.py", "--pred_csv", str(pred), "--ymin", "0.60"])
        run([sys.executable, "scripts/10_directional_eval.py", "--pred_csv", str(pred)])
        run([sys.executable, "scripts/09_plot_feature_importance.py", "--data", str(args.data), "--target", args.target])
    else:
        print("Could not find outputs/preds_randomforest_deltafsi_test.csv; run 07 first.")

if __name__ == "__main__":
    main()
