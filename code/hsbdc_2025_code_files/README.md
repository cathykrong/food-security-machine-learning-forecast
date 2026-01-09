# HSBDC 2025–26: From Harvest to Food Security (USA, 1970–2023)

Pipeline:
1) (Optional) Download USDA QuickStats raw CSVs
2) Clean + reshape crop predictors to annual USA-wide table
3) Build Food Security Index (FSI) from 4 FAOSTAT series (min-max → equal-weight mean)
4) Merge exogenous macro series
5) Engineer shocks (YoY %) and lags (t-1, t-2)
6) Train & evaluate models:
   - Level: predict FSI(t)
   - Change-based: predict ΔFSI(t)=FSI(t)-FSI(t-1), reconstruct FSI_pred(t)=FSI(t-1)+ΔFSI_pred(t)
7) Produce figures: time-series, scatter, residuals, feature importance, ROC + confusion matrix

Quick start (using your already-prepared CSVs in /mnt/data):
python scripts/07_train_models_delta.py --data /mnt/data/us_quickstats_fsi_plus_macro_with_shocks_lags.csv
python scripts/08_plots_core.py --pred_csv outputs/preds_rf_deltafsi_test.csv

All outputs go to ./outputs by default.
