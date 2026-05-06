import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np



import matplotlib.pyplot as plt

# Load merged data
data_path = Path("E:/1Cathy/hsbdc/AI2026/data2/us_quickstats_fsi_merged_1970_2023.csv")
df = pd.read_csv(data_path).sort_values("year").reset_index(drop=True)

target = "FSI_geomean"
pillar_cols = ["A_availability", "X_access", "U_utilization", "S_stability"]

crop_cols = [c for c in df.columns if c not in ["year", target] + pillar_cols]

# Holdout split: last 11 years (2013-2023)
test_year_start = 2013
train = df[df["year"] < test_year_start].copy()
test = df[df["year"] >= test_year_start].copy()

y_train = train[target].values
y_test = test[target].values
years_test = test["year"].values

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, mse, r2

rows = []

# -----------------------
# Baselines
# -----------------------
# Baseline A: train mean
pred_mean = np.full_like(y_test, y_train.mean(), dtype=float)
rmse, mse, r2 = metrics(y_test, pred_mean)
rows.append({"model": "Baseline: Train mean", "features": "(none)", "RMSE": rmse, "MSE": mse, "R2": r2})

# Baseline B: persistence (previous year's actual FSI)
# For 2013, use 2012 actual (available). For later years, use previous year's actual (known historically).
y_prev = df.set_index("year")[target]
pred_persist = np.array([y_prev[yr - 1] for yr in years_test], dtype=float)
rmse, mse, r2 = metrics(y_test, pred_persist)
rows.append({"model": "Baseline: Persistence (t-1)", "features": "FSI(t-1)", "RMSE": rmse, "MSE": mse, "R2": r2})

# Baseline C: linear trend on year
trend_model = LinearRegression()
trend_model.fit(train[["year"]], y_train)
pred_trend = trend_model.predict(test[["year"]])
rmse, mse, r2 = metrics(y_test, pred_trend)
rows.append({"model": "Baseline: Linear trend", "features": "year", "RMSE": rmse, "MSE": mse, "R2": r2})

# -----------------------
# Helper for lag features
# -----------------------
def make_lagged(df_in, cols, lags=(1,2,3)):
    out = df_in.copy()
    for lag in lags:
        for c in cols:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out

# Create lagged dataset (crops only)
df_lag = make_lagged(df, crop_cols, lags=(1,2,3))

# drop rows with NA due to lags
df_lag = df_lag.dropna().reset_index(drop=True)

train_lag = df_lag[df_lag["year"] < test_year_start].copy()
test_lag = df_lag[df_lag["year"] >= test_year_start].copy()

# Feature sets
X_train_year = train[["year"] + crop_cols]
X_test_year = test[["year"] + crop_cols]

X_train_crops = train[crop_cols]
X_test_crops = test[crop_cols]

lag_cols = [c for c in df_lag.columns if any(c.endswith(f"_lag{lag}") for lag in (1,2,3))]
X_train_lags = train_lag[crop_cols + lag_cols]  # include contemporaneous + lags
X_test_lags = test_lag[crop_cols + lag_cols]

y_train_lag = train_lag[target].values
y_test_lag = test_lag[target].values
years_test_lag = test_lag["year"].values

# -----------------------
# Tuned models with TimeSeries CV on TRAIN ONLY
# -----------------------
tscv = TimeSeriesSplit(n_splits=5)

def tune_and_eval(name, base_estimator, param_grid, Xtr, ytr, Xte, yte, features_label):
    gs = GridSearchCV(base_estimator, param_grid=param_grid, cv=tscv, scoring="neg_root_mean_squared_error")
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    pred = best.predict(Xte)
    rmse, mse, r2 = metrics(yte, pred)
    return {
        "model": f"{name} (tuned)",
        "features": features_label,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "best_params": gs.best_params_,
        "pred": pred,
        "estimator": best
    }

results_detail = []

# Ridge (good for small data; also acts like linear + regularization)
ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
ridge_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
results_detail.append(tune_and_eval("Ridge", ridge_pipe, ridge_grid, X_train_year, y_train, X_test_year, y_test, "year + crops"))

# SVR (RBF)
svr_pipe = Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))])
svr_grid = {"model__C": [1, 10, 100], "model__epsilon": [0.005, 0.01, 0.05], "model__gamma": ["scale", "auto"]}
results_detail.append(tune_and_eval("SVR_RBF", svr_pipe, svr_grid, X_train_year, y_train, X_test_year, y_test, "year + crops"))

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf_grid = {
    "n_estimators": [300, 800],
    "max_depth": [None, 3, 5],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.7]
}
results_detail.append(tune_and_eval("RandomForest", rf, rf_grid, X_train_year, y_train, X_test_year, y_test, "year + crops"))

# Gradient Boosting
gbr = GradientBoostingRegressor(random_state=42)
gbr_grid = {
    "n_estimators": [200, 500, 800],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.7, 1.0]
}
results_detail.append(tune_and_eval("GradientBoosting", gbr, gbr_grid, X_train_year, y_train, X_test_year, y_test, "year + crops"))

# Lag-feature models (often best chance to beat persistence)
# Use Ridge and GBR on lags
ridge_pipe_lag = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
results_detail.append(tune_and_eval("Ridge", ridge_pipe_lag, ridge_grid, X_train_lags, y_train_lag, X_test_lags, y_test_lag, "crops + lags(1-3)"))

gbr_lag = GradientBoostingRegressor(random_state=42)
results_detail.append(tune_and_eval("GradientBoosting", gbr_lag, gbr_grid, X_train_lags, y_train_lag, X_test_lags, y_test_lag, "crops + lags(1-3)"))

# -----------------------
# Trend + residual model (legit way to stabilize)
# 1) Fit trend on year, 2) Fit residuals on crop vars + lags, 3) sum
# -----------------------
# Trend on lag-aligned train/test (since residual learner uses lag rows)
trend = LinearRegression()
trend.fit(train_lag[["year"]], y_train_lag)
trend_pred_test = trend.predict(test_lag[["year"]])
trend_pred_train = trend.predict(train_lag[["year"]])

resid_train = y_train_lag - trend_pred_train

# residual learner: tuned GBR on residuals
gbr_resid = GradientBoostingRegressor(random_state=42)
gs_resid = GridSearchCV(gbr_resid, param_grid=gbr_grid, cv=tscv, scoring="neg_root_mean_squared_error")
gs_resid.fit(X_train_lags, resid_train)

resid_pred_test = gs_resid.best_estimator_.predict(X_test_lags)
pred_trend_resid = trend_pred_test + resid_pred_test

rmse, mse, r2 = metrics(y_test_lag, pred_trend_resid)
rows.append({
    "model": "Trend(year) + Residual(GBR on crops+lags) (tuned)",
    "features": "year trend + crops + lags(1-3)",
    "RMSE": rmse, "MSE": mse, "R2": r2
})

# Collect tuned model results
for d in results_detail:
    rows.append({"model": d["model"], "features": d["features"], "RMSE": d["RMSE"], "MSE": d["MSE"], "R2": d["R2"]})

results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

# Find best model overall by RMSE and plot on its corresponding test window
best_row = results.iloc[0]
best_name = best_row["model"]
best_features = best_row["features"]

# Retrieve predictions for plotting if available
pred_series = None
plot_years = None
plot_ytrue = None

if best_name.startswith("Baseline: Train mean"):
    pred_series, plot_years, plot_ytrue = pred_mean, years_test, y_test
elif best_name.startswith("Baseline: Persistence"):
    pred_series, plot_years, plot_ytrue = pred_persist, years_test, y_test
elif best_name.startswith("Baseline: Linear trend"):
    pred_series, plot_years, plot_ytrue = pred_trend, years_test, y_test
elif best_name.startswith("Trend(year) + Residual"):
    pred_series, plot_years, plot_ytrue = pred_trend_resid, years_test_lag, y_test_lag
else:
    # lookup in results_detail
    for d in results_detail:
        if d["model"] == best_name and d["features"] == best_features:
            pred_series = d["pred"]
            # determine which test window
            if "lags" in best_features:
                plot_years, plot_ytrue = years_test_lag, y_test_lag
            else:
                plot_years, plot_ytrue = years_test, y_test
            break

# Plot actual vs predicted over time
plt.figure()
plt.plot(plot_years, plot_ytrue, marker="o")
plt.plot(plot_years, pred_series, marker="o")
plt.xlabel("Year")
plt.ylabel("FSI_geomean")
plt.title(f"Best model by RMSE: {best_name}\n({best_features})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter
plt.figure()
plt.scatter(plot_ytrue, pred_series)
plt.xlabel("Actual FSI_geomean")
plt.ylabel("Predicted FSI_geomean")
plt.title(f"Predicted vs Actual (test): {best_name}")
plt.tight_layout()
plt.show()

# Save results table to CSV for manuscript
out_results = Path("/mnt/data/model_benchmark_results_2013_2023.csv")
results.to_csv(out_results, index=False)

# Also save test predictions for the best model for easy plotting in paper
pred_out = pd.DataFrame({"year": plot_years, "FSI_actual": plot_ytrue, "FSI_pred": pred_series})
out_pred = Path("/mnt/data/best_model_test_predictions.csv")
pred_out.to_csv(out_pred, index=False)

results.head(15), best_row.to_dict(), out_results, out_pred

