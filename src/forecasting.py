from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .preprocessing import load_sales_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"
VALIDATION_DAYS = 90
FORECAST_HORIZON_DAYS = 30
SEASONAL_PERIOD = 7


def load_daily_sales_series(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    raw_df = load_sales_data(csv_path)
    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    daily_df = (
        raw_df.groupby("date", as_index=False)["sales_amount"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    full_dates = pd.date_range(daily_df["date"].min(), daily_df["date"].max(), freq="D")
    missing_dates = full_dates.difference(daily_df["date"])

    daily_df = daily_df.set_index("date").reindex(full_dates)
    daily_df.index.name = "date"
    daily_df["sales_amount"] = daily_df["sales_amount"].interpolate(method="linear").ffill().bfill()
    daily_df = daily_df.reset_index()
    daily_df.attrs["missing_dates"] = int(len(missing_dates))
    return daily_df


def inspect_sales_series(daily_df: pd.DataFrame) -> dict[str, Any]:
    sales = daily_df.set_index("date")
    monthly_sales = sales["sales_amount"].resample("MS").sum()
    weekly_sales = sales["sales_amount"].resample("W").sum()

    inspection = {
        "frequency": "daily",
        "observations": int(len(daily_df)),
        "start_date": str(daily_df["date"].min().date()),
        "end_date": str(daily_df["date"].max().date()),
        "missing_dates": int(daily_df.attrs.get("missing_dates", 0)),
        "trend_start": float(daily_df.iloc[0]["sales_amount"]),
        "trend_end": float(daily_df.iloc[-1]["sales_amount"]),
        "weekly_sales_std": float(weekly_sales.std()),
        "monthly_sales_std": float(monthly_sales.std()),
        "monthly_sales_mean": monthly_sales.round(2).to_dict(),
        "weekday_sales_mean": daily_df.assign(weekday=daily_df["date"].dt.day_name()).groupby("weekday")["sales_amount"].mean().round(2).to_dict(),
    }
    return inspection


def fit_exponential_smoothing(series: pd.Series) -> Any:
    return ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=True)


def build_seasonal_naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    if len(history) < SEASONAL_PERIOD:
        return np.repeat(float(history.iloc[-1]), horizon)

    pattern = history.iloc[-SEASONAL_PERIOD:].to_numpy(dtype=float)
    repeated = np.resize(pattern, horizon)
    return repeated


def evaluate_forecast(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, forecast)))
    mae = float(mean_absolute_error(actual, forecast))
    mape = float(np.mean(np.abs((actual - forecast) / np.clip(np.abs(actual), 1e-8, None))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def run_sales_forecasting(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    validation_days: int = VALIDATION_DAYS,
    forecast_horizon_days: int = FORECAST_HORIZON_DAYS,
) -> dict[str, Any]:
    daily_df = load_daily_sales_series(csv_path)
    inspection = inspect_sales_series(daily_df)

    if len(daily_df) <= validation_days + SEASONAL_PERIOD:
        raise ValueError("Not enough observations for time-aware forecasting.")

    train_df = daily_df.iloc[:-validation_days].copy()
    validation_df = daily_df.iloc[-validation_days:].copy()

    train_series = train_df.set_index("date")["sales_amount"].asfreq("D")
    validation_series = validation_df.set_index("date")["sales_amount"].asfreq("D")

    validation_model = fit_exponential_smoothing(train_series)
    validation_forecast = validation_model.forecast(validation_days)
    baseline_forecast = build_seasonal_naive_forecast(train_series, validation_days)

    validation_frame = pd.DataFrame(
        {
            "date": validation_df["date"].to_list(),
            "actual_sales": validation_series.to_numpy(dtype=float),
            "forecast_sales": validation_forecast.to_numpy(dtype=float),
            "baseline_seasonal_naive": baseline_forecast,
        }
    )

    forecast_metrics = evaluate_forecast(
        validation_frame["actual_sales"].to_numpy(dtype=float),
        validation_frame["forecast_sales"].to_numpy(dtype=float),
    )
    baseline_metrics = evaluate_forecast(
        validation_frame["actual_sales"].to_numpy(dtype=float),
        validation_frame["baseline_seasonal_naive"].to_numpy(dtype=float),
    )

    full_series = daily_df.set_index("date")["sales_amount"].asfreq("D")
    future_model = fit_exponential_smoothing(full_series)
    future_dates = pd.date_range(full_series.index.max() + pd.Timedelta(days=1), periods=forecast_horizon_days, freq="D")
    future_forecast = future_model.forecast(forecast_horizon_days)
    future_frame = pd.DataFrame(
        {
            "date": future_dates,
            "forecast_sales": future_forecast.to_numpy(dtype=float),
        }
    )

    return {
        "inspection": inspection,
        "daily_sales": daily_df,
        "training_frame": train_df,
        "validation_frame": validation_frame,
        "future_forecast": future_frame,
        "validation_model": validation_model,
        "future_model": future_model,
        "selected_method": "Exponential smoothing with additive trend and weekly seasonality",
        "baseline_method": "Seasonal naive (lag-7) baseline",
        "training_period": {
            "start": str(train_df["date"].min().date()),
            "end": str(train_df["date"].max().date()),
        },
        "validation_period": {
            "start": str(validation_df["date"].min().date()),
            "end": str(validation_df["date"].max().date()),
        },
        "forecast_horizon_days": forecast_horizon_days,
        "metrics": {
            "forecast_rmse": forecast_metrics["rmse"],
            "forecast_mae": forecast_metrics["mae"],
            "forecast_mape": forecast_metrics["mape"],
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_mape": baseline_metrics["mape"],
        },
    }


if __name__ == "__main__":
    result = run_sales_forecasting()
    print(f"Frequency: {result['inspection']['frequency']}")
    print(f"Observations: {result['inspection']['observations']}")
    print(f"Missing dates: {result['inspection']['missing_dates']}")
    print(f"Training period: {result['training_period']['start']} to {result['training_period']['end']}")
    print(f"Validation period: {result['validation_period']['start']} to {result['validation_period']['end']}")
    print(f"Forecast horizon: {result['forecast_horizon_days']} days")
    print("Forecast metrics:")
    print(result["metrics"])
    print("Validation preview:")
    print(result["validation_frame"].head().to_string(index=False))
    print("Future forecast preview:")
    print(result["future_forecast"].head().to_string(index=False))