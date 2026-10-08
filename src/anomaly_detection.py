from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import engineer_features, load_sales_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"

NUMERIC_FEATURES = [
    "sales_amount",
    "units_sold",
    "unit_price",
    "discount_pct",
    "ad_spend",
    "customer_traffic",
    "promotion_flag",
    "holiday_flag",
    "month",
    "day_of_week",
    "quarter",
    "is_weekend",
    "price_after_discount",
    "marketing_intensity",
    "traffic_per_unit",
    "sales_to_ad_ratio",
]


def load_processed_business_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    raw_df = load_sales_data(csv_path)
    return engineer_features(raw_df)


def build_isolation_forest(contamination: float = 0.02) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=42,
                ),
            ),
        ]
    )


def run_anomaly_detection(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    contamination: float = 0.02,
) -> dict[str, Any]:
    processed_df = load_processed_business_data(csv_path)
    feature_frame = processed_df[NUMERIC_FEATURES].copy()

    model = build_isolation_forest(contamination=contamination)
    model.fit(feature_frame)

    raw_prediction = model.named_steps["model"].predict(model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(feature_frame)))
    anomaly_flag = np.where(raw_prediction == -1, -1, 1)
    anomaly_score = -model.named_steps["model"].decision_function(
        model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(feature_frame))
    )

    result_df = processed_df[
        ["date", "store", "region", "product_category", "sales_amount", "customer_traffic", "ad_spend"]
    ].copy()
    result_df["anomaly_score"] = anomaly_score
    result_df["anomaly_flag"] = anomaly_flag
    result_df = result_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    anomaly_count = int((result_df["anomaly_flag"] == -1).sum())
    anomaly_percentage = float(anomaly_count / len(result_df) * 100)

    return {
        "processed_data": processed_df,
        "result": result_df,
        "model": model,
        "features_used": NUMERIC_FEATURES,
        "contamination": contamination,
        "observation_count": len(result_df),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_percentage,
    }


if __name__ == "__main__":
    result = run_anomaly_detection()
    print(f"Observations: {result['observation_count']}")
    print(f"Features used: {', '.join(result['features_used'])}")
    print(f"Anomalies detected: {result['anomaly_count']} ({result['anomaly_percentage']:.2f}%)")
    print(result["result"].head(10).to_string(index=False))