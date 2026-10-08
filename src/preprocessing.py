from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "store",
    "region",
    "product_category",
    "units_sold",
    "unit_price",
    "discount_pct",
    "ad_spend",
    "customer_traffic",
    "promotion_flag",
    "holiday_flag",
    "sales_amount",
]

NUMERIC_COLUMNS = [
    "units_sold",
    "unit_price",
    "discount_pct",
    "ad_spend",
    "customer_traffic",
    "promotion_flag",
    "holiday_flag",
]

CATEGORICAL_COLUMNS = ["store", "region", "product_category"]
DATE_COLUMN = "date"
TARGET_COLUMN = "sales_amount"


def load_sales_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a business sales CSV file and ensure the expected columns exist."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple, realistic business features from the raw dataset."""

    processed = df.copy()
    processed[DATE_COLUMN] = pd.to_datetime(processed[DATE_COLUMN])
    processed = processed.sort_values(DATE_COLUMN).reset_index(drop=True)

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        processed[column] = processed[column].astype(str).fillna("Unknown")

    processed["month"] = processed[DATE_COLUMN].dt.month
    processed["day_of_week"] = processed[DATE_COLUMN].dt.dayofweek
    processed["quarter"] = processed[DATE_COLUMN].dt.quarter
    processed["is_weekend"] = processed["day_of_week"].isin([5, 6]).astype(int)
    processed["price_after_discount"] = processed["unit_price"] * (1 - processed["discount_pct"])
    processed["marketing_intensity"] = processed["ad_spend"] / processed["customer_traffic"].replace(0, pd.NA)
    processed["marketing_intensity"] = processed["marketing_intensity"].fillna(0)
    processed["sales_per_customer"] = processed["sales_amount"] / processed["customer_traffic"].replace(0, pd.NA)
    processed["sales_per_customer"] = processed["sales_per_customer"].fillna(0)
    processed["sales_to_ad_ratio"] = processed["sales_amount"] / processed["ad_spend"].replace(0, pd.NA)
    processed["sales_to_ad_ratio"] = processed["sales_to_ad_ratio"].fillna(0)
    processed["traffic_per_unit"] = processed["customer_traffic"] / processed["units_sold"].replace(0, pd.NA)
    processed["traffic_per_unit"] = processed["traffic_per_unit"].fillna(0)

    return processed


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return the numeric and categorical feature sets used by the model."""

    numeric_features = [
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
    ]
    categorical_features = list(CATEGORICAL_COLUMNS)
    return numeric_features, categorical_features


def get_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the features used by the training pipeline."""

    numeric_features, categorical_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features
    return df[feature_columns].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the sales target for supervised learning."""

    return df[TARGET_COLUMN].copy()
