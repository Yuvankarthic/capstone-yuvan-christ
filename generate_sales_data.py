from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_ROWS = 20000
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

STORE_CONFIG = {
    "Store_A": {"region": "North", "traffic_multiplier": 1.12, "sales_multiplier": 1.08, "ad_multiplier": 1.10},
    "Store_B": {"region": "South", "traffic_multiplier": 0.98, "sales_multiplier": 0.96, "ad_multiplier": 0.95},
    "Store_C": {"region": "East", "traffic_multiplier": 1.04, "sales_multiplier": 1.02, "ad_multiplier": 1.00},
    "Store_D": {"region": "West", "traffic_multiplier": 1.08, "sales_multiplier": 1.06, "ad_multiplier": 1.05},
    "Store_E": {"region": "North", "traffic_multiplier": 0.94, "sales_multiplier": 0.93, "ad_multiplier": 0.92},
    "Store_F": {"region": "South", "traffic_multiplier": 1.00, "sales_multiplier": 1.00, "ad_multiplier": 1.00},
}

CATEGORY_CONFIG = {
    "Electronics": {"price_range": (90, 850), "base_demand": 0.88, "margin": 0.11, "traffic_sensitivity": 1.01},
    "Clothing": {"price_range": (20, 180), "base_demand": 1.05, "margin": 0.16, "traffic_sensitivity": 1.00},
    "Home": {"price_range": (35, 420), "base_demand": 0.92, "margin": 0.13, "traffic_sensitivity": 0.99},
    "Grocery": {"price_range": (4, 60), "base_demand": 1.18, "margin": 0.08, "traffic_sensitivity": 1.03},
    "Beauty": {"price_range": (8, 120), "base_demand": 1.00, "margin": 0.15, "traffic_sensitivity": 1.00},
}

MONTH_SEASONALITY = {
    1: 0.92,
    2: 0.95,
    3: 1.00,
    4: 1.02,
    5: 1.05,
    6: 1.00,
    7: 0.98,
    8: 1.01,
    9: 1.06,
    10: 1.08,
    11: 1.16,
    12: 1.24,
}

SPECIAL_HOLIDAYS = {
    (1, 1),
    (7, 4),
    (11, 24),
    (11, 25),
    (11, 26),
    (12, 24),
    (12, 25),
    (12, 26),
    (12, 31),
}


def build_base_frame(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    date_frame = pd.DataFrame({"date": dates})
    date_frame["month"] = date_frame["date"].dt.month
    date_frame["day_of_week"] = date_frame["date"].dt.dayofweek
    date_frame["is_weekend"] = date_frame["day_of_week"].isin([5, 6]).astype(int)
    date_frame["quarter"] = date_frame["date"].dt.quarter
    date_frame["holiday_flag"] = [int((row.month, row.day) in SPECIAL_HOLIDAYS) for row in date_frame["date"]]

    seasonal = date_frame["month"].map(MONTH_SEASONALITY).astype(float)
    weekend = np.where(date_frame["is_weekend"].to_numpy() == 1, 1.10, 1.0)
    holiday = np.where(date_frame["holiday_flag"].to_numpy() == 1, 1.30, 1.0)
    weekday = np.where(date_frame["day_of_week"].to_numpy() == 0, 0.94, 1.0)

    weights = seasonal.to_numpy() * weekend * holiday * weekday
    weights = weights / weights.sum()

    sampled_dates = rng.choice(date_frame["date"].to_numpy(), size=N_ROWS, replace=True, p=weights)
    df = pd.DataFrame({"date": pd.to_datetime(sampled_dates)})
    return df


def assign_business_dimensions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    store_names = np.array(list(STORE_CONFIG))
    store_probs = np.array([0.18, 0.17, 0.16, 0.17, 0.15, 0.17])
    category_names = np.array(list(CATEGORY_CONFIG))
    category_probs = np.array([0.23, 0.21, 0.20, 0.22, 0.14])

    df["store"] = rng.choice(store_names, size=len(df), p=store_probs)
    df["product_category"] = rng.choice(category_names, size=len(df), p=category_probs)
    df["region"] = df["store"].map(lambda store: STORE_CONFIG[store]["region"])

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["holiday_flag"] = [int((dt.month, dt.day) in SPECIAL_HOLIDAYS) for dt in df["date"]]
    return df


def create_traffic(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    store_traffic = df["store"].map({store: cfg["traffic_multiplier"] for store, cfg in STORE_CONFIG.items()}).astype(float)
    category_traffic = df["product_category"].map({category: cfg["traffic_sensitivity"] for category, cfg in CATEGORY_CONFIG.items()}).astype(float)

    weekly_pattern = np.where(df["day_of_week"].isin([4, 5]), 1.13, np.where(df["day_of_week"].isin([0]), 0.94, 1.00))
    seasonal_pattern = df["month"].map(MONTH_SEASONALITY).astype(float)
    holiday_boost = np.where(df["holiday_flag"] == 1, 1.28, 1.00)

    baseline = 980
    traffic = baseline * store_traffic * category_traffic * weekly_pattern * seasonal_pattern * holiday_boost
    traffic = traffic + rng.normal(0, 70, size=len(df))
    traffic = np.clip(np.round(traffic), 120, None).astype(int)
    df["customer_traffic"] = traffic
    return df


def create_pricing_marketing_sales(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    unit_prices: list[float] = []
    discounts: list[float] = []
    ad_spend: list[float] = []
    promotion_flags: list[int] = []
    units_sold: list[int] = []
    sales_amount: list[float] = []

    for _, row in df.iterrows():
        category_cfg = CATEGORY_CONFIG[row["product_category"]]
        store_cfg = STORE_CONFIG[row["store"]]
        month = int(row["month"])
        weekday = int(row["day_of_week"])

        base_price = rng.uniform(*category_cfg["price_range"])
        unit_price = max(1.0, base_price + rng.normal(0, base_price * 0.06))

        discount_base = {
            "Electronics": 0.08,
            "Clothing": 0.14,
            "Home": 0.10,
            "Grocery": 0.04,
            "Beauty": 0.12,
        }[row["product_category"]]
        discount = discount_base + rng.normal(0, 0.03)
        if row["holiday_flag"] == 1:
            discount += rng.uniform(0.03, 0.08)
        if weekday in [4, 5]:
            discount += rng.uniform(0.01, 0.03)
        discount = float(np.clip(discount, 0.0, 0.30))

        promotion_probability = 0.18 + 0.10 * int(weekday in [4, 5]) + 0.16 * int(row["holiday_flag"] == 1) + 0.06 * int(month in [11, 12])
        promotion_flag = int(rng.random() < np.clip(promotion_probability, 0.05, 0.80))

        ad_base = {
            "Electronics": 2400,
            "Clothing": 1800,
            "Home": 1600,
            "Grocery": 800,
            "Beauty": 1300,
        }[row["product_category"]]
        ad_spend_value = ad_base * store_cfg["ad_multiplier"]
        ad_spend_value *= 1.18 if promotion_flag else 0.92
        ad_spend_value *= 1.10 if month in [9, 10, 11, 12] else 1.0
        ad_spend_value *= 1.0 + 0.14 * np.sin(2 * np.pi * month / 12.0)
        ad_spend_value += rng.normal(0, ad_base * 0.12)
        ad_spend_value = max(100.0, ad_spend_value)

        traffic = float(row["customer_traffic"])
        traffic_effect = np.log1p(traffic) / 2.8
        promo_effect = 1.12 if promotion_flag else 1.0
        holiday_effect = 1.18 if row["holiday_flag"] == 1 else 1.0
        seasonal_demand = MONTH_SEASONALITY[month]
        weekend_effect = 1.08 if weekday in [4, 5] else 1.0
        store_effect = store_cfg["sales_multiplier"]
        category_effect = category_cfg["base_demand"]

        units_mean = traffic_effect * category_effect * promo_effect * holiday_effect * weekend_effect * seasonal_demand * store_effect * 8.5
        units_mean += 0.006 * traffic
        units_mean += 0.0009 * ad_spend_value
        units_mean *= 1.0 - (discount * 0.55)
        units = int(np.clip(np.round(rng.normal(units_mean, max(2.0, units_mean * 0.12))), 1, None))

        effective_price = unit_price * (1 - discount)
        gross_sales = units * effective_price
        sales_multiplier = 1.0 + 0.00012 * ad_spend_value + 0.0008 * traffic + 0.0045 * np.log1p(traffic)
        sales_multiplier += 0.04 if promotion_flag else 0.0
        sales_multiplier += 0.06 if row["holiday_flag"] == 1 else 0.0
        sales_multiplier += 0.03 if weekday in [4, 5] else 0.0
        sales_multiplier += rng.normal(0, 0.05)

        sales_value = max(20.0, gross_sales * sales_multiplier)

        unit_prices.append(round(float(unit_price), 2))
        discounts.append(round(float(discount), 4))
        ad_spend.append(round(float(ad_spend_value), 2))
        promotion_flags.append(promotion_flag)
        units_sold.append(units)
        sales_amount.append(round(float(sales_value), 2))

    df["unit_price"] = unit_prices
    df["discount_pct"] = discounts
    df["ad_spend"] = ad_spend
    df["promotion_flag"] = promotion_flags
    df["units_sold"] = units_sold
    df["sales_amount"] = sales_amount
    return df


def inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    anomaly_count = max(1, int(round(len(df) * 0.015)))
    anomaly_indices = rng.choice(df.index, size=anomaly_count, replace=False)

    third = anomaly_count // 3
    for index in anomaly_indices[:third]:
        df.at[index, "customer_traffic"] = int(max(50, df.at[index, "customer_traffic"] * rng.uniform(1.6, 2.4)))
        df.at[index, "units_sold"] = int(max(1, df.at[index, "units_sold"] * rng.uniform(1.35, 1.9)))
        df.at[index, "sales_amount"] = round(float(df.at[index, "sales_amount"] * rng.uniform(1.3, 1.8)), 2)

    for index in anomaly_indices[third : 2 * third]:
        df.at[index, "customer_traffic"] = int(max(30, df.at[index, "customer_traffic"] * rng.uniform(0.35, 0.6)))
        df.at[index, "units_sold"] = int(max(1, df.at[index, "units_sold"] * rng.uniform(0.3, 0.7)))
        df.at[index, "sales_amount"] = round(float(df.at[index, "sales_amount"] * rng.uniform(0.35, 0.65)), 2)

    for index in anomaly_indices[2 * third :]:
        df.at[index, "ad_spend"] = round(float(df.at[index, "ad_spend"] * rng.uniform(1.6, 2.8)), 2)
        df.at[index, "discount_pct"] = float(np.clip(df.at[index, "discount_pct"] + rng.uniform(0.08, 0.12), 0, 0.30))

    return df


def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    missing_rate = 0.01
    missing_columns = ["discount_pct", "ad_spend", "customer_traffic", "promotion_flag"]
    for column in missing_columns:
        sample_size = int(round(len(df) * missing_rate))
        missing_index = rng.choice(df.index, size=sample_size, replace=False)
        df.loc[missing_index, column] = np.nan

    # Small number of rows with slightly messy numeric values for preprocessing testing.
    messy_indices = rng.choice(df.index, size=int(round(len(df) * 0.005)), replace=False)
    df.loc[messy_indices, "discount_pct"] = df.loc[messy_indices, "discount_pct"].clip(lower=0, upper=0.30)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    date_series = pd.to_datetime(df["date"])
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Date range: {date_series.min().date()} to {date_series.max().date()}")
    print("Missing values:")
    print(df.isna().sum().to_string())
    print(f"Duplicate count: {int(df.duplicated().sum())}")
    print(f"Minimum sales: {df['sales_amount'].min():.2f}")
    print(f"Maximum sales: {df['sales_amount'].max():.2f}")
    print(f"Average sales: {df['sales_amount'].mean():.2f}")
    print(f"Total sales: {df['sales_amount'].sum():.2f}")
    print(f"Number of stores: {df['store'].nunique()}")
    print(f"Number of regions: {df['region'].nunique()}")
    print(f"Number of categories: {df['product_category'].nunique()}")

    promo_units = df.groupby("promotion_flag", dropna=True)["units_sold"].mean()
    traffic_sales = df[["customer_traffic", "sales_amount"]].corr(numeric_only=True).iloc[0, 1]
    holiday_sales = df.groupby("holiday_flag", dropna=True)["sales_amount"].mean()
    monthly_sales = df.groupby(date_series.dt.month)["sales_amount"].mean()

    print("Sanity checks:")
    if 0 in promo_units.index and 1 in promo_units.index:
        diff = promo_units.loc[1] - promo_units.loc[0]
        print(f"Promotion effect on units sold (promo - no promo): {diff:.2f}")
    print(f"Traffic-sales correlation: {traffic_sales:.3f}")
    if 0 in holiday_sales.index and 1 in holiday_sales.index:
        diff = holiday_sales.loc[1] - holiday_sales.loc[0]
        print(f"Holiday sales lift: {diff:.2f}")
    print("Average sales by month:")
    print(monthly_sales.round(2).to_string())


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    project_root = Path(__file__).resolve().parent
    output_path = project_root / "data" / "sample_sales.csv"

    df = build_base_frame(rng)
    df = assign_business_dimensions(df, rng)
    df = create_traffic(df, rng)
    df = create_pricing_marketing_sales(df, rng)
    df = inject_anomalies(df, rng)
    df = inject_data_quality_issues(df, rng)

    df = df[
        [
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
    ].copy()

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False)
    validate_dataset(df)


if __name__ == "__main__":
    main()