from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the repository packages importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_detection import DEFAULT_DATA_PATH as ANOMALY_DATA_PATH, run_anomaly_detection
from src.forecasting import DEFAULT_DATA_PATH as FORECAST_DATA_PATH, run_sales_forecasting
from src.model_training import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, load_trained_artifact, train_sales_model
from src.prediction import predict_sales
from src.preprocessing import engineer_features, load_sales_data

st.set_page_config(
    page_title="AI-Powered Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-strong: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --border: rgba(148, 163, 184, 0.22);
            --accent: #0ea5e9;
            --accent-2: #14b8a6;
            --accent-3: #f59e0b;
            --shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.10), transparent 26%),
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.08), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: rgba(255, 255, 255, 0.92) !important;
        }

        [data-testid="stSidebar"] .stButton button {
            background: linear-gradient(135deg, #0ea5e9, #14b8a6);
            color: white;
            border: none;
            border-radius: 14px;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            opacity: 0.92;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.90));
            color: white;
            border-radius: 28px;
            padding: 28px 30px;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            margin-top: 0.45rem;
            color: rgba(226, 232, 240, 0.85);
            font-size: 1rem;
        }

        .hero-note {
            margin-top: 1rem;
            display: inline-flex;
            gap: 0.45rem;
            align-items: center;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: rgba(248, 250, 252, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-size: 0.92rem;
        }

        .section-shell {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.1rem 1.15rem 1.2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }

        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--accent);
            font-weight: 700;
            font-size: 0.74rem;
            margin-bottom: 0.15rem;
        }

        .section-title {
            margin: 0;
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .pill-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.44rem 0.7rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.72);
            color: var(--text);
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem;
            box-shadow: var(--shadow);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.83rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.6rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .metric-detail {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }

        .stDataFrame, [data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
        }

        .stAlert {
            border-radius: 18px;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.18);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <p class="hero-title">AI-Powered Business Intelligence Platform</p>
        <p class="hero-subtitle">A practical retail analytics workspace for sales prediction, time-series forecasting, and anomaly detection.</p>
        <div class="hero-note">Sales Prediction • Forecasting • Anomaly Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data() -> pd.DataFrame:
    return engineer_features(load_sales_data(DEFAULT_DATA_PATH))


@st.cache_resource
def load_artifact() -> dict:
    return load_trained_artifact(DEFAULT_MODEL_PATH)


@st.cache_resource
def load_forecast_bundle() -> dict:
    return run_sales_forecasting(FORECAST_DATA_PATH)


@st.cache_resource
def load_anomaly_bundle() -> dict:
    return run_anomaly_detection(ANOMALY_DATA_PATH)


def build_prediction_samples(featured_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
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
    valid_rows = featured_df.dropna(subset=[column for column in required_columns if column != "date"])
    reference_row = valid_rows.iloc[-1] if not valid_rows.empty else featured_df.iloc[-1]

    sample_inputs = pd.DataFrame(
        [
            {
                "date": reference_row["date"] + pd.Timedelta(days=7),
                "store": reference_row["store"],
                "region": reference_row["region"],
                "product_category": reference_row["product_category"],
                "units_sold": int(reference_row["units_sold"] * 1.05),
                "unit_price": float(reference_row["unit_price"] * 1.02),
                "discount_pct": float(reference_row["discount_pct"] if pd.notna(reference_row["discount_pct"]) else 0.08),
                "ad_spend": float(reference_row["ad_spend"] * 1.08 if pd.notna(reference_row["ad_spend"]) else 1800),
                "customer_traffic": int(reference_row["customer_traffic"] * 1.03 if pd.notna(reference_row["customer_traffic"]) else 1200),
                "promotion_flag": 1,
                "holiday_flag": 0,
                "sales_amount": float(reference_row["sales_amount"] if pd.notna(reference_row["sales_amount"]) else 0),
            },
            {
                "date": reference_row["date"] + pd.Timedelta(days=14),
                "store": "Store_A",
                "region": "North",
                "product_category": "Electronics",
                "units_sold": 55,
                "unit_price": 131,
                "discount_pct": 0.07,
                "ad_spend": 2000,
                "customer_traffic": 1180,
                "promotion_flag": 1,
                "holiday_flag": 0,
                "sales_amount": 0,
            },
            {
                "date": reference_row["date"] + pd.Timedelta(days=14),
                "store": "Store_F",
                "region": "South",
                "product_category": "Home",
                "units_sold": 28,
                "unit_price": 265,
                "discount_pct": 0.05,
                "ad_spend": 1400,
                "customer_traffic": 860,
                "promotion_flag": 0,
                "holiday_flag": 0,
                "sales_amount": 0,
            },
        ]
    )
    return sample_inputs


def get_anomaly_filters(anomaly_df: pd.DataFrame) -> dict[str, object]:
    min_date = pd.to_datetime(anomaly_df["date"]).min().date()
    max_date = pd.to_datetime(anomaly_df["date"]).max().date()

    with st.sidebar:
        st.header("Controls")
        refresh_model = st.button("Train / Refresh Model")
        st.write("Dataset:", DEFAULT_DATA_PATH.name)

        st.subheader("Anomaly Filters")
        store_options = sorted(anomaly_df["store"].dropna().unique().tolist())
        region_options = sorted(anomaly_df["region"].dropna().unique().tolist())
        category_options = sorted(anomaly_df["product_category"].dropna().unique().tolist())

        selected_stores = st.multiselect("Store", options=store_options, default=store_options)
        selected_regions = st.multiselect("Region", options=region_options, default=region_options)
        selected_categories = st.multiselect("Product Category", options=category_options, default=category_options)
        selected_dates = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    return {
        "refresh_model": refresh_model,
        "selected_stores": selected_stores,
        "selected_regions": selected_regions,
        "selected_categories": selected_categories,
        "selected_dates": selected_dates,
    }


def render_section(label: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-label">{label}</div>
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

forecast_bundle = load_forecast_bundle()
anomaly_bundle = load_anomaly_bundle()
anomaly_df = anomaly_bundle["result"].copy()
filter_state = get_anomaly_filters(anomaly_df)

if filter_state["refresh_model"]:
    artifact = train_sales_model(DEFAULT_DATA_PATH)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Model retrained successfully.")
else:
    artifact = load_artifact()

metrics = artifact["metrics"]
overview_total_sales = float(load_data()["sales_amount"].sum())
overview_average_sales = float(load_data()["sales_amount"].mean())
overview_transactions = int(len(load_data()))
overview_anomalies = int(anomaly_bundle["anomaly_count"])

render_section(
    "Section 1",
    "Business Overview",
    "A quick health check of the retail business before drilling into prediction, forecasting, and anomalies.",
)
overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4, gap="small")
with overview_col1:
    render_metric_card("Total Sales", f"{overview_total_sales:,.2f}", "All rows in the current dataset")
with overview_col2:
    render_metric_card("Average Sales", f"{overview_average_sales:,.2f}", "Average sales per transaction")
with overview_col3:
    render_metric_card("Total Transactions", f"{overview_transactions:,}", "Number of sales records analyzed")
with overview_col4:
    render_metric_card("Detected Anomalies", f"{overview_anomalies:,}", "Isolation Forest flagged records")

st.markdown("<div class='pill-row'><span class='pill'>Use the sidebar filters to narrow anomaly inspection</span><span class='pill'>Forecast model beats seasonal-naive baseline</span><span class='pill'>Random Forest prediction remains unchanged</span></div>", unsafe_allow_html=True)

featured_df = load_data()

render_section(
    "Section 2",
    "Sales Prediction",
    "The existing Random Forest model estimates sales_amount from business drivers such as price, traffic, promotions, and seasonality.",
)
prediction_metric_col1, prediction_metric_col2, prediction_metric_col3, prediction_metric_col4 = st.columns(4, gap="small")
with prediction_metric_col1:
    render_metric_card("Model", "RandomForestRegressor", "Sales prediction model")
with prediction_metric_col2:
    render_metric_card("RMSE", f"{metrics['rmse']:.2f}", "Lower is better")
with prediction_metric_col3:
    render_metric_card("MAE", f"{metrics['mae']:.2f}", "Average absolute error")
with prediction_metric_col4:
    render_metric_card("R²", f"{metrics['r2']:.3f}", "Explained variance")

prediction_left_col, prediction_right_col = st.columns([1.15, 1], gap="large")

with prediction_left_col:
    st.markdown("### Sample current predictions")
    prediction_inputs = build_prediction_samples(featured_df)
    predicted_samples = predict_sales(prediction_inputs)
    st.dataframe(
        predicted_samples[["date", "store", "region", "product_category", "predicted_sales_amount"]],
        width="stretch",
        hide_index=True,
    )

with prediction_right_col:
    st.markdown("### Predicted sales scenarios")
    prediction_fig = px.bar(
        predicted_samples,
        x="product_category",
        y="predicted_sales_amount",
        color="store",
        title="Predicted Sales for Sample Scenarios",
        color_discrete_sequence=["#0ea5e9", "#14b8a6", "#f59e0b"],
    )
    prediction_fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10), height=380)
    st.plotly_chart(prediction_fig, width="stretch")

st.markdown("### Model feature importance")
importance_df = pd.DataFrame(
    {
        "feature": artifact["feature_names"],
        "importance": artifact["feature_importances"],
    }
).sort_values("importance", ascending=False).head(12)
importance_fig = px.bar(
    importance_df,
    x="importance",
    y="feature",
    orientation="h",
    title="Top Predictive Features",
    color="importance",
    color_continuous_scale=["#dbeafe", "#0ea5e9", "#0f766e"],
)
importance_fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10), height=430, coloraxis_showscale=False)
st.plotly_chart(importance_fig, width="stretch")

render_section(
    "Section 3",
    "Sales Forecast",
    "A separate time-series model uses only historical sales to project the next 30 days and compare against a seasonal-naive baseline.",
)
forecast_metrics = forecast_bundle["metrics"]
forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
with forecast_col1:
    render_metric_card("Forecast Model", "Exponential Smoothing", "Additive trend + weekly seasonality")
with forecast_col2:
    render_metric_card("Forecast Horizon", f"{forecast_bundle['forecast_horizon_days']} days", "Future period displayed in the chart")
with forecast_col3:
    render_metric_card("Validation Period", f"{forecast_bundle['validation_period']['start']} to {forecast_bundle['validation_period']['end']}", "Time-aware holdout")

forecast_summary_col1, forecast_summary_col2, forecast_summary_col3 = st.columns(3)
with forecast_summary_col1:
    render_metric_card("MAE", f"{forecast_metrics['forecast_mae']:,.2f}", "Forecast validation")
with forecast_summary_col2:
    render_metric_card("RMSE", f"{forecast_metrics['forecast_rmse']:,.2f}", "Forecast validation")
with forecast_summary_col3:
    render_metric_card("MAPE", f"{forecast_metrics['forecast_mape']:.2f}%", "Forecast validation")

historical_sales = forecast_bundle["daily_sales"]
future_forecast = forecast_bundle["future_forecast"]
historical_trace = historical_sales.copy()
historical_trace["series"] = "Historical Sales"
forecast_trace = future_forecast.copy()
forecast_trace["series"] = "Forecasted Sales"

forecast_fig = go.Figure()
forecast_fig.add_trace(
    go.Scatter(
        x=historical_trace["date"],
        y=historical_trace["sales_amount"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="#1d4ed8", width=2.4),
    )
)
forecast_fig.add_trace(
    go.Scatter(
        x=forecast_trace["date"],
        y=forecast_trace["forecast_sales"],
        mode="lines+markers",
        name="Forecasted Sales",
        line=dict(color="#f97316", width=3, dash="dash"),
        marker=dict(size=6),
    )
)
forecast_fig.update_layout(
    title="Historical Sales → Forecasted Sales",
    xaxis_title="Date",
    yaxis_title="Sales Amount",
    legend_title="Series",
    height=520,
    template="plotly_white",
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(forecast_fig, width="stretch")

st.markdown("### Forecast model comparison")
forecast_comparison = pd.DataFrame(
    [
        {
            "Model": "Seasonal Naive",
            "MAE": forecast_metrics["baseline_mae"],
            "RMSE": forecast_metrics["baseline_rmse"],
            "MAPE": forecast_metrics["baseline_mape"],
        },
        {
            "Model": "Exponential Smoothing",
            "MAE": forecast_metrics["forecast_mae"],
            "RMSE": forecast_metrics["forecast_rmse"],
            "MAPE": forecast_metrics["forecast_mape"],
        },
    ]
)
st.dataframe(forecast_comparison, width="stretch", hide_index=True)
st.caption("Lower values are better. The selected forecasting model performs better than the baseline on all displayed metrics.")

render_section(
    "Section 4",
    "Anomaly Detection",
    "Isolation Forest highlights unusual business observations so a manager can investigate them, not automatically label them as problems.",
)
anomaly_kpi_col1, anomaly_kpi_col2, anomaly_kpi_col3 = st.columns(3, gap="small")
with anomaly_kpi_col1:
    render_metric_card("Total Observations", f"{anomaly_bundle['observation_count']:,}", "Rows analyzed by Isolation Forest")
with anomaly_kpi_col2:
    render_metric_card("Number of Anomalies", f"{anomaly_bundle['anomaly_count']:,}", "Flagged as unusual observations")
with anomaly_kpi_col3:
    render_metric_card("Anomaly Percentage", f"{anomaly_bundle['anomaly_percentage']:.2f}%", "Share of rows flagged")

filtered_anomalies = anomaly_df.copy()
selected_stores = filter_state["selected_stores"]
selected_regions = filter_state["selected_regions"]
selected_categories = filter_state["selected_categories"]
selected_dates = filter_state["selected_dates"]

if selected_stores:
    filtered_anomalies = filtered_anomalies[filtered_anomalies["store"].isin(selected_stores)]
if selected_regions:
    filtered_anomalies = filtered_anomalies[filtered_anomalies["region"].isin(selected_regions)]
if selected_categories:
    filtered_anomalies = filtered_anomalies[filtered_anomalies["product_category"].isin(selected_categories)]
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_anomalies = filtered_anomalies[
        (pd.to_datetime(filtered_anomalies["date"]) >= pd.Timestamp(start_date))
        & (pd.to_datetime(filtered_anomalies["date"]) <= pd.Timestamp(end_date))
    ]

filtered_anomalies = filtered_anomalies.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

if filtered_anomalies.empty:
    st.warning("No anomaly rows match the current filters.")
else:
    anomaly_chart_data = filtered_anomalies.sort_values("date").copy()
    anomaly_fig = go.Figure()
    anomaly_fig.add_trace(
        go.Scatter(
            x=anomaly_chart_data["date"],
            y=anomaly_chart_data["sales_amount"],
            mode="lines",
            name="Sales Amount",
            line=dict(color="#16a34a", width=2.4),
        )
    )
    anomaly_points = anomaly_chart_data[anomaly_chart_data["anomaly_flag"] == -1]
    if not anomaly_points.empty:
        anomaly_fig.add_trace(
            go.Scatter(
                x=anomaly_points["date"],
                y=anomaly_points["sales_amount"],
                mode="markers",
                name="Potential Anomaly",
                marker=dict(color="#dc2626", size=9, symbol="x"),
                hovertemplate="Date: %{x}<br>Sales: %{y:,.2f}<extra>Potential anomaly detected</extra>",
            )
        )
    anomaly_fig.update_layout(
        title="Sales Over Time with Potential Anomalies Highlighted",
        xaxis_title="Date",
        yaxis_title="Sales Amount",
        legend_title="Series",
        height=520,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(anomaly_fig, width="stretch")
    st.info("Potential anomaly detected — requires business investigation.")

    st.markdown("### Anomaly table")
    anomaly_table = filtered_anomalies[
        ["date", "store", "region", "product_category", "sales_amount", "anomaly_score", "anomaly_flag"]
    ].copy()
    st.dataframe(anomaly_table, width="stretch", hide_index=True)

st.markdown("---")
st.caption(
    "This dashboard combines descriptive analytics, sales prediction, time-series forecasting, and anomaly detection in one retail decision-support view."
)

