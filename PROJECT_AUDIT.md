# Complete Technical Audit

## Audit Scope

This document describes the current project based on the actual source code, current CSV dataset, saved Joblib model artifact, declared dependencies, and read-only execution checks. No project files were modified during the audit.

Documentation and README statements are treated as unverified unless supported by source code or execution evidence.

## 1. Executive Classification

### Implemented and working

- CSV loading with required-column validation
- Numeric conversion and date conversion
- Feature engineering
- Random Forest sales regression
- Model evaluation using RMSE, MAE, and R²
- Saved model artifact using Joblib
- Row-level sales prediction
- Daily sales aggregation
- Holt-Winters Exponential Smoothing forecasting
- Seasonal-naive forecasting baseline
- Forecast evaluation using RMSE, MAE, and MAPE
- Isolation Forest anomaly detection
- Streamlit dashboard
- Anomaly filters
- Feature-importance visualization
- Dashboard charts and tables

### Partially implemented

- Data cleaning
- Exploratory analytics
- Model refreshing
- Business decision support
- Anomaly investigation
- Data validation
- Testing

### Present in code or requirements but not used

- `xgboost` appears in `requirements.txt`, but no XGBoost model is imported or created.
- `sales_per_customer` is engineered but is not used by the Random Forest model.
- `sales_to_ad_ratio` is engineered but is not used by the Random Forest model. It is used by anomaly detection.
- `margin` values exist in the generator configuration but are not written to the dataset or used later.
- Some imports, such as `Iterable`, are unused.

### Planned or mentioned only

- SHAP explainability
- Automated business recommendations
- Automated natural-language insights
- Production deployment
- Real-time processing
- ERP/SAP integration
- Authentication
- APIs
- Data governance
- Cloud deployment

## 2. Project Structure

| File | Purpose | Important functions/classes | Used by |
|---|---|---|---|
| `generate_sales_data.py` | Generates the sales dataset | `build_base_frame`, `assign_business_dimensions`, `create_traffic`, `create_pricing_marketing_sales`, `inject_anomalies`, `main` | Run manually |
| `src/preprocessing.py` | Loads data and creates features | `load_sales_data`, `engineer_features`, `get_feature_columns`, `get_model_frame`, `get_target` | Training, prediction, forecasting, anomaly detection |
| `src/model_training.py` | Builds and trains the sales model | `build_preprocessor`, `build_model`, `train_sales_model`, `load_trained_artifact` | Dashboard and prediction |
| `src/prediction.py` | Generates row-level sales predictions | `prepare_prediction_input`, `predict_sales` | Dashboard |
| `src/forecasting.py` | Forecasts aggregated daily sales | `load_daily_sales_series`, `fit_exponential_smoothing`, `run_sales_forecasting` | Dashboard |
| `src/anomaly_detection.py` | Detects unusual records | `build_isolation_forest`, `run_anomaly_detection` | Dashboard |
| `app/streamlit_app.py` | Main interactive dashboard | Dashboard rendering and filtering | Application entry point |
| `data/sample_sales.csv` | Current sales dataset | 20,000 records and 12 columns | All models |
| `models/sales_model.joblib` | Saved Random Forest artifact | Pipeline, metrics, feature names, importances | Prediction and dashboard |
| `.streamlit/config.toml` | Streamlit theme configuration | Theme colors and font | Streamlit |
| `requirements.txt` | Declared dependencies | Python package requirements | Environment setup |
| `README.md` | Project documentation | Setup and overview | Human documentation |
| `src/__init__.py` | Source package initializer | None | Python package loading |
| `app/__init__.py` | App package initializer | None | Python package loading |

No project test files were found. The `__pycache__` folders and `.venv` are environment/generated files rather than application logic.

### Entry points

Main dashboard:

```text
streamlit run app/streamlit_app.py
```

Model-training command:

```text
python -m src.model_training
```

## 3. Complete Execution Flow

```text
sample_sales.csv
        |
        v
load_sales_data()
        |
        v
engineer_features()
        |
        +------------------------+
        |                        |
        v                        v
Random Forest             Isolation Forest
row-level prediction      row-level anomaly detection
        |                        |
        v                        v
predicted sales           anomaly score and flag
        |                        |
        +------------+-----------+
                     |
                     v
            Streamlit dashboard

sample_sales.csv
        |
        v
Group transactions by date
        |
        v
Daily aggregated sales
        |
        v
Exponential Smoothing
        |
        v
30-day future forecast
        |
        v
Streamlit forecast chart
```

When Streamlit starts, `app/streamlit_app.py` imports all model modules, loads the forecast bundle, loads the anomaly-detection bundle, creates sidebar controls, loads or trains the saved Random Forest artifact, loads engineered data, and renders the dashboard.

Forecasting and anomaly detection operate on the dataset as a whole before sidebar filtering. The sidebar filters only the displayed anomaly results.

## 4. Dataset Audit

### Dataset facts

- File: `data/sample_sales.csv`
- Rows: `20,000`
- Columns: `12`
- Date range: `2023-01-01` to `2025-12-31`
- Unique dates: `1,096`
- Duplicate rows: `0`
- Stores: `6`
- Regions: `4`
- Product categories: `5`

### Dataset columns

| Column | Data type in CSV | Meaning | Used by |
|---|---|---|---|
| `date` | string | Sales transaction date | All workflows |
| `store` | string | Store identifier | Random Forest, anomaly detection, dashboard |
| `region` | string | Store region | Random Forest, anomaly detection, dashboard |
| `product_category` | string | Product category | Random Forest, anomaly detection, dashboard |
| `units_sold` | integer | Number of units sold | Random Forest, anomaly detection |
| `unit_price` | float | Original unit price | Random Forest, anomaly detection |
| `discount_pct` | float | Discount percentage | Random Forest, anomaly detection |
| `ad_spend` | float | Advertising expenditure | Random Forest, anomaly detection |
| `customer_traffic` | float | Customer traffic measure | Random Forest, anomaly detection |
| `promotion_flag` | float | Promotion indicator, normally 0 or 1 | Random Forest, anomaly detection |
| `holiday_flag` | integer | Holiday indicator, normally 0 or 1 | Random Forest, anomaly detection |
| `sales_amount` | float | Sales value and regression target | Random Forest, forecasting, anomaly detection |

### Missing values

The current CSV contains:

- `discount_pct`: 200 missing values
- `ad_spend`: 200 missing values
- `customer_traffic`: 200 missing values
- `promotion_flag`: 200 missing values
- All other columns: zero missing values

### Important value ranges

| Variable | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| `units_sold` | 6 | 83 | 30.43 |
| `unit_price` | 3.53 | 987.69 | 189.48 |
| `discount_pct` | 0 | 0.30 | 0.1019 |
| `ad_spend` | 362.95 | 9,053.36 | 1,641.29 |
| `customer_traffic` | 293 | 3,205 | 1,110.85 |
| `sales_amount` | 103.12 | 104,653.05 | 10,893.98 |

### Dataset origin

The generator clearly creates synthetic data using random number generation, artificial store and category configurations, programmed seasonality, programmed holidays, programmed anomalies, and programmed missing values.

## 5. Data Generation

`generate_sales_data.py` defines:

```python
RANDOM_STATE = 42
N_ROWS = 20000
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
```

The generator:

1. Samples dates from a three-year period with seasonal, weekend, weekday, and holiday weighting.
2. Assigns one of six stores.
3. Assigns one of five product categories.
4. Derives the region from the store.
5. Generates customer traffic.
6. Generates prices and discounts.
7. Generates advertising expenditure.
8. Generates promotion and holiday flags.
9. Generates units sold.
10. Generates sales amount.
11. Injects artificial anomalies into approximately 1.5% of rows.
12. Injects missing values into four columns.
13. Saves the result to `data/sample_sales.csv`.

The injected anomalies include high or low traffic and sales, plus high advertising spend and unusual discounts.

The generator injects approximately 1% missing values independently into `discount_pct`, `ad_spend`, `customer_traffic`, and `promotion_flag`.

## 6. Preprocessing

### `load_sales_data`

Location: `src/preprocessing.py`.

It checks whether the file exists, reads it with `pandas.read_csv`, and checks that all required columns are present.

It does not remove duplicates, validate numeric ranges, validate date values, directly handle missing values, or check categorical values.

### `engineer_features`

This function:

1. Copies the DataFrame.
2. Converts `date` to pandas datetime.
3. Sorts rows chronologically.
4. Converts numeric fields using `pd.to_numeric(errors="coerce")`.
5. Converts categorical fields to strings.
6. Creates calendar features.
7. Creates pricing and business-ratio features.

### Missing values

For the Random Forest:

- Numeric values use median imputation.
- Categorical values use most-frequent imputation.
- Unknown categories during prediction are ignored by `OneHotEncoder(handle_unknown="ignore")`.

For anomaly detection:

- Numeric values use median imputation.
- Numeric values are standardized.

The preprocessing module itself does not impute values; the model pipelines do.

### Important categorical issue

The code performs:

```python
processed[column] = processed[column].astype(str).fillna("Unknown")
```

Because `astype(str)` converts missing values to the string `"nan"`, the later `fillna("Unknown")` does not replace those values. The current dataset has no missing categorical fields, so this issue does not affect the measured data.

### Duplicate and outlier handling

No duplicate rows are removed. No outliers are removed or capped. The anomaly detector flags unusual observations but does not delete them.

### Scaling

- Random Forest numeric features are standardized.
- Isolation Forest numeric features are standardized.
- Forecasting does not use feature scaling.

## 7. Feature Engineering

| Feature | Logic | Used by |
|---|---|---|
| `month` | Month extracted from `date` | Random Forest, anomaly detection |
| `day_of_week` | Monday = 0 through Sunday = 6 | Random Forest, anomaly detection |
| `quarter` | Calendar quarter extracted from `date` | Random Forest, anomaly detection |
| `is_weekend` | 1 for Saturday/Sunday, otherwise 0 | Random Forest, anomaly detection |
| `price_after_discount` | `unit_price * (1 - discount_pct)` | Random Forest, anomaly detection |
| `marketing_intensity` | `ad_spend / customer_traffic` | Random Forest, anomaly detection |
| `sales_per_customer` | `sales_amount / customer_traffic` | Anomaly detection only |
| `sales_to_ad_ratio` | `sales_amount / ad_spend` | Anomaly detection only |
| `traffic_per_unit` | `customer_traffic / units_sold` | Random Forest, anomaly detection |

Zero denominators are replaced with missing values and then filled with zero.

### Random Forest features

Numeric features:

- `units_sold`
- `unit_price`
- `discount_pct`
- `ad_spend`
- `customer_traffic`
- `promotion_flag`
- `holiday_flag`
- `month`
- `day_of_week`
- `quarter`
- `is_weekend`
- `price_after_discount`
- `marketing_intensity`
- `traffic_per_unit`

Categorical features:

- `store`
- `region`
- `product_category`

The Random Forest excludes `sales_per_customer` and `sales_to_ad_ratio`, both of which are derived from the target `sales_amount`.

## 8. Machine Learning Model: Random Forest

The sales-prediction model is `RandomForestRegressor`, a supervised regression model.

### Hyperparameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 250 |
| `random_state` | 42 |
| `max_depth` | 8 |
| `min_samples_leaf` | 2 |

### Pipeline

```text
Input features
    |
    v
ColumnTransformer
    |
    +--> Numeric pipeline: median imputation, standard scaling
    |
    +--> Categorical pipeline: most-frequent imputation, one-hot encoding
    |
    v
RandomForestRegressor
```

The encoder uses `OneHotEncoder(handle_unknown="ignore")`.

The saved artifact contains 29 transformed features: 14 numeric features and 15 one-hot encoded categorical features.

### Train/test split

```python
test_size=0.2
random_state=42
```

The split is random rather than time-based. This is suitable for a basic row-level demonstration but may produce an optimistic estimate of future performance because repeated dates and synthetic patterns can appear in both partitions.

### Actual model metrics

| Metric | Value |
|---|---:|
| RMSE | 981.37 |
| MAE | 546.03 |
| R² | 0.993765 |

### Feature importance

| Feature | Importance |
|---|---:|
| `unit_price` | 0.885285 |
| `units_sold` | 0.094351 |
| `price_after_discount` | 0.016355 |
| `customer_traffic` | 0.002479 |
| `ad_spend` | 0.000635 |

The dashboard displays the top 12 features. This is Random Forest impurity-based feature importance, not SHAP and not causal analysis.

### Meaning of one prediction

Given a date, store, region, product category, units sold, price, discount, advertising spend, customer traffic, promotion flag, and holiday flag, the model estimates the expected `sales_amount`.

For one checked record:

```text
Actual sales:      4,874.05
Predicted sales:   4,776.47
```

The dashboard does not provide arbitrary user-entered prediction fields. It creates three sample scenarios in code.

## 9. Model Evaluation

### RMSE

RMSE is the square root of the average squared prediction error. Large errors receive more penalty. The stored value is `981.37`.

### MAE

MAE is the average absolute prediction error. The stored value is `546.03`, meaning that predictions differ from actual values by approximately 546 sales units on average in the held-out sample.

### R²

R² measures the proportion of target variation explained by the model. The stored value is `0.993765`.

This is very high and should be interpreted cautiously because the dataset is synthetic, the sales formula depends on model inputs, and the split is random rather than time-based.

### MAPE

MAPE is used for forecasting only. It is not used for Random Forest regression.

### Metrics not implemented

There is no accuracy, precision, recall, F1 score, or confusion matrix because the project does not implement a classification model.

## 10. Time-Series Forecasting

### Model

The project uses additive Holt-Winters Exponential Smoothing with a damped additive trend and weekly additive seasonality:

```python
ExponentialSmoothing(
    series,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated",
)
```

### Input series

Transactions are grouped by date and `sales_amount` is summed. The current series contains 1,096 daily observations from 2023-01-01 to 2025-12-31. No missing calendar dates were found in the current data.

If missing dates exist, the code reindexes to a complete daily date range, linearly interpolates values, and then applies forward/backward filling if necessary.

### Forecasting process

1. Load raw sales data.
2. Convert dates to datetime.
3. Group by date and sum sales.
4. Sort by date.
5. Reindex to a complete daily calendar.
6. Interpolate missing dates.
7. Train on all but the final 90 days.
8. Forecast the 90-day validation period.
9. Compare with a lag-7 seasonal-naive baseline.
10. Fit a final model on the full historical series.
11. Forecast the next 30 days.

### Parameters

| Parameter | Value |
|---|---:|
| Trend | Additive |
| Damped trend | Yes |
| Seasonality | Additive |
| Seasonal period | 7 days |
| Validation period | 90 days |
| Future horizon | 30 days |

The level component is estimated internally by Statsmodels.

### Actual periods

| Period | Dates |
|---|---|
| Training | 2023-01-01 to 2025-10-02 |
| Validation | 2025-10-03 to 2025-12-31 |
| Future forecast | 30 days after the final historical date |

### Forecast metrics

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Seasonal Naive | 159,963.81 | 118,206.70 | 39.65% |
| Exponential Smoothing | 150,804.46 | 107,981.79 | 36.24% |

Exponential Smoothing performs better on all three metrics for the current dataset. The dashboard text states that the selected model is better, but the code does not dynamically check that condition before displaying the statement.

### Prediction versus forecasting

Sales prediction operates on one business scenario and estimates its sales amount.

Sales forecasting operates on aggregated historical daily sales and estimates future total sales for each day.

The Random Forest predicts a row-level business outcome. The Exponential Smoothing model predicts future aggregate time-series values.

## 11. Anomaly Detection

### Algorithm and parameters

The project uses `IsolationForest` with:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `contamination` | 0.02 |
| `random_state` | 42 |

### Features

Isolation Forest uses:

- `sales_amount`
- `units_sold`
- `unit_price`
- `discount_pct`
- `ad_spend`
- `customer_traffic`
- `promotion_flag`
- `holiday_flag`
- `month`
- `day_of_week`
- `quarter`
- `is_weekend`
- `price_after_discount`
- `marketing_intensity`
- `traffic_per_unit`
- `sales_to_ad_ratio`

### Processing

```text
Raw data
   |
   v
Feature engineering
   |
   v
Selected numeric features
   |
   v
Median imputation
   |
   v
StandardScaler
   |
   v
IsolationForest
   |
   v
Anomaly flag and score
```

The model outputs `1` for normal observations and `-1` for anomalies. The code transforms the decision function so larger scores represent more unusual records.

### Actual results

- Total observations: `20,000`
- Anomalies: `400`
- Anomaly percentage: `2.00%`
- Normal observations: `19,600`

The number is exactly consistent with the configured 2% contamination.

### Simple explanation

Isolation Forest repeatedly splits the data randomly. Normal records resemble many other records and take more splits to isolate. Unusual records are different and become isolated quickly.

An anomaly means that a record is unusual compared with the other records. It does not prove that the record is wrong, fraudulent, or invalid.

```text
Anomaly != Error
```

## 12. Streamlit Dashboard

The dashboard has four main sections.

| Dashboard section | Purpose | Source | Model or logic |
|---|---|---|---|
| Business Overview | High-level totals | Full engineered dataset and anomaly bundle | Aggregation and Isolation Forest count |
| Sales Prediction | Model metrics and sample predictions | Saved Joblib model and generated scenarios | Random Forest |
| Feature Importance | Top predictive variables | Saved artifact | Random Forest impurity importance |
| Sales Forecast | Historical and future sales | Forecast bundle | Holt-Winters Exponential Smoothing |
| Forecast Comparison | Model comparison | Forecast metrics | Seasonal Naive versus Exponential Smoothing |
| Anomaly Detection | Unusual records | Anomaly bundle | Isolation Forest |
| Sidebar Controls | Controls anomaly display | Filtered anomaly results | Pandas filtering |

### Business Overview

Displays total sales, average sales, transaction count, and total detected anomalies. These values are calculated from the complete dataset and are not affected by sidebar filters.

### Sales Prediction

Displays the model name, RMSE, MAE, R², three sample predictions, a predicted-sales bar chart, and the top 12 feature importances.

The scenarios are created by `build_prediction_samples()`:

1. A modified version of the latest valid data row.
2. A Store A / North / Electronics scenario.
3. A Store F / South / Home scenario.

### Sales Forecast

Displays the Exponential Smoothing description, 30-day horizon, validation period, MAE, RMSE, MAPE, historical sales, future forecast, and seasonal-naive comparison table.

No forecast filters are provided.

### Anomaly Detection

Displays total observations, anomaly count, anomaly percentage, a time-series chart with potential anomalies marked in red, and a filtered anomaly table.

The anomaly table contains date, store, region, product category, sales amount, anomaly score, and anomaly flag.

## 13. Dashboard Filters

The sidebar includes a Train / Refresh Model button, Store multiselect, Region multiselect, Product Category multiselect, and Date range selector.

### Store filter

Filters anomaly rows by selected store. It affects the anomaly chart and table only.

### Region filter

Filters anomaly rows by selected region. It affects the anomaly chart and table only.

### Product Category filter

Filters anomaly rows by selected category. It affects the anomaly chart and table only.

### Date filter

Filters anomaly rows by date range. It affects the anomaly chart and table only.

### What filters do not affect

The filters do not affect:

- Random Forest training
- Random Forest predictions
- Forecasting
- Overview KPIs
- Anomaly model training

The anomaly model is trained on the full dataset before filtering.

### Empty selections

Filtering is applied only when a selected list is truthy. An empty multiselect therefore behaves as “show all” rather than “show no records.”

### Refresh Model button

The button calls `train_sales_model(DEFAULT_DATA_PATH)` and clears Streamlit caches. It refreshes the Random Forest artifact. It does not explicitly retrain the forecasting or anomaly models in that branch.

## 14. Model-to-Dashboard Connections

### Random Forest

```text
Dashboard sample scenarios
        |
        v
prepare_prediction_input()
        |
        v
engineer_features()
        |
        v
get_model_frame()
        |
        v
Saved Random Forest pipeline
        |
        v
predicted_sales_amount
        |
        v
Table and bar chart
```

### Forecasting

```text
CSV transaction data
        |
        v
Group by date and sum sales
        |
        v
Daily time series
        |
        v
Exponential Smoothing
        |
        v
30-day future forecast
        |
        v
Line chart and comparison table
```

### Anomaly detection

```text
CSV transaction data
        |
        v
Feature engineering
        |
        v
Selected numeric features
        |
        v
Imputation and scaling
        |
        v
Isolation Forest
        |
        v
Anomaly flag and score
        |
        v
Filtered chart and table
```

## 15. Saved Artifacts

### `models/sales_model.joblib`

This is a serialized Joblib dictionary containing:

- Trained preprocessing and model pipeline
- RMSE, MAE, and R²
- Transformed feature names
- Feature importance values
- Full engineered training frame
- Feature column names

It is created by `train_sales_model()` and loaded by `load_trained_artifact()`.

If the artifact does not exist, the code retrains a model using the default dataset.

There is no model version, training timestamp, dataset hash, model registry, or compatibility validation.

Forecast and anomaly models are not saved to disk. They are fitted in memory and cached by Streamlit.

## 16. Dependencies

| Library | Actually used? | Purpose |
|---|---|---|
| `pandas` | Yes | Data loading and manipulation |
| `numpy` | Yes | Numerical operations and random generation |
| `scikit-learn` | Yes | Random Forest, preprocessing, metrics, Isolation Forest |
| `xgboost` | No | Present in requirements but not imported or used |
| `plotly` | Yes | Dashboard charts |
| `streamlit` | Yes | Interactive dashboard |
| `joblib` | Yes | Save and load model artifact |
| `statsmodels` | Yes | Exponential Smoothing forecasting |

Standard-library modules used include `pathlib`, `typing`, `math`, and `sys`.

## 17. SHAP and Explainable AI

SHAP is not currently implemented.

The dashboard feature-importance chart comes from:

```python
RandomForestRegressor.feature_importances_
```

This is not SHAP. The project does not provide local explanations, SHAP values, partial-dependence plots, counterfactual explanations, or feature contributions for an individual prediction.

## 18. Business Insights and Recommendations

Automated business insights and recommendations are not implemented.

The dashboard contains static explanatory text, but there is no code that recommends changing prices, increasing advertising, changing discounts, reordering stock, investigating a specific store, or adjusting promotions.

Decision support is therefore limited to displaying metrics, predictions, forecasts, and unusual records.

## 19. Data Modeling, Security, Governance, and Business Rules

| Area | Status | Evidence |
|---|---|---|
| Data modeling | Partial | Defined tabular columns and derived features, but no database schema or dimensional warehouse model |
| Data security | Not implemented | No authentication, authorization, encryption, or secrets management |
| Data governance | Not implemented | No lineage, ownership, audit logging, retention, or formal data-quality framework |
| Business rules | Partial | Generator contains holiday, promotion, pricing, store, and category rules |
| Business recommendations | Not implemented | No recommendation engine |
| User access control | Not implemented | No users or permissions |

## 20. Testing and Validation

No automated test files were found. There is no `tests/` directory, pytest suite, integration-test configuration, or CI pipeline.

The generator's `validate_dataset()` function prints row count, column count, date range, missing values, duplicate count, sales statistics, store/region/category counts, promotion comparisons, traffic-sales correlation, holiday comparisons, and monthly averages. It is a diagnostic function rather than an automated test suite.

Read-only audit checks confirmed:

- Forecasting import succeeded in the project virtual environment.
- A saved-model prediction succeeded.
- Forecasting executed successfully.
- Anomaly detection executed successfully.
- The Streamlit application previously started successfully on port 8502.
- The existing Joblib artifact loaded successfully.

The model-training command was not rerun because it writes a new artifact and this audit was read-only.

## 21. End-to-End Example

A real transaction row contains date, store, region, product category, units sold, unit price, discount, advertising spend, customer traffic, promotion flag, holiday flag, and sales amount.

For one checked record:

```text
Date: 2025-05-06
Store: Store_E
Region: North
Category: Clothing
Actual sales: 4,874.05
Predicted sales: 4,776.47
```

### Row-level prediction

1. The row is passed to `predict_sales()`.
2. `engineer_features()` creates calendar and ratio features.
3. `get_model_frame()` selects model features.
4. The saved pipeline imputes and encodes values.
5. The Random Forest generates a prediction.
6. The prediction is appended as `predicted_sales_amount`.

### Forecasting

The individual row is not directly forecasted. All transactions on the same date are grouped and summed, producing the daily series used by Exponential Smoothing.

### Anomaly detection

The row is converted into numeric anomaly features and compared with all other rows using Isolation Forest.

This distinction is important:

- Random Forest operates on an individual business scenario.
- Isolation Forest operates on individual records.
- Exponential Smoothing operates on aggregate daily history.

## 22. What the Project Actually Does

This project is a Streamlit retail analytics prototype. It loads a sales CSV file, creates calendar and business-related features, trains or loads a Random Forest regression model to estimate sales for sample business scenarios, aggregates historical sales by day to forecast the next 30 days using Exponential Smoothing, and applies Isolation Forest to flag unusual sales records. The dashboard displays summary statistics, model metrics, prediction examples, forecasts, feature importance, and filtered anomaly records.

It is a functioning prototype, but it is not a complete production Business Intelligence platform.

## 23. What the Project Does Not Do

The current code does not implement:

- SHAP explanations
- Automated recommendations
- Automated natural-language insights
- Classification
- Fraud detection
- Real-time streaming
- Cloud deployment
- ERP integration
- SAP integration
- REST API
- Authentication
- Authorization
- User accounts
- Database storage
- Data warehouse integration
- Data governance
- Audit logging
- Model versioning
- Model monitoring
- Drift detection
- Automated retraining schedules
- Export buttons
- Report generation
- Email alerts
- Notification system
- Automated testing
- Production-grade data validation
- Time-series cross-validation
- Hyperparameter search
- XGBoost implementation

## 24. Current Feature Checklist

| Feature | Status |
|---|---|
| Data ingestion | ✅ Implemented |
| Data cleaning | 🟡 Partially implemented |
| Missing-value handling | ✅ Implemented inside model pipelines |
| Feature engineering | ✅ Implemented |
| Exploratory data analysis | 🟡 Basic descriptive analysis only |
| Sales prediction | ✅ Implemented |
| Sales forecasting | ✅ Implemented |
| Anomaly detection | ✅ Implemented |
| Model evaluation | ✅ Implemented |
| Feature importance | ✅ Implemented |
| SHAP | ❌ Not implemented |
| Business insights | 🟡 Static dashboard descriptions only |
| Recommendations | ❌ Not implemented |
| Dashboard | ✅ Implemented |
| Filters | ✅ Implemented for anomaly displays |
| Export | ❌ Not implemented |
| Reporting | 🟡 Dashboard visualization only |
| Data modeling | 🟡 Basic tabular structure only |
| Data security | ❌ Not implemented |
| Data governance | ❌ Not implemented |
| Business rules | 🟡 Mainly present in synthetic data generation |
| Cloud deployment | ❌ Not implemented |
| Real-time processing | ❌ Not implemented |
| ERP integration | ❌ Not implemented |
| API | ❌ Not implemented |
| Authentication | ❌ Not implemented |
| Testing | ❌ No automated tests |

# Viva Preparation

## 25. Basic Project Questions

### 1. What is the purpose of the project?

It combines sales prediction, time-series forecasting, anomaly detection, and dashboard visualization for retail data.

### 2. What is the main application entry point?

The main entry point is `app/streamlit_app.py`, launched with Streamlit.

### 3. What type of data does the system process?

It processes tabular sales transaction data containing dates, stores, categories, prices, traffic, advertising, and sales.

### 4. Is the project production-ready?

No. It is a working prototype without authentication, automated tests, deployment infrastructure, monitoring, or governance.

## 26. Dataset Questions

### 5. How many rows are in the dataset?

There are 20,000 rows.

### 6. How many columns are present?

There are 12 columns.

### 7. What is the target variable?

The target variable for Random Forest regression is `sales_amount`.

### 8. Is the dataset real or synthetic?

The generator creates synthetic data using programmed business rules and random values.

## 27. Preprocessing Questions

### 9. How are missing numerical values handled?

They are replaced using median imputation inside the scikit-learn pipelines.

### 10. How are categorical variables handled?

They are imputed using the most frequent value and then one-hot encoded.

### 11. Is scaling used?

Yes. Numeric values are standardized before Random Forest and Isolation Forest processing.

### 12. Are duplicate records removed?

No. The code checks duplicates in the generator validation function but does not remove them.

### 13. Are outliers removed?

No. Unusual observations are flagged by Isolation Forest instead.

## 28. Random Forest Questions

### 14. Why was Random Forest used?

It handles nonlinear relationships, mixed business features, and categorical variables after encoding. It also provides feature importance for interpretation.

### 15. Is the Random Forest a classification or regression model?

It is a regression model because it predicts a continuous sales amount.

### 16. What does one Random Forest prediction represent?

It represents the estimated sales amount for one set of business conditions.

### 17. What are the important Random Forest hyperparameters?

The model uses 250 trees, maximum depth 8, minimum leaf size 2, and random state 42.

### 18. Why is the R² value so high?

The dataset is synthetic and sales are generated from many of the same variables used as model inputs. The random split may also produce an optimistic result.

### 19. Is feature importance the same as causality?

No. Feature importance shows predictive usefulness, not that changing a feature will necessarily cause sales to change.

## 29. Forecasting Questions

### 20. What is the difference between prediction and forecasting?

Prediction estimates sales for a business scenario. Forecasting estimates future aggregate sales over time.

### 21. Why is the sales data aggregated daily?

The forecasting model requires a time series, so transaction sales are summed for each date.

### 22. Why is the seasonal period seven?

Seven represents a weekly cycle.

### 23. What forecasting model is used?

Additive Holt-Winters Exponential Smoothing with damped trend and weekly seasonality.

### 24. What is the forecast horizon?

The model forecasts the next 30 days.

### 25. What is the purpose of the seasonal-naive baseline?

It provides a simple benchmark based on repeating the previous seven-day pattern.

### 26. Did Exponential Smoothing outperform the baseline?

Yes. On the current dataset it produced lower RMSE, MAE, and MAPE.

## 30. Anomaly Detection Questions

### 27. What anomaly algorithm is used?

Isolation Forest.

### 28. Why is Isolation Forest suitable here?

It can identify unusual observations without requiring manually labelled anomaly examples.

### 29. What does contamination mean?

It is the expected proportion of anomalies. The project uses 2%.

### 30. How many anomalies were found?

400 out of 20,000 rows, equal to 2%.

### 31. Does an anomaly mean the transaction is wrong?

No. It only means the transaction is unusual compared with the rest of the data.

## 31. Dashboard Questions

### 32. What does the dashboard display?

It displays KPIs, sales predictions, feature importance, forecasts, forecast comparisons, and anomaly records.

### 33. What do the sidebar filters affect?

They affect the anomaly chart and anomaly table.

### 34. Do the filters retrain the models?

No. They filter already-generated anomaly results.

### 35. Are prediction scenarios entered by the user?

No. The dashboard creates three sample scenarios in code.

## 32. Architecture Questions

### 36. How are models connected to the dashboard?

The dashboard imports functions from the `src` modules and displays their returned DataFrames, metrics, and model outputs.

### 37. How is the Random Forest saved?

A dictionary containing the pipeline and metadata is saved using Joblib.

### 38. Are forecast and anomaly models saved?

No. They are fitted in memory and cached by Streamlit.

## 33. Limitations Questions

### 39. Is SHAP implemented?

No. The project uses Random Forest impurity-based feature importance.

### 40. Are automated recommendations implemented?

No. The dashboard provides visual evidence but does not generate actions or recommendations.

### 41. Are there automated tests?

No. There are no project test files.

### 42. Is the train/test split time-aware?

No. The Random Forest uses a random 80/20 split.

## 34. Future Scope Questions

### 43. Why might XGBoost be considered in future work?

It is often powerful for tabular data, but although it is listed as a dependency, it is not currently implemented.

### 44. Why might SHAP be added?

SHAP could explain individual predictions and show how each feature contributed to a prediction.

### 45. How could the project be made production-ready?

It would need robust validation, automated testing, model versioning, monitoring, security, authentication, deployment, and real data integration.

### 46. How could it work with real business data?

The CSV loader could be connected to a database, warehouse, ERP system, or API, provided the real data has compatible fields.

### 47. How could business recommendations be added?

A rules engine or optimization layer could convert predictions and anomalies into suggested actions.

### 48. How could forecasting be improved?

The project could compare multiple models using rolling time-series validation and include store- or category-level forecasts.

### 49. How could domain independence be improved?

The schema, feature engineering, and business rules would need to be configurable rather than hard-coded for retail.

### 50. What is the most important limitation to mention in the viva?

The system demonstrates the complete machine-learning workflow, but it remains a prototype using synthetic data and does not yet provide production governance, security, automated testing, or real-time integration.

# Final Summary

## Project in one paragraph

This project is a Python and Streamlit retail analytics prototype that processes a 20,000-row sales dataset. It engineers calendar, pricing, traffic, and marketing features, trains a Random Forest regression model for row-level sales estimation, uses Holt-Winters Exponential Smoothing to forecast aggregate daily sales for the next 30 days, and applies Isolation Forest to identify unusual business records. The results are presented through a Streamlit dashboard containing business KPIs, sample predictions, feature importance, forecast comparisons, charts, and anomaly filters. SHAP, automated recommendations, APIs, security, governance, deployment, and automated tests are not currently implemented.

## Complete system flow

```text
Synthetic CSV dataset
        ↓
Required-column validation
        ↓
Date conversion and sorting
        ↓
Numeric conversion
        ↓
Calendar and business feature engineering
        ↓
        ├── Random Forest regression → Sales prediction
        ├── Daily aggregation → Exponential Smoothing → 30-day forecast
        └── Numeric anomaly features → Isolation Forest → Anomaly flag and score
        ↓
Streamlit dashboard
```

## Implemented features

CSV loading, schema checking, date processing, numeric conversion, model-pipeline imputation, feature engineering, Random Forest regression, RMSE/MAE/R² evaluation, Joblib model saving, row-level prediction, daily aggregation, Exponential Smoothing, seasonal-naive comparison, RMSE/MAE/MAPE forecast evaluation, Isolation Forest, anomaly scoring, Streamlit dashboard, Plotly charts, anomaly filters, and feature importance.

## Partial and planned features

Data validation is limited. Data cleaning is distributed across model pipelines. Business insights are mainly descriptive. Recommendations are not generated. Feature importance exists, but SHAP does not. Testing is manual rather than automated. Security and governance are absent. Dashboard filters apply mainly to anomaly visualization. The Random Forest uses a random split rather than time-aware validation.

## Model summary

| Model | Purpose | Input | Output | Evaluation |
|---|---|---|---|---|
| Random Forest Regressor | Row-level sales prediction | Business, pricing, traffic, promotion, and calendar features | Predicted sales amount | RMSE 981.37, MAE 546.03, R² 0.993765 |
| Exponential Smoothing | Aggregate daily sales forecasting | Daily total sales | 30 future daily sales values | RMSE 150,804.46, MAE 107,981.79, MAPE 36.24% |
| Seasonal Naive | Forecasting baseline | Previous seven-day seasonal pattern | Baseline future values | RMSE 159,963.81, MAE 118,206.70, MAPE 39.65% |
| Isolation Forest | Unusual-record detection | 16 standardized numeric features | Anomaly flag and score | 400 anomalies, 2.00% of rows |

## Dashboard summary

The dashboard contains Business Overview, Sales Prediction, Sales Forecast, and Anomaly Detection sections. It includes KPI cards, prediction tables, prediction charts, feature-importance charts, historical and future forecast charts, forecast comparison tables, anomaly charts, anomaly tables, and Store, Region, Category, and Date filters.
