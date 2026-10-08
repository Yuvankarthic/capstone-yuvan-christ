from __future__ import annotations

import io
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .anomaly_detection import run_anomaly_detection
from .preprocessing import REQUIRED_COLUMNS

SUPPORTED_EXTENSIONS = (".csv", ".txt", ".tsv", ".xls", ".xlsx", ".xlsm")

DATE_NAME_HINTS = ("date", "time", "day", "month", "year", "created", "order", "invoice")
ID_NAME_HINTS = ("id", "code", "sku", "zip", "postal", "phone", "contact")

SALES_REQUIRED_COLUMNS = list(REQUIRED_COLUMNS)

MAX_CATEGORY_LEVELS = 25
DEFAULT_CONTAMINATION = 0.03


def _read_csv_payload(payload: bytes) -> pd.DataFrame:
    if not payload.strip():
        raise ValueError("The uploaded file is empty.")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The uploaded file contains no columns to parse.") from exc
    raise ValueError(f"Could not decode the CSV file: {last_error}")


def read_uploaded_file(payload: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a cleaned DataFrame."""

    suffix = Path(filename).suffix.lower()

    try:
        if suffix in {".xlsx", ".xlsm", ".xltx"}:
            df = pd.read_excel(io.BytesIO(payload), sheet_name=0, engine="openpyxl")
        elif suffix == ".xls":
            df = pd.read_excel(io.BytesIO(payload), sheet_name=0, engine="xlrd")
        elif suffix == ".tsv":
            df = pd.read_csv(io.BytesIO(payload), sep="\t", encoding="utf-8-sig")
        else:
            df = _read_csv_payload(payload)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file contains no columns to parse.") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")

    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and drop empty or duplicated rows/columns."""

    cleaned = df.copy()
    cleaned.columns = [str(column).replace("\n", " ").strip() for column in cleaned.columns]

    unnamed_columns = [
        column
        for column in cleaned.columns
        if column == "" or column.lower().startswith("unnamed")
    ]
    if unnamed_columns:
        cleaned = cleaned.drop(columns=unnamed_columns)

    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def infer_column_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify every column as date, numeric, or categorical."""

    date_columns: list[str] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        if pd.api.types.is_bool_dtype(series):
            categorical_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        non_null = series.dropna().astype(str)
        if non_null.empty:
            categorical_columns.append(column)
            continue

        numeric_parse = pd.to_numeric(non_null, errors="coerce")
        if numeric_parse.notna().mean() >= 0.9:
            numeric_columns.append(column)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                date_parse = pd.to_datetime(non_null, errors="coerce")
        except (ValueError, TypeError):
            date_parse = pd.Series(dtype="datetime64[ns]")

        if len(date_parse) and date_parse.notna().mean() >= 0.9:
            date_columns.append(column)
        else:
            categorical_columns.append(column)

    return {
        "date": date_columns,
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }


def _looks_like_identifier(series: pd.Series) -> bool:
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return unique_ratio > 0.95


def build_feature_frame(df: pd.DataFrame, roles: dict[str, list[str]]) -> pd.DataFrame:
    """Build the numeric feature matrix used by the detection model."""

    parts: dict[str, pd.Series] = {}

    for column in roles["numeric"]:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() == 0 or series.nunique(dropna=True) <= 1:
            continue
        if _looks_like_identifier(series):
            continue
        parts[column] = series

    for column in roles["categorical"]:
        series = df[column]
        if series.nunique(dropna=True) <= 1:
            continue
        if 1 < series.nunique(dropna=True) <= MAX_CATEGORY_LEVELS:
            parts[f"{column}_code"] = pd.factorize(series.astype(str))[0]

    if roles["date"]:
        dates = pd.to_datetime(df[roles["date"][0]], errors="coerce")
        parts["month"] = dates.dt.month
        parts["day_of_week"] = dates.dt.dayofweek
        parts["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        if dates.notna().any():
            parts["days_since_start"] = (dates - dates.min()).dt.days

    frame = pd.DataFrame(parts, index=df.index)
    return frame.dropna(axis=1, how="all")


def profile_dataset(df: pd.DataFrame, roles: dict[str, list[str]]) -> dict[str, Any]:
    """Summarise the shape and composition of the uploaded dataset."""

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1]) or 1

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_columns": roles["date"],
        "numeric_columns": roles["numeric"],
        "categorical_columns": roles["categorical"],
        "missing_cells": missing_cells,
        "missing_pct": float(missing_cells / total_cells * 100),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def build_data_quality_report(df: pd.DataFrame, roles: dict[str, list[str]]) -> dict[str, Any]:
    """Run standard data-quality checks: missing values, constants, outliers."""

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing["missing_pct"] = missing["missing_count"] / max(len(df), 1) * 100
    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    missing = missing.reset_index(drop=True)

    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=True) <= 1
    ]

    outlier_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []

    for column in roles["numeric"]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())
            if count:
                outlier_rows.append(
                    {
                        "column": column,
                        "outlier_count": count,
                        "outlier_pct": float(count / len(series) * 100),
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                    }
                )

        negative_count = int((series < 0).sum())
        if negative_count:
            negative_rows.append(
                {"column": column, "negative_count": negative_count}
            )

    outliers = pd.DataFrame(outlier_rows)
    negatives = pd.DataFrame(negative_rows)

    return {
        "missing": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": constant_columns,
        "outliers": outliers,
        "negatives": negatives,
    }


def run_file_detection(
    df: pd.DataFrame,
    roles: dict[str, list[str]] | None = None,
    contamination: float = DEFAULT_CONTAMINATION,
) -> dict[str, Any]:
    """Run Isolation Forest anomaly detection on an arbitrary dataset."""

    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    roles = roles or infer_column_roles(df)
    features = build_feature_frame(df, roles)

    if features.shape[1] == 0:
        raise ValueError(
            "No usable numeric features were found in the uploaded file. "
            "Provide at least one numeric column with more than one distinct value."
        )

    pipeline = Pipeline(
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
    pipeline.fit(features)

    transformed = pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["imputer"].transform(features)
    )
    raw_prediction = pipeline.named_steps["model"].predict(transformed)
    anomaly_score = -pipeline.named_steps["model"].decision_function(transformed)

    result = df.copy()
    result["anomaly_score"] = anomaly_score
    result["anomaly_flag"] = [1 if value == 1 else -1 for value in raw_prediction]
    result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    anomaly_count = int((result["anomaly_flag"] == -1).sum())

    return {
        "result": result,
        "features_used": list(features.columns),
        "contamination": contamination,
        "observation_count": int(len(result)),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": float(anomaly_count / max(len(result), 1) * 100),
        "top_anomalies": result.head(25).copy(),
        "roles": roles,
    }


def normalise_column_name(column: object) -> str:
    """Convert a column name to the snake_case form used by the sales schema."""

    name = str(column).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalise_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case column names (first occurrence wins)."""

    normalised = df.copy()
    normalised.columns = [normalise_column_name(column) for column in df.columns]
    if normalised.columns.duplicated().any():
        normalised = normalised.loc[:, ~normalised.columns.duplicated()]
    return normalised


def has_sales_schema(columns: pd.Index | list[str]) -> bool:
    """Check whether the uploaded file matches the full sales dataset schema."""

    normalised = {normalise_column_name(column) for column in columns}
    required = {normalise_column_name(column) for column in REQUIRED_COLUMNS}
    return required.issubset(normalised)


def run_sales_detection(df: pd.DataFrame) -> dict[str, Any]:
    """Run the project's sales-specific detection pipeline on uploaded data."""

    temp_dir = Path(tempfile.mkdtemp(prefix="capstone_upload_"))
    temp_path = temp_dir / "uploaded_sales.csv"
    try:
        df.to_csv(temp_path, index=False)
        bundle = run_anomaly_detection(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
        temp_dir.rmdir()
    return bundle


def analyse_uploaded_file(payload: bytes, filename: str) -> dict[str, Any]:
    """Full upload pipeline: read, profile, quality-check, and detect."""

    df = read_uploaded_file(payload, filename)
    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")

    roles = infer_column_roles(df)
    profile = profile_dataset(df, roles)
    quality = build_data_quality_report(df, roles)
    detection = run_file_detection(df, roles=roles)

    sales_detection = None
    sales_detection_error = None
    if has_sales_schema(df.columns):
        try:
            sales_detection = run_sales_detection(normalise_dataframe_columns(df))
        except Exception as exc:  # noqa: BLE001 - sales pipeline is an optional extra
            sales_detection_error = str(exc) or type(exc).__name__

    return {
        "dataframe": df,
        "roles": roles,
        "profile": profile,
        "quality": quality,
        "detection": detection,
        "sales_detection": sales_detection,
        "sales_detection_error": sales_detection_error,
        "filename": filename,
    }
