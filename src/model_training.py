from __future__ import annotations

from pathlib import Path
from typing import Any
from math import sqrt

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import (
    engineer_features,
    get_feature_columns,
    get_model_frame,
    get_target,
    load_sales_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sales_model.joblib"


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build the preprocessing pipeline used before model training."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_model() -> RandomForestRegressor:
    """Create a simple and reliable regression model for the prototype."""

    return RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Recover transformed feature names for importance reporting."""

    feature_names: list[str] = []

    numeric_features = list(preprocessor.transformers_[0][2])
    feature_names.extend(numeric_features)

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encoder"]
    categorical_features = list(preprocessor.transformers_[1][2])
    encoded_categorical = encoder.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(encoded_categorical)

    return feature_names


def train_sales_model(csv_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Train the sales model and return metrics plus saved artifacts."""

    raw_df = load_sales_data(csv_path)
    featured_df = engineer_features(raw_df)
    features = get_model_frame(featured_df)
    target = get_target(featured_df)

    numeric_features, categorical_features = get_feature_columns()
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    artifact = {
        "pipeline": pipeline,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "feature_names": get_feature_names(pipeline.named_steps["preprocessor"]),
        "feature_importances": pipeline.named_steps["model"].feature_importances_.tolist(),
        "training_frame": featured_df,
        "feature_columns": numeric_features + categorical_features,
    }

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, DEFAULT_MODEL_PATH)
    return artifact


def load_trained_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a saved model artifact, training one if needed."""

    path = Path(model_path)
    if not path.exists():
        return train_sales_model()

    return joblib.load(path)


if __name__ == "__main__":
    artifact = train_sales_model()
    print("Model trained successfully")
    print(f"RMSE: {artifact['metrics']['rmse']:.2f}")
    print(f"MAE: {artifact['metrics']['mae']:.2f}")
    print(f"R2: {artifact['metrics']['r2']:.3f}")
