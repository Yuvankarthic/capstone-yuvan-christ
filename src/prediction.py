from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model_training import DEFAULT_MODEL_PATH, load_trained_artifact
from .preprocessing import engineer_features, get_model_frame


def prepare_prediction_input(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering steps used in training."""

    engineered = engineer_features(sample_df)
    return get_model_frame(engineered)


def predict_sales(sample_df: pd.DataFrame, model_path: str | Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """Generate sales predictions for one or more sample business scenarios."""

    artifact = load_trained_artifact(model_path)
    pipeline = artifact["pipeline"]
    prediction_frame = prepare_prediction_input(sample_df)
    predicted_sales = pipeline.predict(prediction_frame)

    output = sample_df.copy()
    output["predicted_sales_amount"] = predicted_sales
    return output
