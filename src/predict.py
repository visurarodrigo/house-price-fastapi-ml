"""Prediction utilities for inference using a trained sklearn Pipeline artifact.

The pipeline loaded here includes both preprocessing and the model, so callers
only need to supply a raw feature dictionary — no manual transformation needed.

Usage (local demo):
    python -m src.predict
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

# Default artifact path relative to the project root.
DEFAULT_MODEL_PATH = "models/house_price_pipeline.joblib"


def load_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a fitted sklearn Pipeline from a joblib artifact on disk.

    Args:
        model_path: Path to the .joblib pipeline file.

    Returns:
        Fitted sklearn Pipeline ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run src/train.py first to generate the pipeline."
        )
    return joblib.load(path)


def prepare_input(features: dict) -> pd.DataFrame:
    """Convert a raw feature dictionary into a single-row pandas DataFrame.

    The DataFrame column order is determined by the dict key order, which is
    preserved in Python 3.7+. The pipeline's ColumnTransformer selects columns
    by name, so order does not need to match training order exactly.

    Args:
        features: Mapping of feature name -> value for a single house.

    Returns:
        Single-row DataFrame suitable for pipeline.predict().

    Raises:
        ValueError: If the features dict is empty.
    """
    if not features:
        raise ValueError("Feature dictionary must not be empty.")
    return pd.DataFrame([features])


def predict_price(
    features: dict,
    model_path: str = DEFAULT_MODEL_PATH,
) -> float:
    """Predict the house price in LKR for a given set of input features.

    This function is the single public entry point for inference. It loads the
    pipeline, prepares the input, runs prediction, and returns a plain float.

    Args:
        features: Raw feature values for one house (excluding price_lkr).
        model_path: Path to the saved pipeline artifact.

    Returns:
        Predicted house price in LKR as a Python float.

    Raises:
        FileNotFoundError: If the pipeline artifact is missing.
        ValueError: If the features dict is empty or prediction fails.
    """
    pipeline = load_pipeline(model_path)
    input_df = prepare_input(features)

    try:
        prediction = pipeline.predict(input_df)
    except Exception as exc:
        raise ValueError(f"Prediction failed: {exc}") from exc

    return float(prediction[0])


# ── Local demo ────────────────────────────────────────────────────────────────

# A realistic sample drawn from the dataset distribution.
_SAMPLE_INPUT: dict = {
    "district": "Colombo",
    "area": "Colombo 05",
    "perch": 20,
    "bedrooms": 4,
    "bathrooms": 3,
    "kitchen_area_sqft": 180,
    "parking_spots": 2,
    "has_garden": True,
    "has_ac": True,
    "water_supply": "Pipe-borne",
    "electricity": "Three phase",
    "floors": 2,
    "year_built": 2015,
}


def _demo() -> None:
    """Run a single-sample prediction and print the result."""
    print("[predict] Running demo prediction...")
    price = predict_price(_SAMPLE_INPUT)
    print(f"[predict] Estimated price: LKR {price:,.2f}")


if __name__ == "__main__":
    _demo()

