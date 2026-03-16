"""Training entry point for the house price prediction model pipeline.

Usage:
    python -m src.train
    # or from the project root:
    python src/train.py

Outputs:
    models/house_price_pipeline.joblib  — a fitted sklearn Pipeline containing
                                          the preprocessor and the model.
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.model import build_model
from src.preprocess import (
    build_preprocessor,
    get_feature_types,
    load_data,
    split_features_target,
)

# ── Project-level constants ───────────────────────────────────────────────────
DATA_PATH = "data/housing_lk.csv"
TARGET_COLUMN = "price_lkr"
MODEL_OUTPUT = Path("models/house_price_pipeline.joblib")
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── Helper functions ──────────────────────────────────────────────────────────

def load_and_split(data_path: str, target_column: str, test_size: float, random_state: int):
    """Load dataset and return train/test feature and target splits.

    Args:
        data_path: Path to the CSV or parquet file.
        target_column: Name of the regression target column.
        test_size: Fraction of data to reserve for evaluation.
        random_state: Seed for reproducible splits.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    print(f"[train] Loading data from: {data_path}")
    df = load_data(data_path)
    print(f"[train] Dataset shape: {df.shape}")

    # Keep target handling centralized via preprocess helper functions.
    X, y = split_features_target(df, target_column)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_pipeline(X_train) -> Pipeline:
    """Detect feature types from training data and build a full sklearn Pipeline.

    The pipeline encapsulates the preprocessor and the model so that a single
    call to .predict() handles both transforming and scoring new data.

    Args:
        X_train: Feature matrix for the training split.

    Returns:
        Unfitted sklearn Pipeline ready to be trained.
    """
    numeric_features, categorical_features = get_feature_types(X_train)
    print(f"[train] Numeric features  ({len(numeric_features)}): {numeric_features}")
    print(f"[train] Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(random_state=RANDOM_STATE)

    # Single artifact: preprocessing + model in one deployable pipeline.
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate(pipeline: Pipeline, X_test, y_test) -> None:
    """Print MAE, RMSE, and R² metrics on the held-out test set.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test feature matrix.
        y_test: Ground-truth target values for the test split.
    """
    # Predictions are made on raw features; pipeline handles transformations internally.
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n[train] ── Evaluation on test set ──────────────────")
    print(f"  MAE  : {mae:>15,.2f} LKR")
    print(f"  RMSE : {rmse:>15,.2f} LKR")
    print(f"  R²   : {r2:>15.4f}")
    print("[train] ────────────────────────────────────────────\n")


def save_pipeline(pipeline: Pipeline, output_path: Path) -> None:
    """Persist the trained pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline to serialise.
        output_path: Destination path for the .joblib artifact.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"[train] Pipeline saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full training workflow end-to-end."""
    X_train, X_test, y_train, y_test = load_and_split(
        data_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(X_train)

    print("[train] Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    print("[train] Training complete.")

    evaluate(pipeline, X_test, y_test)
    save_pipeline(pipeline, MODEL_OUTPUT)


if __name__ == "__main__":
    main()

