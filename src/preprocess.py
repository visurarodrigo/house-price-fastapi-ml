"""Reusable preprocessing utilities for tabular regression workflows.

This module is intentionally focused on data loading, feature/target splitting,
feature type discovery, and construction of a scikit-learn preprocessor. It does
not include model training logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load a tabular dataset from disk into a DataFrame.

    Supported formats:
    - .csv (via pandas.read_csv)
    - .parquet (via pandas.read_parquet)

    Args:
        file_path: Absolute or relative path to the input dataset.

    Returns:
        Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        "Unsupported file format. Use a .csv or .parquet file for load_data()."
    )


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a full dataset into feature matrix X and target vector y.

    Args:
        df: Input DataFrame containing feature and target columns.
        target_column: Name of the target column to extract.

    Returns:
        A tuple containing:
        - X: DataFrame of input features
        - y: Series of target values

    Raises:
        KeyError: If target_column does not exist in df.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns from a DataFrame.

    Numeric columns are inferred from pandas numeric dtypes. Any remaining
    columns are treated as categorical.

    Args:
        X: Feature DataFrame.

    Returns:
        A tuple of:
        - numeric_features: list of numeric column names
        - categorical_features: list of categorical column names
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a reusable sklearn ColumnTransformer for tabular data preprocessing.

    Numeric preprocessing:
    - median imputation
    - standard scaling

    Categorical preprocessing:
    - most-frequent imputation
    - one-hot encoding with unknown-category handling

    Args:
        numeric_features: Numeric column names.
        categorical_features: Categorical column names.

    Returns:
        Configured ColumnTransformer.

    Raises:
        ValueError: If both numeric_features and categorical_features are empty.
    """
    if not numeric_features and not categorical_features:
        raise ValueError("No features provided to build_preprocessor().")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _demo(file_path: str, target_column: str) -> None:
    """Minimal local demo for quickly checking preprocessing setup."""
    df = load_data(file_path)
    X, _ = split_features_target(df, target_column)
    numeric_features, categorical_features = get_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print(f"Rows: {len(df)}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Preprocessor ready: {preprocessor.__class__.__name__}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick preprocessing module demo")
    parser.add_argument("file_path", type=str, help="Path to CSV or parquet dataset")
    parser.add_argument("target_column", type=str, help="Target column name")
    args = parser.parse_args()

    _demo(file_path=args.file_path, target_column=args.target_column)
