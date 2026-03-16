"""Baseline regression model factory for the house price prediction pipeline.

This module is intentionally thin: it returns a configured estimator that is
then embedded inside a full sklearn Pipeline in train.py. Swap the estimator
here without touching any other module.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_model(random_state: int = 42, n_estimators: int = 200) -> RandomForestRegressor:
    """Return a configured RandomForestRegressor as the baseline regression model.

    The model is intentionally left at sensible defaults so it produces a
    meaningful baseline without tuning. Swap this function's body later to
    experiment with GradientBoostingRegressor, XGBRegressor, etc.

    Args:
        random_state: Seed for reproducibility.
        n_estimators: Number of trees in the forest.

    Returns:
        Configured but unfitted RandomForestRegressor instance.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        # Use all available CPU cores for faster baseline training.
        n_jobs=-1,
    )

