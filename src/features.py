"""
Feature engineering utilities shared between:
- notebooks/02_model_training.ipynb (training-time feature matrix)
- src/inference.py (production inference — must reproduce the exact same features)

Centralizing this here is what guarantees training/serving consistency:
both call the same function instead of maintaining two implementations.
"""
from __future__ import annotations

import pandas as pd


def get_sensor_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Returns the list of sensor/operational columns to engineer features from."""
    exclude = exclude or ["unit", "cycle", "RUL"]
    return [col for col in df.columns if col not in exclude]


def add_temporal_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int = 10,
) -> pd.DataFrame:
    """
    Computes rolling mean and rolling std for each sensor column, grouped by 'unit'
    so that the rolling window never crosses from one engine's history into another's.

    Rolling is trailing (backward-looking) by construction (pandas default), so this
    is safe to apply identically to train and test data without leaking future cycles
    into past rows.

    Note: sensor_cols is an explicit parameter (not a module-level global) so this
    function is safely reusable from inference.py without depending on notebook state.
    """
    df_out = df.copy()
    grouped = df_out.groupby("unit")

    for col in sensor_cols:
        df_out[f"{col}_roll_mean"] = grouped[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        df_out[f"{col}_roll_std"] = grouped[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std().fillna(0)
        )

    return df_out
