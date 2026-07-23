"""Production inference module. Used by:

- notebooks/02_model_training.ipynb (sanity check right after training)
- src/train.py (same sanity check, non-interactive)
- api/main.py (real-time serving — should call this, not reimplement it)

Two entry points:
- predict_unit_health(): returns only the predicted value. Lightweight, no SHAP overhead.
- predict_with_explanation(): returns the predicted value + an out-of-distribution flag +
  the top local SHAP drivers ("RUL: 15 cycles, mainly driven by rising sensor_11"),
  tailored for operator-facing interfaces where a raw number is insufficient to act upon.

Both functions recompute the rolling-window features internally (via
src.features.add_temporal_features) instead of assuming the caller already
passed a pre-engineered dataframe. A real API receives raw-ish sensor
history, not a dataframe with pre-calculated rolling statistics — centralizing
that step here removes a whole class of train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import check_out_of_distribution, compute_shap_values
from features import add_temporal_features, get_sensor_columns


def _get_latest_features(
    raw_history: pd.DataFrame, unit_id: int | str, window_size: int = 10
) -> pd.DataFrame | None:
    """Filters data for a single engine unit, forward-fills missing values,

    recomputes rolling features identically to training time, and returns the
    single most recent row, ready for model inference.

    Returns None if the unit is not found in the raw history.
    """
    unit_data = raw_history[raw_history["unit"] == unit_id].copy()
    if unit_data.empty:
        return None

    if unit_data.isnull().values.any():
        unit_data = unit_data.ffill().fillna(0)

    sensor_cols = get_sensor_columns(unit_data)
    unit_data_eng = add_temporal_features(unit_data, sensor_cols, window_size=window_size)
    return unit_data_eng.tail(1).drop(columns=["unit", "RUL", "cycle"], errors="ignore")


def predict_unit_health(
    raw_history: pd.DataFrame,
    unit_id: int | str,
    model,
    window_size: int = 10,
) -> float | None:
    """raw_history: raw (non feature-engineered) sensor readings for one or more

    units — matching the format produced by data_loader.clean_and_save_data,
    i.e., 'unit', 'cycle', sensor/operational columns (+ 'RUL' if present, ignored).

    Simple entry point: returns predicted RUL, clipped at 0.0 minimum.
    Use predict_with_explanation() when the caller also requires an OOD flag or
    interpretability insights behind the prediction.
    """
    try:
        features = _get_latest_features(raw_history, unit_id, window_size)
        if features is None:
            raise ValueError(f"Unit ID {unit_id} not found in provided history.")

        prediction = model.predict(features)[0]
        return max(0.0, float(prediction))

    except Exception as e:
        print(f"Error during inference for Unit {unit_id}: {str(e)}")
        return None


def predict_with_explanation(
    raw_history: pd.DataFrame,
    unit_id: int | str,
    model,
    explainer,
    reference_ranges: dict,
    window_size: int = 10,
    top_n_reasons: int = 3,
    ood_violation_threshold: float = 0.1,
) -> dict | None:
    """Operator-facing inference: returns predicted RUL + an out-of-distribution flag

    + the top local SHAP drivers for this specific prediction.

    explainer: a shap.TreeExplainer(model) instance, e.g., from
        evaluation.get_shap_explainer(model) — instantiate once at service startup.
    reference_ranges: output of evaluation.compute_feature_reference_ranges,
        persisted at training time and loaded at service startup.

    Returns a dictionary structured for operational decision-making:
        {
            "unit_id": ...,
            "predicted_rul": 15.2,
            "out_of_distribution": False,
            "ood_violation_ratio": 0.0,
            "top_reasons": [{"feature": "s11_roll_mean", "shap_impact": -4.1}, ...]
        }
    A negative shap_impact decreases the prediction (shorter RUL, higher urgency);
    a positive impact increases it.
    """
    try:
        features = _get_latest_features(raw_history, unit_id, window_size)
        if features is None:
            raise ValueError(f"Unit ID {unit_id} not found in provided history.")

        prediction = max(0.0, float(model.predict(features)[0]))
        ood = check_out_of_distribution(features.iloc[0], reference_ranges, ood_violation_threshold)

        # Compute SHAP values for the single row
        shap_row = compute_shap_values(explainer, features)
        if shap_row.ndim == 2:
            shap_row = shap_row[0]
        contributions = pd.Series(shap_row, index=features.columns)

        # Extract top N feature drivers by absolute SHAP magnitude
        top_features = contributions.abs().sort_values(ascending=False).head(top_n_reasons).index
        top = contributions[top_features]

        return {
            "unit_id": unit_id,
            "predicted_rul": prediction,
            "out_of_distribution": ood["is_ood"],
            "ood_violation_ratio": ood["violation_ratio"],
            "top_reasons": [
                {"feature": feat, "shap_impact": round(float(val), 3)}
                for feat, val in top.items()
            ],
        }

    except Exception as e:
        print(f"Error during explained inference for Unit {unit_id}: {str(e)}")
        return None