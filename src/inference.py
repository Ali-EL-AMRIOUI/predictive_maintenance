"""
Production inference. Used by:
- notebooks/02_model_training.ipynb (sanity check right after training)
- src/train.py (same sanity check, non-interactive)
- api/main.py (real-time serving — should call this, not reimplement it)

Two entry points:
- predict_unit_health(): just the number. Cheap, no SHAP overhead.
- predict_with_explanation(): the number + an out-of-distribution flag + the
  top local SHAP drivers ("RUL: 15 cycles, mainly driven by rising sensor_11"),
  for operator-facing output where a bare number isn't enough to act on.

Both recompute the rolling-window features themselves (via
src.features.add_temporal_features) instead of assuming the caller already
passed a pre-engineered dataframe. A real API receives raw-ish sensor
history, not a dataframe with rolling stats already attached — centralizing
that step here removes a whole class of train/serve skew.
"""
from __future__ import annotations

import pandas as pd

from evaluation import check_out_of_distribution
from features import add_temporal_features, get_sensor_columns


def _get_latest_features(raw_history: pd.DataFrame, unit_id, window_size: int = 10) -> pd.DataFrame | None:
    """Filters to one unit, forward-fills gaps, recomputes rolling features the
    same way training did, and returns the single most recent row, model-ready.
    Returns None if the unit isn't present."""
    unit_data = raw_history[raw_history["unit"] == unit_id]
    if unit_data.empty:
        return None

    if unit_data.isnull().values.any():
        unit_data = unit_data.ffill().fillna(0)

    sensor_cols = get_sensor_columns(unit_data)
    unit_data_eng = add_temporal_features(unit_data, sensor_cols, window_size=window_size)
    return unit_data_eng.tail(1).drop(columns=["unit", "RUL", "cycle"], errors="ignore")


def predict_unit_health(
    raw_history: pd.DataFrame,
    unit_id,
    model,
    window_size: int = 10,
) -> float | None:
    """
    raw_history: raw (non feature-engineered) sensor readings for one or more
    units — the same shape/columns produced by data_loader.clean_and_save_data,
    i.e. 'unit', 'cycle', sensor/operational columns (+ 'RUL' if present, ignored).

    Simple entry point: predicted RUL, clipped at 0. Use
    predict_with_explanation() when the caller also needs an OOD flag or the
    reasons behind the number.
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
    unit_id,
    model,
    explainer,
    reference_ranges: dict,
    window_size: int = 10,
    top_n_reasons: int = 3,
    ood_violation_threshold: float = 0.1,
) -> dict | None:
    """
    Operator-facing inference: predicted RUL + an out-of-distribution flag +
    the top local SHAP drivers of this specific prediction.

    explainer: a shap.TreeExplainer(model), e.g. from
        evaluation.get_shap_explainer(model) — build once at service start,
        not per call, it's not cheap.
    reference_ranges: output of evaluation.compute_feature_reference_ranges,
        persisted at training time and loaded once at service start.

    Returns a dict rather than a bare float on purpose — this is meant to be
    read by a person, not just plotted, so it carries enough context to act on:
        {
            "unit_id": ...,
            "predicted_rul": 15.2,
            "out_of_distribution": False,
            "ood_violation_ratio": 0.0,
            "top_reasons": [{"feature": "sensor_11_roll_mean", "shap_impact": -4.1}, ...]
        }
    A negative shap_impact pushes the prediction down (shorter RUL, more urgent);
    positive pushes it up.
    """
    try:
        features = _get_latest_features(raw_history, unit_id, window_size)
        if features is None:
            raise ValueError(f"Unit ID {unit_id} not found in provided history.")

        prediction = max(0.0, float(model.predict(features)[0]))
        ood = check_out_of_distribution(features.iloc[0], reference_ranges, ood_violation_threshold)

        shap_row = explainer.shap_values(features)
        if hasattr(shap_row, "shape") and len(shap_row.shape) == 2:
            shap_row = shap_row[0]
        contributions = pd.Series(shap_row, index=features.columns)
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_n_reasons)

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
