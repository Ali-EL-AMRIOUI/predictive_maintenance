"""
Unit tests for src/inference.py.

Uses a small XGBoost model actually trained on synthetic CMAPSS-shaped data
(not a mock) so the feature-engineering round trip (add_temporal_features)
is exercised for real, the same way it would be in production.

Run: pytest tests/test_inference.py -v
"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from evaluation import compute_feature_reference_ranges, get_shap_explainer
from features import add_temporal_features, get_sensor_columns
from inference import predict_unit_health, predict_with_explanation


@pytest.fixture(scope="module")
def synthetic_history():
    """Raw (pre-feature-engineering) sensor history for 3 units, in the same
    shape data_loader.load_raw_cmapss would produce: unit, cycle, sensor_*."""
    rng = np.random.default_rng(42)
    rows = []
    for unit in [1, 2, 3]:
        n_cycles = 60
        degradation = np.linspace(0, 1, n_cycles)
        for cycle in range(1, n_cycles + 1):
            d = degradation[cycle - 1]
            row = {"unit": unit, "cycle": cycle}
            for s in range(5):
                row[f"sensor_{s}"] = 50 + 10 * d + rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_model_and_context(synthetic_history):
    """Trains a real (tiny) model on the synthetic history, mirroring what
    train.py's engineer_features/train_champion_model do, so the fixtures
    below exercise the actual feature pipeline rather than a mock."""
    sensor_cols = get_sensor_columns(synthetic_history, exclude=["unit", "cycle"])
    engineered = add_temporal_features(synthetic_history, sensor_cols, window_size=10)
    # Fake RUL target: cycles remaining until each unit's last cycle.
    max_cycle = engineered.groupby("unit")["cycle"].transform("max")
    y = (max_cycle - engineered["cycle"]).astype(float)
    X = engineered.drop(columns=["unit", "cycle"])

    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=0)
    model.fit(X, y)

    reference_ranges = compute_feature_reference_ranges(X, lower_q=0.01, upper_q=0.99)
    explainer = get_shap_explainer(model)
    return model, explainer, reference_ranges


def test_predict_unit_health_returns_non_negative_float(synthetic_history, trained_model_and_context):
    model, _, _ = trained_model_and_context
    result = predict_unit_health(synthetic_history, unit_id=1, model=model, window_size=10)
    assert result is not None
    assert isinstance(result, float)
    assert result >= 0.0


def test_predict_unit_health_unknown_unit_returns_none(synthetic_history, trained_model_and_context):
    model, _, _ = trained_model_and_context
    result = predict_unit_health(synthetic_history, unit_id=999, model=model, window_size=10)
    assert result is None


def test_predict_with_explanation_returns_expected_shape(synthetic_history, trained_model_and_context):
    model, explainer, reference_ranges = trained_model_and_context
    result = predict_with_explanation(
        synthetic_history, unit_id=2, model=model, explainer=explainer,
        reference_ranges=reference_ranges, window_size=10, top_n_reasons=3,
    )
    assert result is not None
    assert set(result.keys()) == {
        "unit_id", "predicted_rul", "out_of_distribution", "ood_violation_ratio", "top_reasons",
    }
    assert len(result["top_reasons"]) == 3
    for reason in result["top_reasons"]:
        assert set(reason.keys()) == {"feature", "shap_impact"}
        assert isinstance(reason["shap_impact"], float)


def test_predict_with_explanation_reasons_sorted_by_absolute_impact(synthetic_history, trained_model_and_context):
    model, explainer, reference_ranges = trained_model_and_context
    result = predict_with_explanation(
        synthetic_history, unit_id=3, model=model, explainer=explainer,
        reference_ranges=reference_ranges, window_size=10, top_n_reasons=5,
    )
    impacts = [abs(r["shap_impact"]) for r in result["top_reasons"]]
    assert impacts == sorted(impacts, reverse=True)


def test_predict_with_explanation_unknown_unit_returns_none(synthetic_history, trained_model_and_context):
    model, explainer, reference_ranges = trained_model_and_context
    result = predict_with_explanation(
        synthetic_history, unit_id=999, model=model, explainer=explainer, reference_ranges=reference_ranges,
    )
    assert result is None