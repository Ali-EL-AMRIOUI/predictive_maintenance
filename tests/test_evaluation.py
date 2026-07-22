"""
Unit tests for src/evaluation.py.

Replaces the old tests/test_engine.py: that file tested engine.py, which was
removed when the pipeline was split into data_loader/features/evaluation/
train/inference. There is nothing left for a test_engine.py to test against —
keeping it would mean either an empty file or tests for a module that no
longer exists. This file covers the same evaluation logic instead.

Run: pytest tests/test_evaluation.py -v
"""
import numpy as np
import pandas as pd
import pytest

from evaluation import (
    bootstrap_metric_ci,
    check_out_of_distribution,
    compute_business_impact,
    compute_feature_reference_ranges,
    compute_health_score,
    compute_sliced_metrics,
    get_maintenance_action,
    nasa_score,
    pinball_loss,
)


# ---------------------------------------------------------------------------
# nasa_score
# ---------------------------------------------------------------------------

def test_nasa_score_zero_for_perfect_prediction():
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]
    assert nasa_score(y_true, y_pred) == pytest.approx(0.0)


def test_nasa_score_penalizes_late_prediction_more_than_early():
    """Same |error|, opposite sign -> late (over-)prediction must score higher,
    since it represents the operationally dangerous case (model says 'still
    fine' when the engine is closer to failure)."""
    y_true = [50]
    early = nasa_score(y_true, [40])   # predicted less than truth -> early/safe
    late = nasa_score(y_true, [60])    # predicted more than truth -> late/dangerous
    assert late > early


def test_pinball_loss_zero_for_perfect_prediction():
    assert pinball_loss([10, 20], [10, 20], alpha=0.1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_health_score / get_maintenance_action
# ---------------------------------------------------------------------------

def test_compute_health_score_uses_fixed_cap_not_batch_relative():
    """Same engine (safety_rul=50) must score identically regardless of what
    else is in the batch -- this is the whole point of using a fixed
    max_rul_cap instead of a batch min-max normalization."""
    max_rul_cap = 125
    batch_a = compute_health_score(pd.Series([50, 10]), max_rul_cap)
    batch_b = compute_health_score(pd.Series([50, 125]), max_rul_cap)
    assert batch_a.iloc[0] == pytest.approx(batch_b.iloc[0])


def test_compute_health_score_clipped_to_0_100():
    result = compute_health_score(pd.Series([-10, 500]), max_rul_cap=125)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 100.0


@pytest.mark.parametrize(
    "score,expected",
    [
        (80, "Normal Operation"),
        (50.01, "Normal Operation"),
        (35, "Schedule Inspection"),
        (20.01, "Schedule Inspection"),
        (10, "IMMEDIATE GROUNDING"),
        (0, "IMMEDIATE GROUNDING"),
    ],
)
def test_get_maintenance_action_thresholds(score, expected):
    assert get_maintenance_action(score, inspect_threshold=50, ground_threshold=20) == expected


# ---------------------------------------------------------------------------
# compute_sliced_metrics
# ---------------------------------------------------------------------------

def test_compute_sliced_metrics_perfect_predictions_zero_error():
    y_true = [5, 15, 30, 70, 110]
    result = compute_sliced_metrics(y_true, y_true, bins=((0, 20), (20, 50), (50, 100), (100, None)))
    assert (result["rmse"] == 0).all()
    assert result["n"].sum() == len(y_true)


def test_compute_sliced_metrics_open_ended_last_bin():
    y_true = [150, 200]
    y_pred = [150, 200]
    result = compute_sliced_metrics(y_true, y_pred, bins=((0, 20), (20, 50), (50, 100), (100, None)))
    last_bin = result[result["rul_range"] == "[100, inf)"]
    assert last_bin["n"].iloc[0] == 2


# ---------------------------------------------------------------------------
# compute_business_impact -- covers the fixed "gray zone" boundary
# ---------------------------------------------------------------------------

def test_compute_business_impact_gray_zone_is_counted_as_missed():
    """True RUL between critical_threshold and alert_threshold, with an
    over-confident prediction, must count as a missed failure -- this is the
    exact boundary case a previous version silently left unscored."""
    result = compute_business_impact(
        y_true=[7],       # inside [critical=5, alert=10) -- the gray zone
        y_pred=[15],       # model says "fine", above alert_threshold
        cost_unplanned_failure=50000,
        cost_preventive_maintenance=10000,
        cost_false_positive=2000,
        critical_threshold=5,
        alert_threshold=10,
    )
    assert result["missed_failures"] == 1
    assert result["missed_failures_critical"] == 0  # not in the stricter sub-zone
    assert result["net_savings"] == -50000


def test_compute_business_impact_prevented_failure_has_positive_gain():
    result = compute_business_impact(
        y_true=[3], y_pred=[3],
        cost_unplanned_failure=50000, cost_preventive_maintenance=10000, cost_false_positive=2000,
        critical_threshold=5, alert_threshold=10,
    )
    assert result["prevented_failures"] == 1
    assert result["net_savings"] == pytest.approx(50000 - 10000)


def test_compute_business_impact_false_positive_costs_but_does_not_dominate():
    result = compute_business_impact(
        y_true=[50], y_pred=[3],  # engine was fine, model raised an alert anyway
        cost_unplanned_failure=50000, cost_preventive_maintenance=10000, cost_false_positive=2000,
        critical_threshold=5, alert_threshold=10,
    )
    assert result["false_positives"] == 1
    assert result["net_savings"] == pytest.approx(-2000)


# ---------------------------------------------------------------------------
# bootstrap_metric_ci
# ---------------------------------------------------------------------------

def test_bootstrap_metric_ci_significant_when_one_model_much_better():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 125, size=200)
    y_pred_bad = y_true + rng.normal(0, 40, size=200)   # noisy baseline
    y_pred_good = y_true + rng.normal(0, 2, size=200)   # near-perfect champion
    result = bootstrap_metric_ci(y_true, y_pred_bad, y_pred_good, n_boot=300, seed=0)
    assert result["point_estimate"] > 0          # baseline error - champion error > 0
    assert result["significant"] is True


def test_bootstrap_metric_ci_not_significant_for_identical_predictions():
    rng = np.random.default_rng(1)
    y_true = rng.uniform(0, 125, size=100)
    y_pred = y_true + rng.normal(0, 10, size=100)
    result = bootstrap_metric_ci(y_true, y_pred, y_pred, n_boot=300, seed=1)
    assert result["significant"] is False


# ---------------------------------------------------------------------------
# OOD reference ranges / violation check
# ---------------------------------------------------------------------------

def test_check_out_of_distribution_flags_row_outside_training_range():
    X_train = pd.DataFrame({"sensor_a": np.linspace(0, 10, 100), "sensor_b": np.linspace(100, 110, 100)})
    ranges = compute_feature_reference_ranges(X_train, lower_q=0.01, upper_q=0.99)

    in_range_row = pd.Series({"sensor_a": 5.0, "sensor_b": 105.0})
    ood_row = pd.Series({"sensor_a": 999.0, "sensor_b": 999.0})

    assert check_out_of_distribution(in_range_row, ranges, max_violation_ratio=0.1)["is_ood"] is False
    assert check_out_of_distribution(ood_row, ranges, max_violation_ratio=0.1)["is_ood"] is True


def test_check_out_of_distribution_reports_worst_violations_first():
    X_train = pd.DataFrame({"a": np.linspace(0, 10, 50), "b": np.linspace(0, 10, 50), "c": np.linspace(0, 10, 50)})
    ranges = compute_feature_reference_ranges(X_train)
    # 'a' is wildly out of range, 'c' only slightly -- 'a' must be reported first
    row = pd.Series({"a": 1000.0, "b": 5.0, "c": 10.5})
    result = check_out_of_distribution(row, ranges, max_violation_ratio=0.0)
    assert result["violating_features"][0] == "a"