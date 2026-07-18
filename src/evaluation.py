"""
Evaluation & business-logic utilities shared between:
- notebooks/02_model_training.ipynb (Optuna objective functions)
- notebooks/03_model_evaluation.ipynb (health score, maintenance actions,
  slice analysis, robustness testing, statistical significance, SHAP)
- src/train.py (same objective functions, non-interactive entry point)
- src/inference.py (OOD guard used by predict_with_explanation)

Single source of truth for "what counts as a good prediction" and "what
health score/action a given prediction implies" — every caller imports from
here instead of redefining the logic locally. That local-redefinition pattern
is what caused an earlier 02/03 drift (one notebook computed health_score as a
batch-relative min-max normalization, the other used a fixed scale — same
engine, two different verdicts depending on which notebook you asked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Core scoring
# ----------------------------------------------------------------------------


def nasa_score(y_true, y_pred) -> float:
    """
    Official asymmetric NASA CMAPSS / PHM08 scoring function.

    Penalizes late predictions (predicted RUL > actual RUL, i.e. the model said
    "still fine" when the engine was closer to failure) more heavily than early
    ones, via the asymmetric denominators (10 vs 13).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def pinball_loss(y_true, y_pred, alpha: float = 0.1) -> float:
    """Quantile (pinball) loss, consistent with the quantile_alpha used to train
    the safety model via XGBoost's native reg:quantileerror objective."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


# ----------------------------------------------------------------------------
# Health score & maintenance actions
# ----------------------------------------------------------------------------


def compute_health_score(safety_rul, max_rul_cap: float) -> pd.Series:
    """
    Normalizes the safety RUL against the FIXED piecewise RUL cap used at
    training time (config['data']['max_rul']) — NOT against the min/max of
    whatever batch happens to be in the dataframe.

    A batch-relative min-max score changes for the same engine depending on
    which other engines are in the evaluation set, which makes the score
    meaningless as an absolute safety signal over time. Fixed-scale keeps
    "50% health" meaning the same thing every time it's computed.
    """
    safety_rul = pd.Series(safety_rul)
    return (safety_rul / max_rul_cap * 100).clip(0, 100)


def get_maintenance_action(
    health_score: float,
    inspect_threshold: float = 50,
    ground_threshold: float = 20,
) -> str:
    """Maps a health score to an operational decision. Thresholds come from
    config.yaml so they stay identical everywhere this function is called."""
    if health_score > inspect_threshold:
        return "Normal Operation"
    elif health_score > ground_threshold:
        return "Schedule Inspection"
    else:
        return "IMMEDIATE GROUNDING"


# ----------------------------------------------------------------------------
# Slice analysis — a single global RMSE hides exactly the cases that matter
# most: being off by 10 cycles at RUL=100 is harmless, the same error at
# RUL=5 is dangerous.
# ----------------------------------------------------------------------------


def compute_sliced_metrics(
    y_true,
    y_pred,
    bins=((0, 20), (20, 50), (50, 100), (100, None)),
) -> pd.DataFrame:
    """RMSE/MAE broken down by true-RUL range."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for low, high in bins:
        mask = (y_true >= low) & (y_true < high if high is not None else np.ones_like(y_true, dtype=bool))
        if mask.sum() == 0:
            continue
        label = f"[{low}, {high if high is not None else 'inf'})"
        err = y_true[mask] - y_pred[mask]
        rows.append({
            "rul_range": label,
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mean_bias": float(np.mean(y_pred[mask] - y_true[mask])),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Robustness / stress testing — evaluation on clean held-out data says nothing
# about behavior when a sensor fails or drifts, which is the normal case in
# production, not the exception.
# ----------------------------------------------------------------------------


def inject_sensor_dropout(X: pd.DataFrame, fraction: float = 0.1, seed: int | None = None) -> pd.DataFrame:
    """Simulates sensor failure: zeroes out `fraction` of (row, raw-sensor-column)
    cells. Only raw sensor columns are touched, not the engineered roll_mean/
    roll_std columns — a real sensor fault wouldn't directly overwrite those,
    they'd just reflect the corrupted upstream readings on the next real run."""
    rng = np.random.default_rng(seed)
    X_corrupted = X.copy()
    sensor_cols = [c for c in X.columns if "roll_" not in c]
    if not sensor_cols:
        return X_corrupted
    n_cells = int(len(X) * len(sensor_cols) * fraction)
    row_idx = rng.integers(0, len(X), n_cells)
    col_idx = rng.choice(sensor_cols, n_cells)
    for r, c in zip(row_idx, col_idx):
        X_corrupted.iloc[r, X_corrupted.columns.get_loc(c)] = 0.0
    return X_corrupted


def inject_gaussian_noise(X: pd.DataFrame, sigma_pct: float = 0.05, seed: int | None = None) -> pd.DataFrame:
    """Simulates sensor drift / measurement noise: adds N(0, (sigma_pct * std)^2)
    noise to every numeric column."""
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    for col in X_noisy.columns:
        noise_std = X_noisy[col].std() * sigma_pct
        X_noisy[col] = X_noisy[col] + rng.normal(0, noise_std if noise_std > 0 else 1e-6, size=len(X_noisy))
    return X_noisy


def evaluate_under_perturbation(model, X_test: pd.DataFrame, y_test, perturbations: dict) -> pd.DataFrame:
    """
    perturbations: {"scenario_name": perturbed_X_dataframe, ...}
    Returns RMSE for the clean baseline plus each perturbation, so a reviewer
    can see exactly how much performance degrades under sensor dropout / noise
    instead of just asserting the model is "robust".
    """
    y_test = np.asarray(y_test, dtype=float)
    rows = [{
        "scenario": "clean",
        "rmse": float(np.sqrt(np.mean((y_test - model.predict(X_test)) ** 2))),
    }]
    for name, X_perturbed in perturbations.items():
        preds = model.predict(X_perturbed)
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        rows.append({"scenario": name, "rmse": rmse})
    result = pd.DataFrame(rows)
    result["rmse_degradation_pct"] = (result["rmse"] / result["rmse"].iloc[0] - 1) * 100
    return result


# ----------------------------------------------------------------------------
# Statistical significance — is the champion actually better than baseline,
# or within noise? A single point-estimate RMSE comparison can't answer that.
# ----------------------------------------------------------------------------


def bootstrap_metric_ci(
    y_true,
    y_pred_a,
    y_pred_b,
    metric_fn=None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap on (metric(y_true, y_pred_a) - metric(y_true, y_pred_b)),
    e.g. RMSE_baseline - RMSE_champion, to get a confidence interval on the
    improvement rather than a single point estimate.

    If the CI excludes 0, the improvement is unlikely to be an artifact of
    which engines happened to land in the test split. This is a distribution-
    free, assumption-light stand-in for a formal significance test — chosen
    over e.g. a paired t-test because prediction errors here have no reason to
    be normally distributed.
    """
    if metric_fn is None:
        def metric_fn(yt, yp):
            return float(np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2)))

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    n = len(y_true)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])

    alpha = (1 - ci) / 2
    return {
        "point_estimate": float(metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1 - alpha)),
        "ci_level": ci,
        "significant": bool(np.quantile(diffs, alpha) > 0 or np.quantile(diffs, 1 - alpha) < 0),
    }


# ----------------------------------------------------------------------------
# Out-of-distribution guard — a model will always return *a* number, even for
# input nothing like what it was trained on. This flags that case instead of
# silently returning a confident-looking but meaningless prediction.
# ----------------------------------------------------------------------------


def compute_feature_reference_ranges(X_train: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> dict:
    """
    Per-feature [lower_q, upper_q] percentile ranges from training data,
    persisted at training time and loaded at inference time as a simple,
    interpretable OOD guard.

    A percentile-range check is intentionally simple and explainable — "3 of
    your 18 features are outside their training range" is something an
    operator can act on. A Mahalanobis distance or IsolationForest would give
    a more statistically principled multivariate boundary; a reasonable next
    iteration, not required to demonstrate the concept.
    """
    return {
        col: {"low": float(X_train[col].quantile(lower_q)), "high": float(X_train[col].quantile(upper_q))}
        for col in X_train.columns
    }


def check_out_of_distribution(features_row: pd.Series, reference_ranges: dict, max_violation_ratio: float = 0.1) -> dict:
    """Flags a single feature row as OOD if more than max_violation_ratio of its
    features fall outside the persisted training reference ranges."""
    violations = []
    for col, bounds in reference_ranges.items():
        if col not in features_row.index:
            continue
        val = features_row[col]
        if val < bounds["low"] or val > bounds["high"]:
            violations.append(col)

    violation_ratio = len(violations) / len(reference_ranges) if reference_ranges else 0.0
    return {
        "is_ood": violation_ratio > max_violation_ratio,
        "violation_ratio": round(violation_ratio, 3),
        "violating_features": violations[:5],
    }


# ----------------------------------------------------------------------------
# SHAP — computation lives here (imported by both notebooks and inference.py),
# plotting stays in the notebook where it's actually looked at.
# ----------------------------------------------------------------------------


def get_shap_explainer(model):
    """TreeExplainer setup, factored out so it's built once (e.g. at API
    startup) rather than re-created on every inference call. Imports shap
    lazily so this module has no hard dependency on it for callers who only
    need nasa_score / health score / slice metrics."""
    import shap
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """Returns a (n_samples, n_features) array, normalizing away the
    list-of-arrays shape some shap versions return for single-output models."""
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]
    return values
