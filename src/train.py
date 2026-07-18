"""
CLI-runnable training pipeline — this is what dvc.yaml's `train` stage calls
(`python src/train.py`). Mirrors the orchestration in
notebooks/02_model_training.ipynb; the notebook stays useful for interactive
exploration, this is what runs non-interactively (CI, DVC, cron retraining).

The scoring/feature logic itself (nasa_score, add_temporal_features, ...) is
NOT duplicated here — this script and the notebook both import it from
data_loader / features / evaluation, so there is exactly one place each of
those can drift, instead of two.

Usage:
    python src/train.py
    python src/train.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from data_loader import clean_and_save_data
from evaluation import compute_feature_reference_ranges, get_shap_explainer, nasa_score, pinball_loss
from features import add_temporal_features, get_sensor_columns
from inference import predict_unit_health, predict_with_explanation
from utils import load_config, resolve, set_seed


@dataclass
class FeatureSet:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_data_eng: pd.DataFrame
    test_data_eng: pd.DataFrame


def load_and_split(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_and_save_data(
        input_path=str(resolve(config["data"]["raw_train_path"])),
        output_path=str(resolve(config["data"]["cleaned_train_path"])),
        max_rul=config["data"]["max_rul"],
    )
    df = pd.read_csv(resolve(config["data"]["cleaned_train_path"]))

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
    train_idx, test_idx = next(gss.split(df, groups=df["unit"]))
    train_data = df.iloc[train_idx].reset_index(drop=True).copy()
    test_data = df.iloc[test_idx].reset_index(drop=True).copy()

    print(f"[split] {train_data['unit'].nunique()} train engines / {test_data['unit'].nunique()} test engines")
    return train_data, test_data


def engineer_features(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict) -> FeatureSet:
    sensor_cols = get_sensor_columns(train_data, exclude=["unit", "cycle", "RUL"])
    window_size = config["features"]["window_size"]

    train_data_eng = add_temporal_features(train_data, sensor_cols, window_size=window_size)
    test_data_eng = add_temporal_features(test_data, sensor_cols, window_size=window_size)

    X_train = train_data_eng.drop(["unit", "RUL"], axis=1).drop(columns=["cycle"], errors="ignore")
    X_test = test_data_eng.drop(["unit", "RUL"], axis=1).drop(columns=["cycle"], errors="ignore")
    y_train = train_data_eng["RUL"]
    y_test = test_data_eng["RUL"]

    print(f"[features] {X_train.shape[1]} columns (from {len(sensor_cols)} raw sensors)")
    return FeatureSet(X_train, y_train, X_test, y_test, train_data_eng, test_data_eng)


def train_baseline(features: FeatureSet, config: dict) -> tuple[dict, np.ndarray]:
    baseline = LinearRegression()
    baseline.fit(features.X_train, features.y_train)
    y_baseline = baseline.predict(features.X_test)

    metrics = {
        "rmse_baseline": float(np.sqrt(mean_squared_error(features.y_test, y_baseline))),
        "mae_baseline": float(mean_absolute_error(features.y_test, y_baseline)),
        "nasa_baseline": float(nasa_score(features.y_test, y_baseline)),
    }
    print(f"[baseline] RMSE={metrics['rmse_baseline']:.2f}  NASA={metrics['nasa_baseline']:.0f}")

    with open(resolve("data/processed/baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, y_baseline


def make_objective(X_train, y_train, groups, config, extra_params=None, score_fn=nasa_score):
    """Factory shared by the champion (NASA score) and safety (pinball loss)
    hyperparameter searches — same nested-CV structure (outer GroupKFold +
    inner GroupShuffleSplit for early stopping, so the early-stopping
    validation set never leaks into the CV score), different objective/scorer."""
    extra_params = extra_params or {}
    n_estimators_cap = config["optuna"]["n_estimators_cap"]
    early_stopping = config["optuna"]["early_stopping_rounds"]
    cv_folds = config["optuna"]["cv_folds"]
    random_state = config["split"]["random_state"]

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "n_jobs": -1,
            "random_state": random_state,
            "tree_method": "hist",
            **extra_params,
        }

        gkf = GroupKFold(n_splits=cv_folds)
        fold_scores = []

        for tr_i, score_i in gkf.split(X_train, y_train, groups=groups):
            groups_tr = groups.iloc[tr_i]
            sub_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            sub_tr_i, sub_val_i = next(sub_gss.split(X_train.iloc[tr_i], groups=groups_tr))

            X_sub_tr = X_train.iloc[tr_i].iloc[sub_tr_i]
            y_sub_tr = y_train.iloc[tr_i].iloc[sub_tr_i]
            X_sub_val = X_train.iloc[tr_i].iloc[sub_val_i]
            y_sub_val = y_train.iloc[tr_i].iloc[sub_val_i]

            model = xgb.XGBRegressor(**params, n_estimators=n_estimators_cap, early_stopping_rounds=early_stopping)
            model.fit(X_sub_tr, y_sub_tr, eval_set=[(X_sub_val, y_sub_val)], verbose=False)

            preds = model.predict(X_train.iloc[score_i])
            fold_scores.append(score_fn(y_train.iloc[score_i].values, preds))

        return np.mean(fold_scores)

    return objective


def train_champion_model(features: FeatureSet, train_data: pd.DataFrame, config: dict):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(features.X_train, features.y_train, train_data["unit"], config),
        n_trials=config["optuna"]["n_trials"],
    )
    print(f"[optuna:champion] best CV NASA score = {study.best_value:.0f}")

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config["split"]["random_state"])
    tr_idx, val_idx = next(gss_val.split(features.X_train, features.y_train, groups=train_data["unit"]))
    X_tr, X_val = features.X_train.iloc[tr_idx], features.X_train.iloc[val_idx]
    y_tr, y_val = features.y_train.iloc[tr_idx], features.y_train.iloc[val_idx]

    final_model = xgb.XGBRegressor(
        **study.best_params,
        n_estimators=config["optuna"]["n_estimators_cap"],
        random_state=config["split"]["random_state"],
        early_stopping_rounds=config["optuna"]["early_stopping_rounds"],
    )
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    return final_model, study, X_tr, X_val, y_tr, y_val


def robust_cv_check(features: FeatureSet, train_data: pd.DataFrame, best_params: dict, config: dict) -> tuple[float, float]:
    gkf = GroupKFold(n_splits=config["optuna"]["cv_folds"])
    cv_rmse_scores = []

    for tr_idx_cv, val_idx_cv in gkf.split(features.X_train, features.y_train, groups=train_data["unit"]):
        X_tr_fold = features.X_train.iloc[tr_idx_cv]
        X_val_fold = features.X_train.iloc[val_idx_cv]
        y_tr_fold = features.y_train.iloc[tr_idx_cv]
        y_val_fold = features.y_train.iloc[val_idx_cv]

        model_cv = xgb.XGBRegressor(
            **best_params,
            n_estimators=config["optuna"]["n_estimators_cap"],
            random_state=config["split"]["random_state"],
            early_stopping_rounds=config["optuna"]["early_stopping_rounds"],
        )
        model_cv.fit(X_tr_fold, y_tr_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        fold_preds = model_cv.predict(X_val_fold)
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, fold_preds)))

    mean_cv, std_cv = float(np.mean(cv_rmse_scores)), float(np.std(cv_rmse_scores))
    status = "STABLE" if std_cv < 5 else "HIGH VARIANCE — investigate engine outliers"
    print(f"[robust-cv] mean RMSE={mean_cv:.2f}  std={std_cv:.2f}  ({status})")
    return mean_cv, std_cv


def train_safety_model(features: FeatureSet, train_data: pd.DataFrame, X_tr, y_tr, X_val, y_val, config: dict):
    alpha = config["safety_model"]["quantile_alpha"]
    study_safety = optuna.create_study(direction="minimize")
    study_safety.optimize(
        make_objective(
            features.X_train, features.y_train, train_data["unit"], config,
            extra_params={"objective": "reg:quantileerror", "quantile_alpha": alpha},
            score_fn=lambda yt, yp: pinball_loss(yt, yp, alpha=alpha),
        ),
        n_trials=config["optuna"]["n_trials"],
    )
    print(f"[optuna:safety] best CV pinball loss = {study_safety.best_value:.3f}")

    model_safety = xgb.XGBRegressor(
        **study_safety.best_params,
        n_estimators=config["optuna"]["n_estimators_cap"],
        random_state=config["split"]["random_state"],
        early_stopping_rounds=config["optuna"]["early_stopping_rounds"],
    )
    model_safety.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    safe_limit = np.maximum(model_safety.predict(features.X_test), 0)
    return model_safety, study_safety, safe_limit


def export_artifacts(features: FeatureSet, y_pred, y_baseline, safe_limit, run_info: dict, config: dict) -> None:
    results_export = features.test_data_eng[["unit", "cycle"]].reset_index(drop=True).copy()
    results_export["RUL"] = features.y_test.reset_index(drop=True).values
    results_export["predicted_RUL"] = y_pred
    results_export["baseline_predicted_RUL"] = y_baseline
    results_export["safety_RUL"] = safe_limit
    results_export["absolute_error"] = (results_export["RUL"] - results_export["predicted_RUL"]).abs()
    results_export.to_csv(resolve("data/processed/model_results_final.csv"), index=False)

    features_export = features.test_data_eng.drop(columns=["RUL"]).reset_index(drop=True).copy()
    features_export.to_csv(resolve("data/processed/test_features_engineered.csv"), index=False)

    reference_ranges = compute_feature_reference_ranges(
        features.X_train,
        lower_q=config["evaluation"]["ood_lower_quantile"],
        upper_q=config["evaluation"]["ood_upper_quantile"],
    )
    with open(resolve("data/processed/feature_reference_ranges.json"), "w") as f:
        json.dump(reference_ranges, f, indent=2)

    with open(resolve("data/processed/run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"[export] model_results_final.csv {results_export.shape} | "
          f"test_features_engineered.csv {features_export.shape} | "
          f"feature_reference_ranges.json ({len(reference_ranges)} features) | run_info.json")


def run_sanity_check(test_data: pd.DataFrame, final_model, features: FeatureSet, config: dict) -> None:
    test_id = test_data["unit"].iloc[0]
    prediction = predict_unit_health(test_data, test_id, final_model, window_size=config["features"]["window_size"])
    status = f"{prediction:.2f} cycles" if prediction is not None else "FAILED"
    print(f"[sanity-check] unit {test_id}: {status}")

    explainer = get_shap_explainer(final_model)
    reference_ranges = compute_feature_reference_ranges(
        features.X_train,
        lower_q=config["evaluation"]["ood_lower_quantile"],
        upper_q=config["evaluation"]["ood_upper_quantile"],
    )
    explained = predict_with_explanation(
        test_data, test_id, final_model, explainer, reference_ranges,
        window_size=config["features"]["window_size"],
    )
    if explained is not None:
        top = ", ".join(f"{r['feature']}({r['shap_impact']:+.2f})" for r in explained["top_reasons"])
        print(f"[sanity-check:explained] unit {test_id}: {explained['predicted_rul']:.2f} cycles | "
              f"OOD={explained['out_of_distribution']} | top drivers: {top}")


def main():
    parser = argparse.ArgumentParser(description="Train the predictive maintenance pipeline end to end.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: configs/config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["split"]["random_state"])

    mlflow.set_tracking_uri(f"sqlite:///{resolve(config['mlflow']['tracking_db'])}")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    train_data, test_data = load_and_split(config)
    features = engineer_features(train_data, test_data, config)
    baseline_metrics, y_baseline = train_baseline(features, config)

    final_model, study, X_tr, X_val, y_tr, y_val = train_champion_model(features, train_data, config)
    y_pred = final_model.predict(features.X_test)
    rmse_final = float(np.sqrt(mean_squared_error(features.y_test, y_pred)))
    r2_final = float(r2_score(features.y_test, y_pred))
    nasa_final = nasa_score(features.y_test, y_pred)
    print(f"[champion] RMSE={rmse_final:.2f}  R2={r2_final:.4f}  NASA={nasa_final:.0f}")

    joblib.dump(final_model, resolve(config["models"]["final_model_path"]))

    # run_id is captured so 03_model_evaluation.ipynb can resume logging into
    # THIS exact run later (mlflow.start_run(run_id=...)) instead of creating a
    # disconnected "evaluation" run — evaluation artifacts stay attached to the
    # model version that produced them.
    with mlflow.start_run(run_name="xgboost_final_model") as run:
        champion_run_id = run.info.run_id
        mlflow.log_params(study.best_params)
        mlflow.log_metric("cv_nasa_score", study.best_value)
        mlflow.log_metric("test_rmse", rmse_final)
        mlflow.log_metric("test_r2", r2_final)
        mlflow.log_metric("test_nasa_score", nasa_final)
        mlflow.log_metric("baseline_rmse", baseline_metrics["rmse_baseline"])
        mlflow.xgboost.log_model(final_model, name="model")

    robust_cv_check(features, train_data, study.best_params, config)

    model_safety, study_safety, safe_limit = train_safety_model(features, train_data, X_tr, y_tr, X_val, y_val, config)
    joblib.dump(model_safety, resolve(config["models"]["safety_model_path"]))

    with mlflow.start_run(run_name="xgboost_safety_model") as run:
        safety_run_id = run.info.run_id
        mlflow.log_params(study_safety.best_params)
        mlflow.log_metric("cv_pinball_loss", study_safety.best_value)
        mlflow.log_metric("mean_safety_rul", float(np.mean(safe_limit)))
        mlflow.xgboost.log_model(model_safety, name="model")

    export_artifacts(
        features, y_pred, y_baseline, safe_limit,
        run_info={"champion_run_id": champion_run_id, "safety_run_id": safety_run_id},
        config=config,
    )
    run_sanity_check(test_data, final_model, features, config)
    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
