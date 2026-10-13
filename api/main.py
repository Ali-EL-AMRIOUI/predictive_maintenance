"""
FastAPI serving layer for RUL prediction.

Delegates feature engineering and prediction to src/inference.py and
src/evaluation.py — the exact modules the training pipeline itself uses —
instead of reimplementing them here. The previous version rebuilt rolling
"mean"/"std" features by copying the single raw reading into both columns
(no actual rolling computation), and manually min-max-normalized 3 sensors
that the model was never trained on normalized values for. Both silently
fed the model data shaped nothing like its training distribution. Routing
through src/ instead of duplicating logic here is what makes that class of
bug structurally impossible, not just fixed once.
"""
import json
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from evaluation import compute_health_score, get_maintenance_action, get_shap_explainer
from inference import predict_unit_health, predict_with_explanation
from utils import load_config, resolve

from .schemas import EngineInput, PredictionOutput

app = FastAPI(title="NASA Jet Engine RUL Predictor")

config = load_config()
final_model = joblib.load(resolve(config["models"]["final_model_path"]))
model_safety = joblib.load(resolve(config["models"]["safety_model_path"]))
explainer = get_shap_explainer(final_model)

with open(resolve("data/processed/feature_reference_ranges.json")) as f:
    reference_ranges = json.load(f)

WINDOW_SIZE = config["features"]["window_size"]
MAX_RUL = config["data"]["max_rul"]
INSPECT_THRESHOLD = config["health_score"]["inspect_threshold"]
GROUND_THRESHOLD = config["health_score"]["ground_threshold"]
# Previously not read at all here — predict_with_explanation() silently fell
# back to its own hardcoded default (0.1), which only happened to match
# config.yaml's current value by coincidence. Anyone who tuned
# evaluation.ood_max_violation_ratio in config.yaml would see no effect on
# the live API at all, with no error to indicate why.
OOD_MAX_VIOLATION_RATIO = config["evaluation"]["ood_max_violation_ratio"]


@app.get("/health")
def health():
    """Liveness/readiness probe target (wire this into deployment.yaml's
    livenessProbe/readinessProbe once the HPA/K8s pass happens)."""
    return {"status": "ok", "champion_model_loaded": True, "safety_model_loaded": True}


@app.get("/config")
def get_config():
    """Exposes the thresholds config.yaml already loaded into this process,
    so the frontend (or any other consumer) reads the real values instead of
    keeping its own hardcoded copy that silently drifts from config.yaml —
    the same bug class as the health_score fix, just for thresholds instead
    of a computed field."""
    return {
        "window_size": WINDOW_SIZE,
        "max_rul": MAX_RUL,
        "inspect_threshold": INSPECT_THRESHOLD,
        "ground_threshold": GROUND_THRESHOLD,
        "ood_max_violation_ratio": OOD_MAX_VIOLATION_RATIO,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: EngineInput):
    try:
        raw_history = pd.DataFrame([
            {"unit": input_data.unit_id, "cycle": r.cycle, **r.sensors}
            for r in input_data.history
        ])

        explained = predict_with_explanation(
            raw_history, input_data.unit_id, final_model, explainer, reference_ranges,
            window_size=WINDOW_SIZE,
            ood_violation_threshold=OOD_MAX_VIOLATION_RATIO,
        )
        if explained is None:
            raise ValueError(
                f"Could not compute a prediction for unit {input_data.unit_id} — "
                f"check that sensor keys match training column names."
            )

        safety_rul = predict_unit_health(raw_history, input_data.unit_id, model_safety, window_size=WINDOW_SIZE)
        if safety_rul is None:
            raise ValueError(f"Safety model failed for unit {input_data.unit_id}.")

        # If the input is out-of-distribution, the score is forced to 0.0 —
        # worst case, not "unknown" — and the status names the real problem
        # (bad/unusual sensor data), not a generic maintenance action. An
        # operator scanning a fleet table for the lowest health scores must
        # not be able to miss this row by filtering on a null/blank value.
        if explained["out_of_distribution"]:
            health_score = 0.0
            status = "🚨 INVALID SENSOR DATA - INSPECT SENSORS"
        else:
            health_score = float(compute_health_score(pd.Series([safety_rul]), max_rul_cap=MAX_RUL).iloc[0])
            status = get_maintenance_action(health_score, INSPECT_THRESHOLD, GROUND_THRESHOLD)
        return PredictionOutput(
            unit=input_data.unit_id,
            predicted_RUL=round(explained["predicted_rul"], 2),
            safety_RUL=round(safety_rul, 2),
            health_score=round(health_score, 2),
            status=status,
            out_of_distribution=explained["out_of_distribution"],
            ood_violation_ratio=explained["ood_violation_ratio"],
            ood_violating_features=explained["ood_violating_features"],
            top_reasons=explained["top_reasons"],
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)