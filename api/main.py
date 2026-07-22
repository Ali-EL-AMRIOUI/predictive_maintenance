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
from evaluation import compute_health_score, get_maintenance_action, get_shap_explainer  # noqa: E402
from inference import predict_unit_health, predict_with_explanation  # noqa: E402
from utils import load_config, resolve  # noqa: E402

from schemas import EngineInput, PredictionOutput  # noqa: E402

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


@app.get("/health")
def health():
    """Liveness/readiness probe target (wire this into deployment.yaml's
    livenessProbe/readinessProbe once the HPA/K8s pass happens)."""
    return {"status": "ok", "champion_model_loaded": True, "safety_model_loaded": True}


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
        )
        if explained is None:
            raise ValueError(
                f"Could not compute a prediction for unit {input_data.unit_id} — "
                f"check that sensor keys match training column names."
            )

        safety_rul = predict_unit_health(raw_history, input_data.unit_id, model_safety, window_size=WINDOW_SIZE)
        if safety_rul is None:
            raise ValueError(f"Safety model failed for unit {input_data.unit_id}.")

        health_score = float(compute_health_score(pd.Series([safety_rul]), max_rul_cap=MAX_RUL).iloc[0])
        status = get_maintenance_action(health_score, INSPECT_THRESHOLD, GROUND_THRESHOLD)

        return PredictionOutput(
            unit=input_data.unit_id,
            predicted_RUL=round(explained["predicted_rul"], 2),
            safety_RUL=round(safety_rul, 2),
            health_score=round(health_score, 2),
            status=status,
            out_of_distribution=explained["out_of_distribution"],
            top_reasons=explained["top_reasons"],
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)