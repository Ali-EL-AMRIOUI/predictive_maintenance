# API

> **Note on scope:** this documents the contract `api/main.py` should expose,
> derived from `src/inference.py`'s two entry points. It wasn't generated
> from the actual current contents of `api/main.py` / `api/schemas.py` (those
> weren't available when this doc was written) — reconcile the routes below
> against what's actually implemented there before treating this as a source
> of truth, or regenerate this doc once `main.py` is finalized.

## Design principle

`api/main.py` should contain **no prediction logic of its own** — it's a thin
FastAPI wrapper around `src/inference.py`. The reason both entry points
recompute features internally (`_get_latest_features` calls
`features.add_temporal_features`, the same function `train.py` uses) is
specifically so the API never needs its own feature-engineering code path —
one less place train/serve skew could creep in.

## Endpoints

### `GET /health`

Liveness check.

```json
{"status": "ok", "model_version": "<champion_run_id from run_info.json>"}
```

### `POST /predict`

Thin wrapper around `inference.predict_unit_health`. Cheap — no SHAP
overhead, use this when only the number is needed.

**Request**
```json
{
  "unit_id": 47,
  "history": [
    {"unit": 47, "cycle": 1, "sensor_2": 642.1, "sensor_3": 1589.7, "...": "..."},
    {"unit": 47, "cycle": 2, "sensor_2": 642.4, "sensor_3": 1590.1, "...": "..."}
  ]
}
```
`history` is raw (pre-feature-engineering) sensor readings — the same shape
`data_loader.load_raw_cmapss` produces. The API recomputes rolling features
itself; it does not accept pre-engineered input.

**Response**
```json
{"unit_id": 47, "predicted_rul": 38.4}
```

**Errors:** `404` if `unit_id` isn't present in `history`, `500` for any
other failure during feature computation or inference (see note on error
handling below).

### `POST /predict/explain`

Wraps `inference.predict_with_explanation`. Use when the caller needs to act
on the number, not just display it — includes an out-of-distribution flag
and the top local SHAP drivers.

**Request:** same shape as `/predict`, plus optional:
```json
{"unit_id": 47, "history": [...], "top_n_reasons": 3}
```

**Response**
```json
{
  "unit_id": 47,
  "predicted_rul": 15.2,
  "out_of_distribution": false,
  "ood_violation_ratio": 0.0,
  "top_reasons": [
    {"feature": "sensor_11_roll_mean", "shap_impact": -4.1},
    {"feature": "sensor_4_roll_std", "shap_impact": -2.3},
    {"feature": "sensor_9_roll_mean", "shap_impact": 1.2}
  ]
}
```
A negative `shap_impact` pushes the prediction down (shorter RUL, more
urgent); positive pushes it up.

## Reference implementation sketch

```python
# api/main.py
from fastapi import FastAPI, HTTPException
import joblib
import json
import pandas as pd

from evaluation import get_shap_explainer
from inference import predict_unit_health, predict_with_explanation
from utils import load_config, resolve

app = FastAPI(title="Predictive Maintenance API")
config = load_config()

model = joblib.load(resolve(config["models"]["final_model_path"]))
explainer = get_shap_explainer(model)
with open(resolve("data/processed/feature_reference_ranges.json")) as f:
    reference_ranges = json.load(f)
with open(resolve("data/processed/run_info.json")) as f:
    run_info = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": run_info["champion_run_id"]}


@app.post("/predict")
def predict(payload: PredictRequest):  # see api/schemas.py
    history = pd.DataFrame(payload.history)
    result = predict_unit_health(history, payload.unit_id, model, config["features"]["window_size"])
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unit {payload.unit_id} not found in history")
    return {"unit_id": payload.unit_id, "predicted_rul": result}


@app.post("/predict/explain")
def predict_explain(payload: PredictExplainRequest):
    history = pd.DataFrame(payload.history)
    result = predict_with_explanation(
        history, payload.unit_id, model, explainer, reference_ranges,
        window_size=config["features"]["window_size"],
        top_n_reasons=payload.top_n_reasons,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unit {payload.unit_id} not found in history")
    return result
```

## A known gap worth fixing before this is "real" production code

Both `inference.predict_unit_health` and `predict_with_explanation` currently
catch all exceptions internally and return `None` — which collapses "unit
not found" (expected, should be a clean `404`) and "unexpected internal
error" (should be logged/alerted, arguably a `500`) into the same signal.
The sketch above re-raises `404` for the "not found" case based on a `None`
return, but genuinely can't distinguish it from an internal error with the
current `inference.py` contract. Fixing this properly means having
`inference.py` raise a specific exception type (e.g. `UnitNotFoundError`) for
the "not found" case instead of swallowing everything into `None`.