# API

> This documents the actual contract implemented in `api/main.py` /
> `api/schemas.py` as of the production-readiness pass (2026-10). The
> previous version of this file was a design sketch written before
> `main.py` existed and no longer matched it — wrong endpoints, wrong
> response shapes, wrong `/health` payload. This replaces it.

## Design principle

`api/main.py` contains **no prediction or feature-engineering logic of its
own** — it's a thin FastAPI wrapper around `src/inference.py` and
`src/evaluation.py`, the exact modules the training pipeline itself uses.
An earlier version of this file reimplemented rolling features and sensor
normalization by hand inside the API and got both wrong (see git history /
`CHANGELOG`); routing everything through `src/` is what makes that class of
train/serve skew bug structurally impossible, not just fixed once.

Thresholds (`window_size`, `max_rul`, `inspect_threshold`, `ground_threshold`,
`ood_max_violation_ratio`) are loaded from `configs/config.yaml` at startup —
the same file `src/train.py` reads — and never hardcoded in `main.py` or in
the Streamlit frontend. See `GET /config` below.

## Endpoints

### `GET /health`

Liveness/readiness probe.

```json
{"status": "ok", "champion_model_loaded": true, "safety_model_loaded": true}
```

### `GET /config`

Exposes the live threshold values this process loaded from `config.yaml`,
so any consumer (the Streamlit frontend, a future dashboard, a smoke test)
reads the real values instead of keeping its own hardcoded copy.

```json
{
  "window_size": 10,
  "max_rul": 125,
  "inspect_threshold": 50,
  "ground_threshold": 20,
  "ood_max_violation_ratio": 0.1
}
```

### `POST /predict`

The only prediction endpoint — there is no separate `/predict/explain`.
Every response includes the champion point estimate, the conservative
safety bound, the out-of-distribution flag, and the top SHAP drivers in one
call, computed via `inference.predict_with_explanation` (champion model) and
`inference.predict_unit_health` (safety/quantile model).

**Request** (`EngineInput`, see `api/schemas.py`)
```json
{
  "unit_id": 47,
  "history": [
    {"cycle": 1, "sensors": {"s2": 642.1, "s3": 1589.7, "os1": 0.001}},
    {"cycle": 2, "sensors": {"s2": 642.4, "s3": 1590.1, "os1": 0.001}}
  ]
}
```
`history` is raw (pre-feature-engineering) sensor readings, oldest cycle
first — the API recomputes rolling features itself via `src/features.py`,
the same code path training uses. Sensor key names in `sensors` must match
whatever column names your `src/data_loader.py` / `src/features.py`
actually produce at training time (not hardcoded in the schema on purpose).
Send at least `window_size` cycles (10, see `GET /config`) for the rolling
features to be fully meaningful; fewer still works (`min_periods=1`, same
as training) but is based on less history than the model saw at training
time.

**Response** (`PredictionOutput`)
```json
{
  "unit": 47,
  "predicted_RUL": 38.42,
  "safety_RUL": 35.10,
  "health_score": 70.08,
  "status": "Normal Operation",
  "out_of_distribution": false,
  "ood_violation_ratio": 0.0,
  "ood_violating_features": [],
  "top_reasons": [
    {"feature": "s11_roll_mean", "shap_impact": -4.1},
    {"feature": "s4_roll_std", "shap_impact": -2.3},
    {"feature": "s9_roll_mean", "shap_impact": 1.2}
  ]
}
```

Field notes:
- `predicted_RUL` — champion model's point estimate. Informational; **not**
  the number to threshold grounding decisions on.
- `safety_RUL` — the conservative lower-quantile bound. Use this one for
  maintenance/grounding decisions.
- `health_score` — `safety_RUL` normalized to the training RUL cap (0–100).
  **Forced to `0.0` whenever `out_of_distribution` is `true`** — worst-case,
  not a real computed value, and not `null`. The input looks unlike
  anything the models were trained on, so the actual number is not
  reliable; `0.0` (rather than `null`) keeps the row visible to anything
  that sorts or filters a fleet table on `health_score` instead of letting
  it silently disappear. Always check `out_of_distribution` before treating
  a low `health_score` as a genuine maintenance signal — `0.0` here means
  "inspect the sensors", not "this engine is about to fail".
- `status` — `"Normal Operation"` / `"Schedule Inspection"` /
  `"IMMEDIATE GROUNDING"` normally; **`"🚨 INVALID SENSOR DATA - INSPECT
  SENSORS"`** whenever `out_of_distribution` is `true`, replacing the
  threshold-based status entirely (not a prefix on it) since a
  maintenance-action label computed from an unreliable `health_score`
  would itself be unreliable.
- `out_of_distribution` / `ood_violation_ratio` / `ood_violating_features` —
  from a percentile-range check (`evaluation.compute_feature_reference_ranges`
  / `check_out_of_distribution`) against the training data's 1st/99th
  percentile per feature. A row is flagged OOD when more than
  `ood_max_violation_ratio` (default 0.1, from `config.yaml`, see
  `GET /config`) of features fall outside that range.
- `top_reasons` — local SHAP drivers for this specific prediction. Negative
  `shap_impact` shortens predicted RUL (more urgent); positive extends it.
  `null` is possible if the explainer failed for this input — guard for it.

**Errors:** `422` if a prediction can't be computed for `unit_id` (e.g. no
matching rows in `history`, or sensor keys that don't match training column
names) or if the safety model fails for that unit; `500` for any other
unexpected internal error.

## Known limitations

- `inference.predict_with_explanation` / `predict_unit_health` currently
  collapse "unit not found in history" and "unexpected internal error"
  into the same `None` return, which this layer maps to a single `422` for
  both. Distinguishing them cleanly would mean `inference.py` raising a
  specific exception (e.g. `UnitNotFoundError`) instead of swallowing both
  cases into `None`.
- The Kubernetes manifest (`deployment.yaml`) is out of scope for this
  pass; see `docs/deployment.md`.