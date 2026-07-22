"""
Request/response contracts for the RUL prediction API.

Key change from the previous version: EngineInput now carries a short HISTORY
of recent cycles, not a single reading. Rolling-window features (mean/std
over the last `window_size` cycles) are mathematically undefined from one
reading — the API needs real history to reproduce the exact features the
model was trained on, the same way notebooks/02 and src/train.py do.

Sensor key names (e.g. "s2"/"os1" vs "sensor_2"/"op_setting_1") are NOT
hardcoded here on purpose — they must match whatever column names your
actual src/data_loader.py + src/features.py produce at training time. This
API passes them straight through to src/inference.py, which reproduces the
training feature pipeline exactly regardless of what those names are.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CycleReading(BaseModel):
    """One cycle of raw sensor/operational-setting readings for one engine."""
    cycle: int
    sensors: Dict[str, float]  # e.g. {"s2": 642.1, "s3": 1589.7, "os1": 0.001, ...}


class EngineInput(BaseModel):
    unit_id: int
    history: List[CycleReading] = Field(
        ..., min_length=1,
        description="Recent cycles for this engine, oldest first. Send at "
                     "least `window_size` cycles (see config.yaml) for the "
                     "rolling features to be fully meaningful — fewer still "
                     "works (min_periods=1, same as training) but the "
                     "rolling stats will be based on less history than the "
                     "model saw during training."
    )


class ShapReason(BaseModel):
    feature: str
    shap_impact: float  # negative shortens predicted RUL, positive extends it


class PredictionOutput(BaseModel):
    unit: int
    predicted_RUL: float          # champion model's point estimate
    safety_RUL: float             # conservative quantile-model bound — use THIS for grounding decisions, not predicted_RUL
    health_score: float           # safety_RUL normalized to the training RUL cap, 0-100
    status: str                   # "Normal Operation" / "Schedule Inspection" / "IMMEDIATE GROUNDING"
    out_of_distribution: bool     # True if input looks unlike anything in training data
    top_reasons: Optional[List[ShapReason]] = None