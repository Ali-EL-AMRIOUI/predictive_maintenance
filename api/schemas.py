from pydantic import BaseModel
from typing import Dict

class EngineInput(BaseModel):
    # L'API attend un dictionnaire avec les noms des capteurs et leurs valeurs
    # Exemple : {"s1": 518.67, "s2": 641.82, ...}
    sensors: Dict[str, float]

class PredictionOutput(BaseModel):
    unit: int
    predicted_RUL: float
    health_score: float
    status: str