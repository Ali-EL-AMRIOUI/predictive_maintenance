from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="NASA Jet Engine RUL Predictor")

# --- CHARGEMENT DU MODÈLE ---
MODEL_PATH = "models/xgboost_model_safety.pkl"
model = joblib.load(MODEL_PATH)

# --- CONFIGURATION DE LA NORMALISATION (MIN-MAX) ---
# Ces valeurs permettent de transformer tes données réelles en 0-1
SCALING_PARAMS = {
    's2':  {'min': 641.21, 'max': 643.52},
    's4':  {'min': 1394.8, 'max': 1423.2},
    's11': {'min': 47.27,  'max': 48.13}
}

EXPECTED_FEATURES = [
    'cycle', 'os1', 'os2', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 
    's12', 's13', 's14', 's15', 's16', 's17', 's20', 's21', 
    'os1_roll_mean', 'os1_roll_std', 'os2_roll_mean', 'os2_roll_std', 
    's2_roll_mean', 's2_roll_std', 's3_roll_mean', 's3_roll_std', 
    's4_roll_mean', 's4_roll_std', 's5_roll_mean', 's5_roll_std', 
    's6_roll_mean', 's6_roll_std', 's7_roll_mean', 's7_roll_std', 
    's8_roll_mean', 's8_roll_std', 's9_roll_mean', 's9_roll_std', 
    's11_roll_mean', 's11_roll_std', 's12_roll_mean', 's12_roll_std', 
    's13_roll_mean', 's13_roll_std', 's14_roll_mean', 's14_roll_std', 
    's15_roll_mean', 's15_roll_std', 's16_roll_mean', 's16_roll_std', 
    's17_roll_mean', 's17_roll_std', 's20_roll_mean', 's20_roll_std', 
    's21_roll_mean', 's21_roll_std'
]

class SensorInput(BaseModel):
    sensor_data: dict

@app.post("/predict")
def predict(input_data: SensorInput):
    try:
        data = input_data.sensor_data
        
        # 1. NORMALISATION MANUELLE (Min-Max Scaling)
        # On ne normalise PAS le cycle, seulement les capteurs physiques
        for s in ['s2', 's4', 's11']:
            if s in data:
                val = data[s]
                s_min = SCALING_PARAMS[s]['min']
                s_max = SCALING_PARAMS[s]['max']
                # Formule : (x - min) / (max - min)
                data[s] = (val - s_min) / (s_max - s_min)
        
        # 2. PRÉPARATION DU DATAFRAME (61 colonnes)
        final_df = pd.DataFrame(columns=EXPECTED_FEATURES)
        row = {}
        for col in EXPECTED_FEATURES:
            base_name = col.split('_')[0]
            # On remplit avec la valeur normalisée si dispo, sinon 0.0
            row[col] = data.get(base_name, 0.0)
            
        # Le cycle reste brut (important !)
        row['cycle'] = data.get('cycle', 1)
        
        final_df = pd.DataFrame([row])[EXPECTED_FEATURES]

        # 3. PRÉDICTION
        prediction = model.predict(final_df)[0]
        return {"predicted_RUL": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)