# Predictive Maintenance System for Turbine Engines

![Predictive Maintenance](https://img.shields.io/badge/Predictive-Maintenance-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
This project implements a **predictive maintenance system** for turbine engines using sensor data to predict **Remaining Useful Life (RUL)**. The system combines:
- **Machine Learning** (XGBoost, Quantile Regression)
- **Advanced Feature Engineering** (rolling features, normalization)
- **Model Interpretability** (SHAP analysis, correlations)
- **FastAPI** for real-time predictions
- **Streamlit Dashboard** for visualization

## Project Structure
predictive_maintenance/

├── api/                  # FastAPI for real-time predictions

├── app/                  # Streamlit application (dashboard)

├── data/

│   ├── raw/              # Raw data (CMAPSS dataset)

│   └── processed/        # Cleaned data with engineered features

├── models/               # Saved models (XGBoost)

├── notebooks/            # EDA, training, evaluation

├── reports/              # Reports and visualizations

├── src/                  # Modular source code

├── tests/                # Unit tests

├── Dockerfile            # Containerization

├── requirements.txt      # Python dependencies

└── README.md             # This file

## Key Features
| Feature                     | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **Feature Engineering**     | Added **rolling mean/std** to capture sensor degradation patterns         |
| **Optimized XGBoost Model** | RMSE = **29.99 cycles** (vs 32.21 for baseline)                            |
| **Safety Model**            | Predicts conservative lower bound (Quantile Regression, α=0.1)           |
| **SHAP Analysis**           | Identifies critical sensors (e.g., `s15_roll_mean`, `s3_roll_mean`)       |
| **FastAPI**                 | `/predict` endpoint for real-time predictions                             |
| **Streamlit Dashboard**     | Visualizes Health Scores and maintenance actions                          |
| **Robust Cross-Validation** | GroupKFold validation prevents data leakage between engines               |

## Dataset (CMAPSS)
- **Source**: [NASA Turbofan Engine Degradation Simulation](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)
- **Files**:
  - `train_FD001.txt` to `train_FD004.txt` (training data)
  - `test_FD001.txt` to `test_FD004.txt` (test data)
  - `RUL_FD00*.txt` (true RUL values)
- **Variables**:
  - **24 sensors** (`s1` to `s24`) + 3 operational settings (`os1`, `os2`, `os3`)
  - **Target**: `RUL` (Remaining Useful Life in cycles)

## Installation

### 1. Clone the repository
```
git clone https://github.com/your-username/predictive_maintenance.git
cd predictive_maintenance
2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run services
FastAPI:

uvicorn api.main\:app --reload
Access: http://localhost:8000/docs

Streamlit Dashboard:

streamlit run app/streamlit_app.py
Access: http://localhost:8501

Usage
1. Train the model
jupyter notebook notebooks/Model_training.ipynb
Output: Models saved in models/ (xgboost_model_optimized.pkl, xgboost_model_safety.pkl)

2. Make prediction via API
import requests

data = {
    "s2": 641.80,   # Temperature
    "s4": 1400.60,  # Pressure
    "s11": 47.40,   # Speed
    # ... (other sensors)
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
Example response:

{
  "Predicted RUL": 174.7,
  "Global Health Level": 69,
  "Status": "No maintenance required"
}
3. Visualize results
Open notebooks/model_evaluation.ipynb for:

Sensor correlation matrix
SHAP analysis
Operational dashboard (e.g., reports/figures/engine_0.0_operational_report.png)
Model Performance


Metric
Value
Baseline (Linear Regression)
Improvement
RMSE
29.99 cycles
32.21 cycles
+6.9%
R² Score
0.7914
0.7593
+4.2%
MAE
23.06 cycles
25.67 cycles
+10.2%
Prevented Failures
25
-
$1M saved
Key EDA Insights
Critical Sensors:

s11 (Temperature), s12, s4 (Pressure), s7 show strongest correlation with RUL
s15_roll_mean and s3_roll_mean are top contributors (SHAP)
Operational Thresholds:

Health Score > 50%: Normal operation
20% < Health Score ≤ 50%: Schedule inspection
Health Score ≤ 20%: Immediate grounding (failure risk)
Model Error:

Error increases when RUL is high (new engine)
Model is more accurate when engine is near failure (RUL < 50)
Operational Dashboard
Operational Dashboard

Health Score (green): Engine health index (0-100%)
Safety Margin (red): Uncertainty between main and safety models
Thresholds:
50%: Inspection threshold
20%: Immediate grounding limit
Safety & Robustness
Safety Model: Predicts conservative lower bound (RUL=65 vs 108 for main model)
Average Safety Margin: 43 cycles (buffer to prevent failures)
Cross-Validation:
Mean RMSE: 39.72 cycles (stable across folds)
Std Dev: 2.90 (robust model)
Business Impact


Metric
Value
Critical Failures Prevented
25
Potential Savings
$1,000,000
Preventive Maintenance Cost
$126,600
Net Savings
$873,400
Production Cycles Saved
12,660
Deployment
1. With Docker
docker build -t predictive_maintenance .
docker run -p 8000:8000 predictive_maintenance
2. Unit Tests
pytest tests/test_engine.py
3. Monitoring
Sensor Drift: Monitor sensor distributions (e.g., s11)
Model Performance: Retrain monthly with notebooks/Model_training.ipynb
Limitations
Simulated Data: Performance may vary on real-world data
Heuristic Thresholds: Health Score thresholds (20%, 50%) need fleet-specific adjustment
Linearity Assumption: RUL assumed to decrease linearly (simplification)
Sensor Dependency: Missing sensor data degrades predictions
Next Steps
Production Deployment:
Add logging for predictions (audit trail)
Implement real-time alerts (Slack/Email for Health Score < 20%)
Model Improvement:
Test LSTM for temporal patterns
Add external features (e.g., environmental conditions)
Expand to Other Equipment (e.g., compressors, pumps)
References
NASA Prognostics Data Repository
XGBoost Documentation
SHAP Values for Model Interpretability
Contributing
Contributions are welcome! Open an issue or submit a pull request to:

Add new features
Improve notebooks
Fix bugs
License
This project is licensed under MIT. See LICENSE for details.

### Instructions pour l'utiliser:
1. **Copie-colle** ce bloc entier dans ton fichier `README.md`
2. **Remplace** `https://github.com/your-username/predictive_maintenance.git` par ton vrai URL de dépôt
3. **Ajoute** un fichier `LICENSE` avec le texte standard MIT (disponible sur [choosealicense.com](https://choosealicense.com/licenses/mit/))
4. **Vérifie** que les chemins des images (`reports/figures/...`) correspondent à tes fichiers

Ce format est:
- **100% en anglais** comme demandé
- **Propre** sans séparateurs inutiles
- **Prêt à l'emploi** avec toutes les sections essentielles
- **Optimisé** pour GitHub (badges, tableaux, images)