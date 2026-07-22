#!/usr/bin/env bash
# Runs the full training pipeline (src/train.py): ETL, feature engineering,
# baseline, Optuna-tuned champion model, quantile safety model, MLflow
# logging, and artifact export for notebooks/03_model_evaluation.ipynb.
#
# Usage:
#   ./scripts/train.sh                      # uses configs/config.yaml
#   ./scripts/train.sh --config path/to.yaml
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "[train.sh] no virtualenv active — activating .venv/"
  if [ ! -f ".venv/bin/activate" ]; then
    echo "[train.sh] .venv/ not found. Create it first: python3 -m venv .venv && pip install -r requirements.txt" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ ! -f "data/raw/train_FD001.txt" ]; then
  echo "[train.sh] data/raw/train_FD001.txt not found. Place the CMAPSS raw data there first." >&2
  exit 1
fi

echo "[train.sh] starting training pipeline..."
python src/train.py "$@"
echo "[train.sh] done. Models in models/, artifacts in data/processed/, run tracked in mlflow.db."