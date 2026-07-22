"""
Predictive Maintenance — shared pipeline package.

Modules:
- data_loader : raw CMAPSS ingestion + piecewise RUL capping
- features    : rolling-window feature engineering (shared by training & inference)
- evaluation  : NASA score, pinball loss, health score, maintenance actions
- train       : end-to-end training pipeline (CLI-runnable: `python src/train.py`)
- inference   : production inference (mirrors the training feature pipeline)
- utils       : config loading, project-root path resolution, reproducibility

Import style: modules use flat imports (`from features import ...`), matching
notebooks/*.ipynb which add src/ to sys.path directly rather than treating it as
an installed package. Running from repo root (scripts, pytest) also works via
`from src.evaluation import ...` since this __init__.py marks src/ as a package.
"""
__version__ = "0.1.0"
