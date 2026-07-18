"""
Cross-cutting helpers used by train.py and (optionally) the notebooks.

The main problem this solves: notebooks run with cwd=notebooks/ (so '../data/...'
resolves correctly), but a script invoked as `python src/train.py` or via DVC runs
with cwd=repo root — the same '../data/...' string would then point OUTSIDE the
repo. `resolve()` anchors every path on the project root itself, so the same
config values work regardless of where the process was launched from.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Loads config.yaml. Defaults to <project_root>/configs/config.yaml."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(relative_path: str) -> Path:
    """Resolves a repo-root-relative path (e.g. 'data/raw/train_FD001.txt')
    to an absolute path, independent of the current working directory."""
    return PROJECT_ROOT / relative_path


def set_seed(seed: int) -> None:
    """Seeds numpy/random for reproducibility, on top of the per-model
    random_state already passed explicitly to sklearn/xgboost/optuna calls."""
    np.random.seed(seed)
    random.seed(seed)
