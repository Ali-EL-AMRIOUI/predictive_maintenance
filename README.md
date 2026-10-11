# Predictive Maintenance — Turbofan Remaining Useful Life (RUL) Estimation

Production-oriented MLOps pipeline that predicts how many operating cycles a
turbofan engine has left before failure (NASA C-MAPSS FD001), and translates
that prediction into an operational decision — not just a number.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![DVC](https://img.shields.io/badge/pipeline-DVC-945DD6)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Methodology](#methodology)
- [Evaluation & Robustness](#evaluation--robustness)
- [Testing](#testing)
- [Results](#results)
- [Inference API](#inference-api)
- [Collaboration Workflow](#collaboration-workflow)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)
- [Author](#author)

---

## Problem Statement

Unplanned failure of a jet engine is enormously more expensive than
maintaining it early — but maintaining every engine on a fixed schedule
wastes the remaining useful life it still had. The standard framing is
**Remaining Useful Life (RUL) prediction**: given an engine's sensor history
up to the current cycle, estimate how many operating cycles remain before it
degrades past a safe operating threshold.

This project trains on **NASA's C-MAPSS FD001** subset (single operating
condition, single fault mode — the simplest of the four C-MAPSS subsets, a
stated assumption, not something the project hides — see
[Limitations](#limitations)). The deliverable isn't a notebook that prints an
RMSE; it's a full pipeline that goes from raw sensor logs to a maintenance
decision with a dollar figure attached, built and reviewed the way a small
ML team actually ships one.

## Key Features

- **Two-model architecture, not one.** A point-estimate champion model
  (XGBoost, hyperparameters tuned against the official NASA/PHM08 asymmetric
  score) paired with a separate quantile-regression **safety model** that
  gives a conservative lower bound on RUL — the number an operator should
  actually ground an engine on, not the point estimate.
- **Leak-safe nested cross-validation.** Hyperparameter search uses an outer
  `GroupKFold` (by engine, so no engine's cycles appear in both train and
  validation) with an *inner* `GroupShuffleSplit` for early stopping — the
  early-stopping validation set never leaks into the score being optimized.
- **Domain-correct objective.** Optuna optimizes the actual NASA/PHM08
  scoring function (asymmetric — late predictions cost more than early ones),
  not a generic RMSE that treats both directions of error as equally bad.
- **Interpretability that reaches production.** SHAP isn't confined to a
  notebook plot — `predict_with_explanation()` returns the top local SHAP
  drivers behind every single prediction, in the same call an API would make.
- **Statistical rigor on the business claim.** The "XGBoost beats the linear
  baseline" claim is backed by a paired bootstrap confidence interval, not
  two point estimates — same for the net-savings dollar figure, which
  separately tracks missed failures *in the critical zone* rather than a
  single blended miss count.
- **Robustness testing.** Sensor dropout and Gaussian noise are injected into
  the test set to measure how much RMSE degrades under realistic sensor
  failure/drift, instead of only ever evaluating on clean data.
- **Out-of-distribution guard.** Inference checks incoming features against
  the training distribution and flags low-confidence predictions instead of
  silently returning a confident-looking number for input unlike anything
  the model was trained on.
- **Config-driven, not magic-number-driven.** Every threshold, cost
  assumption, and hyperparameter search setting lives in one
  `configs/config.yaml`, imported by the notebooks, `src/train.py`, and
  `dvc.yaml` alike.
- **MLOps, wired together, not bolted on.** DVC pipeline stages enforce that
  evaluation can only run against artifacts training actually produced;
  MLflow logs live inside the exact run that trained the model being
  evaluated — not a disconnected "evaluation" run.

## Architecture

```mermaid
flowchart TD
    A["Raw C-MAPSS data\ntrain_FD001.txt"] --> B["ETL\ndata_loader.py\npiecewise RUL capping"]
    B --> C["GroupShuffleSplit\nby engine unit"]
    C --> D["Feature Engineering\nfeatures.py\nrolling mean / std per unit"]
    D --> E["Baseline\nLinear Regression"]
    D --> F["Optuna\nChampion Model\nNASA-score objective"]
    D --> G["Optuna\nSafety Model\nquantile regression"]
    F --> H[("xgboost_final_model.pkl")]
    G --> I[("xgboost_model_safety.pkl")]
    F --> K["MLflow run\n(champion)"]
    G --> L["MLflow run\n(safety)"]
    H --> M["model_results_final.csv\ntest_features_engineered.csv\nfeature_reference_ranges.json\nrun_info.json"]
    I --> M
    E --> M
    M --> N["03_model_evaluation.ipynb"]
    N --> O["SHAP · Slice Analysis\nStress Testing · Bootstrap CI\nOOD Validation"]
    O --> P["Business Impact\n+ Operational Dashboard"]
    K -. resumes same run .-> N
```

`dvc.yaml` encodes this as two stages — `train` (`python src/train.py`) and
`evaluate` (the notebook above) — so `dvc repro` refuses to evaluate against
stale or missing training artifacts, and automatically re-runs training when
`config.yaml` or the raw data changes. `01_eda.ipynb` sits outside this
dependency chain by design — it's exploratory, and documents *why* the
pipeline's modeling choices (piecewise capping, rolling windows) are the
right ones, rather than gatekeeping `02`.

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | XGBoost, scikit-learn |
| Hyperparameter search | Optuna (nested Group K-Fold CV) |
| Interpretability | SHAP (global + local) |
| Experiment tracking | MLflow (params, metrics, models, figures, tables) |
| Pipeline / data & model versioning | DVC |
| Testing | pytest |
| CI | GitHub Actions |
| Serving | FastAPI (`api/`) |
| UI / demo | Streamlit (`app/`) |
| Containerization | Docker (`Dockerfile`, built and runnable) |
| Language | Python 3.10+ |

Kubernetes (`deployment.yaml`) is checked in as a manifest but not yet an
active deployment target — see [Roadmap](#roadmap).

## Project Structure

```
predictive-maintenance/
├── api/                     # FastAPI serving layer
│   ├── main.py               # delegates to src/inference.py — no duplicated logic
│   └── schemas.py
├── app/                      # Streamlit demo UI
├── configs/
│   └── config.yaml           # single source of truth for every parameter
├── data/
│   ├── raw/                  # C-MAPSS train/test/RUL files
│   └── processed/            # cleaned data + exported evaluation artifacts
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
├── models/                   # persisted .pkl artifacts (DVC-tracked, see dvc.yaml)
├── notebooks/
│   ├── 01_eda.ipynb           # exploratory — documents the "why" behind 02's choices
│   ├── 02_model_training.ipynb    # orchestrates src/, produces artifacts
│   └── 03_model_evaluation.ipynb  # loads artifacts only, no shared state with 02
├── reports/figures/          # saved evaluation plots
├── scripts/                  # train.sh / test.sh / deploy.sh
├── src/
│   ├── data_loader.py         # raw ingestion, piecewise RUL capping
│   ├── features.py            # rolling-window feature engineering
│   ├── evaluation.py          # NASA score, health score, slice/stress/bootstrap/OOD/SHAP, business impact
│   ├── train.py                # CLI-runnable training pipeline (what DVC calls)
│   ├── inference.py           # production inference, mirrors the training feature pipeline
│   └── utils.py               # config loading, project-root path resolution
├── tests/
│   ├── test_evaluation.py     # bootstrap CI, OOD detection, slice metrics, business impact
│   └── test_inference.py      # SHAP explanation path, feature reconstruction
├── .github/workflows/ci.yml
├── .env.example
├── dvc.yaml                    # train -> evaluate pipeline definition
├── Dockerfile
├── LICENSE
└── requirements.txt
```

`02` and `03` share zero in-memory state. Everything `03` needs — models,
predictions, feature matrices, baseline metrics, OOD reference ranges, MLflow
run IDs — is written to disk by `02` / `train.py` and reloaded from a fresh
kernel. This is enforced structurally by `dvc.yaml`, not just by convention.

## Quickstart

```bash
git clone <this-repo>
cd predictive-maintenance
pip install -r requirements.txt
pip install dvc mlflow pytest

# Reproducible pipeline (recommended) — .dvc/config already ships with this
# repo, so skip `dvc init` on a fresh clone; it would error on an
# already-initialized repo.
dvc pull                                   # fetch DVC-tracked outputs, if a remote is configured
dvc repro                                  # runs train -> evaluate end to end

# Or run pieces individually
python src/train.py                        # training only
jupyter notebook notebooks/03_model_evaluation.ipynb   # evaluation only, once 02/train.py has run once

# Tests
pytest tests/ -v

# Inspect experiment tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Methodology

**RUL target — piecewise capping.** Raw RUL (cycles until failure) decreases
linearly and unboundedly from the start of an engine's life, but degradation
is only reliably observable in roughly the last 125 cycles. Cycles further
from failure are capped at `max_rul` (config-driven, default 125) rather than
given a large, mostly-noise target — the standard C-MAPSS convention (Heimes,
2008).

**Features — grouped rolling statistics.** Rolling mean/std per sensor,
grouped by engine `unit` so the window never crosses from one engine's
history into another's, and trailing (backward-looking) by construction so
no future cycle ever leaks into a past row's features.

**Objective — the actual NASA score, not RMSE.** The PHM08 scoring function
penalizes late predictions (model says "still fine" when the engine is
closer to failure) more heavily than early ones via asymmetric exponential
terms. Optuna optimizes this directly, so the selected hyperparameters
reflect the real cost structure of the problem, not a proxy for it.

**Validation — nested, not single-split.** Hyperparameter search uses an
outer `GroupKFold` for the score being optimized, with an *inner*
`GroupShuffleSplit` carved out purely for early stopping. Without this
separation, the early-stopping set would leak information into the CV score,
producing an optimistic estimate of how good the chosen hyperparameters
actually are. The same nested pattern is used for the post-training
stability check (`robust_cv_check`) — an earlier version of that check
scored each fold on the same rows used for early stopping, which made it
mildly optimistic about its own variance; this is fixed in `src/train.py`
and still pending in the equivalent notebook cell (see
[Roadmap](#roadmap)).

**Safety layer — quantile regression, not `prediction - margin`.** The
safety bound is a second model trained on XGBoost's native
`reg:quantileerror` objective (default: 10th percentile), not the champion's
prediction minus an arbitrary buffer. It answers a different question than
the champion model does — "how low could this realistically be?" — which is
the number that should actually drive a grounding decision.

**Health score — fixed scale, not batch-relative.** Health score is
`safety_RUL / max_rul_cap`, anchored to the same fixed cap used to build the
training target — not a min-max normalization over whatever batch of engines
happens to be in the evaluation set. A batch-relative score would make the
same engine's status depend on which other engines were being evaluated
alongside it, which makes "50% health" meaningless as a signal over time.

**Business impact — the full alert zone, not just the critical sub-zone.**
`compute_business_impact` counts a missed failure whenever true RUL is
inside the alert zone and the model predicted above it — not only when true
RUL was inside the stricter critical sub-zone. An earlier version only
checked the critical sub-zone, which left true RUL between the critical and
alert thresholds, paired with an over-confident prediction, completely
unscored in the cost model — exactly the case where a false "still fine" is
most dangerous. `missed_failures_critical` is still reported separately for
visibility into the more severe subset.

## Evaluation & Robustness

A model that looks good on a single global RMSE, on clean data, compared to
baseline with no confidence interval, is not evaluated — it's demoed. This
project evaluates on four axes beyond the basics:

| Axis | What it checks | Where |
|---|---|---|
| **Slice analysis** | RMSE/MAE broken down by true-RUL range — is the model actually accurate where it matters (near failure), or just accurate on average? | `evaluation.compute_sliced_metrics` |
| **Robustness** | RMSE degradation under simulated sensor dropout and Gaussian noise | `evaluation.evaluate_under_perturbation` |
| **Statistical significance** | Paired bootstrap CI on (RMSE\_baseline − RMSE\_champion) and on net savings — is the improvement real, or noise from the test split? | `evaluation.bootstrap_metric_ci` |
| **Out-of-distribution guard** | Percentile-range check against training feature distributions, validated to confirm it stays quiet on clean held-out data | `evaluation.check_out_of_distribution` |

All four are computed in `03_model_evaluation.ipynb` and logged as MLflow
metrics/figures inside the training run that produced the model being
evaluated.

## Testing

```bash
pytest tests/ -v
```

- `test_evaluation.py` — bootstrap CI behavior (does it correctly flag a
  known-significant difference vs. a null case), OOD guard behavior (flags
  engineered outliers, stays quiet on in-range data, ranks violations by
  severity), slice metric correctness against hand-computed expected values,
  and the business-impact boundary case (true RUL between the critical and
  alert thresholds with an over-confident prediction must count as a missed
  failure).
- `test_inference.py` — the SHAP local-explanation path end to end (correct
  shape, sorted by absolute impact), unknown-unit handling, and that
  `predict_with_explanation` reconstructs the exact same rolling features
  `src/features.py` would produce at training time (the specific class of
  train/serve skew this project is built to make structurally impossible).

Runs automatically via `.github/workflows/ci.yml` on push. The training
pipeline itself (`src/train.py`) is decomposed into named, single-purpose
functions (`engineer_features`, `make_objective`, `robust_cv_check`, ...)
specifically so each is independently testable, rather than only checkable
by running the whole pipeline end to end — extending `pytest` coverage to
these is a known gap (see [Roadmap](#roadmap)).

## Results

*From a full run against real FD001 data (100 engines, 20 held out for test,
25 Optuna trials per model). Numbers shift on every re-run — Optuna's sampler
isn't seeded, so re-running `dvc repro` will land on slightly different
hyperparameters and metrics; treat this table as one verified data point, not
a fixed benchmark. `mlflow ui` is the live source of truth for whatever the
current run shows.*

*The business-impact figures below (missed/prevented failures, net savings)
were computed before the critical-zone boundary fix described in
[Methodology](#methodology) — they'll shift somewhat once `dvc repro` is
re-run with the corrected `compute_business_impact`, most likely upward on
the missed-failures count for the reason described there. Treat this table
as the last verified pre-fix run, not the current one.*

| Metric | Baseline (Linear) | Champion (XGBoost) |
|---|---|---|
| RMSE | 19.78 | **16.59** (−16.1%) |
| R² | — | 0.8419 |
| NASA Score | 26,644 | 34,059 (**−27.8%, worse**) |
| Avg Predicted RUL | — | 86.1 cycles |
| Safety Bound (avg) | — | 86.2 cycles |

Three findings from this run, reported as found rather than smoothed over:

**RMSE improved, NASA score didn't — slice analysis explains why, and the
weakest slice isn't the one you'd expect.** The champion beats baseline on
RMSE (bootstrap 95% CI on the gap: `[2.76, 3.63]` cycles — genuinely
significant), but scores worse on the asymmetric NASA metric. Slicing error
by true-RUL range shows the critical zone (RUL < 20) is actually the
model's *best* slice by RMSE (6.55 cycles) — but with a mean bias of
**+4.83** (systematically over-predicting, the direction NASA score punishes
hardest). The real weak spot is the `[50, 100)` slice: worst RMSE (23.1
cycles) *and* worst bias (+10.7) of any range. A global RMSE alone surfaces
neither of these; it's exactly what the slice analysis in
[Evaluation & Robustness](#evaluation--robustness) exists to catch.

**Net savings: real, large, and not yet statistically proven.** Point
estimate (pre-fix, see note above): **$4,450,000** ahead of baseline (128
critical failures prevented, 13 missed, 10 false alarms). The 95% bootstrap
CI on that advantage: **`[-$60,100, $1,760,600]`** — it crosses zero.
Reporting the point estimate alone would overstate the claim; the honest
read is "likely positive, not yet proven at 20 test engines" — a small
sample for a metric built from comparatively rare tail events (critical
failures, false alarms).

**Robustness held up well under this run.** Worst-case RMSE degradation
under simulated sensor dropout/Gaussian noise was **0.5%** — the model
degrades gracefully rather than falling apart under imperfect input.

To reproduce or refresh these numbers:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Starts a local dashboard (open it in a browser) listing every run with its
params, metrics, and logged figures — the source these numbers were pulled
from. `dvc repro` re-runs `train` → `evaluate` end to end, skipping stages
whose inputs haven't changed.

## Inference API

```python
from evaluation import get_shap_explainer, compute_feature_reference_ranges
from inference import predict_unit_health, predict_with_explanation

# Cheap path — just the number
rul = predict_unit_health(raw_sensor_history, unit_id=42, model=final_model)

# Operator-facing path — number + confidence + reasons
explainer = get_shap_explainer(final_model)
result = predict_with_explanation(
    raw_sensor_history, unit_id=42, model=final_model,
    explainer=explainer, reference_ranges=reference_ranges,
)
# {
#   "unit_id": 42,
#   "predicted_rul": 15.2,
#   "out_of_distribution": False,
#   "ood_violation_ratio": 0.0,
#   "top_reasons": [{"feature": "s11_roll_mean", "shap_impact": -4.1}, ...]
# }
```

Both recompute rolling features from raw sensor history internally (via
`src/features.py`), so the API layer never has to reimplement — or drift
from — the exact feature pipeline training used. `api/main.py` requires a
short *history* of recent cycles per engine (not a single reading) — rolling
features are mathematically undefined from one data point, and a model
prediction on too little history is flagged by the OOD guard rather than
returned silently at face value.

## Collaboration Workflow

Built solo, using a two-account GitHub setup to practice and demonstrate a
real team code-review workflow:

- **`Ali-EL-AMRIOUI`** — ML Engineer Lead: writes code, opens pull requests.
- **`elamrioui-png`** — MLOps Reviewer: reviews, comments, approves.

Feature and fix work goes through a branch, a PR, and a review before
merging — `refactor/predictive-pipeline`, `feat/bug-fixes-and-model-artifacts`,
`chore/cleanup-and-readme`, among others, each squash-merged after
`elamrioui-png`'s approval. A handful of small maintenance commits (dead-file
removal, the LFS-to-DVC migration below) landed directly on `main` during
early setup rather than through a PR — worth naming rather than glossing
over, since the PR history itself is what's checkable.

- **Conventional Commits** throughout (`feat(src): ...`, `fix(notebooks): ...`,
  `test: ...`, `docs: ...`, `chore: ...`).
- **Squash-and-merge** on every PR, keeping `main`'s history one clean commit
  per reviewed change instead of a raw stream of WIP commits.
- **Git LFS, tried and deliberately replaced.** Model artifacts were
  initially tracked with Git LFS. Once DVC's pipeline stages were in place,
  it became clear DVC already replaces what LFS was doing (large-file
  storage via a pointer file) *and additionally* knows these files are
  reproducible outputs of `dvc.yaml`'s `train` stage — running two systems
  that solve the same storage problem for the same files was redundant, not
  additive, so LFS was dropped in favor of DVC alone
  (`chore: leave model tracking to DVC`).
- Real review comments on substance (nested-CV correctness, MLflow logging
  choices, business-logic duplication) — approved after actual back-and-forth,
  not `LGTM` on every PR.

## Limitations

- Trained on FD001 only: a single operating condition, a single fault mode —
  the simplest of the four C-MAPSS subsets. Generalization to multi-condition
  or multi-fault data (FD002/FD003/FD004) is untested.
- The out-of-distribution guard is a percentile-range heuristic, not a
  calibrated statistical test — it catches obviously unusual input, it does
  not bound false-negative risk. It also checks each feature independently,
  so a combination of individually-in-range features can still be anomalous
  without being flagged.
- Health score thresholds (`inspect_threshold`, `ground_threshold`) are
  heuristic starting points, not calibrated against real fleet outcome data.
- Robustness testing covers sensor dropout and Gaussian noise; it does not
  cover correlated multi-sensor failure modes or adversarial input.
- The net-savings advantage over baseline is not yet statistically
  significant at the current test-set size (see [Results](#results)) —
  treat the dollar figure as directional, not a proven claim.
- MLflow artifact storage currently resolves relative to whatever directory
  the logging process was started from — `mlruns/` at the repo root when
  `train.py`/`dvc repro` runs, `notebooks/mlruns/` when the evaluation
  notebook runs directly — even though both write run *metadata* to the
  same `mlflow.db`. Not a correctness bug (every run is still fully logged
  and queryable), but a rough edge worth fixing before treating the
  artifact store as a single canonical location.

## Roadmap

- [ ] Apply the nested-CV fix now in `src/train.py`'s `robust_cv_check` to
      the equivalent cell in `notebooks/02_model_training.ipynb` as well —
      currently only the CLI path has the corrected (non-optimistic)
      stability check.
- [ ] Resolve the scaler built in `01_eda.ipynb` (`MinMaxScaler` →
      `models/scaler.joblib` → `train_features.csv`): either wire
      `02_model_training.ipynb` to actually consume it, per `01`'s own
      closing note, or remove the step — XGBoost and the linear baseline are
      both scale-invariant, so it currently has no effect on model quality
      either way, just an unresolved instruction.
- [ ] Point MLflow's artifact store at a single absolute path regardless of
      which directory a logging process starts from (see Limitations).
- [ ] Lightweight feature store pattern (versioned feature definitions +
      registry) — full Feast is disproportionate for a single-dataset batch
      pipeline, but the concept is worth demonstrating.
- [ ] Evidently for production data-drift monitoring.
- [ ] Kubernetes HPA (autoscaling) on `deployment.yaml`.
- [ ] MkDocs site generated from `docs/` + module docstrings.
- [ ] Extend `pytest` coverage to `src/train.py`'s pipeline functions
      (`engineer_features`, `make_objective`, ...) — evaluation and
      inference are covered; training is the remaining gap.
- [ ] Investigate the `[50, 100)` RUL-range slice — currently the model's
      worst-performing range by both RMSE and bias, not the critical zone.

## Web Interface

The project includes a Streamlit web application for interactive Remaining Useful Life (RUL) prediction.

The frontend communicates directly with the FastAPI REST API and uses the same inference pipeline as the training notebooks. No feature engineering or prediction logic is duplicated in the user interface.

### Main Features

- Manual engine history creation (1–20 flight cycles)
- JSON-based sensor input
- Remaining Useful Life (RUL) prediction
- Conservative Safety RUL estimation
- Health score computation
- Maintenance recommendation
- Out-of-Distribution (OOD) detection
- SHAP explanation of predictions
- Prediction latency measurement

### Running the Web Interface

Start the API:

```bash
uvicorn api.main:app --reload
```

Launch the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

By default:

```
Frontend: http://localhost:8501
API:      http://localhost:8000
```

### Prediction Workflow

```
Streamlit
    │
    ▼
FastAPI
    │
    ▼
Inference Pipeline
    │
    ├── Feature Engineering
    ├── Champion Model
    ├── Safety Model
    ├── OOD Detection
    └── SHAP Explanation
```

### Input

The application accepts an engine history containing between **1 and 20 flight cycles**.

Each cycle includes:

- Operational settings (`os1`–`os3`)
- Sensor measurements (`s1`–`s21`)

Although predictions are supported with a single cycle (`min_periods=1`), providing at least the rolling window size used during training (default: 5 cycles) produces more representative temporal features.

### Output

For each prediction, the interface displays:

- Predicted RUL
- Safety RUL
- Health Score
- Maintenance Recommendation
- Out-of-Distribution Status
- OOD Violation Ratio
- Violating Features
- Top SHAP Features
- Prediction Latency

### Screenshots

```
docs/images/dashboard.png
docs/images/prediction.png
docs/images/shap.png
```

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Ali El Amrioui** — AI/ML Engineer
[github.com/Ali-EL-AMRIOUI](https://github.com/Ali-EL-AMRIOUI)