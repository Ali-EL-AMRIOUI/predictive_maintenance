# Deployment

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env              # fill in any local values
```

## Running the pipeline (DVC)

`dvc.yaml` defines two stages — `train` and `evaluate` — wired to
`configs/config.yaml` so a parameter change automatically invalidates the
right stage.

```bash
dvc init                                  # once per repo
dvc add data/raw                          # start tracking raw data with DVC
dvc remote add -d storage <remote-url>    # S3 / GDrive / local path
dvc repro                                 # runs whichever stages are stale
```

`dvc repro` re-runs `train` if `configs/config.yaml`'s tracked params or the
raw data change, and always re-runs `evaluate` afterward since it depends on
`train`'s outputs. Running only the training step directly, without DVC:

```bash
./scripts/train.sh
```

## MLflow tracking

Training logs to a local SQLite-backed MLflow store (`mlflow.db`, resolved
to an absolute path via `utils.resolve` so it's the same file whether MLflow
is invoked from `notebooks/` or from the repo root).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://localhost:5000` — champion and safety runs are logged
separately per training run, linked by `run_info.json`'s `champion_run_id`
so the evaluation notebook resumes logging into the same run rather than
creating a disconnected one.

## Serving the model

```bash
uvicorn api.main:app --reload --port 8000
```
See `docs/api.md` for the endpoint contract. In a real deployment this would
run behind a process manager (systemd, gunicorn+uvicorn workers, or a
container orchestrator) rather than `--reload`, and the model artifacts
(`models/*.pkl`, `feature_reference_ranges.json`, `run_info.json`) would be
pulled via `dvc pull` as part of the image build/startup rather than assumed
to already be on disk.

## Retraining

There's no scheduled retraining wired up yet — `dvc repro` is a manual
trigger. The natural next step is a scheduled CI job (cron-triggered GitHub
Actions workflow) that runs `dvc repro` and opens a PR with the updated
metrics if performance regresses, rather than deploying automatically.

## Monitoring & drift

Not implemented yet. The two things worth tracking once this serves real
traffic:
- **Input drift**: are incoming sensor readings statistically consistent
  with the training distribution? `evaluation.check_out_of_distribution`
  already gives a per-request signal; aggregating that signal's flag rate
  over time (e.g. in a dashboard, or via a tool like Evidently) turns it
  into a fleet-level drift indicator instead of only a per-prediction one.
- **Prediction drift**: is the distribution of `predicted_RUL` across the
  fleet shifting over time in a way that isn't explained by the fleet
  actually aging? Compare against `data/processed/model_results_final.csv`'s
  distribution at training time as the reference.

## Rollback

Since `train.py` never overwrites `models/*.pkl` conditionally — every run
produces a new model file and a new MLflow run — rolling back means pointing
`config.yaml`'s `models.final_model_path` / `models.safety_model_path` at a
previous run's logged model (`mlflow.xgboost.load_model` with the old
`run_id`, re-exported to the expected `.pkl` path) rather than re-running
training. No automated rollback tooling exists yet; this is a manual
config-and-redeploy step today.