# Production readiness — fixes applied

This documents issues found and fixed directly in the code (not just
flagged) ahead of deployment. Each one is a case where one part of the
system silently disagreed with another; the fix in every case makes that
disagreement structurally impossible rather than patching the one place it
was first noticed.

## 1. Health score bug: 99.3% shown for an out-of-distribution engine

**Symptom:** a live prediction with `predicted_RUL=122.83`,
`safety_RUL=124.15`, `out_of_distribution=true` (20.8% of features out of
training range) still displayed a **Health Score of 99.3%** — a number that
looked fine while the input was already flagged as unlike anything the
model was trained on.

**Root cause:** `evaluation.compute_health_score()` normalizes
`safety_RUL / max_rul_cap * 100` and has no awareness of the OOD flag at
all (124.15 / 125 × 100 = 99.32%, matching the bug exactly). The OOD check
and the health score were computed independently and never reconciled.

**Fix:** `health_score` in the API response (`PredictionOutput`,
`api/schemas.py`) is forced to `0.0` — worst-case, not a computed value —
whenever `out_of_distribution` is `true`, and `status` is replaced entirely
with `"🚨 INVALID SENSOR DATA - INSPECT SENSORS"` instead of a
threshold-based maintenance action. Decided once, in `api/main.py`, rather
than leaving every consumer to separately check `out_of_distribution`
before trusting the score. `0.0` rather than `null`: a fleet table sorted
or filtered on `health_score` must not be able to silently drop an invalid
row. `app/streamlit_app.py` was updated in **both** places that render it:
- the single-engine prediction page (metric, gauge color, and a warning
  caption keyed off `out_of_distribution`, not off the score itself)
- the fleet analysis page, where `Average Health` explicitly excludes
  `Out_of_Distribution` rows (rather than relying on a missing/NaN value)
  so their forced `0.0`s don't drag the fleet average down, and an OOD
  count + warning banner is shown when any engines are flagged.

(An earlier iteration of this fix used `null`/`None` instead of `0.0`,
matching a different convention from the original, unmodified `api/main.py`
this session was given. That diverged from the convention the user had
already established in their own working copy of the file, which was
caught and corrected to the `0.0` + explicit-status version documented
above.)

## 2. Quantile crossing: the "safety" bound wasn't always the safer one

**Symptom:** `safety_RUL` (the conservative lower-quantile model, meant to
underestimate remaining life) exceeded `predicted_RUL` (the point
estimate) on some rows — e.g. the live example above
(124.15 > 122.83) — and in aggregate during training
(`notebooks/02_model_training.ipynb`: mean champion RUL 86.08 vs. mean
safety RUL 86.26, buffer −0.2, logged as "WARNING: Safety limit too high").
This is the textbook quantile-crossing failure mode, and it matters here
specifically because `safety_RUL` is the number this system tells
operators to ground engines on.

**Root cause:** the check existed only as a print statement in a notebook
cell nobody monitors in CI — not run automatically, not logged anywhere
queryable, and with no visibility into whether it's "every row crosses
slightly" vs. "a few rows cross badly" (those call for different fixes).

**Fix:** added `evaluation.check_quantile_crossing()` — computes, per row,
`predicted_RUL - safety_RUL` and returns the crossing rate, mean buffer,
and worst-case crossing, not just the aggregate mean. `src/train.py` now
calls this automatically right after training the safety model and logs
`quantile_crossing_rate`, `quantile_crossing_mean_buffer`, and
`quantile_crossing_worst` as real MLflow metrics on every run, with a
printed warning (pointing to the standard fixes: lower `quantile_alpha`,
add a monotonicity constraint, or rearrange post-hoc per Chernozhukov et
al. 2010) whenever the crossing rate is above zero. This makes the failure
visible in the MLflow UI and comparable across runs instead of relying on
someone re-reading a notebook.

## 3. Frontend thresholds didn't match the models they were describing

**Symptom:** `app/streamlit_app.py`'s Settings page and internal constants
used `WINDOW_SIZE=5`, `INSPECT_THRESHOLD=30`, `GROUND_THRESHOLD=10` — all
different from `configs/config.yaml`'s actual values (`10`, `50`, `20`),
which is what the models were actually trained and tuned against.

**Fix:** the frontend no longer keeps its own copy of these values at all.
`api/main.py` gained a `GET /config` endpoint that returns the thresholds
it already loads from `config.yaml` at startup; `streamlit_app.py` fetches
it once (cached 60s) and falls back to config.yaml's real defaults (not the
old wrong ones) only if the API is unreachable. One source of truth,
consistent with the `health_score` fix above.

## 4. `ood_max_violation_ratio` in config.yaml had no effect on the API

**Symptom:** `config.yaml` declares `evaluation.ood_max_violation_ratio:
0.1`, but `api/main.py` never read it — `inference.predict_with_explanation`
silently used its own hardcoded default (also `0.1`, by coincidence).
Anyone who tuned this value in `config.yaml` would see zero effect on the
live API, with no error to explain why.

**Fix:** `api/main.py` now reads `config["evaluation"]["ood_max_violation_ratio"]`
at startup and passes it explicitly as `ood_violation_threshold` to
`predict_with_explanation`. Also exposed via `GET /config`.

## 5. Docker images built from a fresh clone would crash at first request

**Symptom:** `models/*.pkl` and `data/processed/feature_reference_ranges.json`
are DVC-tracked and `.gitignore`'d. `Dockerfile` / `Dockerfile.streamlit`
only ran `COPY . .`, which does not materialize DVC-tracked files — a fresh
clone + `docker build` produces an image that starts, passes `/health`, and
then throws `FileNotFoundError` on the first `/predict` call when
`joblib.load(...)` runs against a `.pkl` that was never actually copied in.
(`docs/deployment.md` already flagged this as a known gap before this
pass.)

**Fix:** `Dockerfile` (the API image — the only one that needs model
artifacts; `app/streamlit_app.py` only calls the API over HTTP and has no
DVC-tracked dependency of its own) now installs `dvc[s3]`, runs
`dvc pull` for the specific tracked artifacts the API loads, using a
BuildKit secret mount for remote credentials (`--secret id=dvc_config`)
rather than a build arg, which would leak credentials into the image's
layer history — then uninstalls `dvc` again so it isn't in the runtime
image. Adjust the remote backend in `.dvc/config` for your actual storage
(S3/GCS/Azure/etc.) before building; see the comments in `Dockerfile` for
the exact build command.

## Operational note: rebuilding after a code change

Replacing a file on disk inside a cloned repo does **not** update a
already-built Docker image — `COPY . .` runs once, at build time. After
changing any file under `src/`, `api/`, or `app/`:

```bash
docker-compose up -d --build
```

`docker-compose up` alone (without `--build`) reuses the existing image and
will silently keep running the old code.

## Not fixed — explicitly out of scope

- `deployment.yaml` (Kubernetes): a single container currently claims both
  the API's port 8000 and the frontend's port 8501, but no single Docker
  image runs both processes — this manifest won't actually deploy correctly
  as written. Left as-is per project decision; not a current priority.
- `inference.py`'s collapsing of "unit not found" and "internal error" into
  a single `None` return (both currently surface as the API's `422`) — see
  `docs/api.md`'s "Known limitations" section.