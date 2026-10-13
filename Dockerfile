FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

# models/*.pkl and data/processed/* are DVC-tracked and .gitignore'd for GIT
# (see models/.gitignore, .gitignore) — but `.gitignore` has no effect on
# `docker build`. COPY . . copies whatever is physically on disk in this
# build context, so as long as you run `./scripts/train.sh` (or `dvc repro`)
# locally BEFORE `docker build`/`docker-compose build`, the trained .pkl
# files are already on disk and get copied in fine. No `dvc pull` needed
# for that normal local workflow.
#
# dvc pull would only be needed to build this image WITHOUT training
# locally first — e.g. a CI runner or a fresh clone on another machine that
# only has dvc.lock's pinned hashes, not the actual model files. That setup
# needs a configured DVC remote (`dvc remote add -d ...`) and BuildKit
# (`DOCKER_BUILDKIT=1` + the `docker buildx` plugin) for `--mount=type=secret`
# to pass remote credentials into the build without leaking them into the
# image's layer history. Not wired in here since no DVC remote is configured
# in this project yet — add it back (see git history) once one exists.

EXPOSE 8000

# Start ONLY the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]