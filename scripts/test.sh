#!/usr/bin/env bash
# Runs the test suite. Same command used locally and in CI
# (.github/workflows/ci.yml) so "passes on my machine" and "passes in CI"
# mean the same thing.
#
# Usage:
#   ./scripts/test.sh            # run all tests
#   ./scripts/test.sh -k health  # pass extra args straight to pytest
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

pytest tests/ -v --tb=short "$@"