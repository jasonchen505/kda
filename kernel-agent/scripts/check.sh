#!/usr/bin/env bash
# CPU review path; no model service or GPU is started.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
uv sync --extra dev --frozen
uv run --frozen pytest -q
uv run --frozen ruff check flows scripts src tests
python3 scripts/verify_evidence.py
bash -n scripts/reproduce.sh scripts/reproduce_existing_selector.sh scripts/start_cached_qwen27.sh scripts/start_local_model.sh
