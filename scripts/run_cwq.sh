#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="${CWQ_RUN_DIR:-$PROJECT_ROOT/outputs/cwq_$(date +%Y%m%d_%H%M%S)}"
QUESTION_WORKERS="${QUESTION_WORKERS:-4}"
if [[ -e "$RUN_DIR" ]]; then echo "Choose a new CWQ_RUN_DIR; no overwrite." >&2; exit 2; fi
mkdir -p "$RUN_DIR"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -u -m semantic_guided_kbqa.cli \
  --config "$PROJECT_ROOT/configs/pipeline.cwq.json" \
  --start 0 --limit "${CWQ_LIMIT:-0}" --question-workers "$QUESTION_WORKERS" \
  --output "$RUN_DIR/results.json" --errors-output "$RUN_DIR/errors.json" \
  --no-write-only-nonperfect-json "$@" 2>&1 | tee "$RUN_DIR/run.log"
echo "CWQ results: $RUN_DIR (Git ignored)"
