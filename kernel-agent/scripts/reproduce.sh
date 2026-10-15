#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
HUMANIZE_PYTHON=${HUMANIZE_PYTHON:-/data/home/chenyizhou/humanize/.venv/bin/python}
export PATH="/data/home/chenyizhou/.local/bin:$PATH"
export PYTHONPATH="$TASK_ROOT/src"
cd "$TASK_ROOT"
mkdir -p runs results
PROFILE=${1:-strategy}
RUN_SCRIPT=scripts/run_pilot.py
case "$PROFILE" in
  qwen27|strategy|selector-validation)
    GENERATION_GPUS=${KERNEL_MODEL_GPUS:-4,5}
    MODEL_PORT=18118
    MODEL_ID=kernel-qwen-local-27b
    MODEL_STARTER=scripts/start_cached_qwen27.sh
    MODEL_SEED=${KERNEL_MODEL_SEED:-20261011}
    RESULT_NAME=agent-pilot-qwen27
    RUNNER_ARGS=(--modes reference_guided --transport-policies json-lines --config data/opencode-qwen27.json
      --model-id "$MODEL_ID" --model-name 'Cached Qwen 27B FP8 (config: qwen3_5; path alias: Qwen3.8-27B-FP8)'
      --model-asset-manifest results/qwen27-asset-manifest.json --generation-gpus "$GENERATION_GPUS")
    if [[ "$PROFILE" == strategy ]]; then
      RESULT_NAME=agent-strategy-study
      RUN_SCRIPT=scripts/run_strategy_study.py
      RUNNER_ARGS=()
    elif [[ "$PROFILE" == selector-validation ]]; then
      RESULT_NAME=agent-selector-validation
      RUN_SCRIPT=scripts/run_selector_validation.py
      MODEL_SEED=${KERNEL_MODEL_SEED:-20261012}
      RUNNER_ARGS=(--seed "$MODEL_SEED")
    fi
    ;;
  lines4b)
    GENERATION_GPUS=5
    MODEL_PORT=18117
    MODEL_ID=kernel-qwen3-4b
    MODEL_STARTER=scripts/start_local_model.sh
    MODEL_SEED=${KERNEL_MODEL_SEED:-20261011}
    RESULT_NAME=agent-pilot-lines
    RUNNER_ARGS=(--modes reference_guided --transport-policies json-schema json-lines --config data/opencode-proposer.json)
    ;;
  *) echo 'Usage: bash scripts/reproduce.sh [strategy|selector-validation|qwen27|lines4b]' >&2; exit 2 ;;
esac
"$HUMANIZE_PYTHON" - "$GENERATION_GPUS" "$MODEL_PORT" <<'PY'
import socket,subprocess,sys
generation={int(v)for v in sys.argv[1].split(',')}
if 6 in generation:raise RuntimeError('GPU6 is reserved for independent kernel evaluation')
targets=generation|{6}
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(a):int(b)for a,b in (row.split(',')for row in rows.splitlines())}
busy={gpu:used[gpu]for gpu in targets if used[gpu]>256}
if busy:raise RuntimeError(f'required GPUs are occupied: {busy}; existing processes will not be stopped')
for port in (18116,int(sys.argv[2])):
    with socket.socket()as probe:probe.bind(('127.0.0.1',port))
PY
if [[ "$PROFILE" == qwen27 || "$PROFILE" == strategy || "$PROFILE" == selector-validation ]]; then
  "$HUMANIZE_PYTHON" scripts/inventory_model.py \
    "${KERNEL_MODEL_DIRECTORY:-/data/home/chenyizhou/my-research-project-v11/artifacts/p1/models/Qwen3.8-27B-FP8}" \
    --output results/qwen27-asset-manifest.json
fi
ARCHIVE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -f "results/$RESULT_NAME.json" ]]; then
  if [[ "$PROFILE" == strategy || "$PROFILE" == selector-validation ]]; then
    mv "results/$RESULT_NAME.json" "results/$RESULT_NAME-$ARCHIVE_STAMP.json"
  else
    cp "results/$RESULT_NAME.json" "results/$RESULT_NAME-$ARCHIVE_STAMP.json"
  fi
fi
KERNEL_MODEL_GPUS="$GENERATION_GPUS" KERNEL_MODEL_PORT="$MODEL_PORT" bash "$MODEL_STARTER" > runs/vllm-reproduce.log 2>&1 &
MODEL_SERVER_PID=$!
PROXY_BASE_ARGS=()
if [[ "$MODEL_STARTER" == scripts/start_cached_qwen27.sh ]]; then
  PROXY_BASE_ARGS=(--upstream-model-id qwen38-27b-fp8)
fi
"$HUMANIZE_PYTHON" scripts/local_proxy.py --model-id "$MODEL_ID" --seed "$MODEL_SEED" --upstream "http://127.0.0.1:$MODEL_PORT" \
  "${PROXY_BASE_ARGS[@]}" \
  --trace-dir runs/reproduce-wire > runs/proxy-reproduce.log 2>&1 &
PROXY_SERVER_PID=$!
trap 'kill "$PROXY_SERVER_PID" "$MODEL_SERVER_PID" 2>/dev/null || true' EXIT
"$HUMANIZE_PYTHON" - "$MODEL_PORT" <<'PY'
import sys,time,urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models', timeout=2).close()
        break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError('local model did not become ready; read runs/vllm-reproduce.log')
PY
"$HUMANIZE_PYTHON" -m pytest tests/unit tests/integration -q
if [[ "$PROFILE" == strategy ]]; then
  "$HUMANIZE_PYTHON" "$RUN_SCRIPT"
  "$HUMANIZE_PYTHON" scripts/summarize_strategy.py
elif [[ "$PROFILE" == selector-validation ]]; then
  "$HUMANIZE_PYTHON" "$RUN_SCRIPT" "${RUNNER_ARGS[@]}"
  "$HUMANIZE_PYTHON" scripts/summarize_strategy.py --input "results/$RESULT_NAME.json"
else
  "$HUMANIZE_PYTHON" "$RUN_SCRIPT" "${RUNNER_ARGS[@]}" --output-name "$RESULT_NAME.json"
fi
"$HUMANIZE_PYTHON" scripts/build_report.py --input "results/$RESULT_NAME.json"
