#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
VLLM_EXECUTABLE=${VLLM_EXECUTABLE:-/data/home/chenyizhou/my-research-project-v11/artifacts/p1/vllm-0271-venv/bin/vllm}
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:-/data/home/chenyizhou/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
export PATH="$(dirname "$VLLM_EXECUTABLE"):$PATH"
export CUDA_VISIBLE_DEVICES=${KERNEL_MODEL_GPUS:-5}
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V2_MODEL_RUNNER=0
mkdir -p "$TASK_ROOT/runs"
exec "$VLLM_EXECUTABLE" serve "$MODEL_SNAPSHOT" --served-model-name kernel-qwen3-4b \
  --host 127.0.0.1 --port 18117 --max-model-len 16384 --max-num-seqs 2 \
  --gpu-memory-utilization 0.65 --enforce-eager --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
