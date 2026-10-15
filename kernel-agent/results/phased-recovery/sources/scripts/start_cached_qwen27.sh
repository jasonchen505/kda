#!/usr/bin/env bash
set -euo pipefail
VLLM_EXECUTABLE=${VLLM_EXECUTABLE:-/data/home/chenyizhou/my-research-project-v11/artifacts/p1/vllm-0271-venv/bin/vllm}
MODEL_DIRECTORY=${KERNEL_MODEL_DIRECTORY:-/data/home/chenyizhou/my-research-project-v11/artifacts/p1/models/Qwen3.8-27B-FP8}
export PATH="$(dirname "$VLLM_EXECUTABLE"):$PATH"
export CUDA_VISIBLE_DEVICES=${KERNEL_MODEL_GPUS:-5,6}
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V2_MODEL_RUNNER=0
exec "$VLLM_EXECUTABLE" serve "$MODEL_DIRECTORY" --served-model-name qwen38-27b-fp8 \
  --host 127.0.0.1 --port "${KERNEL_MODEL_PORT:-18118}" --tensor-parallel-size 2 --max-model-len 32768 \
  --max-num-seqs 1 --gpu-memory-utilization 0.85 --enforce-eager \
  --kv-cache-dtype fp8 --enable-prefix-caching \
  --attention-backend TRITON_ATTN --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
