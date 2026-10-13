# Multi-seed execution on user-selected GPU4

The 36-job prospective design in `multiseed-study-plan.md` is unchanged: six development
families, three stochastic seeds and feedback/blind refinement, three proposals each.
This document records execution changes separately from the frozen design and old evidence.

## Preserved pilots and why a guarded plan is needed

The original v1 plan was prepared on occupied GPU6 and started zero jobs. A runtime review
found that `Path(...).resolve()` in its planner discarded the evaluation virtualenv's
`bin/python` symlink, storing a bare interpreter without Torch. This did not affect any
historical generation campaign: that new runner had not executed a job. The original
checkpoint and published resource preflight are retained, rather than silently repaired.

The current runner preserves the absolute invocation path and records the actual fresh
process's `sys.prefix`, `sys.executable`, Python/Torch/Triton versions. Readiness verifies
that same invocation and environment before each task. A real temporary symlinked-venv
regression proves the invocation retains its environment; another test rejects environment
drift before service requests. This is a correctness fix to the experiment driver.

GPU4 became idle while GPU6 remained occupied. A new v2 checkpoint was frozen on GPU4,
before any new proposals. All arms/seeds use the same device and local base service;
no process belonging to another task was stopped. The original v1 plan does not contribute
results to v2. Its original source/runtime fingerprint is expected to differ.

V2 completed one Softmax feedback job (three actual model proposals and independent
GPU checks), then stopped before the blind pair when GPU4 was occupied. A foreign user's
CUDA process started during that job; its start time and next readiness failure are
preserved in `results/multiseed-gpu4-pilot-qualification.json`. The raw evaluator recorded
correctness/promotion, but the promotion is not qualified as clean performance evidence.
Its raw checkpoint, three candidate files, six request bodies and frozen sources remain
separate. No outcome is deleted or rewritten.

The user selected GPU4 again. The new v3 plan adds the symmetric evaluator boundary guard
and bounded resource waiting described in `measurement-guard.md`. It starts a fresh full
schedule; v2 is an explicitly retained integration pilot, not pooled into the guarded
two-arm analysis. No failed/unknown job in v3 is selectively retried.

## Frozen guarded v3 identity

- Checkpoint: `/data/home/chenyizhou/kernel-agent/runs/multiseed-v3-gpu4-guarded/checkpoint.json`.
- Plan SHA: `c4c4a5f852179e6b59674540860de8e8905197389858bc2ce085950958a31864`.
- Existing user-owned Qwen base service: loopback port 8005, PID 3813970, generation GPUs2/3.
  Base ID `qwen38-27b-fp8`, no requested adapter. Directory alias is not an official revision;
  config is `qwen3_5`, upstream weight revision remains unverified.
- Evaluation GPU4, RTX4090; exact invocation
  `/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python`, Torch2.6.0+cu124/Triton3.2.0.
- Recorded OpenCode1.19.0, vLLM0.27.1+cu129, updated pinned Humanize public-source fingerprints.
  This is not the historical OpenCode1.18.32 runtime, and results are not pooled with it.
- Temperature0.2, per-seed request settings, gate-first search selection, immutable source
  snapshots and pre/post foreign-CUDA-PID checks. A single sequential scheduler owns the
  campaign lock. Each pending task can wait up to 10800 seconds for resource readiness,
  without reserving its budget or calling the model.

The source hashes and actual runtime values in the checkpoint are authoritative. Recreating
a service or moving the workspace requires a new plan. A manually launched, one-shot
supervisor records the runner exit code; it does not automatically retry failed/unknown jobs.

## Resume and inspect

```bash
cd /data/home/chenyizhou/kernel-agent
export PATH=/data/home/chenyizhou/.local/bin:$PATH
# Only run when the previous scheduler has exited; the ownership lock enforces this.
.venv/bin/python scripts/run_campaign.py --run --pid 3813970 --gpu-index 4 \
  --resource-wait-seconds 10800 --checkpoint runs/multiseed-v3-gpu4-guarded/checkpoint.json
# Read-only audit/analysis can run while a scheduler holds its lock.
.venv/bin/python scripts/summarize_campaign.py \
  --checkpoint runs/multiseed-v3-gpu4-guarded/checkpoint.json \
  --sources runs/multiseed-v3-gpu4-guarded/sources \
  --resource-state runs/multiseed-v3-gpu4-guarded/resource-wait.json \
  --output runs/multiseed-v3-gpu4-guarded/analysis.json
```

## Analysis boundaries

`scripts/summarize_campaign.py` checks hashes, schedule, allocations, original model
proposals, search selection and independent final verdicts before aggregating. It reports
task/arm counts, paired seed differences, first-source matches, duplicates, reported usage
and explicit failures. Pending/running jobs have no verdict and are never counted as failed.
Missing usage remains unknown. Failed and interrupted-unknown terminal jobs remain zero
successes in the conservative denominator.

Only after all 36 scheduled jobs terminate does it emit full scheduled-denominator rates
and an exploratory percentile interval by resampling the **six task clusters**, never the
108 dependent proposals. The small development-task set and unmatched first-source/system
contexts limit generalization and causal claims. A partial snapshot shows progress, not a
feedback-benefit estimate. Historical GPU scores stay in their own report section.
