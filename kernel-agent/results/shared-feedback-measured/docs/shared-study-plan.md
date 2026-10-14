# Observed-sharing continuation and prospective feedback replication

The user authorizes GPU4 sharing. Resident memory or a CUDA PID does not prevent execution
when at least 1024 MiB is free. The trusted evaluator retains its numerical contract, the
three search timing shapes, randomized baseline/candidate order, three CUDA-graph repeats,
gate-first selection and independent final veto. No generated source is hand-corrected.

Each evaluator records memory, utilization, SM clock, power and CUDA PIDs before, during
(nominally every 0.5 seconds) and after execution. Overall utilization includes this
evaluator's work; it does not identify another user's load or establish isolation. Sampling
also has overhead. Ratios answer a conditional comparison in the observed shared environment,
not a claim about standalone speedups. Numerical failures and resource/monitor faults remain
distinct. OOM or timeout ends a model job without pretending it is code-repair evidence.

## Complete committed generation first

The two Softmax generation jobs from the deferred pilot have six proposals and three unique
numerically eligible sources. A new measurement plan reuses their exact committed sources,
with **zero new model calls**. The original guarded queues remain unchanged. The new queue
plan explicitly says `shared_observed`; it cannot turn guarded evidence into shared evidence
or vice versa. Search timings select once, selection is persisted, and final data only vetoes.

## Prospective shared feedback study

Six previously developed families: vector_add, softmax, rmsnorm, transpose, row_sum, silu_mul.
One fresh seed 20261025 × blind/feedback = **12 jobs / 36 proposal opportunities**. This is a
small, complete first replication under the new execution condition, not held-out evaluation.
Fixed cached base model, no requested adapter, temperature 0.2, 4096 tokens/request, three
proposals per job; 900 seconds / 20000 SDK output tokens per job. Schedule is deterministically
randomized before any proposals using the existing planner. Same source/runtime applies to
both arms; feedback includes actual search numerical and shared timing results. Final shapes
are never fed back or used to reselect. Unknown/failed jobs keep their scheduled denominator.

No old job is retried or replaced. The original 23/36 guarded experiment remains an incomplete
historical experiment; shared measurements are analyzed separately. Full rates and exploratory
task-clustered differences appear after all twelve jobs terminate. One seed and six reused
families cannot establish generalization or stable model success rates; uncontrolled background
load and differing first sources limit causal attribution to feedback.

Run in the new frozen workspace, preserving both previous server workspaces:

```bash
cd /data/home/chenyizhou/kernel-agent-coexecution
export PATH=/data/home/chenyizhou/.local/bin:$PATH
PYTHONPATH=src:. /data/home/chenyizhou/kernel-agent/.venv/bin/python scripts/finish_deferred.py \
  --generation /data/home/chenyizhou/kernel-agent-shared/runs/deferred-softmax-v1/generation.json \
  --directory runs/shared-completion-v1
PYTHONPATH=src:. /data/home/chenyizhou/kernel-agent/.venv/bin/python scripts/run_shared_campaign.py \
  --pid 3813970 --gpu-index 4 --checkpoint runs/shared-feedback-v1/checkpoint.json
```

Only model-service busy readiness or insufficient free memory can cause finite preflight
waiting (120 seconds per pending job). Presence of another CUDA process is recorded, not a
readiness veto. Existing task-boundary locking, immutable plans and budget reservations apply.
