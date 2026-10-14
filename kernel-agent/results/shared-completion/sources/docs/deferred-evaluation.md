# Shared generation and independently resumable performance evaluation

The integrated multi-seed experiment remains frozen in its original server workspace.
This improvement changes the harness execution architecture and runs separately. It does
not replace resource-interrupted trials or merge a different feedback policy into that study.

## Execution contract

1. Humanize/OpenCode requests three unedited local-model proposals per arm. Search numerical
   checks allow resident CUDA processes when at least 1024 MiB is free. The child checks FP32
   and BF16 values, input preservation and the output contract; it never calls a benchmark or
   examines final shapes. The free-memory threshold is a preflight heuristic, not a reservation.
2. The feedback arm receives those search numerical results; the blind arm receives the
   previous source only. Neither arm receives timing feedback. OOM and resource timeouts
   end the generation job as infrastructure problems rather than prompting code repair.
3. Unique numerically eligible source SHAs enter an atomic, single-writer timing queue.
   Each queued file must exactly equal the parsed model proposal. Generation checkpoints
   retain their budgets and never replay uncertain model calls.
4. A separate worker checks every queued source on search shapes through the guarded
   evaluator. Busy/contended observations remain in the ledger and return the item to the
   queue. This is a retry of pure evaluation, with no new model request. Unknown pure
   evaluations may be repeated after a crash; committed measurements are reused.
5. Once all search timings are resolved, the worker persists a gate-first selection before
   opening final shapes. The selected source receives the independent final veto. Final
   evidence cannot prompt repairs or select a different source.

This removes the need to own an empty evaluation device throughout generation. Timing still
uses the existing conservative before/after occupancy check. It is not continuous monitoring
or hardware isolation. Numerical passes under sharing do not imply a performance improvement.
The queued policy trades immediate timing feedback for scheduler throughput; its effect on
generation quality requires a separate comparison and is not established by the pilot.

## Engineering pilot

One previously used Softmax task, seed 20261024, two arms, three proposal opportunities each,
temperature 0.2, 4096 output tokens/request, 900 seconds / 20000 SDK output tokens per job.
Use the existing owned cached base-model service, no requested adapter or paid provider.
Generation runs in a separate workspace so the original 36-job campaign can resume unchanged.

```bash
cd /data/home/chenyizhou/kernel-agent-shared
PYTHONPATH=src:. /data/home/chenyizhou/kernel-agent/.venv/bin/python scripts/run_deferred.py \
  generate --pid 3813970 --gpu-index 4 --directory runs/deferred-softmax-v1
# Reuse the same frozen workspace and GPU Python environment; no model calls:
PYTHONPATH=src:. /data/home/chenyizhou/kernel-agent/.venv/bin/python scripts/run_deferred.py \
  measure --directory runs/deferred-softmax-v1
```

The measurement worker exits 2 while resources or generation results are pending. Resume
the same queue when a measurement window is available. A completed *generation* checkpoint
means the proposal phase ended, not that kernels have passed final checks or been promoted.
The report must show these statuses separately. This is a phase-separation integration pilot,
not a new held-out success rate or a matched performance-feedback experiment.
