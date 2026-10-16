# Measurement qualification before the guarded multi-seed campaign

The previous GPU4 campaign (v2) stopped after one completed Softmax feedback job. A
foreign user's CUDA process appeared during the job, and the next readiness check
recorded GPU4 occupied. Raw outputs are preserved separately; its recorded promotion
cannot establish clean GPU performance. Do not pool that partial pilot into a new study.
The earlier zero-start GPU6 v1 plan is preserved too.

## Guarded experiment rule, fixed before new proposals

The guarded campaign still fixes six development families, seeds20261021/22/23, both
blind/feedback arms, temperature0.2, three proposal opportunities, gate-first selection,
and the same cached base model. GPU4 is the user-selected evaluation device. This is a
new source/runtime plan rather than a modification of a previously measured checkpoint.
Its scheduler waits up to 10800 seconds per pending task for readiness, checking every
30 seconds. Waiting does not reserve a job budget or make a model call. Source/runtime
drift fails immediately, rather than being retried until it disappears. Failed/unknown
model jobs are never replayed. This is a single finite campaign process, not a recurring job.

Before each scheduled job the runner checks the model queue and idle evaluation GPU.
Each search/final evaluator additionally queries CUDA PIDs immediately **before and
after** its child process, using `nvidia-smi -i GPU --query-compute-apps=pid`.

- Foreign PID before evaluation: `resource_busy`, no evaluator process started.
- Foreign PID after evaluation: `resource_contended`, original child output preserved
  under `discarded_evaluation`; its timings cannot satisfy the promotion gate.
- Query/monitor failure: `resource_monitor_error`; absence of a valid occupancy check
  cannot be treated as permission to promote a candidate.
- Normal evaluation: retains the GPU verdict and the two boundary observations.

Search resource failures stop the inner repair loop; no new code is requested in
response to an infrastructure failure. Any retained valid candidate still receives
the guarded independent final check, which can veto it. These jobs are not selectively
replayed. Failed/unknown outcomes retain their original denominator and budget allocation.

Boundary checks cannot detect an interfering process that starts AND ends entirely
between observations. They are not continuous NVML monitoring, reservation against other
users, or a proof of exclusive hardware. Unobserved interference remains a limitation.
The two-arm comparison applies this exact guard symmetrically; CPU guard tests are
mechanism tests, not new model successes.
