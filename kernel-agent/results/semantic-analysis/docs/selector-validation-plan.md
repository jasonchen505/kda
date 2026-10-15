# Fresh gate-first selector validation

Question: with the fixed candidate-selection rule active in the real local 27B agent,
does the end-to-end controller retain the search-best gate-eligible candidate through its
final check?

Run one new feedback-policy replication for RMSNorm, row_sum, and silu_mul, using three
proposal opportunities, reference-guided fresh sessions, line-array transport, local cached
27B Qwen, 4096 tokens per request, and a new proxy/model seed of 20261012. The current
controller's `gate_first` policy ranks correct candidates by existing promotion eligibility,
then by search geometric mean, with stable ties. Final evaluator shapes are never returned
to the model. Preserve each unique search-correct candidate's final evaluation for audit.

This is a fresh integration check of the selection policy, not a feedback-versus-blind
comparison: it has no blind arm. It repeats prior task families rather than calling them
new held-out tasks. Three runs cannot estimate stable success, cross-seed reliability, or
the causal benefit of evaluator feedback. A run with no candidate-ranking divergence still
checks the production path but does not independently exercise the exact failure branch;
the fake-flow regression test covers that branch deterministically.

Write only to `results/agent-selector-validation.json`; refuse an existing result. Freeze
the listed controller, evaluator, task, reference, model-manifest, proxy, runner, and plan
hashes before generation and verify them after completion. Keep this artifact distinct from
the frozen original strategy study and from its fixed-source recheck.

Resource amendment frozen before generation: GPUs 4/5 are occupied by other jobs. Reuse
the owner's idle loopback vLLM service on port 8016 (generation GPUs 2/3), requesting its
explicit base model `qwen38-27b-fp8`, never the registered LoRA model. Record PID, process
start ticks, argv, model listing, runtime version, adapter-routing implementation hashes,
and initial queue metrics. Do not modify or terminate this existing service. The task's
proxy alone maps `kernel-qwen-local-27b` to the recorded base ID and traces each request's
seed, upstream model, schema and tools. Evaluation stays on the idle GPU6.

This shared service uses a 65536 context, FP8 KV cache, prefix caching, LoRA-enabled
runtime and memory utilization 0.90; the original dedicated service used a 16384 context,
default KV dtype, no LoRA and utilization 0.85. Calling the base ID selects no adapter,
but these runtime differences preclude a matched before/after causal claim. Report this
as an integration replication. Requests are serial and capped; fail preflight if the
existing endpoint has active or queued work. Queue state during the campaign is not
guaranteed to remain idle, so elapsed time is not a controlled comparison.
