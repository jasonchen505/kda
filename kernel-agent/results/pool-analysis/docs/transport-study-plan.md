# Proposal transport ablation, frozen before the complete campaign

Request tracing confirmed that Humanize's read-only OpenCode sessions offer `read` and
`webfetch` with `tool_choice=auto`, despite the proposal prompt forbidding tool use.
Both trace and actual replies retain this evidence. The existing custom proposer system
prompt is active; it was not silently ignored.

Compare three transports on the same four development tasks, controller mode
`reference_guided`, model snapshot, temperature/seed, three-turn cap, 4096-token cap,
reference examples, evaluator and final-shape checks:

1. `passthrough`: retain the original offered tools and automatic tool choice.
2. `tool-free`: remove tool declarations/choice/parallel settings for proposer requests.
3. `json-schema`: tool-free plus vLLM structured decoding for draft/plan/code string fields.

Unrelated OpenCode title requests remain unchanged. No extra kernel implementation hints
are added to a condition, and generated code is never manually repaired. The schema cannot
guarantee valid Python, real APIs or correct math. Truncated generations remain failures.
All conditions use the same local model and reference-assisted agent; no paid providers.

Run one session per task/condition, shuffle the 12-run order with seed 20261011. Report final
correctness/performance counts and proposal/admission/compile/runtime failures separately.
Actual usage and timing may differ; equal limits do not mean equal consumption. Unique
OpenCode directories appear in each system prompt, so requests are matched on task/settings,
not byte-identical. This small development ablation gives observed counts, not a population
success estimate or proof of generalization. Existing studies and the diagnostic vector-add
probe stay separate. Freeze source hashes until the complete campaign ends.

Structured-output interface: https://docs.vllm.ai/en/latest/features/structured_outputs/
