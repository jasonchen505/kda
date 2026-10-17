# Independent phase-interface correction cohort

The first `staged-study-v2` cohort is preserved. Its basic staged arms rejected all research
stages. Inspection found controller-interface defects: the reviewer was told all three
stages' conditions at once (affine-SiLU was rejected for lacking shape tuning despite correct
GPU search), and its evidence packet omitted read/research events, causing false "no tools"
claims. Fresh workspaces also returned unhelpful empty-source errors, encouraging retries.
Do not interpret that cohort as clean evidence that staged methods or feedback transfer fail.

This independent cohort uses NEW seed2026101702 on the same three development families,
three arms/nine outcomes, same24-call/6-submission/9-search/2-profile ceilings, same local
models and runtime, same GPU4 evaluator/selection/final isolation. No pooling or replacing
the first cohort. First-cohort results are not passed to model prompts. All earlier kernel
candidates remain inaccessible. This is a controller correction, not a new kernel strategy.

The corrected flow is additive (`staged_kernel_corrected.py`). Each planner/writer/reviewer
receives only the active phase's acceptance goal; research explicitly requires correctness,
not profiling, speedup or routing. The reviewer sees actual read/research/failed-tool events
as well as numerical/profile evidence. Missing-source and empty-research responses explain
the available next actions; they never manufacture code or successful measurements.
The original deterministic gates, complete-source preservation, budgeting, action limits,
same-model/fresh-context reviewer and final accept/veto rules remain in force.
Another first-cohort path received a truncated2048-token source response, after which the
typed SDK stopped the entire arm. Port the old harness's budgeted protocol recovery: only
a hash-bound fully received INVALID writer response consumes one action and can be followed
by a new remaining action. Do not reconstruct partial code, replay a request, retry unknown
responses, or extend24/6 limits. Planner/reviewer or external faults still retain their
recorded failure. This correction applies to fixed and staged writers in the new cohort.

Primary descriptive endpoints remain final correct/3, promoted/3, actual total tokens/calls,
stage acceptance and real tool traces. Also record phase-confusion and missing-event claims.
No comparison of old vs new seeds is a clean causal correction effect. Within the corrected
cohort, workflow/feedback contrasts remain exploratory bundled interventions, with just
three developed families/one seed and unequal actual cost under common ceilings.

Freeze executed controller/evaluator/flow sources, references, tests and this plan before
new model calls. Reporting-only tools are separate and bound by the closing raw manifest;
editing a report cannot change the running agent. Wait for the first supervisor to exit
before using the same services/GPU for this cohort. Preserve original source freezes
and raw data on the server; publish only code/docs/bounded summaries/latest HTML. API
unknowns are never replayed. Do not revise cohort failures in place or keep trying seeds
until scores improve. Full raw auditing uses the frozen original numerical/proposal
auditor plus a versioned reporting-only adapter if needed.
