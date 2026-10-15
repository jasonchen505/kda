# Kernel Agent Harness: project summary

This project turns a local coding agent into an auditable kernel-generation experiment.
It adapts KDA's contract → draft → plan → implementation workflow through Humanize and
OpenCode, then adds a downstream controller whose verdict comes from a separate GPU
evaluator. The work targets LLM research engineering and agent evaluation for systems.

## Contribution

- **Execution contract:** a structured line-array proposal protocol; basic AST admission;
  controller-owned writes; real Triton kernels with fresh outputs and no Torch compute fallback.
- **Feedback and budgets:** typed numerical, compiler, protocol and harness failures;
  bounded context and fresh repair sessions; proposal/time/output budgets; duplicate handling.
- **Candidate lifecycle:** source and parent SHA bindings, preservation of correct candidates,
  gate-first search ranking, and a separate final veto on distinct shapes. Final audits never
  enter the prompt or change the search-selected source.
- **Evaluation discipline:** complete denominators, equal proposal opportunities for blind
  refinement versus measured feedback, versioned plans, negative results, frozen sources,
  request traces and an interactive offline report.
- **Maintenance:** compatibility with two pinned Humanize APIs, install metadata/lockfile,
  CPU mechanism checks and a standard-library published-evidence verifier.
- **Durable experiments:** atomic task-boundary checkpoints, single-owner scheduling,
  immutable source/runtime plans, persistent budget allocations and conservative unknown
  outcomes. A real SIGKILL/restart test reuses completed results and does not replay an
  uncertain call; it uses actual Humanize flows with fake model/GPU drivers.
- **Measurement conditions:** preserve the evaluation virtualenv invocation and keep the
  original uncontended-boundary studies frozen. The explicit sharing condition allows
  resident CUDA processes while recording memory, utilization, clock, power and PIDs;
  conditional ratios are separate from isolated speedups. Infrastructure faults stop
  repair, and task-level rates require every scheduled job to terminate.
- **Phase separation:** share the GPU for generation/search numerical validation, persist
  unedited eligible candidates in a timing queue, and resume pure measurements without
  repeating model calls. Search selection is persisted before final-data access; resource
  failures leave measurements queued rather than erasing generated work.

The engineering contribution is the orchestration and evidence contract. KDA, Humanize,
OpenCode, Triton and Qwen are reused components; this project does not claim to implement
them or to reproduce the complete original KDA research pipeline.

## Evidence and result

| Experiment | Correct final outputs | Performance promotion | Interpretation |
|---|---:|---:|---|
| Cached 27B capacity pilot, four task families | 4/4 | 1/4 | Feasibility; Softmax mean speedup 1.235× |
| Equal-budget blind refinement, three tasks | 3/3 | 0/3 | One run per task, three proposal opportunities |
| Equal-budget measured feedback, same tasks | 3/3 | 0/3 | Does not establish a feedback advantage |
| Fresh repaired-controller integration, three tasks | 3/3 | 2/3 | SiLU 1.291×, RMSNorm 1.073×; changed service configuration |
| Observed-sharing blind refinement, six families | 6/6 | 1/6 | One fresh seed, three opportunities each |
| Observed-sharing measured feedback, six families | 5/6 | 1/6 | One 900-second budget stop, no final verdict; retained denominator |

Speedups are geometric means over three BF16 shapes against `torch.compile`, on an
RTX 4090 evaluator. Promotion requires mean ≥1.05× and worst shape ≥0.90×, plus correctness.
Results are separate single-run studies, not pooled success-rate trials or end-to-end LLM speedups.
Official Triton examples assist generation; some references contain complete implementation fragments.

The most useful defect was in the controller itself: a mean-only selector discarded a
candidate that passed the promotion gate for one whose slightly higher mean hid a 0.773×
worst-shape regression. Gate-first ranking fixes the inconsistency. Three fresh evaluator
processes per frozen source confirmed 0/3 promotions for the discarded selection versus
3/3 for the eligible alternative. This is a fixed-source measurement contrast.

The later real-generation run exercises the repaired path, but its final selected sources
also win under the old rule after proposal three. Its 2/3 promotions cannot be attributed
to the selector alone. This distinction is part of the result, not an omitted caveat.

## Review in five minutes

1. Read `flows/kernel_loop.py` and `src/kernel_agent/protocol.py` for the actual control loop.
2. Run `python3 scripts/verify_evidence.py` for snapshot, proposal and decision integrity.
3. Open `reports/kernel-agent-report.html` for experiments, trajectory and the selector defect.
4. Read `tests/integration/test_strategy_study.py` for failure retention and no final-data leakage.
5. Check `docs/publication-verification.md` and `docs/humanize-update.md` for delivery status.
6. Read `docs/campaign-recovery.md` and run the CPU fault-injection regression for durability.

## Limits and next research question

This is a resume-ready harness prototype, with working code and real evidence. It is not
a production sandbox or a demonstrated general-purpose kernel optimizer. The task set is
small, families are reused, and the cached 27B weights' upstream revision is unverified.
The six-family × three-seed × two-arm study has 23/36 terminal jobs in the published
snapshot: 12 final-correct and 6 promoted, with 8 early resource stops and another final
resource veto. Thirteen jobs remain pending; incomplete, unequally observed arms cannot
establish a feedback benefit. Failures remain in the scheduled denominator.

That experiment exposed an architectural bottleneck. The new mode splits shared generation
and numerical validation from guarded performance evaluation. This improves resource use
without inventing speedups, changing old trials or replaying uncertain model calls.
Its separate two-job pilot tests the execution chain, not whether numerical-only feedback
outperforms timing feedback. A completed multi-seed comparison, held-out task families and
profiler-assisted repair remain research extensions.

The phase-split pilot completed six real proposals with three unique numerically eligible
sources while GPU4 had another resident CUDA process. All three unique checks passed;
the actual timing preflight left the queue pending. It proves shared generation/checking
and durable queue creation. A subsequent separately frozen shared measurement plan reused
those sources with zero new model calls: five independent evaluations, 2/2 final-correct and
2/2 passing the shared performance gate. Means 1.240× / 1.232× are conditional ratios against
`torch.compile`, not isolated speedups or new model-generation trials.

The observed-sharing replication has terminated all twelve jobs across six families and
two arms. Blind refinement is 6/6 final-correct and 1/6 promoted; feedback is 5/6 final-correct
and 1/6 promoted. Its failed transpose task exhausted the 900-second flow budget after three
correct search records; no final result is credited and no retry replaces it. All 36 candidate
files and 72 request bodies, including the uncommitted trace, remain auditable. Complete-job
SDK output totals are 19,200 tokens; the failed task lacks a complete total.

Promotion difference is zero; end-to-end correctness difference is −1/6 due to the budget
stop. Exploratory task-bootstrap intervals are [0, 0] and [−0.5, 0], respectively, not proof
of equivalence or stable causal effects. The five pairs with complete results had different
first sources. The old 23/36 study remains incomplete and is not patched with this condition.
CPU checks pass 119 cases on both SDKs; model/GPU counts are separate.

## Online feedback with durable finalization (2026-10-15)

The previous budget failure motivated a new `search_only` mode: retain online numerical
and timing feedback, commit the source pool, original search measurements and selected
source before final-data access, then import those measurements into a durable final
queue. Only the selected source is finalized; final evidence cannot trigger reselection.

A newly started cached 27B service on GPUs 5/6 generated six proposals across transpose
and Softmax, reporting 5,325 SDK output tokens. All six independent search checks passed.
A real SIGKILL after durable selection, before evaluator-child launch, left the final
evaluation in `running`. Recovery imported all six search measurements and ran two final
checks, with zero new model requests or search evaluations. Reopening completed queues
left request/checkpoint hashes unchanged. Final correctness is 2/2; Softmax passes the
conditional performance gate (mean 1.257×, worst 1.014×), transpose does not (0.979×, 0.957×).

An initial `--prepare` CLI exit-code defect stopped the supervisor before measurement.
After correcting it, a separate frozen finalization plan reused the unchanged committed
generation. Both the initial failure and sources are preserved. This engineering pilot
does not replace the old failed job, establish a feedback benefit, compare budget-failure
rates, or demonstrate recovery during an in-flight model request/GPU kernel.
