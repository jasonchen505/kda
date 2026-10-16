# Kernel Agent Harness: project summary

This project turns a local coding agent into an auditable kernel-generation experiment.
It adapts KDA's contract → draft → plan → implementation workflow through Humanize and
OpenCode, then adds a downstream controller whose verdict comes from a separate GPU
evaluator. The work targets LLM research engineering and agent evaluation for systems.


## Latest iteration: operational evidence and transactional patches

The semantic review's null result motivated a repair architecture that derives finite
numeric relations and coordinate witnesses directly from trusted probes, routes actual
errors, and requests exact model-authored before/after edits against a source SHA. The
controller applies changes atomically and retains the base after rejection; it never
invents a kernel fix. Both arms get two repair calls with the same initial and evaluator.

The completed 22-block study has **44 final outcomes and 96 actual calls**, all confirmed.
Known-failure regression: patch **9/14**, rewriting **7/14**; new four-family/two-seed
validation: patch **6/8**, rewriting **7/8**. Family-equal differences are +25pp
(exploratory interval −30 to +70, five families) and −12.5pp (−37.5 to 0, four families).
The raw regression count difference is +14.3pp; it uses different weighting. Strata are
separate and new tasks are project-defined, not an external blind benchmark.

Patch repair rescues old SiLU and one transpose source, but fails on row-mean diagnosis
and a new L2 dtype repair. **17/42 patch transactions are rejected**; output representation
is a measured bottleneck. Keep ordinary generation/repair as default, with the patch
policy experimental. No post-outcome retries or hand fixes replace failures.
Three Qwen processes reach three concurrent ports; 107 GPU4 windows peak at one and
all 4847 older evidence files retain their hashes. CPU mechanisms pass **191 tests**.
See [results and failure analysis](operational-study-results.zh.md),
[reproduction](operational-reproduction.md), and the
[HTML report](../reports/kernel-agent-report.html#operational-study).

## Previous iteration: machine-grounded semantic repair

The preceding factorial's negative result motivated a separate reviewer and bounded AST
interpreter over complete asymmetric FP32 search probes. Verified expressions and factual
validation traces replace unverified review narratives for the grounded writer. A failed
check falls back without another request; kernels are never hand-edited.

The new 22-block, three-arm study resolved all 66 outcomes with 115 physical dispatches.
Known-failure regression: rewrite 7/14, critic 7/14, grounded 7/14.
Fresh generation: rewrite 7/8, critic 7/8, grounded 7/8.
Each arm has two extra model opportunities, with one review shared between the critic
variants; this is not an equal-kernel-count experiment. Strata are reported separately,
clustered by five/eight families, and are not a blind external benchmark. Keep ordinary
generation/repair as the default; semantic variants remain experimental. Mathematical
agreement does not prove layout or API correctness.

Three-service idle admission/EWMA scheduling and physical GPU4 proof remain active.
The CPU suite passes 176 tests on both pinned SDKs and a fresh locked installation.
See [results](semantic-study-results.zh.md), [failure analysis](semantic-failure-analysis.zh.md),
[reproduction](semantic-reproduction.md), and `results/semantic-study/`.


## Preceding generation and repair: paired factorial results (2026-10-15)

The new native local-chat backend compares two generation policies and two repair
policies on eight families × two seeds: **64/64 terminal arm outcomes**, **152/160
actual dispatched calls**. Each generation condition runs once per block; both repair
arms import the byte-identical initial source and search verdict, with two further
opportunities each. All four search selections commit before final-data access.

First-proposal search correctness: ordinary prompting **12/16**, contract checklist
**11/16**. Full contract+counterexample configuration is **13/16 final-correct**,
versus **15/16** for ordinary generation+repair. Of 8 fully observed shared-failure
pairs, ordinary repair rescues 5, counterexample repair 5. Failures remain visible;
these exploratory small-task outcomes do not establish stable generalization. Neither experimental policy earns promotion; ordinary generation/repair remains the recommendation. A degenerate repair interval is not an equivalence test.

Generation checks output coordinates, offsets, masking and casts in the existing draft/
plan. Repair receives synthetic search-only input/expected/actual witnesses, failed-source
memory and an optional correct anchor. A tokenizer drops only whole optional context
blocks; generated source is never edited. Both arms share the backend, so old OpenCode
results are not a causal latency comparison. Native requests have no auxiliary titles.

Dynamic discovery and latency-EWMA work allocation cover 3 GPU groups and
3 process identities; active blocks stay on one process. Completed model-request
intervals reach 3 concurrent ports; known independent GPU windows peak at
1. The CPU suite passes 159 cases on both pinned SDKs and a fresh locked install.
The row_mean/layer_norm additions were seen in an invalid device pre-run after policy freeze; they are not claimed as blind held-out tasks.
The invalid device pre-run preserves 48 dispatches (three unconfirmed), excluded from effects; the evaluator now verifies physical GPU UUID binding. The analysis-only eight-cluster correction changes no generated proposal or evaluator.
See [the reproduction guide](repair-reproduction.md), [the frozen protocol](repair-study-plan.md), [detailed results](repair-study-results.zh.md)
and `results/repair-study/`. Earlier releases below keep independent denominators.

## Earlier engineering evidence: a three-service pool (2026-10-15)

The harness completed all **24 scheduled jobs**: six development families × two seeds ×
two strategies. Final correctness is **20/24**, with **6/24** passing the observed-sharing
performance gate. Blind refinement is 10/12 correct and 3/12 promoted; feedback is
10/12 and 3/12. These are observed outcomes, not evidence of a feedback advantage.

The pool gives each explicitly authorized service an isolated process, workspace, proxy
and ledger. Every dispatch checks its HTTP declaration, idle metrics and exact tokenized
context (8192 common cap, 2048 output cap). Completed proposer intervals overlap across
three ports; 80 known independent GPU4 windows have peak overlap one under the shared
project lock. No end-to-end throughput speedup was measured. The dataset preserves
67 dispatched proposer requests, 66 committed replies and 66 admitted source files;
proposal opportunities, duplicate sources and auxiliary calls are counted separately.

Two borrowed endpoints disappeared mid-run. A separately frozen gateway restored their
logical entrances after declaration/GPU-mapping checks. All **15 committed jobs** were
reused; only **nine untouched jobs** proceeded. The 178 old request/admission files retain
their hashes and budget reservations never reset. Four transpose jobs had no eligible
source (three numerical failures and one interface failure), so no final evaluation ran
for those jobs. Three jobs stopped on interface failures; two still finalized a previously retained correct
source, while the transpose job had none. Two uncertain response receipts remain visible
and were not retried by
the quarantined job proxies. This does not promise exactly-once external calls.

Five strategy pairs cross a service restart. Pooled feedback contrasts and intervals are
therefore suppressed. Service versions differ; cross-root weight equality and process/cache
continuity are unverified. This release validates concurrent scheduling, durable evidence
and recovery coverage, not a continuous-condition causal study. Raw snapshots and the
handoff are retained separately. The CPU suite passes **134 cases** on both pinned SDKs
and a fresh locked installation; no model/GPU calls are part of those CPU counts.

See [the handoff](service-pool-handoff.md), [results](service-pool-results.zh.md),
`results/pool-study/` and the report's service-pool panel.

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
That release passed 119 CPU cases; its model/GPU counts remain separate.

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
