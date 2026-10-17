# Evidence-driven Kernel Agent Harness

A portfolio research engineering project about the **agent that writes kernels**. It
adapts KDA workflow references, integrates an early Humanize/OpenCode route, and studies
a downstream generation/repair controller using local Qwen. Kernels are assessment artifacts and are never hand-repaired.
Original KDA documentation remains in [README-kda-upstream.md](README-kda-upstream.md).

## Latest: RTX4090 staged-agent port

The new route runs real Humanize SDK planner/writer/reviewer roles through LiteLLM and
an audited local Qwen gateway. Separate draft and plan calls precede model-chosen hosted
tools; evidence-citing stage review controls advancement. Existing source memory, SHA
patches, actual-token packing, three-service scheduling and independent final acceptance
are ported into that route. All role turns use fresh sessions with host-managed memory.

See [results and migration limits](kernel-agent/docs/staged-results.zh.md),
[execution and reconstruction](kernel-agent/docs/staged-reproduction.md), and
[interactive HTML](kernel-agent/reports/kernel-agent-report.html#staged-study).
This is a bounded method port, not equivalent reproduction of the original Claude/Codex
plugin sessions, B200/FlashInfer contest workloads or Humanize upstream core. NCU hardware
counters are permission-denied; real CUDA activity is an explicitly limited fallback.
The [earlier workflow audit](kernel-agent/docs/workflow-comparison.zh.md) remains historical.

| Corrected port cohort: three development families / one NEW seed | Final correct | Calls |
|---|---:|---:|
| Fixed controller | 3/3 | 18 |
| Staged agent | 2/3 | 41 |
| Staged + transferred feedback/repair | 2/3 | 40 |

All nine outcomes terminate with99 confirmed calls and zero unknown requests. No arm earns
performance promotion; no real trajectory completes all three stages. Each staged arm
passes two research stages, then stops at refinement. Source facts enter15 actual requests,
but the model uses no patch actions. The port establishes executable integration, not an
efficacy gain. Ordinary repair stays default. The initial68-call cohort preserves interface
defects separately; it is not pooled or replaced. Common ceilings are not equal actual cost.

## Previous source-grounded results and the controller failure

Source-SHA-bound Python/JIT signatures, launch bindings and bounded store dependencies
are added to indexed/fallback repair. Same source admission, independent evaluator,
selector and two repair opportunities per arm.

| Separate cohort | Indexed control | Source-grounded | Interpretation |
|---|---:|---:|---|
| Eight development families, one fresh generation seed | 7/8 | 8/8 | One affine-SiLU discordance |
| Initial selected diagnostic parents | 0/6 | 3/6 | Two context failures cancelled controls; not clean efficacy |
| Budget/isolation follow-up: same parents, two NEW repair seeds | 3/6 | 5/6 | Separate diagnostic cohort; no fresh generation |

The initial run dispatched56/64 calls. Its preflight missed the longest actual patch
mode; policy context refusal incorrectly stopped both arms. Keep its failures and raw
labels. The follow-up tokenizes actual prompts and chooses full/compact/exact-baseline
views, preserving complete parent source/common evidence; a policy refusal cannot cancel
its counterpart. It dispatched24/24 calls. No extra repair slot or unknown replay.

Do not pool cohorts, claim stable generalization, or equate source facts with numerical
proof. Keep ordinary repair as default. The earlier indexed study separately achieved13/14
known-failure correctness versus9/14 rewriting and7/14 text patches; recent fresh starts
were6/8 in all three arms. Earlier negative/null studies remain unchanged.

## Review

- [KDA versus current workflow and interview boundaries](kernel-agent/docs/workflow-comparison.zh.md)
- [Resume and interview notes](kernel-agent/docs/project-summary.zh.md)
- [Initial results and failure analysis](kernel-agent/docs/grounded-study-results.zh.md)
- [Separate follow-up results](kernel-agent/docs/grounded-budget-results.zh.md)
- [Prospective follow-up protocol](kernel-agent/docs/grounded-budget-study-plan.md)
- [Reproduction and scope](kernel-agent/docs/grounded-budget-reproduction.md)
- [Interactive HTML](kernel-agent/reports/kernel-agent-report.html#workflow-comparison)
- [Canonical server workflow](kernel-agent/docs/server-workflow.md)

Core contributions: [source facts](kernel-agent/src/kernel_agent/grounded_repair.py),
[context adaptation](kernel-agent/src/kernel_agent/grounded_budget.py),
[controller](kernel-agent/scripts/run_grounded_budget_study.py),
[request/selection reconstruction](kernel-agent/scripts/summarize_grounded_budget.py),
[model pool](kernel-agent/src/kernel_agent/model_pool.py) and
[independent evaluator](kernel-agent/scripts/evaluate_candidate.py).

## Validation and storage

236 CPU mechanism/evidence checks and lint pass in the controller and isolated SDK runtimes, including self-contained fake-parent
integration and forced context-failure isolation. These are not model successes. Three
authorized Qwen services use idle admission/EWMA, pinning one process per pair. No paid
model API is used. GPU4 evaluations are cooperatively serialized with physical UUID proof;
sharing is allowed, and cached cross-root weight equality is unverified. Do not claim
isolated kernel performance or end-to-end model throughput gains.

New raw evidence stays on the owner's server, outside Git and Mac. This branch publishes
core code, documentation, compact derived summaries/bounded diffs and latest HTML. Public
summary checks validate self-hashes/counts, not absent raw records or CUDA numerics:

```bash
cd kernel-agent
python3 scripts/verify_staged_summary.py results/staged-summary.json
python3 scripts/verify_staged_summary.py results/staged-initial-summary.json
```

All server scripts run through `/data/home/chenyizhou/kernel-agent-storage/bin/run.py`;
canonical repository `/data/home/chenyizhou/kda-kernel-agent-harness`. Supporting runtimes,
upstreams/services/archives stay in sibling `kernel-agent-storage/`. Use
`build_staged_report.py` for the current report. Never restore the full historical corpus
on the lean Mac for routine publication. Do not restart completed/unknown experiment work.

This project does not run complete KernelBench, train a kernel model or claim an arbitrary
code sandbox. KDA/Humanize/OpenCode/Qwen/Triton are reused components with attribution.

## License

Original KDA files retain root notices. Downstream code uses [its license](kernel-agent/LICENSE);
Triton examples retain MIT [attribution](kernel-agent/THIRD_PARTY_NOTICES.md). Model weights
and Humanize/OpenCode source are not redistributed.
