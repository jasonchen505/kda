# Evidence-driven Kernel Agent Harness

A portfolio research engineering project about the **agent that writes kernels**.
It adapts KDA's contract/draft/plan workflow through Humanize/OpenCode and a local Qwen
backend, then studies the downstream generation/repair controller. Candidate kernels are
assessment artifacts; none are hand-repaired to obtain model scores. Original KDA docs
and workflows remain in [README-kda-upstream.md](README-kda-upstream.md) and their directories.

## Latest controlled result

An indexed source patch binds 1-based line ranges to a source SHA. On first-patch rejection,
the agent spends its second existing slot on complete-source recovery. Both controls and
the new policy share the same initial source, API preflight, independent GPU evaluator
and selector; each arm has two repair calls capped at 2048 output tokens.

| Independent stratum | Two rewrites | Old text patches | Indexed patch + fallback |
|---|---:|---:|---:|
| Known failed sources: 7 parents × 2 seeds, 5 families | 9/14 | 7/14 | **13/14** |
| Fresh starts on 4 existing recent families × 2 seeds | 6/8 | 6/8 | 6/8 |

All **22 paired blocks / 66 final outcomes / 140 physical calls** completed, with no unknown
responses or infrastructure failures. First-patch application is 13/21 vs 18/21; total
refusals are 17/43 vs 7/40. Fallbacks use the existing budget, so patch-attempt denominators
differ. Protocol acceptance is not compilation or numerical correctness.

The primary family-equal regression difference vs old patches is +35 pp (exploratory 95%
interval [+10, +60], five families); vs rewriting +30 pp [0, +70]. Raw-count effects have
different weights. Recent-family fresh starts show no net gain. These are small development
sets, not an external blind benchmark or a stable generalization claim. Keep ordinary
repair as default; indexed/fallback remains an experimental option.

## Review the project

- [Chinese resume and interview summary](kernel-agent/docs/project-summary.zh.md)
- [English project summary](kernel-agent/docs/project-summary.md)
- [Latest results, failure analysis and limitations](kernel-agent/docs/span-study-results.zh.md)
- [Prospective design](kernel-agent/docs/span-study-plan.md) and [reproduction](kernel-agent/docs/span-reproduction.md)
- [Interactive HTML](kernel-agent/reports/kernel-agent-report.html#span-study)
- [Canonical server / lean local workflow](kernel-agent/docs/server-workflow.md)
- [Implementation](kernel-agent/README.md), [Humanize compatibility](kernel-agent/docs/humanize-update.md)

The contributions are budgeted repair policies, exact candidate lineage, independent
search/final evaluation, durable state and failure handling, and three-service scheduling.
KDA, Humanize/OpenCode, Qwen and Triton are reused components, not original implementations.

Core entry points are [indexed repair](kernel-agent/src/kernel_agent/span_repair.py),
[the study controller](kernel-agent/scripts/run_span_study.py),
[raw reconstruction](kernel-agent/scripts/summarize_span.py),
[the local model pool](kernel-agent/src/kernel_agent/model_pool.py) and
[the independent evaluator](kernel-agent/scripts/evaluate_candidate.py).

## Run and audit

Current canonical code is `/data/home/chenyizhou/kda-kernel-agent-harness/kernel-agent`.
Run scripts through `/data/home/chenyizhou/kernel-agent-storage/bin/run.py`:

```bash
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py check_project.py
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py verify_evidence.py
```

The 205-test CPU suite and lint pass. Fake HTTP/GPU fixtures test mechanisms, not agent
success rates. Three authorized Qwen services use idle admission and latency EWMA, keeping
one process per paired block.156 independent GPU4 windows peak at one, with child physical
UUID proof. GPU sharing is allowed; performance ratios are conditional, not isolated or
end-to-end LLM speedups. Cross-root cached weight equality is unverified. No paid API is used.

**New raw experiment data stay on the owner's server, outside Git and Mac.** This branch
publishes code, documentation, compact derived summaries, bounded diff excerpts and the
latest HTML. Raw manifest identity and locations are in the summary. Public summary
verification checks self-hashes/counts; complete raw reconstruction requires server access:

```bash
cd kernel-agent
python3 scripts/verify_span_summary.py results/span-summary.json
```

Do not restore the full historical corpus on the lean Mac as a routine publishing step.
Historical frozen results remain unchanged on the server/in Git history. Previous negative
and null studies, selector fixes, endpoint handoffs and real SIGKILL recovery keep their own
denominators. This project does not run the complete KernelBench or reproduce all KDA
research results. Further work should test structural/semantic repair on additional families.

## License

Original KDA files retain the root license and notices. Downstream code uses
[kernel-agent/LICENSE](kernel-agent/LICENSE); referenced Triton tutorials retain MIT
attribution in [the notices](kernel-agent/THIRD_PARTY_NOTICES.md). Model weights and
Humanize/OpenCode source are not redistributed.
