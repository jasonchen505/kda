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

The novel engineering work is the orchestration and evidence contract. KDA, Humanize,
OpenCode, Triton and Qwen are reused components; this project does not claim to implement
them or to reproduce the complete original KDA research pipeline.

## Evidence and result

| Experiment | Correct final outputs | Performance promotion | Interpretation |
|---|---:|---:|---|
| Cached 27B capacity pilot, four task families | 4/4 | 1/4 | Feasibility; Softmax mean speedup 1.235× |
| Equal-budget blind refinement, three tasks | 3/3 | 0/3 | One run per task, three proposal opportunities |
| Equal-budget measured feedback, same tasks | 3/3 | 0/3 | Does not establish a feedback advantage |
| Fresh repaired-controller integration, three tasks | 3/3 | 2/3 | SiLU 1.291×, RMSNorm 1.073×; changed service configuration |

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

## Limits and next research question

This is a resume-ready harness prototype, with working code and real evidence. It is not
a production sandbox or a demonstrated general-purpose kernel optimizer. The task set is
small, repeated families are used, seeds are not replicated, and the cached 27B weights'
upstream revision is unverified. Broader tasks/multiple seeds and profiler-assisted repair
are the next experiments; they are not presented as completed work.
