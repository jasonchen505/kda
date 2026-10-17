# Reproduce or audit budget-aware source-grounded repair

Read [the prospective protocol](grounded-budget-study-plan.md). This is a separate
six-pair diagnostic cohort after the [initial run](grounded-study-results.zh.md), with
new seeds20261221/20261222, twelve outcomes and a24-completion cap. There is no fresh
generation here, no replacement of initial failures and no pooling of efficacy scores.

## Public CPU and summary review

With the project's locked Python dependencies, the added fake-parent fixtures are
self-contained and do not require private remote parent kernels:

```bash
python -m pytest -q tests/unit/test_grounded_repair.py tests/unit/test_grounded_budget.py \
  tests/integration/test_grounded_study.py tests/integration/test_grounded_budget_study.py
python3 scripts/verify_grounded_summary.py results/grounded-summary.json
python3 scripts/verify_grounded_budget_summary.py results/grounded-budget-summary.json
```

The current full suite has219 passing CPU checks plus lint. Tests include forced context
overflow while the control still receives both opportunities, first-fit full/compact/baseline
views, exact source/prompt SHA bindings, same-slot patch fallback, final-source reuse and
forged prompts with re-signed request receipts. Fake model/GPU results are never agent scores.
Summary checks prove self-hash/count consistency, not unavailable raw records or CUDA numerics.

## Full remote audit

```bash
LAUNCH=/data/home/chenyizhou/kernel-agent-storage/bin/run.py
RUN=/data/home/chenyizhou/kernel-agent-storage/experiments/grounded-budget-v2
python3 "$LAUNCH" summarize_grounded_budget.py "$RUN"
# Only after a completed supervisor exit:
python3 "$LAUNCH" export_grounded_budget.py --directory "$RUN" --finalize
python3 "$LAUNCH" verify_grounded_budget_summary.py results/grounded-budget-summary.json
python3 "$LAUNCH" build_grounded_report.py
```

Full reconstruction checks frozen source/parent inputs, request/response identity,
source-derived facts and actual patch decoding, per-policy context selection receipts,
token counts used for dispatch, both selections before final access, exact final source,
physical GPU4 identity and serialized windows. It makes no new model/GPU calls. Context
records establish the recorded tokenizer measurements; they do not independently attest
a third-party tokenizer. Cached cross-root weight equality is unverified.

Raw requests, replies, parent records, candidates, measurements, tokenization records,
full reconstruction and source freezes stay in the remote experiment directory outside
Git/Mac. The immutable manifest records their identity. Its compact export retains only
counts, bounded diffs and remote locators. New data are not made public by a manifest hash.

## New executions

```bash
python3 "$LAUNCH" check_project.py
python3 "$LAUNCH" check_grounded_budget.py \
  --output /data/home/chenyizhou/kernel-agent-storage/checks/NEW-contexts.json
python3 "$LAUNCH" run_grounded_budget_study.py \
  --directory /data/home/chenyizhou/kernel-agent-storage/experiments/NEW-ID --background
```

Every new experiment requires an unused directory. Do not restart a completed supervisor,
replay unknown work, change an active generation source or fill a failed ledger with later
successes. Use the consolidated launcher/runtimes; old workspace paths are retired.

First-fit adaptation changes only auxiliary facts: full -> explicitly partial compact ->
the original indexed prompt. Complete parent source and common numerical evidence remain
mandatory. A refusal consumes that policy's opportunity, not its counterpart's. Unknown
dispatched calls and service identity failures retain block-wide stop/no-replay handling.
The separate full-source repair fallback still spends only the original second call slot.

For initial-run resources, `summarize_grounded_costs.py` is a post-run accounting/closing
attestation, separately retained under its raw manifest's analysis-tools. The initial
reporting-only exporter corrects opportunities vs received replies and constructed vs
dispatched packets; it never changes generation/evaluation/selection/outcomes. Use each
cohort's frozen code and receipts when interpreting historical attestations.
