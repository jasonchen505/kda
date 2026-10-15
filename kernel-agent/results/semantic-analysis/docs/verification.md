# Delivery verification

Verified on 2026-10-12. Both the original strategy campaign and the subsequent fresh
selector integration replication are complete.

## Latest selector replication

- `results/agent-selector-validation.json` is complete, SHA
  `ead614f0f5fbf25e8a743790539b68e4749e037c689bfdb845f72616e029543f`.
  Three scheduled feedback-agent replications each consumed three proposals: 3/3
  final-correct, 2/3 promoted. SiLU mean/worst 1.291/0.968x; RMSNorm 1.073/0.954x;
  row_sum 0.949/0.851x, not promoted. No failed result was dropped.
- Eighteen measured files matched current delivery and frozen snapshots both remotely
  and locally. Nine candidate files match the original model proposal, source SHA and
  evaluator bindings. All actual final choices match gate-first search ranking.
- SiLU diverges between the two reconstructed rankings after proposal two; proposal
  three removes that difference. Final selections match geomean in all three runs.
  This is not evidence of an incremental final-promotion gain from the selector.
- Nine real proposer requests and nine background requests are recorded separately.
  All request the local base ID and seed 20261012; proposer tools are stripped and the
  source-line array schema is present. Initial service queue was empty. Runtime differs
  from the original campaign; these repeat prior task families, with no blind arm.
- The complete `reproduce_existing_selector.sh` path passed its preflight, 54 tests,
  three real generation runs, report build and cleanup. Ruff and shell syntax passed.
  Its task-owned proxy exited; the pre-existing model services and other users' jobs
  were preserved. The dedicated-service wrapper was not rerun on the occupied GPUs.
- Original KDA/Humanize clones remain clean. No paid model API, new weights, training,
  requested runtime LoRA or hand correction of generated candidates was used.
- HTML and download checks for this final delivery are recorded separately in
  `results/selector-report-browser-audit.json`; phase-seven checks below are historical.

See `docs/selector-validation-results.md` and `results/selector-validation-delivery-audit.json`.

## Original strategy delivery snapshot

- Six original runs: blind and feedback each 3/3 final-correct, 0/3 final-promoted.
  RMSNorm is a development task; prospective row_sum and silu_mul are each correct
  under both policies. All six consume three proposal opportunities. These are single
  runs per task/policy, not a stable-success estimate or an external benchmark.
- All 16 original measured files matched both local and remote delivery before the
  later selector patch, and are archived in `results/study7-measured/`. All 18 remote
  candidate files matched the proposal SHA; all selected/final SHA bindings matched.
  Twelve unique source hashes within runs include comment changes: eleven are search-
  correct with final checks, one is search-rejected. Duplicates reuse measurement.
- Generated-source review confirms actual Triton launches, newly allocated output,
  and no PyTorch computation fallback. No candidate was manually corrected.
- First user prompts match across all three task pairs. System work directories and
  first sources differ. Request audit records 18 local proposer calls with no forwarded
  tools and line-array schema, plus 18 background calls; the latter are not trials.
- Selection audit found SiLU feedback round two search-eligible and final-promoted
  (mean 1.287x, worst 0.990x), discarded by the original mean-only selector. The selector
  now prioritizes the existing performance gate, then search mean, with stable ties.
  The original strategy runner explicitly retains geomean-only selection. Current
  default flow and frozen measured flow are different; hashes/patch are recorded.
- Gate-first is implemented and tested. Its search-only post-hoc counterfactual on the
  original candidate pool gives feedback 1/3 versus blind 0/3 promoted. This is not
  a fresh model campaign; original 0/6 headline promotions are unchanged. Do not claim
  statistically significant feedback gains or independently validated selector gains.
- Fixed-source recheck: each of the two SiLU feedback sources ran through three new GPU6
  evaluator processes with SHA verification. The old selected source is correct 3/3 and
  promoted 0/3 (0.926–0.932x mean); the discarded source is correct/promoted 3/3
  (1.289–1.299x mean). `results/silu-selector-recheck.json` contains all six raw runs.
  This confirms the recorded measurement contrast, not a fresh agent-generation result.
- Remote controller tests: 53 passed in 0.46s; Ruff passed after an import-order fix.
  `bash -n scripts/reproduce.sh` passed. The wrapper defaults to the strategy profile.
  Its entire service-start/generation path was not rerun after wrapper changes; the
  campaign used the component services directly. Old results are archived on rerun.
- HTML: six rows, separate 3/3 correctness and 0/3 promotion counts; new-task timing
  selector; three-round SiLU trajectory; first/last/best and gate-first audit; selection
  defect case plus fixed-source recheck; current 53-test validation and previous capacity
  study are verified.
- Complete browser JSON download equals all final embedded evidence and the original
  study, including large integer model-file timestamps. Raw JSON is downloaded directly
  to avoid JavaScript precision loss. Request audit and prior studies remain separate.
- No report console errors. Page width fits 1280px and 390px; tables scroll within their
  wrappers. Temporary viewport overrides were reset. The report tab is retained.
  `reports/report-preview.png` shows the selector case. The phase-seven HTML is saved.
- Original KDA and Humanize clone `git status --short` outputs are empty. Dedicated
  model/proxy PIDs 3244986/3248482 and workers 3245916/3245917 have exited; existing
  services 3097364/3097365 and unrelated later GPU jobs are retained. No paid API,
  new weights, model training or runtime LoRA was used for this campaign. We did not
  continuously record all other-user GPU activity, so do not claim complete isolation.

Historical evidence stays separate: 8 initial, 8 API, 4 reference, 12 transport,
8 representation and 4 cached-27B capacity runs. Capacity pilot: 4/4 correct, 1/4
promoted, Softmax mean 1.235x. Its measured files remain in `results/study6-measured/`.
Diagnostic probes and selection audits are not pooled into model-success denominators.
Upstream Humanize's earlier full suite still has 17 failures. Model weights' upstream
revision/full hashes remain unverified; no claim of official Qwen3.8 provenance is made.

Delivery checks establish the recorded prototype and evidence, not broad generalization.
See `results/strategy-delivery-audit.json`, `results/agent-strategy-study.json`,
`results/agent-strategy-summary.json` and `results/selector-fix.patch`.
