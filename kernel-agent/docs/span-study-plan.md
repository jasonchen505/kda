# Prospective study: indexed patch transactions with a budget-neutral fallback

The research object is the kernel-writing agent's repair architecture. Previous exact-text
patch repair improved known-source regression9/14 vs7/14, but regressed on recent new tasks
6/8 vs7/8. Of42 patch replies,17 were rejected: twelve matching/no-op errors, four embedded
line breaks and one truncated JSON. Preserve that complete study unchanged.

## Hypothesis and policies

Reducing representation friction should increase usable repair opportunities. Compare:

- `rewrite`: two ordinary complete-source revisions.
- `operational_patch`: the unchanged exact-before-text patch policy, two attempts.
- `span_repair`: numbered source plus SHA-bound, inclusive1-based line-range edits;
  if the first indexed reply is rejected, use the SECOND existing slot for complete-source
  recovery with the same trusted evidence. No third call, parser-model retry or human fix.

All three share the same initial source and newly measured verdict, installed API preflight,
search evaluator, selector and final evaluator. Each receives two calls, output2048 each,
common8192 context, temperature0.2, no thinking/tools. Old/new patch arms share all diagnostic
routing, trusted search probes, computed obligations and API inventory. Only edit protocol,
source presentation and budgeted fallback differ. This tests that bundled interface policy;
it cannot isolate the fallback or numbered lines as an individual causal component.

Indexed patches need not copy old text. The source SHA and bounds bind each1-based range.
Apply all edits atomically to the original source, never sequentially; reject stale SHA,
noninteger/out-of-bounds/overlapping ranges, invalid source and wholly unchanged transactions.
Unchanged sub-edits may coexist with a real edit. Replacement strings are joined with LF
and a final LF; an embedded LF remains verbatim. No auto-indentation, escape guessing,
semantic rewrite, inferred missing token or human candidate repair is performed. This
prospective exact decoding rule differs from the old policy's stricter single-line arrays.

## Fixed schedule and independent strata

Regression: ALL seven admitted noninfrastructure initial failures from the original bank,
five task families, two NEW repair seeds20261201/20261202. Fourteen blocks,42 arm outcomes,
84 physical calls. Imported sources are re-evaluated; unexpected successes remain visible.

Fresh starts: four EXISTING recent families (column sum, L2 normalization, affine SiLU,
scaled transpose), two NEW generation seeds20261203/20261204. Eight shared initial calls,
eight blocks,24 outcomes,56 calls. These families participated in the last study and are
not new held-out or external tasks. Initial success and conditional rescue stay distinct.

Total22 blocks,66 outcomes, physical cap140 and reserved output286720 tokens. New seeds
are identifiers, not dates. No post-outcome adjustment of policies, order or denominators.
At every round, seed-randomize all three arms; both rounds use s+1/s+2 for every arm.
Commit ALL THREE selections before any final access. Independent final data veto selection;
never use them to choose a replacement source. Exact-source evaluations may be reused
within a block; source admission/API failure saves GPU work but still consumes the call.

Primary contrast: indexed/fallback vs old patch final correctness, separately within each
stratum. Secondary: indexed/fallback vs rewriting; old patch vs rewriting supplies context.
Average seeds/parents within family before exact family-cluster bootstrap (five/four
clusters). Intervals are exploratory, not equivalence tests or stable generalization claims.
Also report first-repair admission, all patch refusals, budgeted fallbacks, full-source
failures, conditional rescues, actual calls/output and independent GPU windows. Protocol
success is not numerical success. Failures/unknown requests remain in planned denominators.

## Runtime, storage and publication

Canonical code: `/data/home/chenyizhou/kda-kernel-agent-harness/kernel-agent`.
Invoke scripts ONLY through `/data/home/chenyizhou/kernel-agent-storage/bin/run.py`.
Runtime/controller and GPU paths are supplied by that launcher; old paths are retired.
Use all three authorized cached Qwen services when observed idle, dynamic discovery plus
EWMA allocation, one active block per process, fixed process identity inside each block.
GPU4 sharing is allowed; serialize project evaluations and require child physical UUID proof.
Do not edit/restart foreign services or use paid APIs. Unknown physical requests are not replayed.

Freeze sources/tasks/plan BEFORE first proposal in a NEW
`kernel-agent-storage/experiments/span-study-v1` directory. Canonical generation sources
must match the freeze before/after every block; don't edit them during execution. Existing
historical raw results/frozen sources remain byte-identical and retain their old paths.

ALL new raw requests, replies, kernels, measurements, source freezes and full ledgers stay
on the SERVER outside Git. Publish code, documentation, compact derived summaries, manifest
roots and the latest HTML. Mac must stay lean: no evidence restore, new raw-result copy,
experiment export tar or historical source snapshot expansion. Reports may show bounded
analysis diffs/statuses and remote evidence locators; no full reply/kernel corpus is copied.

Retain ordinary repair as default absent convincing fresh-start benefit. Even positive
outcomes are development-family evidence. Do not replace negative cases or run extra
unplanned retries to obtain a positive resume claim.
