# Prospective source-grounded repair comparison

Freeze before any model completion. The object is the kernel-writing agent. No candidate
is hand-edited and no raw experiment is copied to Mac or Git.

## Hypothesis and policy

Previous indexed repair has three final failures: duplicate JIT parameters after an edit,
wrong dtype-interface repair, and unresolved transposed store axes. A deterministic,
source-SHA-bound packet reports Python compilation, direct JIT signatures, launch bindings,
partial unresolved names and bounded store dependency slices. It adds explicit obligations
to repair declaration/use and value/pointer/mask axes together. It is partial evidence,
not a numerical proof, a reference implementation or automatic source correction.

Compare `indexed` (unchanged indexed/fallback policy) with `grounded` (that same policy plus
mandatory source facts and obligations before each repair with an admitted parent). Same
reply schema, API audit, source admission, evaluator, selector and final veto. Static facts
never admit, reject or rewrite a candidate. Both arms have exactly two model opportunities,
at most 2048 output tokens each, temperature .2, no thinking or tools, common context 8192.
The packet replaces neither numeric evidence nor evaluator diagnostics. Protocol failures
may spend the second existing slot on full-source fallback; no additional calls.

## Samples and fixed budget

- Diagnostic regression: ALL three final failures of `span_repair` in span-study-v1. Use the
  initial source, or the first admitted repair if the initial has no code. The selection
  rule and hashes are in `data/grounded-failure-bank.json`; complete prior attempts stay remote.
  Three task families × two new repair seeds 20261211/20261212 = six paired blocks, 24 calls.
- Primary fresh generation: four recent families (column_sum, l2_normalize, affine_silu,
  scale_transpose) plus transpose, row_mean, rmsnorm and silu_mul; one new shared generation
  seed 20261213. Eight paired blocks, each one shared initial + four repairs = 40 calls.
  All families have development history. No external/held-out benchmark claim.

Total 14 blocks, 28 final outcomes, cap 64 physical completions and 131072 reserved output
tokens. Do not conflate allocated tokens with observed usage. Blocks are shuffled with
fixed seed 20261210; interleave arm order per repair round. One assigned model process per
block; use all three authorized idle Qwen services with existing admission/EWMA scheduling.
Project GPU4 measurement is cooperatively serialized, physical UUID-bound; sharing allowed.
Services may differ in runtime version and cached weight equality is unverified.

## Outcomes and audit

Primary: final correct on eight fresh starts; compare grounded minus indexed paired by
family. Secondary: all six known-failure repairs, separately; average seeds within each of
three families before exploratory cluster intervals. Report all scheduled denominators,
conditional rescues, protocol/compile/numeric failures, real input/output token counts,
source-packet coverage and bound final selections. Small samples and a single fresh seed
cannot establish broad generalization or equivalence; do not promote the strategy merely
because mechanisms exist or a diagnostic case improves.

Freeze generation code, task contracts, plan, bank and parent input identities. Persist raw
requests/responses/candidates, admission receipts, prompt reconstruction and source packets
on the server. Check generation source hashes around each block. Commit all selections
before final data. Unknown calls are not replayed, completed outcomes are not replaced,
and no policy tuning or extra retry follows intermediate results. Reconstruct exact policy
prompts/replies/patches and GPU identity/windows from the full remote record; publish only
bounded derived summaries and latest HTML. Preserve all older studies and literal paths.
