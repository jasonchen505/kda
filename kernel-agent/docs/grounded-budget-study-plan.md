# Prospective context-budget and arm-isolation follow-up

This is a NEW cohort after grounded-study-v1, not a replay or replacement. V1 completed
56 calls; two policy-caused pre-dispatch context refusals also cancelled the counterpart.
Its fresh-start7/8 vs8/8 result is retained; its0/6 vs3/6 regression contrast is not a clean
paired efficacy estimate. Frozen v1 records, source and original labels remain unchanged.

## Controller correction

Before each real request, tokenize the ACTUAL policy/mode/history. Source-grounded repair
tries full facts, then compact partial facts with explicit omissions, then the exact indexed
baseline prompt. Pick the first view fitting input+2048 <=8192. Each tokenization and
selected view/context SHA is persisted and reconstructible offline. This makes no model
completion call and edits no candidate. The independent evaluator/selector remain shared.

A context refusal is local to its policy opportunity. It must not cancel the other policy
or trigger extra model calls. External service identity failures and unknown dispatched
responses still stop the block; no cross-process migration or unknown replay. Ordinary
patch rejection retains its original second-slot full-source fallback. No new repair slot.

## Cohort and budget

Use ALL three same diagnostic parents from data/grounded-failure-bank.json, with NEW repair
seeds20261221/20261222. Six paired blocks, twelve final outcomes, two repair opportunities
per arm, cap24 physical model calls/49152 reserved output tokens. No fresh generation in
this cohort. Shuffle blocks with seed20261220, interleave policy order by repair round,
pin a model process per block and use all three authorized idle services. GPU4 allows
sharing; this project's measurements remain serialized and physically UUID-bound.

Primary engineering criterion: per-request context coverage, first-fit view binding,
no budget-triggered cancellation of the counterpart, no replay/additional model calls.
Report final correctness and conditional rescues separately on these known diagnostic
parents; cluster seeds within each of three task families. This is not external validation,
not another fresh-generation seed and not grounds to pool scores with v1. Keep every
scheduled outcome, include context failures in end-to-end denominators and explicitly
distinguish policy-caused failures from external infrastructure faults.

Freeze source/task/parent identities before inference. Do not tune after intermediate
results. Keep all raw tokenizations, requests, replies, candidates, source freezes and
measurements remote outside Git/Mac. Publish compact summaries and latest HTML only.
