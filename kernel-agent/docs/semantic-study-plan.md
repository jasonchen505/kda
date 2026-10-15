# Grounded semantic review: prospective repair study

Freeze this protocol, source, parent bank and model-call ceiling before generation.
The preceding 2x2 study remains unchanged: ordinary generation/repair 15/16 final-correct,
contract+diagnostic repair 13/16, and both repairs rescued 5/8 fully observed failures.
The new study tests a different architecture; it does not replace those negative results.

## Mechanism and question

Observed failures include missing SiLU factors, wrong output addressing, dtype/API errors
and invented precision explanations. The previous checklist did not mechanically test
the model's stated mathematics or enforce correspondence between diagnosis and code.
Inferring an attention shift from those trajectories is a hypothesis, not a measured
internal-model mechanism.

A separate reviewer now emits a reference expression, candidate expression and narrative.
A bounded AST interpreter checks expressions against two trusted, asymmetric FP32 search
probes with all inputs and complete expected/actual values. Only supported arithmetic,
last-axis reductions, broadcasting and transposition are admitted. No Python eval,
attribute access, tools, model-authored reference outputs or manually patched kernels.
The grounded writer receives only verified expressions, validation results and trusted
evidence. Unverified free-text hypotheses/plans are withheld. Rejected expressions cause
a factual fallback, not another model call. FP32 probe agreement is not symbolic
equivalence, BF16 proof, layout proof or full-kernel verification.

## Three paired arms

Every block shares exactly one initial source and new search evaluation. After that:

1. `rewrite`: two ordinary full-kernel revisions.
2. `critic`: one semantic review and one full-kernel revision using its unverified output.
3. `grounded`: the exact same review and one revision using machine-validated context.

Each arm has two additional model opportunities, 2048 output tokens each. The review
is executed once and shared between the last two arms: five physical repair calls per
block, rather than six. Critics get only one new kernel proposal, while rewrite gets
two; that allocation tradeoff is part of this architecture comparison. It is not an
equal-number-of-kernels experiment. The ablation isolates grounding conditional on
that shared reviewer, without claiming every component is independently identified.

First-round review/rewrite order and second-round arm order are seeded and randomized.
The reviewer and first rewrite use seed+1; all second-round writers use seed+2. No arm
receives another arm's candidates or later evaluation. All three search selections
commit before any independent final check. Final results only accept or veto the
selected source, with exact-source search/final caching recorded within each block.

## Strata, denominators and budgets

Regression uses **all seven admitted non-infrastructure failed initial sources** from
the preceding study. Selection depends only on initial search failure and source
admission, not repair/final outcomes. One protocol failure has no admitted source and
one infrastructure failure is excluded from the source bank by this declared rule.
Seven parents span five task families. Two new repair seeds (20261111/20261112) give
14 blocks and 42 terminal arm outcomes. Every imported source is rechecked under the
new evaluator before repair; unexpected correct or infrastructure outcomes remain
visible and are not silently dropped. This is a known-failure regression set.

Fresh-start validation uses all eight existing task families with seed 20261113:
eight blocks, eight shared plain initial generations and 24 terminal arm outcomes.
It uses new generation draws, not unseen task families or an external benchmark.
Results remain separate from the regression stratum.

Total: **22 blocks, 66 terminal arm outcomes, at most 118 physical model dispatches**:
14×5 regression repair calls plus 8×6 fresh initial+repair calls. Reserved output
allocation is 241664 tokens, not measured consumption. Each arm's effective budget
remains two repair calls; imported generation costs belong to the previous study.
Per-block caps and process identity are durable before dispatch. A model-phase
admission deadline is 1800 seconds; total supervisor admission deadline is 10800.
Infrastructure-stopped and unknown calls retain their original denominators, and
uncertain calls are never replayed. No additional attempts to replace negative results.

## Runtime and evaluation

All arms use local authorized Qwen FP8 services, common 8192 context, temperature .2,
no thinking, no tools and 2048 output ceiling. Optional reference cards can be removed
whole using actual tokenization; source, contract and active evidence cannot be sliced.
Dynamic idle admission and latency EWMA distribute blocks over GPU0/1, 2/3 and 5/6.
A running block stays on one process fingerprint; newly ready services may take only
untouched blocks. Roots/versions are recorded, with no cross-root weight equality claim.

GPU4 allows sharing and cooperatively serializes this project's workers. Physical UUID
binding and child proof remain mandatory. Numerical/contract failures, proposal errors,
model infrastructure failures and monitoring failures are separately recorded. The two
new probe shapes ([5]/[7] or [2,5]/[3,4]) are search-only and contain no final random data.
The original full correctness cases, BF16 tolerances, performance gates and final
shape/random-seed checks remain unchanged. Every arm gets the same evaluator/probe work;
ordinary rewrite does not receive the semantic probe fields in its feedback.

## Analysis and policy decision

Primary: final correctness on all 14 regression blocks per arm; report reproducing
initial failures and fully observed conditional rescue separately. Secondary: final
correctness on eight fresh blocks per arm, critic expression admission, protocol and
repetition failures, model usage and independent GPU work. Keep all negative outcomes.

Compute paired effects separately per stratum, average seeds/parents within each task
family, and use exact weighted task-cluster bootstrap intervals: five regression families
and eight fresh families. Parent/source repeats are not independent task clusters.
These are small, exploratory intervals, not equivalence tests or confirmation of broad
generalization. Fully observed subsets may describe infrastructure sensitivity but do
not replace the predeclared denominators. Performance is secondary and conditional on
observed sharing, not isolated or end-to-end LLM acceleration.

No policy edits after results. A positive result may justify further validation;
null/negative evidence means retaining the previous recommendation. Reproduction,
semantic arithmetic traces, exact requests/responses, source lineage and final vetoes
must be reviewable in HTML and the independent portfolio branch.
