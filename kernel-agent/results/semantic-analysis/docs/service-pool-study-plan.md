# Service-blocked paired feedback replication

Frozen before new generation requests, 2026-10-15 (America/Los_Angeles).

The user explicitly authorizes idle use of loopback 54575 (GPU0/1) and 60953 (GPU2/3),
alongside the owned 18118 (GPU5/6). Their existing processes are never modified or stopped.
Foreign services are accessed through HTTP only, without reading their environment,
private files, weights or other users' requests. Their HTTP declarations identify base
models with no requested adapter; equal directory names do not prove equal weights.

The study has six development families, seeds 20261027/20261028 and blind/feedback
refinement: 24 jobs, 72 proposal opportunities. Pairs stay on one service:

| Endpoint | Declared generation GPUs | Families | vLLM / native context |
|---|---|---|---|
| 18118 | 5/6 | vector_add, row_sum | 0.27.1 / 32768 |
| 54575 | 0/1 | softmax, rmsnorm | 0.25.1 / 8192 |
| 60953 | 2/3 | transpose, silu_mul | 0.25.1 / 8192 |

All requests use temperature 0.2, no thinking, a common 8192-token context ceiling,
at most 2048 output tokens per request, reference-guided line-array proposals and
three opportunities. Exact chat tokenization precedes dispatch; oversized requests
are rejected without truncating prompts or charging a generation. The largest prior
request checked before this study tokenized to 4619 on all three services.

Each endpoint has its own OS process, workspace, proxy and ledger. Own HTTP requests
are serialized; requests are dispatched only after running/waiting metrics are zero.
Observation cannot reserve another user's service or exclude arrivals immediately
after preflight. Uncertain upstream responses quarantine that job's proxy and are
not retried by it. Complete, rejected and unknown admissions remain recorded.

GPU4 evaluation allows sharing, records telemetry, and serializes this project's
evaluators with a shared OS lock. The evaluator child inherits the lock FD so a parent
crash cannot immediately release an active child's device reservation. Lock waiting
is bounded at 180 seconds; each flow evaluator call is bounded at 400 seconds.
Search timing feedback is online. Generation and search-only selection commit before
the independent final check, which can veto but cannot reselect or repair.

Every job reserves 900 seconds / 20000 SDK output tokens (21600 / 480000 total allocations).
These are conservative allocations, not measured GPU time or actual token totals.
Failed/unknown generation jobs retain their denominator and are never automatically
replayed. Pure final evaluations may resume. Busy services can leave untouched jobs
pending. Only complete denominators support full rates and descriptive contrasts.

Report contrasts within task/service and average seed outcomes per task before a
six-task exploratory bootstrap. Never treat 24 runs as 24 independent task clusters.
Do not pool older 4096-token, integrated-final or different-runtime studies. Two seeds,
reused tasks, reference assistance, uncontrolled device load and unverified cross-root
weight equivalence prohibit stable generalization, causal superiority or service-quality
claims. Count model proposals, auxiliary title requests, GPU evaluations and CPU tests
separately. Preserve every original candidate, request, admission and failed trace.
