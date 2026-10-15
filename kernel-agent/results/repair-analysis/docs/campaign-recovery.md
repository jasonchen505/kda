# Task-boundary recovery for kernel-agent experiments

The next useful portfolio contribution is durability of the **agent harness**, paired
with a better replication design. It is implemented in `src/kernel_agent/campaign.py`
and `scripts/run_campaign.py`; the original inner generation/repair policy is unchanged.

## Implemented and verified

One checkpoint owns an immutable schedule, source/runtime fingerprint, result ledger,
and SHA-linked events. A POSIX ownership lock prevents concurrent schedulers; fsync/atomic
rename protects checkpoint replacement. Started jobs reserve their full configured
duration/output-token allocation before the external call. Reservations cannot reset
across restarts. Completed results must agree with the trusted final decision and
selected source SHA. Plan drift, edited checkpoints and false headlines are rejected.

`scripts/validate_recovery.py` executed a **real SIGKILL** against a task-owned child:

| Task | At interruption | After restart | Worker invocations |
|---|---|---|---:|
| A | Completed, result committed | Reused, exact result SHA unchanged | 1 |
| B | External-call boundary, running persisted | `interrupted_unknown`; never replayed | 1 |
| C | Pending | Completed after restart | 1 |

A and C execute the actual Humanize kernel flow with fake model and evaluator drivers.
No new Qwen proposal or GPU timing is measured in this test. A second restart makes no
worker calls. All three jobs stay in the denominator; the allocation remains 2700 seconds /
60000 output tokens. The synthetic correctness flags are test fixtures, never added to
the real 3/3 correctness and 2/3 promotion campaign.

Raw transitions, event hashes, worker calls, assertions and frozen source are in
`results/campaign-recovery-validation.json` and `results/recovery-measured/`.
`scripts/verify_evidence.py` audits them offline alongside the historical GPU evidence.
Other mechanism tests cover resource blocking, lock contention, budget exhaustion,
source/plan drift, result binding and an injected atomic-replacement failure.

## Prepared, awaiting resources

The [prospective design](multiseed-study-plan.md) schedules six development task families,
three stochastic seeds, two feedback arms and three proposals each: 36 jobs / 108 opportunities.
The server prepared its exact plan and attempted the first readiness check. It stopped
with `awaiting_resources`, **zero started jobs and zero reserved allocations**; the
read-only record is `results/multiseed-preflight.json`. This is a preparation result,
not a completed generation experiment. Resume commands are in the plan document.

## Practical limits

Recovery is at task boundaries, not inside a candidate or native Humanize session.
An interrupted external request may finish without a committed result; exactly-once
model execution is not guaranteed. Unknown outcomes remain visible rather than receiving
invented results or selective retries. Allocations track configured scheduler budgets,
not an exact count of provider usage; missing interrupted/background usage is unknown.
Checksums are accidental-integrity checks, not signatures. Readiness does not reserve a
shared GPU against other users. Runtime changes require a fresh plan rather than mixing
two execution environments into one claimed comparison.

For interviews, this adds a concrete reliability story: checkpoint before side effects,
recover without double-counting, preserve failure denominators, and refuse invalid
experimental continuation. Empirical feedback gains still require the pending matched study.
