# Latest delivery: online feedback and durable finalization

The 2026-10-15 engineering pilot has two committed local-Qwen search jobs, six unedited
proposals, six imported search measurements, two independent final evaluations, and a
real SIGKILL before the final evaluator child started. Recovery added zero model calls
and zero search evaluations. Final correctness is 2/2 and conditional promotion is 1/2.
Completed-queue reopening preserved request and checkpoint hashes.

A preparation-only CLI exit-code error stopped the initial supervisor. Its failure,
initial queued plan and original sources are retained. A corrected finalization plan
completed the same committed generation with no model retry. The experiment does not
replace the old budget-failed task or estimate a feedback/budget-failure advantage.

The current CPU suite passes 119 tests on both pinned Humanize SDKs and a fresh locked
installation. Ruff, each shell script's syntax and offline artifact verification pass.
The new audit also checks that imported search observations cannot change even if a
checkpoint's outer hash is recomputed. Raw checks: `results/phased-checks.json`.
Frozen engineering evidence: `results/phased-recovery/`.

The new cached-base service uses loopback port 18118, generation GPUs 5/6 and evaluation
GPU4. It is a new runtime identity; the old port-8005 process is gone. Weight metadata and
shard inventory identify the cached asset; its upstream revision/full digests remain
unverified. No paid inference, requested adapter or redistributed model weights.

The entries below describe previous releases and keep their original counts.

---

# Publication verification

## Explicit observed-sharing condition, 2026-10-14

The current CPU suite passes **114 tests** on both pinned Humanize SDKs. A fresh installed
copy also passes all 114 tests, Ruff, shell syntax and offline artifact verification through
`bash scripts/check.sh`. Raw outputs are in `results/coexecution-checks.json` and
`results/coexecution-installed-checks.json`. These checks are separate from model/GPU results.

The added cases exercise actual child execution with resident CUDA PIDs, monitor-failure
rejection, minimum-free-memory readiness, immutable measurement-policy boundaries and the
search-selected source's final veto. The sharing wrapper records memory, utilization,
clock, power and process observations; this qualifies conditional measurements, not isolation.

An independently frozen completion plan reused the two committed Softmax generation jobs
and their unedited sources. It executed five GPU evaluations with **zero new model calls**:
**2/2 final-correct and 2/2 passing the shared performance gate**, means 1.240× / 1.232×.
`results/shared-completion/` contains the separate plan, measured source files, queue
checkpoints and exact candidate bytes. The original occupied-GPU rejection remains intact.
These outcomes are not new generation trials, a feedback-benefit estimate or isolated speedups.

The prospective paired replication uses six development families, two feedback arms and
one new seed, with twelve scheduled jobs and three proposal opportunities each. Its plan
was frozen before generation at
`43010765447ee3b0db97982f079ab3d4ab4836675cd808d58302a17dcaab9eb9`.
`results/shared-feedback-checkpoint.json`, `results/shared-feedback-measured/` and
`evidence/shared-feedback-requests/` preserve the actual status and committed evidence.
Final-data access does not enter repair prompts or replace the search-selected source.
The old 23/36 guarded study is preserved separately; no pending or failed outcome is rewritten.

All twelve scheduled jobs terminated: blind 6/6 final-correct and 1/6 promoted; feedback
5/6 final-correct and 1/6 promoted. The feedback transpose job exhausted its 900-second
flow budget after three correct search records. It has no committed final result, remains
in the denominator and was not retried. All 36 candidate files and 72 request bodies are
verified, including that failed task's uncommitted trace. This does not establish a feedback
benefit. Complete-job SDK totals exclude the failed task's missing overall usage.

## Phase-split harness and partial multi-seed evidence, 2026-10-13

The current CPU suite passes **107 tests** on both pinned Humanize environments, plus Ruff
and offline artifact checks. The new cases verify shared numerical execution, deferred
candidate eligibility, resource-blocked queue recovery, committed measurement reuse,
interrupted pure-evaluation retry, source binding and final-veto selection protection.
Those tests use SDK fakes / CPU children and are separate from model and GPU evidence.

A separate real local-Qwen pilot completed two generation jobs / six proposals on GPU4
while a foreign CUDA process remained resident (9757 MiB, observed utilization 100%).
Three unique sources passed 28 search numerical checks each; three duplicate proposals
reused the already recorded checks. The controller preserved all six unedited candidate
files, twelve request bodies and 3727 reported SDK output tokens. The timing worker's
real preflight refused occupied GPU4, retained the queue and exited 2. **No speedup or
final-correctness claim is made for these pending sources.** The queue resume path and
final veto are covered by CPU mechanism tests, not a new clean GPU performance run.

Frozen plan: `39545f36f9403e84c7f7d0d2d8226c3aedc582b82ddc5c0b52abbe0963cd1f3e`.
Artifacts: `results/deferred-generation-checkpoint.json`, `results/deferred-measured/`,
`results/deferred-timing/`, `evidence/deferred-requests/`, `results/deferred-checks.json`.
This numerical-feedback execution mode is not pooled with the immediate-performance
feedback experiment. The original server workspace keeps its measured v3 source intact.

The guarded multi-seed snapshot now contains 23/36 terminal jobs, 57 candidate files and
114 request bodies: 12 final-correct, 6 promoted, 8 early resource stops and another final
resource veto. Thirteen jobs remain pending. The original same-plan runner was resumed
in a finite supervisor on user-selected GPU4; no terminal/unknown job is replayed and no
other user's process is stopped. Report status is an offline snapshot, not a live monitor.

The HTML now leads with a five-minute project review and an interactive phase-separation
explanation. Raw checkpoint downloads preserve the original bytes, including floating-point
representations used in integrity hashes. Earlier measured sources and full-study scores
retain their original hashes and their separate result scopes.

## Guarded multi-seed follow-up, 2026-10-13

The current installed CPU suite passes **99 tests** on both pinned Humanize environments,
plus Ruff, shell syntax and offline integrity checks. Outputs are preserved separately in
`results/multiseed-checks.json`. New checks cover actual symlinked-venv binding, resource
waiting, before/after GPU occupancy vetoes, infrastructure-stop behavior and paired analysis.

A real local Qwen GPU4 pilot executed three proposals and passed Softmax final numerical
checks. A foreign CUDA process started during the job; next-task readiness stopped the run.
Original candidate files, six request bodies and the raw evaluator verdict are preserved,
with an explicit interference qualification. Its recorded promotion is not added to clean
performance results or pooled with historical campaigns.

The guarded full 36-job plan is independently frozen at
`c4c4a5f852179e6b59674540860de8e8905197389858bc2ce085950958a31864`.
The user chose GPU4; a one-shot supervisor waits up to 10800 seconds per pending job for
readiness. An actual busy-GPU wrapper preflight rejected the evaluator before CUDA execution;
recorded in `results/evaluation-guard-real-preflight.json`. The published checkpoint is an
offline progress snapshot, not a claim that the full comparison has completed. Existing
measured sources/raw GPU outcomes remain immutable. See `multiseed-execution.md`.

## Harness reliability follow-up, 2026-10-12 local date

The follow-up adds task-boundary campaign recovery and a prepared multi-seed design.
Installed frozen CPU checks pass **85 tests**, Ruff, shell syntax and the expanded offline
evidence audit. The same 85 tests pass on the archived original Humanize environment.
Raw command outputs are in `results/harness-upgrade-checks.json`.

A real SIGKILL/restart test passed with the actual Humanize flow and synthetic model/GPU
drivers; completed task A is reused, uncertain task B is not replayed, and task C runs once.
Its frozen evidence is separate in `results/campaign-recovery-validation.json`. Historical
GPU studies and snapshots retain their original hashes and 3/3 correct / 2/3 promotion.

The six-family × three-seed × two-arm plan has 36 pending jobs / 108 proposal opportunities.
The server prepared the plan and then blocked at readiness: GPU6 occupied (10037 MiB),
zero model jobs started and zero budget allocated. `results/multiseed-preflight.json`
preserves that exact plan and event. It is not a completed model experiment. No model
service was started, stopped or modified. See `docs/campaign-recovery.md` for boundaries.

The existing optional CI template remains inactive; no hosted harness CI run is claimed.

Release target: the user-owned `jasonchen505/kda` fork, independent branch
`codex/kernel-agent-harness`, with the downstream project in `kernel-agent/`.
The original KDA workflow files retain their scope and license; the branch landing README
links to the upstream README and the downstream harness.

## Checks completed on 2026-10-12

- Frozen dependency installation: `uv sync --extra dev --frozen` on Linux/Python 3.12.
- Installed project path: `uv run --frozen pytest -q` — **57 passed**.
- Current controller with the original Humanize environment — **57 passed**.
- Current controller with the updated Humanize environment — **57 passed**.
- `ruff check flows scripts src tests` and shell syntax checks — passed.
- Standard-library offline evidence verification — passed: 18 measured files,
  nine candidate files, nine proposer requests, nine separate background requests,
  3/3 correct final outputs and 2/3 promotions.
- Updated Humanize commit's published CI — successful `ci-ok`, static and Linux/macOS checks;
  recorded in `results/humanize-upstream-ci.json`.
- Real updated-SDK → OpenCode → cached Qwen3-4B local smoke — passed, with the exact
  requested JSON reply. It used a temporary task-owned service on initially idle GPU7,
  without a LoRA adapter, and was shut down afterward. This is an integration check,
  not a new kernel-generation experiment.

The measured generation campaign passed 54 checks; publication adds three integrity checks.
They verify real controller behavior against SDK fakes and published artifacts, not model
generation reliability. No historical generation result or frozen source was modified.

## Version boundary

Measured Humanize: `77e3d0658d9dd4c54cd9497e45940593bf6a43ff`.
Updated/installable Humanize: `d7f732e935100047d0faa5fc7c4d6b03ce350fc7`.
Only the public workspace-binding compatibility adapter changes the generation flow;
the selection/evaluation policy remains the same. The 27B campaign was not rerun under
the new SDK, and its 3/3 and 2/3 metrics remain attributed to the measured snapshots.

`scripts/check_updated_humanize.py` provides a small local SDK/CLI smoke entry, separate
from GPU kernel trials. The passed result is stored in
`results/humanize-current-smoke.json`. Model-service ownership and idle
queue preflights prevent silently substituting another user's service or an adapter.

The optional GitHub Actions template in `ci/github-actions.yml` runs the same installed CPU
tests, lint, shell syntax and offline integrity checks. It is deliberately a template rather
than an active workflow: the publishing credential can write repository contents but does
not have GitHub's `workflow` scope. The initial push with an active workflow was rejected;
the unpublished commit was corrected before publication. Hosted harness CI has not run.
The complete publication directory was installed independently on the server and passed
all 57 checks, lint, shell syntax and offline verification. This is local verification,
separate from the successful upstream Humanize CI.

## Review scope

The branch includes source code, lockfile, tests, plans, referenced MIT tutorials, raw
results, frozen measured and analysis source files, local-model request bodies and the
offline HTML. HTTP authorization headers, private keys, model weights and native personal
session stores are not part of the published evidence. Paths and runtime metadata describe
the measured server. The original unsanitized study JSON is kept intact so its hash remains
verifiable.
