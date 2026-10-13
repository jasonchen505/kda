# Publication verification

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
