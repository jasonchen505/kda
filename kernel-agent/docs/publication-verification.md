# Publication verification

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
