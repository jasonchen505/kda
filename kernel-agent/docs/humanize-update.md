# Humanize upstream update, 2026-10-12

## Source and environment update

The server clone `/data/home/chenyizhou/humanize` now tracks the user fork at
[`d7f732e935100047d0faa5fc7c4d6b03ce350fc7`](https://github.com/jasonchen505/humanize/commit/d7f732e935100047d0faa5fc7c4d6b03ce350fc7).
No upstream implementation was edited.

The remote `main` was force-updated relative to the server's old `77e3d065` lineage;
there were 941 old-only and 1122 remote-only commits. A fast-forward was impossible.
Because the old clone was clean, its `main` was retained as
`codex/archive-humanize-77e3d065`, and a new tracking `main` was checked out at `origin/main`.
This aligns the clone to the published tree without mixing two generations of internal APIs.

The measured source remains in `/data/home/chenyizhou/humanize-measured-77e3d065`.
The original `.venv`'s editable import points there; it keeps the original dependency set.
The new clone uses `.venv-current`, installed with its frozen upstream lockfile and dev group.
Thus the old GPU results remain attributed to the old SDK and frozen controller snapshots.

## Applied compatibility change

[The public flow API](https://github.com/jasonchen505/humanize/blob/d7f732e935100047d0faa5fc7c4d6b03ce350fc7/src/hmz/flows/agents.py)
now takes `env` on `Agent.run`, rather than `Agent.spawn`. Unmodified downstream tests
on the new SDK gave **37 passed / 17 failed**, all flow failures at `spawn(env=...)`.
`src/kernel_agent/humanize_api.py` detects this public signature and binds the workspace
at the correct call. It does not inspect private drivers or change selection/evaluation.

The original 54 checks then passed on both SDKs. The publication adds three artifact
tests, giving **57 passed on each**. Versioned compatibility evidence and the separate
real local-model smoke check are linked in `publication-verification.md`.
Current flow files therefore differ from the measured snapshots; old raw studies and
source hashes were not rewritten. The offline verifier checks frozen source files.

## Changes worth using next

| Upstream capability | Relevance to this kernel harness | Current status |
|---|---|---|
| [LiteLLM direct-model harness](https://github.com/jasonchen505/humanize/blob/d7f732e935100047d0faa5fc7c4d6b03ce350fc7/src/hmz/coganchor/agents/litellm.py) and protocol gateways | A proposer has no filesystem tools; direct local chat could remove CLI startup/background requests and simplify structured output. Measure against the current OpenCode route with the same model/budget. | Useful next ablation; not switched into historical experiments. |
| [Ralph loop](https://github.com/jasonchen505/humanize/blob/d7f732e935100047d0faa5fc7c4d6b03ce350fc7/src/hmz/flows/builtin/ralph_loop/__init__.py) versus [stateful Ralph](https://github.com/jasonchen505/humanize/blob/d7f732e935100047d0faa5fc7c4d6b03ce350fc7/src/hmz/flows/builtin/stateful_ralph/__init__.py) | Ready examples of fresh versus persistent sessions; supports explicit context-policy experiments. | Current harness already uses fresh bounded repair sessions; no new benefit claimed. |
| Resumable `FlowState` and daemon hosting | Preserve a selected candidate, budget and evaluation ledger across a stopped frontend, rather than restarting an interrupted study. | Main extension to consider; this harness logs partial traces but does not yet implement full resumable execution. |
| [RLAR actor/reviewer flow](https://github.com/jasonchen505/humanize/blob/d7f732e935100047d0faa5fc7c4d6b03ce350fc7/src/hmz/flows/builtin/rlar/__init__.py) | Structured review can suggest repairs, while deterministic numerical/performance verdicts retain acceptance authority. | Design reference; no replacement of the trusted evaluator by an LLM judge. |
| Unified FakeTool/FakeCli/FakeCommand and web Exomyth timelines | Clear tool boundaries and session timelines can support richer evaluator integration and trajectory inspection. | Current fake-driver tests and HTML remain; richer tool/timeline integration is future work. |

For a portfolio release, the API port and environment separation are the necessary changes.
Adding all new backends or interfaces would enlarge scope without strengthening the present evidence.

## Upstream validation status

On 2026-10-12, GitHub reported completed successful `ci-ok`, static and Linux/macOS
tests on this commit in the user fork. The upstream organization also reported successful
checks for the same commit. Raw check names, conclusions and URLs are preserved in
`results/humanize-upstream-ci.json`.

This is verification of published CI, not a claim that this machine reran Humanize's whole
suite. The old 17 upstream failures in the historical report belong to the previous version.
