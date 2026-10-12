# Evidence-driven Kernel Agent Harness

A job-portfolio prototype that reproduces the KDA task workflow through a real Humanize
coding-agent session and improves the downstream controller. The research object is the
agent that generates kernels, not manually authored operator optimization.

**Portfolio scope:** a working research-engineering harness, with local inference,
independent GPU verdicts, bounded repair, search-only selection, frozen evidence and an
interactive report. The latest three-task integration campaign is **3/3 final-correct,
2/3 promoted**; this is a small reference-assisted replication, not a general success rate.

Start with [the project summary](docs/project-summary.md),
[Chinese resume and interview notes](docs/project-summary.zh.md), and
[the offline HTML report](reports/kernel-agent-report.html). Generated candidates are
never hand-corrected. Negative experiments and the original selector defect remain visible.

```text
Task contract → local Qwen proposal → protocol / AST admission → independent GPU process
                         ↑                                         ↓
                bounded typed feedback ← search verdict → gate-first retained candidate
                                                                  ↓
                                                  separate final shapes → accept / veto
```

## Quick review without a GPU

```bash
python3 scripts/verify_evidence.py
python3 -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765/reports/kernel-agent-report.html
```

The standard-library evidence verifier checks 18 frozen measured files, nine proposed
candidate files, the recorded selection/final verdicts and nine local proposer requests.
It verifies recorded artifacts; it does not execute the model or recompute GPU timings.

For controller development, use Python 3.12+ in this directory:

```bash
uv sync --extra dev --frozen
uv run pytest
uv run ruff check flows scripts src tests
```

`pyproject.toml` and `uv.lock` pin the updated Humanize source. GPU evaluation remains in
a separate Linux environment with Torch 2.6.0+cu124 and Triton 3.2.0; vLLM 0.27.1 and
OpenCode 1.18.32 are separate executables. No model weights are included.

## Background

KDA is a workflow/prompt reference at `ef6ce617693ef0782b3ecb9f37e39bbf10226a90`.
Measured experiments used Humanize at `77e3d0658d9dd4c54cd9497e45940593bf6a43ff`,
driving OpenCode 1.18.32. The server clone is now updated to
`d7f732e935100047d0faa5fc7c4d6b03ce350fc7`; the small public-API adapter supports both.
The existing local Qwen3-4B-Instruct-2507 snapshot supplies the small-model development
studies. A separate cached 27B FP8 Qwen capacity pilot uses the latest controller.
No paid provider, requested LoRA adapter, model training or upstream source-code edit is required.
Both source clones retain their own licenses and notices. The prompt adaptation is
attributed to KDA's CC BY 4.0 documentation; this project does not redistribute weights.

## Install

This server already contains the frozen environments. The measured Humanize environment
remains pinned to the archived source worktree; the updated clone uses `.venv-current`.
The sibling `kda-humanize-portfolio` environment supplies Torch 2.6.0 and Triton
3.2.0. The local vLLM runtime is 0.27.1. The 4B model revision is
`cdbee75f17c01a7cc42f958dc650907174af0554`. GPU 5 serves the 4B model; GPU 6 evaluates kernels.
The 27B service uses GPUs 4/5 after the 4B service stops. Its directory alias says
`Qwen3.8-27B-FP8`, but the actual config is `qwen3_5`; the upstream weight revision is
unverified. The recorded metadata hashes and weight-shard inventory identify the cached
asset without asserting an official release or full weight digest.
Paths in the scripts target this server. Allocate equivalent devices and adjust runtime
paths before porting elsewhere.

## Usage

```bash
cd /data/home/chenyizhou/kernel-agent
bash scripts/reproduce.sh strategy
# Fresh integration validation of the repaired selector (new local-model run):
# bash scripts/reproduce.sh selector-validation
# Reuse an idle existing owner-provided local base service (no server restart):
# bash scripts/reproduce_existing_selector.sh 8016 3097365
# Previous model-capacity pilot: bash scripts/reproduce.sh qwen27
# To reproduce the separate 4B representation ablation:
# bash scripts/reproduce.sh lines4b
```

The script starts its own loopback-only model service and local protocol adapter, runs
tests and the generation pilot, then builds the HTML report. It archives an existing
pilot result before replacing it. Ports 18116 and 18118 (27B), or 18116 and 18117 (4B), must be free; shut
down an earlier instance you own first. The recorded model process uses the existing
vLLM environment and the old model-runner/non-FlashInfer sampling path to avoid a local
sampler startup dependency issue.

`reproduce_existing_selector.sh` is a separate integration-replication entry. It checks
ownership, the base model ID, initial queue state and GPU6; records the existing server's
configuration; starts only this project's proxy; and leaves the model server running.
Its runtime differs from the original dedicated strategy service, so results are not a
matched before/after comparison. It refuses an existing validation result. After a completed
run, `scripts/audit_selector_validation.py` verifies sources, actual selection, final SHA
bindings, unmodified model proposals and all recorded base-model requests.

Read [the HTML report](reports/kernel-agent-report.html), [the frozen pilot plan](docs/plan.md)
and the latest complete raw evidence in `results/agent-selector-validation.json`. The original
strategy evidence remains in `results/agent-strategy-study.json`. Earlier capacity evidence remains
in `results/agent-pilot-qwen27.json`. The controller stores model
prompts, replies, draft/plan files, candidate sources, parent SHA links, external results,
terminal states and actual output-token usage. Runtime artifacts also remain in `runs/`
and Humanize's per-epic native session records.

## Design

`flows/kernel_loop.py` uses an actual Humanize `Agent`, scoped as a read-only generator.
The controller writes model-proposed draft, plan and code in that order. It independently
executes `scripts/evaluate_candidate.py` through `LocalEnv`. Its decision uses correct
GPU outputs and timings, never an agent-written success flag. Basic AST admission is
defensive validation, not a proof that arbitrary Python is sandbox-safe.

Failures become typed feedback. Each new candidate is source-hashed and linked to its
parent. In ordinary runs duplicates terminate immediately; fixed-budget studies reuse the measured
source while consuming all three proposal opportunities. A later failed optimization does not overwrite
the best correct candidate. Current default selection is gate-first: eligible candidates
outrank correct but gate-ineligible candidates, then compare search geometric mean.
The strategy runner explicitly retains the original geomean-only rule for its frozen design. The selected source receives a final check on distinct shapes
and random inputs, never fed back to the model; headline metrics use that final check.

The initial pilot compares a one-turn KDA-prompt adaptation with up to three feedback
turns on four tasks. It is descriptive and has unequal compute budgets. Further API-aware
feedback studies, if present in the report, retain their separate controls and evidence.
Hand-written sibling-project kernels never count as agent-generated successes.

The latest mode adds task-routed, MIT-licensed Triton 3.2 examples, an explicit proposal
role and fresh repair sessions carrying bounded feedback and the previous source. The
vector-add reference includes a complete kernel and wrapper. Outputs are reference-assisted,
not evidence of unassisted or novel synthesis. Source URLs, licenses and excerpt hashes
are preserved in `references/` and each run. This is a combined development intervention,
not an isolated causal comparison.

Proposal transport is now explicit: passthrough, tool-free, code-string JSON schema or
source-line array JSON schema. Only proposer requests are rewritten; background title
requests are preserved. The proxy can only forward to a loopback HTTP server and its
configured local model. The controller reconstructs line arrays without heuristic code
repair. Rejected proposed source is retained and fed into the next repair session, while
admitted candidates remain separate. Duplicates and harness failures retain elapsed time,
usage and trace records too.

## Measured result

The first three development studies completed: initial prompt/feedback 0/8 final-correct,
equal-round feedback/API-card comparison 0/8, and latest reference-assisted evaluation
0/4. None passed the performance promotion gate. A separate vector-add development probe
passed final correctness after repair and retained its correct candidate after a later
failed proposal, but its final geometric mean speedup was 0.959x (worst 0.827x), so it was
not promoted. The successful probe is not pooled into the complete-task headline.

The 12-run transport ablation found 1/4 final-correct under passthrough, 1/4 tool-free,
and 0/4 with code-string JSON schema; no performance promotions. The subsequent source
representation ablation found 0/12 versus 9/12 replies admitted as Python/basic-policy
source for strings versus lines, but both arms remained 0/4 final-correct. These candidate
turns are dependent within four runs; do not call them independent success-rate trials.
The completed cached 27B pilot is separate: **4/4 final-correct and 1/4 performance-promoted**.
Softmax achieved 1.235x final geometric mean speedup (worst 1.010x) against the compiled
reference on three BF16 shapes. Vector-add fell from 1.461x during search to 1.011x at final
validation and was rejected for promotion; RMSNorm was 1.031x (worst 0.878x), transpose
0.968x (worst 0.937x). Every selected source actually launches Triton and allocates a fresh
output; no candidate was hand-edited. These four single runs cannot establish a
controller-only effect, reliability across seeds, or full KernelBench performance.

These results validate execution and evidence handling, but do not establish a reliable
kernel generator or a causal success-rate improvement. Observed limitations include invented
Triton APIs, malformed proposals, ineffective repair and accumulated-context overflow.
The latest bounded-context controller has regression tests; a larger matched study is
needed before asserting an empirical benefit. No manual changes were made to generated
candidate code. See `docs/results-zh.md` for an interview-oriented account of the limits.

## Main research line: equal-budget strategy study

The agent study fixes the same cached 27B asset and references, comparing evaluator
feedback with blind refinement. Both use three proposal opportunities, previous source,
fresh sessions, the same output caps and the same search-best selection rule. Duplicate
proposals consume a request but reuse their earlier measurement. Final data never changes
which candidate the search selected. The generation policies differ in whether actual
search evaluation is returned to proposals two and three.

RMSNorm is a development task; row_sum and silu_mul were reserved before this campaign as
new task families. Their specifications and generic official example routes were frozen
before generation. They are prospective task checks, not independently curated benchmark
coverage. Every unique search-correct candidate gets a final check so first/last/best
selection can be audited on the same artifacts. Those audits are not new independent runs.
One run per task/policy cannot estimate stable success or statistical significance.

The completed study is **3/3 final-correct and 0/3 promoted in each arm**; both new task
families are correct in both arms. Across nine proposals each, blind / feedback produce
4 / 8 distinct source hashes and 5 / 1 duplicates (comments count as source changes).
First-round user prompts match, but workspace/system context and first sources differ.
This small descriptive study does not establish a causal feedback benefit.

The selection audit exposed a controller defect: SiLU feedback round two met the search
gate and achieved 1.287x final mean / 0.990x worst, but the old mean-only selector discarded
it for a slightly higher mean with a 0.773x worst-shape ratio. Gate-first selection is now
implemented and regression-tested. Its **post-hoc search-only counterfactual** on the same
candidate pool is 1/3 promoted for feedback, 0/3 blind. A separate fixed-source recheck runs
each of the two SiLU sources through three new evaluator processes: the old selected source is
0/3 promoted (0.926–0.932x mean), while the discarded source is 3/3 promoted (1.289–1.299x
mean). This validates the recorded measurement contrast, not a new model-generation trial.
The subsequent fresh integration campaign is described below; original 0/6 promotion
results remain immutable. All 16 measured source files and 18 proposed sources are archived in
`results/study7-measured/`; use this snapshot for the exact measured implementation. Current
code includes the later selector fix and Humanize API adapter.

See `docs/strategy-study-plan.md`, `scripts/run_strategy_study.py` and the interview project
card at `docs/project-card.zh.md`. `docs/selector-validation-plan.md` and
`scripts/run_selector_validation.py` define a separate fresh integration run of gate-first
selection, using a new local seed and output file. Results remain separate from the preceding
model-capacity pilot. The goal is an evidence-backed kernel-agent method, not a gallery of
hand-tuned kernels.

## Fresh selector integration replication

Completed on 2026-10-12 with local seed 20261012: **3/3 final-correct, 2/3 promoted**.
SiLU achieved 1.291x geometric mean / 0.968x worst shape; RMSNorm 1.073x / 0.954x;
row_sum 0.949x / 0.851x and was not promoted. The nine proposals produced nine different
source hashes, all search-correct; each unique source has a separate final audit. Actual
output was 6082 tokens; total agent-flow time was 1985.0 seconds, including evaluator audits.

The existing owner's idle base-model service was reused on GPUs 2/3; evaluation stayed
on GPU6. Runtime differences (FP8 KV, prefix cache, context limit, LoRA-enabled runtime
with no adapter requested) are recorded in `results/selector-runtime-manifest.json`.
This repeats prior task families with no blind arm; it is not a controlled before/after
success-rate experiment.

Search evidence reconstructs a ranking divergence after SiLU proposal two. Proposal
three becomes best under both rules, so final selections match geomean-only in all three
runs. The fresh result exercises the repaired production path but does not demonstrate
an incremental final-promotion benefit from the selector. All actual selections match
gate-first search ranking; the original failure branch also has deterministic flow tests.

The delivery audit verifies 18 measured files, nine actual candidate files and nine
proposer requests (plus nine separate background requests), including explicit base-model
routing, seed and line schema. Snapshots are in `results/study8-measured/` and
`results/study8-analysis/`. See `docs/selector-validation-results.md` and
`results/selector-validation-delivery-audit.json`.

Those source hashes describe the measured snapshot. Current flow code has a subsequent
workspace-binding compatibility change; it is not the exact code used in the campaign.
`scripts/verify_evidence.py` checks the immutable snapshots, while
`scripts/audit_selector_validation.py` is the original delivery-time audit.
See [Humanize update notes](docs/humanize-update.md) for compatibility evidence.

## Checks

```bash
PYTHONPATH=src /data/home/chenyizhou/humanize/.venv/bin/python -m pytest tests/unit tests/integration -q
/data/home/chenyizhou/humanize/.venv/bin/ruff check flows scripts src tests
```

The measured campaign passed 54 controller/interface checks. Publication adds three
artifact-integrity tests; current results are recorded in `docs/publication-verification.md`.
These tests check mechanism behavior, not model success. Historical upstream failures
belong to the old audit; the updated Humanize commit's own CI status is recorded separately
in `docs/humanize-update.md`.

## Maintainer

This workspace is maintained for the user's kernel-agent reproduction project.

## License

Original downstream code is MIT; see `LICENSE`. KDA and Humanize retain their respective
upstream terms. Referenced Triton documentation is linked at its pinned version.
