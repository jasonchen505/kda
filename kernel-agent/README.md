# Evidence-driven Kernel Agent Harness

## Latest: RTX4090 staged-agent port

A real Humanize SDK flow now separates drafting, executable planning, writer-selected
hosted tools and fresh-context review across research, profiling/refinement and shape
comparison. It runs through LiteLLM and an audited local Qwen gateway. The existing
source/evidence memory, budgeted patches, tokenizer packing, three-service scheduling,
independent GPU oracle and committed-selection/final-veto boundary are reused.

This is a bounded adaptation of KDA/Humanize methods to RTX4090 and local models. It is
not an equivalent reproduction of the original Claude/Codex plugin sessions, B200 contest
workloads or Humanize upstream core. All role turns are fresh; memory is host-managed.
NCU counters are denied by server permissions; real CUDA activity is explicitly a fallback.

Read [the port results and migration limits](docs/staged-results.zh.md),
[execution and audit](docs/staged-reproduction.md),
[the historical workflow comparison](docs/workflow-comparison.zh.md), and the
[latest HTML](reports/kernel-agent-report.html#staged-study).
236 CPU checks and lint pass in controller and isolated SDK environments. Synthetic
three-phase wiring tests are not model successes. New raw evidence remains remote.

The corrected nine-outcome cohort finishes with99 confirmed calls: fixed3/3 correct
(18 calls), staged2/3 (41), transferred2/3 (40). No performance promotion or full
three-stage completion. Source facts enter15 requests; no patch action is used. Ordinary
repair stays default. The initial68-call interface-defect cohort is preserved separately.
Common ceilings do not imply equal actual cost; these are three development families/one
new seed per cohort, not external generalization or an isolated planning effect.

## Previous iteration: source facts, actual context budgets and arm isolation

Initial fresh generation: indexed7/8, source-grounded8/8, eight development families/one
seed. Initial diagnostic0/6 vs3/6 is contaminated by two policy context refusals cancelling
controls; retain it, do not claim clean efficacy. A separate two-NEW-seed controller
follow-up records indexed3/6, source-grounded5/6, 24/24 physical calls.
Same parents and two repair slots; no candidate is hand-fixed. Do not pool cohorts.

Read [results/failure](docs/grounded-study-results.zh.md), [follow-up](docs/grounded-budget-results.zh.md),
[reproduction](docs/grounded-budget-reproduction.md), [resume](docs/project-summary.zh.md).
The preceding release passed219 CPU checks/lint. New raw evidence stays remote; only code/docs/derived summaries
and latest HTML are synchronized. Ordinary repair remains default.


A job-portfolio prototype that adapts KDA workflow references, integrates real
Humanize execution routes, and implements a downstream controller. The research object is the
agent that generates kernels, not manually authored operator optimization.

**Portfolio scope:** local-model agent orchestration and evaluation for LLM research
engineering. Generated kernels are assessment tasks; candidates are never hand-repaired.
The contribution is the downstream controller, candidate lifecycle, resource scheduling
and auditable evidence. KDA, Humanize/OpenCode, Qwen and Triton are reused components.

Start with [the project summary](docs/project-summary.md),
[Chinese resume and interview notes](docs/project-summary.zh.md), and
[the offline HTML report](reports/kernel-agent-report.html#staged-study).

## Previous iteration: indexed repair with a budget-neutral fallback

The agent now supports SHA-bound, numbered source ranges and complete-source recovery
inside its second existing repair slot. It never invents an edit or adds a retry. The
completed three-arm comparison has22 paired blocks,66 final outcomes and140 confirmed
physical calls. Known failed sources: indexed13/14, rewriting9/14, old text patches7/14.
Fresh starts on four existing recent families: all arms6/8. Strata stay separate.

First-patch application rises from13/21 to18/21; total refusals fall from17/43 to7/40.
The primary family-equal regression effect vs old patches is+35pp, exploratory interval
[+10,+60] over five families; vs rewriting+30pp [0,+70]. Raw-count effects use different
weights. These are development-family results, not an external benchmark or stable
probability of success. Keep ordinary repair as default; indexed/fallback remains an
experimental option pending broader validation. Protocol success is not numerical success.

All three authorized Qwen services participate;156 independent GPU4 windows peak at one.
The205-test CPU suite and lint pass. All98 generation-freeze files match after execution;
a separate reporting-only exporter handles missing-source initials without changing outcomes.
New raw records stay on the server, outside Git and Mac. Only code/docs, compact derived
summaries and the latest HTML are synchronized. Full raw reconstruction runs remotely;
local verification checks published summary hashes/counts only.

See [results](docs/span-study-results.zh.md), [prospective protocol](docs/span-study-plan.md),
[reproduction](docs/span-reproduction.md), [server workflow](docs/server-workflow.md), and
[the HTML report](reports/kernel-agent-report.html#span-study).

## Previous iteration: operational evidence and transactional patches

The semantic review's null result motivated a repair architecture that derives finite
numeric relations and coordinate witnesses directly from trusted probes, routes actual
errors, and requests exact model-authored before/after edits against a source SHA. The
controller applies changes atomically and retains the base after rejection; it never
invents a kernel fix. Both arms get two repair calls with the same initial and evaluator.

The completed 22-block study has **44 final outcomes and 96 actual calls**, all confirmed.
Known-failure regression: patch **9/14**, rewriting **7/14**; new four-family/two-seed
validation: patch **6/8**, rewriting **7/8**. Family-equal differences are +25pp
(exploratory interval −30 to +70, five families) and −12.5pp (−37.5 to 0, four families).
The raw regression count difference is +14.3pp; it uses different weighting. Strata are
separate and new tasks are project-defined, not an external blind benchmark.

Patch repair rescues old SiLU and one transpose source, but fails on row-mean diagnosis
and a new L2 dtype repair. **17/42 patch transactions are rejected**; output representation
is a measured bottleneck. Keep ordinary generation/repair as default, with the patch
policy experimental. No post-outcome retries or hand fixes replace failures.
Three Qwen processes reach three concurrent ports; 107 GPU4 windows peak at one and
all 4847 older evidence files retain their hashes. CPU mechanisms pass **191 tests**.
See [results and failure analysis](docs/operational-study-results.zh.md),
[reproduction](docs/operational-reproduction.md), and the
[HTML report](reports/kernel-agent-report.html#operational-study).

## Previous iteration: machine-grounded semantic repair

The preceding factorial's negative result motivated a separate reviewer and bounded AST
interpreter over complete asymmetric FP32 search probes. Verified expressions and factual
validation traces replace unverified review narratives for the grounded writer. A failed
check falls back without another request; kernels are never hand-edited.

The new 22-block, three-arm study resolved all 66 outcomes with 115 physical dispatches.
Known-failure regression: rewrite 7/14, critic 7/14, grounded 7/14.
Fresh generation: rewrite 7/8, critic 7/8, grounded 7/8.
Each arm has two extra model opportunities, with one review shared between the critic
variants; this is not an equal-kernel-count experiment. Strata are reported separately,
clustered by five/eight families, and are not a blind external benchmark. Keep ordinary
generation/repair as the default; semantic variants remain experimental. Mathematical
agreement does not prove layout or API correctness.

Three-service idle admission/EWMA scheduling and physical GPU4 proof remain active.
The CPU suite passes 176 tests on both pinned SDKs and a fresh locked installation.
See [results](docs/semantic-study-results.zh.md), [failure analysis](docs/semantic-failure-analysis.zh.md),
[reproduction](docs/semantic-reproduction.md), and `results/semantic-study/`.


**Preceding generation/repair evaluation: 64/64 terminal outcomes**, eight families ×
two seeds × a 2×2 factorial. Contract planning is **11/16 first-proposal correct**
versus **12/16** for ordinary prompting. Full contract+counterexample configuration
is **13/16 final-correct**, versus **15/16** for ordinary generation+repair.
Neither new policy improves final correctness here; keep ordinary generation/repair as the recommended configuration. Both repair arms share the identical initial source and verdict; 8 fully observed
initial-failure pairs yield 5 versus 5 rescues. Actual calls: **152/160**. See
[the frozen design](docs/repair-study-plan.md) and [results/limits](docs/repair-study-results.zh.md).

The new native Chat backend, tokenizer-based context packing and dynamic Qwen discovery
are common to all arms. Idle admission, process binding and latency-EWMA queue allocation
cover 3 GPU groups. Independent GPU4 evaluations remain cooperatively serialized.
No manually fixed kernels, model-call replay or final-data reselection. The preceding release passed **159 cases**
on both pinned SDKs and a fresh locked installation; model/GPU evidence is counted separately.

Earlier evidence remains separate: the 24-job service-pool engineering validation,
[recorded endpoint handoff](docs/service-pool-handoff.md), real SIGKILL final-stage recovery,
[12-job feedback study](docs/shared-study-plan.md), [selector recheck](docs/selector-recheck.md)
and the incomplete [23/36 guarded plan](docs/multiseed-execution.md). The legacy
Humanize/OpenCode entry remains available; this native-backend study is not a causal
comparison with its historical latencies. References assist generation; task-cluster
intervals are exploratory, cross-root weight equality is unverified, and shared ratios
against torch.compile are not isolated or end-to-end LLM speedups.

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
It also audits the new 22-block semantic study: shared starts/reviews, exact assigned
request contexts, numerical expression validation and selection-before-final bindings.
The separate CPU SIGKILL recovery artifact and its frozen sources are checked as well.
The new pilot's candidate/request artifacts, interference qualification and guarded
campaign snapshot are checked separately; pending jobs do not become failed trials.
It verifies recorded artifacts; it does not execute the model or recompute GPU timings.

For controller development, use Python 3.12+ in this directory:

```bash
uv sync --extra dev --frozen
uv run pytest
uv run ruff check flows scripts src tests
# Separate CPU fault-injection evidence (refuses to overwrite an existing artifact):
uv run python scripts/validate_recovery.py --output runs/recovery-check/validation.json
```

`pyproject.toml` and `uv.lock` pin the updated Humanize source. GPU evaluation remains in
a separate Linux environment with Torch 2.6.0+cu124 and Triton 3.2.0; vLLM 0.27.1 and
OpenCode are separate executables. The current shared study records OpenCode 1.19.0;
earlier generation snapshots record 1.18.32. No model weights are included.

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

This server already contains the frozen environments. The old measured Humanize environment
remains pinned to the archived source worktree; the installed project pins the updated SDK.
The sibling `kda-humanize-portfolio` environment supplies Torch 2.6.0 and Triton
3.2.0. The completed 2026-10-14 sharing study used a loopback vLLM 0.27.1 service on GPUs 2/3,
port 8005. That service is no longer live. The 2026-10-15 engineering pilot uses a newly
started owned service on GPUs 5/6, port 18118, and evaluates on GPU4. The launcher uses
the existing cached weights, TP2, a 32,768-token context, FP8 KV cache and eager execution;
no adapter is registered. Earlier dedicated services used different devices; their
runtime records remain in their own frozen studies. The 4B model revision is
`cdbee75f17c01a7cc42f958dc650907174af0554`. The cached 27B directory alias says
`Qwen3.8-27B-FP8`, but the actual config is `qwen3_5`; the upstream weight revision is
unverified. The recorded metadata hashes and weight-shard inventory identify the cached
asset without asserting an official release or full weight digest.
Paths in the scripts target this server. Allocate equivalent devices and adjust runtime
paths before porting elsewhere.

For the new recoverable campaign, `scripts/run_campaign.py --prepare --pid SERVICE_PID`
freezes the source/runtime plan without model generation; `--run` resumes pending tasks.
Use the same arguments and source for every restart. The runner checks the existing owned
base service and separate evaluation GPU before allocating a job, and exits 2 if work is
pending. It never starts or kills the model service. See the study plan for exact commands.

The phase-split entry is `scripts/run_deferred.py generate|measure|summary`. See
`docs/deferred-evaluation.md` for its separate workspace, frozen settings and commands.
Generation-only completion cannot be interpreted as final correctness or promotion.
On this server, prepend `/data/home/chenyizhou/.local/bin` to `PATH` so that the recorded
OpenCode executable is available in a non-interactive SSH session.
`scripts/finish_deferred.py` completes committed generation under a new frozen sharing
plan; `scripts/run_shared_campaign.py` runs the prospective shared feedback study.
`scripts/capture_shared.py --checkpoint PATH` exports a self-contained progress/result
snapshot without GPU or model execution. It copies one atomic checkpoint read, committed
sources, request bodies and the measured implementation before computing the report metrics.

## Usage

The service-pool entry prepares the fixed two-seed schedule before any model calls:

```bash
uv run python scripts/run_pool.py --pid OWNED_SERVICE_PID \
  --directory runs/my-service-pool --prepare
# Remove --prepare to run three isolated workers; GPU4 work is cooperatively serialized.
# Capture evidence without requesting the model or running GPU evaluation:
uv run python scripts/capture_pool.py --directory runs/my-service-pool --output runs/my-pool-export
```

The two borrowed services are HTTP-only. They are never started or stopped by this runner.
Endpoint changes require an explicit, recorded declaration check and handoff; frozen plans
are never rewritten. `service_gateway.py` records the observed replacement routes for this
case and must not be reused blindly for unrelated ports or declarations.


Reviewers can run `bash scripts/check.sh` without starting a model or a GPU experiment.
For a new observed-sharing experiment, provide the owned base-service PID, an evaluation
device and a fresh checkpoint path:

```bash
# If no owned service exists, inspect occupancy and choose two available generation GPUs first:
KERNEL_MODEL_GPUS=5,6 bash scripts/start_cached_qwen27.sh
# The line above runs the loopback server in the foreground; use another terminal below.
uv run python scripts/run_phased.py --pid SERVICE_PID --port 18118 \
  --gpu-index 4 --directory runs/my-phased-pilot --prepare
# Remove --prepare to generate; finalization is an independent pure-evaluation phase:
uv run python scripts/finish_deferred.py --generation runs/my-phased-pilot/generation.json \
  --directory runs/my-phased-final

# Separate full two-arm study (use the owned service's actual port and a fresh checkpoint):
uv run python scripts/run_shared_campaign.py --pid SERVICE_PID --port 18118 \
  --gpu-index 4 --checkpoint runs/my-shared-study/checkpoint.json --prepare
# Remove --prepare to execute the frozen twelve-job plan; requires the recorded runtime.
# Export only committed evidence into a separate review directory:
uv run python scripts/capture_shared.py --checkpoint runs/my-shared-study/checkpoint.json \
  --output-root runs/my-shared-export
```

The completed experiment's frozen sources and checkpoint are evidence. Use a fresh plan
for replication; do not overwrite those artifacts or switch a live plan's measurement policy.
`scripts/build_report.py` rebuilds the packaged report from its `results/` evidence.
The following entries reproduce earlier dedicated-service studies and have different
runtime requirements:

Use their frozen `results/*/sources` or measured snapshots for exact historical launcher
settings. The current cached-model launcher has the new service configuration; legacy
dedicated entries explicitly map its base-model alias and keep their own GPU allocation.

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
