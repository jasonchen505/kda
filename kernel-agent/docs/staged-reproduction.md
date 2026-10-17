# RTX4090 staged-agent port: execution and audit

The port runs a real Humanize SDK flow through its LiteLLM backend, an audited loopback
gateway and authorized local Qwen services. No paid model account is used. The writer
chooses typed hosted tools; it has no native CLI/shell or unrestricted repository tools.
All model turns use fresh sessions with host-persisted source/evidence memory; this differs
from the original long-lived writer session. This is a bounded port of KDA/Humanize methods, not the original Claude/Codex plugin or
B200/FlashInfer contest workflow. No upstream Humanize implementation is modified.

## Runtimes and storage

Canonical code: `/data/home/chenyizhou/kda-kernel-agent-harness/kernel-agent`.
Storage: `/data/home/chenyizhou/kernel-agent-storage`.

- Controller launcher: `bin/run.py`, using the existing controller environment.
- Humanize/LiteLLM worker: separate `runtimes/multistage/bin/python`; old environments
  were not upgraded. Its source dependency is Humanize d7f732e9. The portable
  [controller dependency snapshot](../requirements-staged.txt) pins installed versions;
  private installed-requirements bytes are additionally SHA-bound in each study plan.
- Numerical evaluation: existing `runtimes/gpu/bin/python`, Torch2.6/Triton3.2.
- Profiler: Nsight Compute2024.1.1. NCU2021 on default PATH is not used. Actual collection
  is denied by performance-counter permissions. The explicit CUDA-activity fallback uses
  torch.profiler and retains the original permission error. No SM/DRAM/stall counters exist.

All requests, replies, exact source proposals (including rejected ones), complete plans,
GPU measurements, source freezes, Humanize workspaces and activity traces stay outside
Git/Mac under `storage/experiments/`. Copy only code/docs, compact public summaries and the
latest HTML to Mac. Do not restore the historical full corpus for routine work.

## Independent cohorts

`staged-study-v2` is the first scored port cohort, seed2026101701: three development
families × three arms. It exposed phase-goal leakage, invisible read/research events and
fatal treatment of a fully received truncated writer reply. Preserve its9 outcomes; do not
use it to claim a clean full-workflow effect.

`staged-study-v3` is the independent phase-interface correction cohort, NEW seed2026101702,
same three families/arms and ceilings. See [its prospective protocol](staged-confirmation-plan.md)
and [initial protocol/limits](staged-study-plan.md). Within each block the arms share one
model process. Equal ceilings do not imply equal actual compute; report total role calls,
tokens and tool usage. Both cohorts are small development studies, not blind generalization.

Start a NEW directory only when deliberately running another cohort; do not rerun these:

```bash
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py run_staged_corrected.py \
  --directory /data/home/chenyizhou/kernel-agent-storage/experiments/NEW-ID --background
```

The existing pool admits idle replicas, uses unused replicas before EWMA, and pins a whole
block to a process identity. It is admission, not a reservation of someone else's service.
GPU4 sharing is allowed; project evaluations/profile windows use a cooperative lock and
physical UUID binding. Timing remains conditional on observed sharing. Unknown model
requests are never replayed, and an interrupted block is terminal unknown rather than a
restored task-internal Humanize session.

The corrected flow uses per-phase acceptance criteria and includes all real host event
types in the review packet. Only a fully received, hash-matched invalid writer response
can consume an existing opportunity and be followed by a new action. It cannot invent a
decoded source, retry an unknown request or add a call. A stage failure blocks advancement.
All three search selections in a block commit before any final evaluation; final results
are never fed to models or used to reselect.

## Reconstruction and report

After the supervisor exits, reconstruct exact requests, typed SDK replies, source/patch
decoding, stage documents/gates, search selection and final source identity:

```bash
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py export_staged_report.py \
  --directory /data/home/chenyizhou/kernel-agent-storage/experiments/staged-study-v3 --finalize
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py build_staged_report.py
```

The raw manifest freezes server records and versioned reporting tools; it excludes
ephemeral Humanize runtime databases, supervisor console log and locks. Raw reconstruction does not
rerun Qwen or remeasure CUDA. It checks recorded identity and decisions, not a formal
semantic proof or arbitrary-code sandbox. Reviewer independence means fresh context on
the same Qwen process; it does not mean independent errors or model weights.

On lean Mac/public checkout, run only the summary checker:

```bash
python3 scripts/verify_staged_summary.py results/staged-summary.json
python3 scripts/verify_staged_summary.py results/staged-initial-summary.json
```

These verify self-hashes and count consistency. Full raw auditing requires server access.
HTML includes bounded plan/review/tool excerpts; companion JSON downloads require the
published summary files. Preflights and the unscored pilot remain separate in
`results/staged-environment.json`; their handwritten profiler fixture is never an agent score.

References: [KDA common stages and Humanize commands](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/prompts/README.md),
[Humanize public flow API](https://github.com/jasonchen505/humanize/blob/d7f732e935100047d0faa5fc7c4d6b03ce350fc7/src/hmz/flows/agents.py).
