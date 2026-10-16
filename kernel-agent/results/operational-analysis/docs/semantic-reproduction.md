# Reproducing the grounded semantic-repair study

The new study changes the agent's review/repair mechanism. It does not supply model
answers from a task-to-equation lookup table, edit proposed kernels, or replace the
preceding negative factorial study. Its protocol is `docs/semantic-study-plan.md`.

## Review without a model or GPU

From this directory:

```bash
uv sync --extra dev --frozen
uv run pytest -q
uv run ruff check flows scripts src tests
python3 scripts/verify_evidence.py
python3 scripts/summarize_semantic.py results/semantic-study > /tmp/semantic-summary.json
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `reports/kernel-agent-report.html#semantic-study`. The self-contained report embeds
one exact ledger snapshot and the separately reported regression/fresh strata. The
standard-library audit reconstructs the assigned policy's request content, verifies
raw responses and expression checks, shared initial/review bindings, call allocations,
source lineage, search selections and final access after their commit. It verifies
recorded evidence; it does not regenerate model output or GPU timings.

`results/semantic-study/` contains the plan, checkpoint, executed-source snapshot,
wire requests/receipts/responses, exact candidates, review results, search evidence and
independent final verdicts. `data/semantic-failure-bank.json` declares all seven eligible
old failed initial sources, selected without conditioning on repair outcomes. The source
plan binds that bank, original summary and exact imported candidates. The old experiment's
records remain under `results/repair-study/`, with their own original source snapshot.

## Execute a fresh copy on the recorded server

Use Python 3.12+ for the controller and the separate Linux evaluation environment
(Torch 2.6.0+cu124, Triton 3.2.0). Keep the virtualenv invocation path, not its resolved
interpreter symlink. Use a fresh empty experiment directory; never modify frozen policy
files after the first dispatch:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_semantic_study.py \
  --directory /data/home/chenyizhou/kernel-agent-new-semantic-study \
  --gpu-python /data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python
uv run python scripts/capture_semantic.py \
  --directory /data/home/chenyizhou/kernel-agent-new-semantic-study \
  --output /data/home/chenyizhou/kernel-agent-new-semantic-export
```

The frozen settings target physical GPU4 and authorized cached model roots on this
server. `model_pool.py` discovers live declarations, public process arguments and GPU
worker ancestry, checks idle metrics, and balances untouched blocks using latency EWMA.
The current ports are not hardcoded. One block stays on one process fingerprint; two
writer arms share one review. The wrapper explicitly binds evaluator children to GPU4's
physical UUID, checks one visible device and local device index zero, and cooperatively
serializes this project's GPU work. It does not reserve foreign services or prevent
unrelated shared-GPU jobs. Record conditions before interpreting performance.

Replication on other hardware requires a new source/plan snapshot for its paths, roots
and GPU settings; the current configuration is a documented deployment, not a universal
cluster layer. Run the mechanism oracle check before model work when adapting it:
`validate_semantic_oracle.py` must execute inside the same UUID-proven, cooperatively
locked evaluator wrapper. Its hand-specified *test* equations never enter agent prompts.

The ceiling is 118 physical dispatches, 241664 reserved output tokens, 22 paired blocks
and 66 terminal outcomes. Each arm has two extra model opportunities, not two extra
kernel proposals. Imported generation belongs to the historical ledger. Unknown calls
are never replayed; infrastructure and source/protocol failures retain their denominators.
This supervisor intentionally does not resume interrupted model work: preserve it as
unknown rather than creating an unrecorded second attempt.

## Interpret separately

Regression: 14 blocks/arm from seven known failed sources in five families, two repair
seeds. Fresh: eight blocks/arm from existing families, one new generation seed. Average
parents/seeds within task before the exact task-cluster bootstrap; do not treat 66 arm
outcomes as independent or pool the two strata. Both are small exploratory evaluations.
A zero-width interval is not equivalence; no external blind benchmark is claimed.

The current analyzer canonicalizes timeline order across filesystems. This post-run
analysis-only correction changes no metric, record, request, kernel or evaluation. The
original summary and executed analyzer are preserved; hashes and equality checks are in
`results/semantic-analysis-correction.json`. A reversed-filesystem-order test reproduces
the same complete summary. See the completion audit for the one disclosed source exception.
