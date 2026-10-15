# Reproducing the generation and repair study

This release studies the agent that writes kernels. The two changes are a contract
checklist in the existing generation draft/plan, and a bundled repair context containing
the last complete source, search evidence, a tiny diagnostic sample, failed-source memory
and an optional correct-source anchor. The model emits the entire replacement source.
The controller never edits a kernel to make it pass.

## Review the evidence without a GPU

From `kernel-agent/`, with Python 3.12+ and uv:

```bash
uv sync --extra dev --frozen
uv run pytest -q
uv run ruff check flows scripts src tests
python3 scripts/verify_evidence.py
python3 scripts/summarize_repair.py results/repair-study > /tmp/repair-summary.json
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/reports/kernel-agent-report.html#repair-study`.
The report embeds its data and also works as an offline file. Its four cells are final
outcomes with shared starts, not four independent initial generations. The verifier
checks recorded artifacts and decisions; it does not regenerate kernels or GPU timings.

The wire receipts, model responses, exact candidates, initial verdicts, pre-final search
selections and independent final verdicts live in `results/repair-study/`. The plan includes
the executed generation source hashes; `sources/` preserves those files. Analysis and
presentation changes made after execution are separately frozen in `results/repair-analysis/`.
`repair-delivery-audit.json` binds the delivered files. Historical studies retain their
own sources, denominators and runtime identities.

## Run a fresh study on the recorded server

Use an empty output directory and a separate GPU evaluator environment with Torch
2.6.0+cu124 and Triton 3.2.0. The controller environment does not need Torch. The recorded
server command is:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_repair_study.py \
  --directory /data/home/chenyizhou/kernel-agent-new-study \
  --gpu-python /data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python
```

The wrapper itself binds every evaluator child to the physical GPU4 UUID and rejects
missing or mismatched device proof. It does not depend on the caller's CUDA visibility.
The GPU lock only serializes this project's workers; unrelated workloads can share GPU4.
Record those conditions when interpreting performance ratios.

This release's deployment settings are explicit in `model_pool.ROOTS` and the frozen plan:
two authorized model-family roots, loopback vLLM servers, at least 8192 context, physical
GPU4 for evaluation, eight task families and two fixed seeds. Dynamic discovery follows
public process arguments, HTTP metadata and public GPU worker ancestry, so changed ports
can take untouched blocks. Each active block retains one model-process identity. An
external operator adapting roots, GPU, tasks or seeds must freeze a new plan and use a
new output directory; this artifact is not a universal hardware configuration layer.

Each block reserves ten calls: two initial proposals and two repairs for each of four
arms. Both repair arms import the exact same initial source and verdict. All four search
selections are written before any final evaluation. Equal sources within a block reuse
measurements and retain explicit reuse references. An uncertain dispatched call is
quarantined rather than retried; interrupted blocks remain in the planned denominator.

To export a live or finished ledger into a different review directory:

```bash
uv run python scripts/capture_repair.py \
  --directory /data/home/chenyizhou/kernel-agent-new-study \
  --output /data/home/chenyizhou/kernel-agent-new-export
```

## Interpret the result

First-proposal correctness uses 16 initial opportunities per generation policy. Final
correctness uses 16 outcomes per factorial cell. Conditional repair rescue only compares
the two arms sharing an initially failed candidate, and separately reports whether both
received their two repair opportunities. End-to-end outcomes retain infrastructure
failures. Average seeds and the other factor within each task, then resample the eight
task families; treating 64 outcomes as independent would overstate the evidence.

The fixed tiny diagnostic probe is useful for inspecting index/layout mistakes but may
pass when the original search failure depends on width, dtype or scale. It must not be
called a reproducing counterexample unless its actual output contradicts its expected
output. The original search verdict remains authoritative. The treatment combines
instructions and context; this experiment does not isolate each component or establish
latent reasoning. Shared kernel ratios against torch.compile do not measure end-to-end
LLM acceleration, and cached directory names do not prove identical weights.

The invalid device-integration pre-run is retained in `results/repair-invalid-device-v1/`.
Its 48 dispatches, including three unconfirmed responses, are excluded from efficacy.
The corrected formal run has a separate 160-call ceiling, new seeds and unchanged policy
source. The two additional task families were already observed in the invalid pre-run;
they are not claimed as blind held-out tasks. No negative trial is silently replaced.
