# Reproduce the operational repair comparison

This is a downstream agent experiment. The model writes all candidate changes; the
controller performs exact transaction application, source checks and independent testing.
No hand-repaired candidate is credited as model output.

## Read without a model or GPU

```bash
uv sync --extra dev --frozen
uv run python scripts/summarize_operational.py results/operational-study
uv run python scripts/verify_evidence.py
uv run pytest -q
uv run ruff check src scripts tests flows
python3 scripts/build_report.py
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `reports/kernel-agent-report.html#operational-study`. The single HTML includes
its data; switching cases, inspecting model patches and downloading the raw ledger
works offline. Offline reconstruction verifies source and reply hashes, exact policy
messages/schema/seeds, shared initial, patch base and replacement spans, both committed
selections before final access, final veto and physical GPU proof. It makes no new calls.

`results/operational-study/sources/` contains the actual executed source freeze.
`results/operational-analysis/` is the separate delivery-analysis freeze. Tests using
fake HTTP/GPU responses validate controller wiring; their success is not an agent score.
`results/operational-smoke/` holds independent NumPy oracle checks and unchanged old
failed-source probes. These are also separate from model-effect results.

## New controlled run

The existing ledger is immutable. Choose a NEW directory and a newly reviewed prospective
plan for any new run. Do not replay a completed study or retry unknown physical requests.
The original design is `docs/operational-study-plan.md`; its SHA is bound in the run plan.

The recorded runtime uses Python3.12, torch2.6.0+cu124 and Triton3.2.0. Read the public
Triton symbol inventory from the actual GPU environment, not from a language-model guess:

```bash
GPU_PYTHON=/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python
"$GPU_PYTHON" scripts/collect_triton_capability.py > data/runtime-triton-api.json
PYTHONPATH=src:. uv run python scripts/run_operational_study.py \
  --directory /path/to/NEW-operational-study --gpu-python "$GPU_PYTHON"
```

The runner discovers only the explicitly allowed local cached Qwen roots. It validates
process start identity, HTTP model declaration and context length, observes idle admission,
then chooses unused replicas before latency EWMA. One active paired block per process;
all requests in a block remain on the same process. Common context8192, output2048,
no thinking/tools, temperature0.2. Three services are used when observed idle. HTTP
metadata cannot prove cross-root weight equality, and idle observations do not reserve
another user's service. The runner does not edit or restart those services.

Evaluation binds physical GPU4 by UUID in each child, retains before/during/after samples,
and serializes project evaluations with a cooperative lock. Sharing is allowed. A child
failure or monitoring error stays in the denominator; a final failure never selects a
new candidate. The particular GPU Python path must remain its original environment
entry point: resolving its executable symlink can lose torch/Triton dependencies.

Capture an atomic offline export with the frozen analyzer:

```bash
uv run python scripts/capture_operational.py \
  --directory /path/to/NEW-operational-study --output /path/to/NEW-export
```

The shared initial is generated once on new tasks; known regression sources are imported
byte-for-byte and re-evaluated. Both arms get two repair calls. A no-source patch arm uses
its same slot for complete-source recovery. Malformed/stale/ambiguous patches consume an
opportunity; the controller never repairs them. Keep the public evidence and all failures.
