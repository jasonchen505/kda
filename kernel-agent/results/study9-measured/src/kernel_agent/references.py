"""Task-routed, licensed example retrieval with explicit source provenance."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTES = {"vector_add": "01-vector-add.py", "transpose": "01-vector-add.py",
          "softmax": "02-fused-softmax.py", "rmsnorm": "05-layer-norm.py",
          "row_sum": "05-layer-norm.py", "silu_mul": "01-vector-add.py"}


def retrieve_example(task_id):
    ledger = json.loads((ROOT / "references/triton-3.2/manifest.json").read_text())
    name = ROUTES[task_id]
    result = {"name": name, **ledger[name]}
    result["context"] = ("Retrieved MIT-licensed Triton 3.2 reference excerpt. It is an example, not necessarily "
                         "the solution for this task. Adapt its API and indexing patterns to the actual contract. "
                         "For RMSNorm remove centering/bias and preserve the required cast order. "
                         "For transpose adapt the vector memory pattern to transposed indices. "
                         "You must still produce the complete run wrapper and pass independent tests.\n"
                         + result["excerpt"])
    return result
