"""Differential mechanism check, no model calls and no generated or hand-fixed kernel."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch  # noqa: E402
from kernel_agent.semantic import check_expression  # noqa: E402
from scripts.evaluate_candidate import semantic_probes  # noqa: E402


def main():
    # These independent test equations never enter model contexts or controller policies.
    centered = "(x0-mean_last(x0,True))"
    programs = {
        "vector_add": "x0+x1", "row_sum": "sum_last(x0,False)", "row_mean": "mean_last(x0,False)",
        "softmax": "exp(x0-max_last(x0,True))/sum_last(exp(x0-max_last(x0,True)),True)",
        "rmsnorm": "x1*x0*rsqrt(mean_last(x0*x0,True)+0.000001)",
        "silu_mul": "x0*sigmoid(x0)*x1", "transpose": "transpose(x0)",
        "layer_norm": f"{centered}*rsqrt(mean_last({centered}*{centered},True)+0.00001)*x1+x2",
    }
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for task, expression in programs.items():
            probes = semantic_probes(task, Path(directory) / "no-candidate.py")
            validation = check_expression(expression, probes, "expected")
            if not validation["valid"]:
                raise ValueError((task, validation))
            rows.append({"task": task, "expression": expression, "validation": validation, "probes": probes})
    bad = check_expression("sigmoid(x0)*x1", next(r["probes"] for r in rows if r["task"] == "silu_mul"), "expected")
    wrong_layout = check_expression("x0", next(r["probes"] for r in rows if r["task"] == "transpose"), "expected")
    assert not bad["valid"] and not wrong_layout["valid"]
    index = torch.cuda.current_device()
    return {"status": "semantic_mechanisms_passed", "oracle_probe_checks": 16, "rows": rows,
            "missing_factor_rejected": bad, "wrong_shape_rejected": wrong_layout,
            "scope": "Trusted Torch FP32 oracle versus bounded interpreter; no candidate success counts or model work",
            "device_audit": {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_index": index,
                             "device_count": torch.cuda.device_count(), "gpu_uuid": str(torch.cuda.get_device_properties(index).uuid)}}


if __name__ == "__main__":
    print(json.dumps(main(), allow_nan=False))
