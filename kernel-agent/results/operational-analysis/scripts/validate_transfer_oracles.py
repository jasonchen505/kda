"""Independent NumPy checks of new trusted GPU references, never agent scores."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scripts.evaluate_candidate import inputs, reference  # noqa: E402


def main():
    rows = []
    for task in json.loads((ROOT / "data/repair-transfer-tasks.json").read_text()):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ([2, 5], [3, 4]):
                for scale in (1., 10.):
                    torch.manual_seed(551234)
                    args = inputs(task, shape, dtype)
                    args[0].mul_(scale)
                    before = [t.clone() for t in args]
                    x = args[0].float().cpu().numpy().astype(np.float64)
                    if task == "column_sum":
                        independent = x.sum(axis=0)
                    elif task == "l2_normalize":
                        independent = x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-6)
                    elif task == "affine_silu":
                        z = x * args[1].float().cpu().numpy() + args[2].float().cpu().numpy()
                        independent = z / (1 + np.exp(-z))
                    else:
                        independent = (x * args[1].float().cpu().numpy()).T.copy()
                    expected = torch.as_tensor(independent, device="cuda").to(dtype).contiguous()
                    actual = reference(task, *args)
                    assert actual.is_contiguous() and actual.shape == expected.shape and actual.dtype == dtype
                    assert all(torch.equal(a, b) for a, b in zip(args, before))
                    atol, rtol = (.015, .015) if dtype == torch.bfloat16 else (2e-5, 2e-4)
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    rows.append({"task": task, "dtype": str(dtype), "shape": shape, "scale": scale, "status": "passed"})
    index = torch.cuda.current_device()
    return {"status": "transfer_oracles_passed", "independent_checks": len(rows), "rows": rows,
            "scope": "Trusted Torch reference vs independent NumPy float64 equations; no model or candidate kernel",
            "device_audit": {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_index": index,
                             "device_count": torch.cuda.device_count(), "gpu_uuid": str(torch.cuda.get_device_properties(index).uuid)}}


if __name__ == "__main__":
    print(json.dumps(main(), allow_nan=False))
