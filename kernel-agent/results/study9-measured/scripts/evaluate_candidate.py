"""Trusted, fresh-process GPU evaluator. Candidate files never contain the reference."""
import importlib.util
import json
import random
import statistics
import sys
import traceback
from functools import partial
from pathlib import Path

import torch
import triton.testing

from kernel_agent.protocol import admit

ROOT = Path(__file__).resolve().parents[1]


def reference(task, *args):
    x = args[0]
    if task == "vector_add":
        return (x.float() + args[1].float()).to(x.dtype)
    if task == "softmax":
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
    if task == "transpose":
        return x.T.contiguous()
    if task == "row_sum":
        return x.float().sum(-1).to(x.dtype)
    if task == "silu_mul":
        return (torch.nn.functional.silu(x.float()) * args[1].float()).to(x.dtype)
    if task == "rmsnorm":
        return args[1] * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    raise ValueError(f"unknown task {task}")


def inputs(task, shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if task in ("vector_add", "silu_mul"):
        return (x, torch.randn_like(x))
    if task == "rmsnorm":
        return (x, torch.randn(shape[-1], dtype=dtype, device="cuda"))
    return (x,)


def main(task, source, final=False):
    code = Path(source).read_text()
    try:
        source_sha = admit(code)
    except (ValueError, SyntaxError) as error:
        return {"status": "policy_error", "message": str(error)}
    spec = importlib.util.spec_from_file_location("generated_candidate", source)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 -- generated code can raise arbitrary exceptions
        return {"status": "compile_error", "message": str(error), "traceback": traceback.format_exc()[-4000:]}
    task_spec = json.loads((ROOT / "data/tasks.json").read_text())[task]
    test_shapes = ([[1], [33], [1025], [65537]] if task in ("vector_add", "silu_mul") else
                   [[1, 1], [3, 33], [7, 257], [17, 1536]])
    if task == "transpose":
        test_shapes[-1] = [17, 769]
    if final:
        task_spec["timing_shapes"] = {
            "vector_add": [[513], [131071], [1000003]],
            "softmax": [[13, 511], [96, 1025], [192, 2048]],
            "rmsnorm": [[13, 511], [96, 2047], [192, 4096]],
            "transpose": [[13, 511], [257, 769], [1000, 1001]],
            "row_sum": [[13, 511], [96, 2047], [192, 4096]],
            "silu_mul": [[513], [131071], [1000003]],
        }[task]
        test_shapes = ([[3], [8193], [65539], [99991]] if task in ("vector_add", "silu_mul") else
                       [[2, 3], [9, 65], [5, 509], [13, 1009]])
    test_shapes += task_spec["timing_shapes"]
    torch.manual_seed(992771 if final else 771019)
    checks = 0
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            for shape in test_shapes:
                for scale in (1.0, 10.0):
                    args = inputs(task, shape, dtype)
                    args[0].mul_(scale)
                    originals = [t.clone() for t in args]
                    expected = reference(task, *args)
                    try:
                        actual = module.run(*args)
                        torch.cuda.synchronize()
                        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                                or actual.dtype != dtype or actual.device != args[0].device
                                or not actual.is_contiguous()):
                            return {"status": "contract_error", "shape": shape, "dtype": str(dtype),
                                    "message": "output shape/dtype/device/contiguity mismatch"}
                        if any(not torch.equal(t, old) for t, old in zip(args, originals)):
                            return {"status": "contract_error", "message": "input mutation"}
                        if not torch.isfinite(actual).all():
                            return {"status": "numeric_error", "message": "nonfinite output", "shape": shape}
                        atol = 0.015 if dtype == torch.bfloat16 else 2e-5
                        rtol = 0.015 if dtype == torch.bfloat16 else 2e-4
                        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                        torch.testing.assert_close(module.run(*args), expected, atol=atol, rtol=rtol)
                        checks += 1
                    except AssertionError as error:
                        return {"status": "numeric_error", "shape": shape, "dtype": str(dtype),
                                "scale": scale, "message": str(error)[-2500:], "checks_passed": checks}
                    except Exception as error:  # noqa: BLE001 -- classify candidate failures without losing the run
                        status = "compile_error" if "Compilation" in type(error).__name__ else "runtime_error"
                        return {"status": status, "shape": shape, "dtype": str(dtype), "message": str(error)[-2500:],
                                "traceback": traceback.format_exc()[-3000:], "checks_passed": checks}
        compiled = torch.compile(lambda *args: reference(task, *args), fullgraph=True,
                                 options={"emulate_precision_casts": True})
        rows = []
        rng = random.Random(20261011)
        for shape in task_spec["timing_shapes"]:
            args = inputs(task, shape, torch.bfloat16)
            funcs = {"compiled": partial(compiled, *args), "candidate": partial(module.run, *args)}
            samples = {name: [] for name in funcs}
            for fn in funcs.values():
                fn()
            torch.cuda.synchronize()
            orders = []
            for _ in range(3):
                order = list(funcs)
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    samples[name].append(triton.testing.do_bench_cudagraph(funcs[name], rep=70) * 1000)
            rows.append({"shape": shape, "dtype": "bfloat16", "compiled_us": statistics.median(samples["compiled"]),
                         "candidate_us": statistics.median(samples["candidate"]), "samples_us": samples, "orders": orders})
    return {"status": "correct", "source_sha256": source_sha, "checks_passed": checks,
            "timings": rows, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


if __name__ == "__main__":
    try:
        result = main(sys.argv[1], sys.argv[2], final="--final" in sys.argv)
    except Exception as error:  # noqa: BLE001 -- always return machine-readable evaluator failure
        result = {"status": "evaluator_error", "message": str(error), "traceback": traceback.format_exc()[-4000:]}
    print(json.dumps(result, allow_nan=False))
