"""Trusted, fresh-process GPU evaluator. Candidate files never contain the reference."""
import importlib.util
import json
import os
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
    if task == "row_mean":
        return x.float().mean(-1).to(x.dtype)
    if task == "layer_norm":
        fp = x.float()
        centered = fp - fp.mean(-1, keepdim=True)
        return (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)
                * args[1].float() + args[2].float()).to(x.dtype)
    if task == "silu_mul":
        return (torch.nn.functional.silu(x.float()) * args[1].float()).to(x.dtype)
    if task == "rmsnorm":
        return args[1] * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    if task == "column_sum":
        return x.float().sum(0).to(x.dtype)
    if task == "l2_normalize":
        return (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
    if task == "affine_silu":
        z = x.float() * args[1].float() + args[2].float()
        return (z * torch.sigmoid(z)).to(x.dtype)
    if task == "scale_transpose":
        return (x.float() * args[1].float()).T.to(x.dtype).contiguous()
    raise ValueError(f"unknown task {task}")


def inputs(task, shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if task in ("vector_add", "silu_mul"):
        return (x, torch.randn_like(x))
    if task in ("rmsnorm", "scale_transpose"):
        return (x, torch.randn(shape[-1], dtype=dtype, device="cuda"))
    if task in ("layer_norm", "affine_silu"):
        return (x, torch.randn(shape[-1], dtype=dtype, device="cuda"),
                torch.randn(shape[-1], dtype=dtype, device="cuda"))
    return (x,)


def counterexample(task, module, dtype):
    """A tiny, deterministic search-only witness, identical for both repair arms."""
    shape = [7] if task in ("vector_add", "silu_mul") else [2, 3]
    args = inputs(task, shape, dtype)
    for i, tensor in enumerate(args):
        tensor.copy_(((torch.arange(tensor.numel(), device=tensor.device).reshape(tensor.shape)
                       + 1 + i) / 3).to(dtype))
    def snapshot(tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "values": tensor.float().flatten()[:64].cpu().tolist(),
                "values_truncated": tensor.numel() > 64}
    value = {"phase": "search_only", "inputs": [snapshot(t) for t in args],
             "expected": snapshot(reference(task, *args))}
    try:
        actual = module.run(*args)
        torch.cuda.synchronize()
        if isinstance(actual, torch.Tensor):
            # JSON never contains NaN/Infinity, including a failed diagnostic probe.
            value["actual"] = (snapshot(actual) if torch.isfinite(actual).all()
                               else {"nonfinite": True, "shape": list(actual.shape)})
        else:
            value["actual"] = {"python_type": type(actual).__name__}
    except Exception as error:  # noqa: BLE001 -- diagnostics do not replace the original verdict
        value["probe_error"] = str(error)[-600:]
    return value


def semantic_probes(task, source):
    """Search-only asymmetric FP32 probes; reference values are never model authored."""
    module, load_error = None, None
    try:
        admit(Path(source).read_text())
        spec = importlib.util.spec_from_file_location("semantic_candidate", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 -- a broken implementation still has a task oracle
        load_error = str(error)[-500:]
    def snapshot(t):
        if not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() or t.numel() > 64:
            return {"unobservable": True}
        return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
                "values": t.float().flatten().cpu().tolist(), "values_truncated": False}
    rows = []
    vector = task in {"vector_add", "silu_mul"}
    for index, shape in enumerate(([5], [7]) if vector else ([2, 5], [3, 4])):
        args = inputs(task, shape, torch.float32)
        for i, t in enumerate(args):
            values = torch.arange(t.numel(), device=t.device).reshape(t.shape)
            t.copy_((((values * (i + 2) + 1 + 2 * index) % (17 + 6 * index))
                     - (8 + 3 * index)) / (3 + i + index))
        expected = reference(task, *args)
        row = {"id": f"fp32-asymmetric-{index}", "phase": "search_only",
               "inputs": [snapshot(t) for t in args], "expected": snapshot(expected)}
        try:
            if load_error:
                raise ValueError(load_error)
            actual = module.run(*(t.clone() for t in args))
            torch.cuda.synchronize()
            row["actual"] = snapshot(actual)
        except Exception as error:  # noqa: BLE001 -- never replace the authoritative search verdict
            row["actual_error"] = str(error)[-500:]
        rows.append(row)
    return rows


def main(task, source, final=False, timing=True, diagnostics=False):
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
    tasks = json.loads((ROOT / "data/tasks.json").read_text())
    transfer = json.loads((ROOT / "data/repair-transfer-tasks.json").read_text())
    task_spec = {**tasks, **transfer}[task]
    test_shapes = ([[1], [33], [1025], [65537]] if task in ("vector_add", "silu_mul") else
                   [[1, 1], [3, 33], [7, 257], [17, 1536]])
    if task in ("transpose", "scale_transpose"):
        test_shapes[-1] = [17, 769]
    if final:
        task_spec["timing_shapes"] = task_spec["final_timing_shapes"] if task in transfer else {
            "vector_add": [[513], [131071], [1000003]],
            "softmax": [[13, 511], [96, 1025], [192, 2048]],
            "rmsnorm": [[13, 511], [96, 2047], [192, 4096]],
            "transpose": [[13, 511], [257, 769], [1000, 1001]],
            "row_sum": [[13, 511], [96, 2047], [192, 4096]],
            "row_mean": [[13, 511], [96, 2047], [192, 4096]],
            "layer_norm": [[13, 511], [96, 2047], [192, 4096]],
            "silu_mul": [[513], [131071], [1000003]],
        }[task]
        test_shapes = (task_spec["final_correctness_shapes"] if task in transfer else
                       [[3], [8193], [65539], [99991]] if task in ("vector_add", "silu_mul") else
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
                        failure = {"status": "numeric_error", "shape": shape, "dtype": str(dtype),
                                   "scale": scale, "message": str(error)[-2500:], "checks_passed": checks}
                        if diagnostics and not final:
                            failure["counterexample"] = counterexample(task, module, dtype)
                        return failure
                    except Exception as error:  # noqa: BLE001 -- classify candidate failures without losing the run
                        status = ("resource_exhausted" if isinstance(error, torch.OutOfMemoryError) else
                                  "compile_error" if "Compilation" in type(error).__name__ else "runtime_error")
                        return {"status": status, "shape": shape, "dtype": str(dtype), "message": str(error)[-2500:],
                                "traceback": traceback.format_exc()[-3000:], "checks_passed": checks}
        if not timing:
            return {"status": "numeric_correct", "source_sha256": source_sha, "checks_passed": checks,
                    "evaluation_phase": "search_correctness", "timings": None,
                    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
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
    device_audit = None
    try:
        index = torch.cuda.current_device()
        device_audit = {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_index": index,
                        "device_count": torch.cuda.device_count(),
                        "gpu_uuid": str(torch.cuda.get_device_properties(index).uuid)}
        if "--correctness-only" in sys.argv and "--final" in sys.argv:
            raise ValueError("deferred generation may not inspect final shapes")
        if "--semantic-probes" in sys.argv and "--final" in sys.argv:
            raise ValueError("semantic probes are search-only")
        result = main(sys.argv[1], sys.argv[2], final="--final" in sys.argv,
                      timing="--correctness-only" not in sys.argv, diagnostics="--diagnostics" in sys.argv)
        if "--semantic-probes" in sys.argv:
            result["semantic_probes"] = semantic_probes(sys.argv[1], sys.argv[2])
    except Exception as error:  # noqa: BLE001 -- always return machine-readable evaluator failure
        result = {"status": "resource_exhausted" if isinstance(error, torch.OutOfMemoryError) else "evaluator_error",
                  "message": str(error), "traceback": traceback.format_exc()[-4000:]}
    result["device_audit"] = device_audit
    print(json.dumps(result, allow_nan=False))
