"""Candidate admission, feedback and terminal decisions are owned by the controller."""
import ast
import hashlib
import json
import math

ALLOWED_IMPORTS = {"torch", "triton", "math", "typing"}
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "globals", "locals"}
TORCH_HOST_CALLS = {"empty", "empty_like", "zeros", "zeros_like", "full", "full_like", "tensor", "device"}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def proposal(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and all(isinstance(value.get(k), str) and value[k].strip()
                                           for k in ("draft", "plan", "code")):
            return value
    raise ValueError("reply must contain a JSON object with nonempty draft, plan and code strings")


def admit(source):
    if len(source) > 60000:
        raise ValueError("candidate exceeds source limit")
    tree = ast.parse(source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "run" not in functions:
        raise ValueError("missing run entry point")
    has_jit = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        if any(name.split(".")[0] not in ALLOWED_IMPORTS for name in names):
            raise ValueError("only torch, triton, math and typing imports are allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            raise ValueError("dynamic execution and file access are not allowed")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                and node.func.attr not in TORCH_HOST_CALLS):
            raise ValueError("torch is for allocation and metadata, not compute")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder access is not allowed")
        if isinstance(node, ast.Attribute) and node.attr == "jit":
            has_jit = True
    if not has_jit:
        raise ValueError("a Triton JIT implementation is required")
    return sha(source)


def decision(result):
    if result.get("status") != "correct":
        return {"correct": False, "promoted": False, "reason": result.get("status", "invalid_evidence")}
    try:
        rows = result["timings"]
        if len(rows) != 3 or result.get("checks_passed", 0) < 28:
            raise ValueError("incomplete evaluation")
        if len({tuple(row["shape"]) for row in rows}) != 3:
            raise ValueError("duplicate timing shapes")
        if any(not math.isfinite(row[key]) or row[key] <= 0 for row in rows
               for key in ("compiled_us", "candidate_us")):
            raise ValueError("invalid timing")
        ratios = [r["compiled_us"] / r["candidate_us"] for r in rows]
        if any(not math.isfinite(r) or r <= 0 for r in ratios):
            raise ValueError("invalid timing")
        mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        return {"correct": True, "promoted": mean >= 1.05 and min(ratios) >= 0.90,
                "geomean_speedup": mean, "worst_speedup": min(ratios), "reason": "independent GPU evaluator"}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"correct": False, "promoted": False, "reason": "invalid_timing_evidence"}


def prefers(candidate, incumbent=None, policy="gate_first"):
    """Rank search decisions only; never consult final measurements, keep stable ties."""
    if policy not in ("gate_first", "geomean"):
        raise ValueError("unknown selection policy")
    if not candidate.get("correct"):
        return False
    if incumbent is None or not incumbent.get("correct"):
        return True
    def rank(item):
        return (bool(item["promoted"]) if policy == "gate_first" else False,
                item["geomean_speedup"])
    return rank(candidate) > rank(incumbent)


def feedback(result):
    state = result.get("status", "evaluator_error")
    actions = {"numeric_error": "Check cast points, reductions and masks. Preserve the exact task semantics.",
               "contract_error": "Fix shape/dtype/device and input mutation before optimizing.",
               "policy_error": "Use only the admitted imports and a real Triton JIT kernel.",
               "compile_error": "Fix Triton compilation/API errors before benchmarking.",
               "runtime_error": "Check pointer bounds, strides and launch grid.",
               "correct": "Correctness passed. Optimize the worst workload without breaking other cases.",
               "protocol_error": "Return valid JSON with draft, plan and complete code strings.",
               "timeout": "Reduce resource use and avoid unbounded launches."}
    return (f"Independent evaluator status: {state}. {actions.get(state, 'Inspect the failure and revise.')}\n"
            + json.dumps(result, ensure_ascii=False)[:7000]
            + "\nReturn a new complete proposal JSON. Never change the evaluator or its thresholds.")
