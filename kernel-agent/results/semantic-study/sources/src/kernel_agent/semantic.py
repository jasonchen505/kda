"""Bounded, non-eval mathematical interpretation and grounded reviewer contexts."""
import ast
import itertools
import json
import math
from dataclasses import dataclass

from kernel_agent import repair

ARMS = ("rewrite", "critic", "grounded")
REVIEW_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in
                 ["hypothesis", "reference_expression", "candidate_expression", "repair_plan"]},
                 "required": ["hypothesis", "reference_expression", "candidate_expression", "repair_plan"],
                 "additionalProperties": False}
GRAMMAR = (
    "Return hypothesis, reference_expression, candidate_expression and repair_plan strings. "
    "Expressions describe FP32 logical tensor mathematics, not Python/Triton source. Names x0,x1,x2 "
    "refer to probe inputs. Only finite numeric literals, + - * /, unary minus, exp(x), sigmoid(x), "
    "rsqrt(x), transpose(x), sum_last(x, keepdim), mean_last(x, keepdim), max_last(x, keepdim) are allowed. "
    "keepdim is True or False. Reductions use the actual final dimension; standard trailing-dimension "
    "broadcasting applies. No attributes, subscripts, assignments or tools. Write complete mathematics "
    "including every multiplicative factor before considering layout/casts. reference_expression must "
    "match every expected probe; candidate_expression should describe actual candidate output if possible, "
    "otherwise use the empty string. Do not blame BF16 for FP32 failures or invent a test result. "
    "Keep hypothesis and repair_plan under 100 words each."
)


@dataclass(frozen=True)
class Tensor:
    shape: tuple
    values: tuple

    def __post_init__(self):
        if len(self.shape) > 2 or math.prod(self.shape) != len(self.values) or len(self.values) > 64:
            raise ValueError("only complete rank<=2 tensors of at most 64 values")
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError("nonfinite semantic value")


def tensor(value):
    if value.get("values_truncated"):
        raise ValueError("truncated semantic input")
    return Tensor(tuple(value["shape"]), tuple(value["values"]))


def binary(left, right, function):
    n = max(len(left.shape), len(right.shape))
    a, b = (1,) * (n - len(left.shape)) + left.shape, (1,) * (n - len(right.shape)) + right.shape
    if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
        raise ValueError("incompatible broadcast shapes")
    shape = tuple(max(x, y) for x, y in zip(a, b))
    def at(t, padded, coordinates):
        offset = 0
        for d, index in zip(padded, coordinates):
            offset = offset * d + (index if d != 1 else 0)
        return t.values[offset]
    return Tensor(shape, tuple(function(at(left, a, c), at(right, b, c))
                               for c in itertools.product(*(range(d) for d in shape))))


def interpret(expression, inputs):
    if not isinstance(expression, str) or not 0 < len(expression) <= 1200:
        raise ValueError("expression length must be 1..1200")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 160:
        raise ValueError("semantic expression too large")
    names = {f"x{i}": tensor(x) for i, x in enumerate(inputs)}
    def visit(node, depth=0):
        if depth > 24:
            raise ValueError("semantic expression too deep")
        def child(x):
            return visit(x, depth + 1)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1e6:
                raise ValueError("unsupported constant")
            return Tensor((), (float(node.value),))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            t = child(node.operand)
            return Tensor(t.shape, tuple(-v if isinstance(node.op, ast.USub) else v for v in t.values))
        if isinstance(node, ast.BinOp):
            functions = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                         ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}
            if type(node.op) in functions:
                return binary(child(node.left), child(node.right), functions[type(node.op)])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in {"exp", "sigmoid", "rsqrt", "transpose"} and len(node.args) == 1:
                t = child(node.args[0])
                if name == "transpose":
                    if len(t.shape) != 2:
                        raise ValueError("transpose requires rank two")
                    m, n = t.shape
                    return Tensor((n, m), tuple(t.values[i * n + j] for j in range(n) for i in range(m)))
                functions = {"exp": math.exp, "sigmoid": lambda v: 1 / (1 + math.exp(-v)),
                             "rsqrt": lambda v: 1 / math.sqrt(v)}
                return Tensor(t.shape, tuple(functions[name](v) for v in t.values))
            if name in {"sum_last", "mean_last", "max_last"} and len(node.args) == 2:
                keep = node.args[1]
                if not isinstance(keep, ast.Constant) or type(keep.value) is not bool:
                    raise ValueError("keepdim must be a literal boolean")
                t = child(node.args[0])
                if not t.shape or t.shape[-1] <= 0:
                    raise ValueError("reduction requires a nonempty last dimension")
                width = t.shape[-1]
                groups = [t.values[i:i + width] for i in range(0, len(t.values), width)]
                values = tuple(max(v) if name == "max_last" else math.fsum(v) /
                               (width if name == "mean_last" else 1) for v in groups)
                return Tensor(t.shape[:-1] + ((1,) if keep.value else ()), values)
        raise ValueError("unsupported semantic syntax")
    return visit(tree.body)


def check_expression(expression, probes, target):
    rows = []
    if not probes:
        return {"valid": False, "reason": "no trusted probes", "probes": []}
    for probe in probes:
        expected = probe.get(target)
        if not expected or "values" not in expected:
            rows.append({"probe": probe["id"], "status": "unobservable"})
            continue
        try:
            predicted = interpret(expression, probe["inputs"])
            actual = tensor(expected)
            matching = predicted.shape == actual.shape and all(
                abs(a - b) <= 2e-5 + 2e-4 * abs(b) for a, b in zip(predicted.values, actual.values))
            rows.append({"probe": probe["id"], "status": "match" if matching else "mismatch",
                         "predicted": {"shape": list(predicted.shape), "values": list(predicted.values)},
                         "target_shape": list(actual.shape)})
        except (ValueError, SyntaxError, TypeError, OverflowError, ZeroDivisionError) as error:
            rows.append({"probe": probe["id"], "status": "invalid", "error": str(error)[:300]})
    return {"valid": all(r["status"] == "match" for r in rows), "probes": rows}


def validate_review(review, probes):
    if (not isinstance(review, dict) or set(review) != set(REVIEW_SCHEMA["required"])
            or not all(isinstance(v, str) and len(v) <= 4000 for v in review.values())):
        return {"status": "invalid_review", "reference": {"valid": False}, "candidate": {"valid": False}}
    reference = check_expression(review["reference_expression"], probes, "expected")
    candidate = check_expression(review["candidate_expression"], probes, "actual")
    return {"status": "reference_validated" if reference["valid"] else "reference_rejected",
            "reference": reference, "candidate": candidate,
            "scope": "FP32 finite-probe agreement, not symbolic equivalence or full-kernel proof"}


def reviewer_sections(contract, start):
    evidence = {k: v for k, v in start["evaluation"].items() if k in {
        "status", "message", "dtype", "shape", "scale", "semantic_probes", "traceback"}}
    return [("review_contract", "Review a kernel's mathematical semantics independently.\n" + contract + "\n" + GRAMMAR, True),
            ("source", "Exact candidate source:\n" + start.get("code", ""), True),
            ("evidence", json.dumps(evidence, sort_keys=True), True)]


def baseline_sections(task_id, contract, history=None):
    # Canonical nested key order makes recorded request reconstruction byte-exact.
    history = json.loads(json.dumps(history or [], sort_keys=True))
    return repair.sections(task_id, contract, "baseline", history)


def writer_sections(task_id, contract, start, review, validation, grounded):
    parts = baseline_sections(task_id, contract, [start])
    probes = start["evaluation"].get("semantic_probes", [])
    if grounded:
        admitted = {"validation": validation, "trusted_search_probes": probes}
        if validation["reference"]["valid"]:
            admitted["validated_reference_expression"] = review["reference_expression"]
        if validation["candidate"]["valid"]:
            admitted["validated_candidate_expression"] = review["candidate_expression"]
        context = ("Machine-checked semantic review. Unvalidated hypotheses/plans are withheld. "
                   "Implement every factor and axis of the validated reference expression. If rejected, use only "
                   "the contract, numeric evidence and validation errors. Probe agreement is not a correctness proof; "
                   "the full contract and independent evaluator still govern BF16, masks and layout.\n" + json.dumps(admitted, sort_keys=True))
    else:
        context = "Unverified independent review:\n" + json.dumps(review, sort_keys=True) + "\nTrusted search probes:\n" + json.dumps(probes, sort_keys=True)
    # Insert before optional API/example context; the entire grounding evidence is mandatory.
    parts.insert(3, ("semantic_review", context, True))
    return parts
