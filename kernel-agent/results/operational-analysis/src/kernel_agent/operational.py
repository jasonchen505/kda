"""Task-independent evidence routing and transactional model-authored source edits."""
import ast
import json

from kernel_agent import repair, semantic
from kernel_agent.protocol import admit, sha
from kernel_agent.line_protocol import FORMAT_RULES

ARMS = ("rewrite", "operational_patch")
ROLE = "Repair the actual Triton source with an exact, minimal patch transaction."
PATCH_SCHEMA = {"type": "object", "properties": {
    "draft": {"type": "string"}, "plan": {"type": "string"},
    "base_sha256": {"type": "string"},
    "edits": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
        "type": "object", "properties": {
            "before_lines": {"type": "array", "items": {"type": "string"}},
            "after_lines": {"type": "array", "items": {"type": "string"}}},
        "required": ["before_lines", "after_lines"], "additionalProperties": False}}},
    "required": ["draft", "plan", "base_sha256", "edits"], "additionalProperties": False}


def apply(reply, source):
    """Exact unique matches against one SHA; reject the entire transaction on any error."""
    value = json.loads(reply)
    if (not isinstance(value, dict) or set(value) != set(PATCH_SCHEMA["required"])
            or not all(isinstance(value[k], str) and value[k].strip() for k in ("draft", "plan", "base_sha256"))
            or value["base_sha256"] != sha(source)):
        raise ValueError("patch must name the exact current base_sha256 and nonempty draft/plan")
    edits = value["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 4:
        raise ValueError("patch requires 1..4 edits")
    spans = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"before_lines", "after_lines"}:
            raise ValueError("edit requires before_lines and after_lines")
        before, after = edit["before_lines"], edit["after_lines"]
        if (not isinstance(before, list) or not 1 <= len(before) <= 160 or not isinstance(after, list)
                or len(after) > 200 or not all(isinstance(line, str) and "\n" not in line and "\r" not in line
                                             for line in before + after)):
            raise ValueError("edit arrays contain complete Python lines only")
        old, new = "\n".join(before) + "\n", "\n".join(after) + ("\n" if after else "")
        positions = [i for i in range(len(source)) if (i == 0 or source[i - 1] == "\n") and source.startswith(old, i)]
        if len(positions) != 1 or old == new:
            raise ValueError("before_lines must match exactly one complete-line span; no no-op edits")
        start = positions[0]
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping edits are forbidden; all edits refer to the original base")
    result = source
    for start, end, new in reversed(spans):
        result = result[:start] + new + result[end:]
    source_sha = admit(result)
    return {"draft": value["draft"], "plan": value["plan"], "code": result,
            "source_sha256": source_sha, "transaction": {
                "base_sha256": sha(source), "result_sha256": source_sha,
                "spans": [{"start": a, "end": b, "replacement_sha256": sha(c)} for a, b, c in spans]}}


def api_audit(source, capability):
    tree = ast.parse(source)
    aliases = {n.asname or n.name.split(".")[-1] for node in tree.body if isinstance(node, ast.Import)
               for n in node.names if n.name == "triton.language"}
    aliases.update(n.asname or n.name for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == "triton" for n in node.names if n.name == "language")
    shadowed = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    aliases -= shadowed
    used = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                   and isinstance(n.value, ast.Name) and n.value.id in aliases})
    missing = sorted(set(used) - set(capability["language_symbols"]))
    return {"status": "api_error" if missing else "api_admitted", "used_language_symbols": used,
            "missing_language_symbols": missing,
            "message": "Installed triton.language has no: " + ", ".join(missing) if missing else "",
            "scope": "Direct imported language aliases only; not full compilation or type proof"}


def _close(a, b):
    return a.shape == b.shape and all(abs(x - y) <= 2e-5 + 2e-4 * abs(y) for x, y in zip(a.values, b.values))


def relations(probes):
    """Find bounded observable correction relations, without a task ID or answer table."""
    usable = [p for p in probes if all(k in p and "values" in p[k] for k in ("expected", "actual"))]
    if len(usable) != len(probes) or len(usable) < 2:
        return []
    if all(_close(semantic.tensor(p["actual"]), semantic.tensor(p["expected"])) for p in usable):
        return []
    count = min(len(p["inputs"]) for p in usable)
    programs = [("-actual", lambda a, inputs: semantic.Tensor(a.shape, tuple(-v for v in a.values)))]
    for index in range(count):
        for symbol, function in [("*", lambda a, b: a * b), ("+", lambda a, b: a + b),
                                 ("-", lambda a, b: a - b), ("/", lambda a, b: a / b)]:
            def calculate(a, inputs, i=index, fn=function):
                return semantic.binary(a, semantic.tensor(inputs[i]), fn)
            programs.append((f"actual {symbol} x{index}", calculate))
    matched = []
    for expression, function in programs:
        try:
            if all(_close(function(semantic.tensor(p["actual"]), p["inputs"]), semantic.tensor(p["expected"])) for p in usable):
                matched.append({"expected_matches": expression, "probes": [p["id"] for p in usable],
                                "scope": "Finite search-probe relation, not symbolic equivalence"})
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
    return matched[:4]


def movement(probes):
    """Infer coordinate witnesses from a finite oracle permutation, including input products."""
    if len(probes) < 2:
        return []
    count = min(len(p["inputs"]) for p in probes)
    expressions = [f"x{i}" for i in range(count)] + [f"x{i}*x{j}" for i in range(count) for j in range(i + 1, count)]
    matches = []
    for expression in expressions:
        witnesses = []
        try:
            for p in probes:
                source = semantic.interpret(expression, p["inputs"])
                target = semantic.tensor(p["expected"])
                if source.shape == target.shape or len(source.values) != len(target.values):
                    raise ValueError("not a layout change")
                remaining = set(range(len(source.values)))
                examples = []
                def coord(index, shape):
                    return [index // shape[1], index % shape[1]] if len(shape) == 2 else [index]
                for out, value in enumerate(target.values):
                    candidates = [i for i, v in enumerate(source.values) if abs(v - value) <= 2e-5 + 2e-4 * abs(value)]
                    available = [i for i in candidates if i in remaining]
                    if not available:
                        raise ValueError("not a finite permutation")
                    chosen = available[0]
                    remaining.remove(chosen)
                    if len(candidates) == 1 and len(examples) < 6:
                        examples.append({"output_coordinate": coord(out, target.shape), "output_flat": out,
                                         "source_coordinate": coord(chosen, source.shape), "source_flat": chosen,
                                         "expected_value": value})
                if len(examples) < 3:
                    raise ValueError("ambiguous values")
                witnesses.append({"probe": p["id"], "source_shape": list(source.shape),
                                  "output_shape": list(target.shape), "examples": examples})
            matches.append({"source_expression": expression, "witnesses": witnesses,
                            "scope": "Complete finite multiset agreement; coordinates shown only for unique values"})
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            continue
    return matches[:2]


def diagnose(evaluation):
    probes = evaluation.get("semantic_probes", [])
    status = evaluation.get("status", "unknown")
    move = movement(probes)
    algebra = relations(probes)
    route = ("preserve_correct" if status == "correct" else "api_or_compile" if status in {
        "api_error", "compile_error", "runtime_error"} else "layout" if move else
        "algebraic_relation" if algebra else "numeric_or_contract")
    return {"route": route, "status": status, "probe_relations": algebra, "coordinate_witnesses": move,
            "scope": "Evidence-derived repair hints. No task-ID dispatch, generated reference, or automatic kernel edit."}


def anchor(history):
    return next((a for a in reversed(history) if a.get("code") and a.get("source_sha256")), None)


ROUTE_RULES = {
    "api_or_compile": "Fix the concrete compiler/API/type error with the smallest edits. Do not redesign the arithmetic. "
                      "A torch.dtype is not a Triton conversion type. Infer the type from a pointer element_ty or use a "
                      "supported tl dtype. Do not introduce APIs absent from the installed symbol inventory.",
    "layout": "The trusted coordinate examples show output-to-source correspondence on non-square arrays. Derive "
              "input/output flat offsets from their own shapes, not a shared row stride. Align pointer, loaded-value "
              "and mask axes. If the 2D tile is error-prone, replace its JIT body with simple output-linear indexing "
              "and adapt only the necessary launch arguments. A semantic plan alone is insufficient: fix the actual offsets.",
    "algebraic_relation": "The expected/actual relation is computed from both probes. Apply it to the final stored "
                          "value in the actual source; retain the existing loads, launch and API implementation. "
                          "Do not replace working exp/add/multiply with an unrelated new API. Check every input factor.",
    "numeric_or_contract": "Use every input, expected and actual output to locate the first incorrect operation. "
                           "Preserve actual reduction widths, output allocation/strides and the contracted cast order. "
                           "Fix one evidence-supported cause; avoid speculative precision changes on FP32 failures.",
    "preserve_correct": "Retain the correct arithmetic and layout. Make at most one localized performance change. "
                        "A correct retained source remains available if the patch fails.",
}


def sections(task_id, contract, history, capability):
    current = anchor(history)
    if current is None:
        raise ValueError("no admitted source exists for patching")
    evidence = {k: v for k, v in current["evaluation"].items() if k in {
        "status", "message", "shape", "dtype", "scale", "traceback", "semantic_probes"}}
    diagnosis = diagnose(current["evaluation"])
    transaction = {"base_sha256": current["source_sha256"]}
    failed = [{"request_id": r["request_id"], "source_sha256": r.get("source_sha256"),
               "status": r["evaluation"]["status"], "message": r["evaluation"].get("message")}
              for r in history if r["evaluation"]["status"] != "correct"]
    base = repair.base(contract).replace(FORMAT_RULES, "")
    instructions = ("Return draft, plan, base_sha256 and edits. Each edit has before_lines and after_lines arrays, "
                    "one Python line per item including original indentation. Each before span must occur exactly once "
                    "as complete lines in the ORIGINAL source. All edits apply atomically to that SHA, not sequentially. "
                    "Use 1..4 nonoverlapping edits; copy before_lines exactly; no tools, no complete-code response. "
                    "You may replace a full JIT function when required, while retaining unrelated code. "
                    "The controller never invents or fixes your patch. Rejection consumes this attempt and retains the base. "
                    "Keep draft/plan under 60 words each.\n" + ROUTE_RULES[diagnosis["route"]])
    return [("contract", base, True), ("patch_protocol", instructions, True),
            ("transaction", "Patch transaction:\n" + json.dumps(transaction, sort_keys=True), True),
            ("source", "Current admitted source (exact):\n" + current["code"], True),
            ("evidence", "Trusted search-only evidence:\n" + json.dumps(evidence, sort_keys=True), True),
            ("diagnosis", "Computed repair obligations:\n" + json.dumps(diagnosis, sort_keys=True), True),
            ("memory", "Prior failures and patch errors:\n" + json.dumps(failed, sort_keys=True), True),
            ("api", "Available direct triton.language symbols (installed runtime, not a model guess):\n" +
             ", ".join(capability["language_symbols"]), True)]
