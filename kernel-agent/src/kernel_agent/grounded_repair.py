"""Read-only source facts for bounded indexed repair; never edits model candidates."""
import ast
import builtins
import json

from kernel_agent import operational, span_repair
from kernel_agent.protocol import sha

ARMS = ("indexed", "grounded")
RULES = (
    "Use the source facts to repair the implementation, not just its explanation. "
    "Before editing a signature, check existing parameter names and every recorded launch binding; "
    "do not add an existing parameter twice. Preserve tl.constexpr annotations unless evidence requires a change. "
    "If a conversion receives a wrapper tensor.dtype, fix the value passed at launch or derive a Triton "
    "conversion type from the pointer; changing only the annotation does not convert a torch.dtype. "
    "For each store, trace the value, pointer and mask dependencies. Their lane axes must describe the "
    "same output coordinates, including on non-square inputs. Check that against the trusted coordinate "
    "witnesses; changing strides alone may leave the value axes wrong. "
    "Facts are partial source observations, not a semantic proof. Do not invent missing facts or claim tests ran. "
    "Keep the existing two-call budget and output schema."
)


def facts(source):
    """Compile without execution; expose direct signatures, launch bindings and store slices."""
    result = {"source_sha256": sha(source), "scope": "Partial direct AST facts; no execution, no numerical proof",
              "python_compile": "passed", "functions": [], "launches": [], "stores": []}
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as error:
        result["python_compile"] = {"line": error.lineno, "message": error.msg}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result
    aliases = {n.asname or n.name.rsplit(".", 1)[-1] for item in tree.body if isinstance(item, ast.Import)
               for n in item.names if n.name == "triton.language"}
    aliases |= {n.asname or n.name for item in tree.body if isinstance(item, ast.ImportFrom) and item.module == "triton"
                for n in item.names if n.name == "language"}
    def text(n):
        value = ast.unparse(n)
        return value if len(value) <= 110 else value[:107] + "..."
    def language_call(n, name):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name
                and isinstance(n.func.value, ast.Name) and n.func.value.id in aliases)
    functions = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    module_names = set(functions)
    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            module_names.update(n.asname or n.name.split(".")[0] for n in item.names)
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            module_names.update(n.id for n in ast.walk(item) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    for f in functions.values():
        jit = any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in f.decorator_list)
        args = f.args.posonlyargs + f.args.args + f.args.kwonlyargs
        if jit:
            local = {a.arg for a in args} | {n.id for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            unresolved = sorted({n.id for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                                - local - module_names - set(dir(builtins)))
            result["functions"].append({"name": f.name, "line": f.lineno,
                                        "parameters": [a.arg + (":" + text(a.annotation) if a.annotation else "") for a in args],
                                        "unresolved_names_partial": unresolved[:8]})
        assigned = {}
        # Only straight-line statements are resolved. Branch/loop assignments are not claimed as bindings.
        for statement in f.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        assigned[target.id] = statement.value
            for n in ast.walk(statement):
                if not isinstance(n, ast.Call):
                    continue
                if isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name) and n.func.value.id in functions:
                    target = functions[n.func.value.id]
                    positional = target.args.posonlyargs + target.args.args
                    bindings = {a.arg: text(v) for a, v in zip(positional, n.args) if not isinstance(v, ast.Starred)}
                    bindings.update({k.arg: text(k.value) for k in n.keywords if k.arg})
                    result["launches"].append({"kernel": target.name, "line": n.lineno, "bindings": bindings,
                        "partial": any(isinstance(v, ast.Starred) for v in n.args) or any(k.arg is None for k in n.keywords)})
                if language_call(n, "store"):
                    expressions = [text(v) for v in n.args[:3]] + [k.arg + "=" + text(k.value) for k in n.keywords if k.arg]
                    needed = {q.id for v in n.args[:3] + [k.value for k in n.keywords] for q in ast.walk(v) if isinstance(q, ast.Name)}
                    deps = {}
                    for _ in range(3):
                        for name in sorted(needed):
                            if name in assigned and name not in deps and len(deps) < 10:
                                deps[name] = text(assigned[name])
                        needed = {q.id for name in deps for q in ast.walk(assigned[name]) if isinstance(q, ast.Name)}
                    result["stores"].append({"line": n.lineno, "arguments": expressions, "dependencies": deps})
    # Explicit bounded packet: omitted facts are counted, not silently asserted as complete.
    for name, cap in (("functions", 2), ("launches", 2), ("stores", 2)):
        result[name + "_omitted"] = max(0, len(result[name]) - cap)
        result[name] = result[name][:cap]
    while len(json.dumps(result, separators=(",", ":"))) > 3300 and result["stores"]:
        result["stores"].pop()
        result["stores_omitted"] += 1
    return result


def assignment(arm, history):
    if arm not in ARMS:
        raise ValueError("unknown source-grounded arm")
    return span_repair.assignment("span_repair", history)


def context(task_id, contract, history, capability, mode, arm):
    if arm not in ARMS:
        raise ValueError("unknown source-grounded arm")
    parts, settings = span_repair.context(task_id, contract, history, capability, mode)
    parent = operational.anchor(history)
    if arm == "grounded" and parent:
        packet = json.dumps(facts(parent["code"]), sort_keys=True, separators=(",", ":"))
        # Mandatory and source-bound. No candidate rewriting or extra critic call.
        parts.append(("source_facts", "Verified source facts:\n" + packet + "\nRepair obligations:\n" + RULES, True))
    return parts, settings
