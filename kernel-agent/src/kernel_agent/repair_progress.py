"""Conservative rename equivalence and source-bound compiler observations.

This is a refusal/feedback heuristic, not a Triton type checker or semantic proof.
Changed annotations, constants, imports, launch arguments and external names remain
different. Unsupported binding constructs disable equivalence rather than guessing.
"""
import ast
import copy
import hashlib
import re


def alpha_identity(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if any(isinstance(n, (ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp,
                          ast.DictComp, ast.GeneratorExp, ast.Global, ast.Nonlocal,
                          ast.AsyncFunctionDef)) for n in ast.walk(tree)):
        return None
    # Restrict to module functions with ordinary local bindings. Nested scopes,
    # reflection and dynamic argument binding are deliberately not normalized.
    if any(isinstance(n, ast.FunctionDef) for f in tree.body if isinstance(f, ast.FunctionDef)
           for n in ast.walk(f) if n is not f):
        return None
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
           n.func.id in {'locals', 'globals', 'eval', 'exec', 'getattr', 'setattr'}
           for n in ast.walk(tree)):
        return None
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if any(n.args.vararg or n.args.kwarg or n.args.kwonlyargs for n in functions.values()):
        return None
    # Imports are external API bindings. Keep them literal to avoid alias mistakes.
    module = {n.name: f'fn{i}' for i, n in enumerate(functions.values()) if n.name != 'run'}
    for n in tree.body:
        for item in ast.walk(n) if not isinstance(n, ast.FunctionDef) else []:
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                module.setdefault(item.id, f'global{len(module)}')
    scopes = {}
    for name, fn in functions.items():
        mapping = {a.arg: f'arg{i}' for i, a in enumerate(fn.args.posonlyargs + fn.args.args)}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                mapping.setdefault(n.id, f'local{len(mapping)}')
        scopes[name] = mapping

    class Normalize(ast.NodeTransformer):
        current = {}

        def visit_FunctionDef(self, node):
            old = self.current
            # Defaults/decorators are evaluated in the enclosing module scope.
            node.decorator_list = [self.visit(x) for x in node.decorator_list]
            node.args.defaults = [self.visit(x) for x in node.args.defaults]
            original_name = node.name
            self.current = scopes[original_name]
            for arg in node.args.posonlyargs + node.args.args:
                arg.arg = self.current[arg.arg]
                if arg.annotation:
                    arg.annotation = self.visit(arg.annotation)
            node.body = [self.visit(x) for x in node.body]
            node.name = module.get(original_name, original_name)
            self.current = old
            return node

        def visit_Name(self, node):
            node.id = self.current.get(node.id, module.get(node.id, node.id))
            return node

        def visit_Call(self, node):
            fn = node.func.value if isinstance(node.func, ast.Subscript) else node.func
            target = fn.id if isinstance(fn, ast.Name) else None
            if target in scopes:
                for kw in node.keywords:
                    kw.arg = scopes[target].get(kw.arg, kw.arg)
            return self.generic_visit(node)

    normalized = Normalize().visit(copy.deepcopy(tree))
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def compiler_observation(source, verdict):
    if verdict.get('status') not in {'compile_error', 'runtime_error'}:
        return None
    message = str(verdict.get('message', ''))
    lines = message.splitlines()
    # A unique exact source-line match binds the diagnostic without trusting the
    # compiler's function-relative line number as a module-relative line number.
    caret = next((i for i in range(len(lines)-1, 0, -1)
                  if re.fullmatch(r'\s*\^+\s*', lines[i])), None)
    excerpt = lines[caret-1].strip() if caret else None
    matches = [i for i, line in enumerate(source.splitlines(), 1)
               if excerpt and line.strip() == excerpt]
    anchor = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'diagnostic_tail': '\n'.join(lines[-2:])[-400:],
              'source_line': matches[0] if len(matches) == 1 else None,
              'failing_line': excerpt if len(matches) == 1 else None,
              'scope': 'Observed compiler location only; dependency/type causes remain hypotheses.'}
    if anchor['source_line']:
        tree = ast.parse(source)
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.lineno <= anchor['source_line'] <= n.end_lineno), None)
        if fn:
            anchor['function'] = fn.name
            anchor['parameters'] = [ast.unparse(a) for a in fn.args.args]
    return anchor
