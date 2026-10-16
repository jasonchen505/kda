# Explicit source-line representation

The JSON-schema preflight returned three syntactically valid proposal objects but every
decoded code string concatenated imports and statements without newlines. Their sources
failed AST admission. A JSON grammar constrains serialization, not Python source syntax.

After the frozen 12-run transport study ends, compare the code-string JSON schema against
an explicit `code_lines` array schema on the same four development tasks. Both disable
tools and use the same reference-assisted controller, model revision and evaluator.
Both arms also retain syntactically rejected proposed source and send it to the next repair
session. The older controller omitted it because it only saved admitted source; a fresh
session could see the syntax error without seeing the actual source. This shared controller
change means the new string arm must be rerun; earlier string results are not a matched control.
Only source representation and its formatting instructions change. Code lines are joined
with newline characters, preserving all leading spaces and literal backslashes. Reject
embedded newlines, non-string entries, ambiguous code/code_lines fields or oversized source.
There is no heuristic import separation, indentation repair or manual candidate editing.

Keep three-turn/4096-token limits, shuffle the eight task/condition runs with seed 20261011,
retain every result and use distinct final shapes. The same token limit does not mean
equal effective source capacity because arrays add serialization overhead. This is a
small development ablation, not a held-out benchmark or stable-success-rate estimate.
Archive both earlier and current measured sources before switching implementations.
