"""SHA-bound indexed source edits and a budget-neutral protocol-error fallback."""
import json

from kernel_agent import operational, semantic
from kernel_agent.line_protocol import FORMAT_RULES
from kernel_agent.protocol import admit, sha

ARMS = ("rewrite", "operational_patch", "span_repair")
ROLE = "Repair actual Triton source using SHA-bound, 1-based source-line ranges."
SCHEMA = {"type": "object", "properties": {
    "draft": {"type": "string"}, "plan": {"type": "string"}, "base_sha256": {"type": "string"},
    "edits": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
        "type": "object", "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"},
                                       "replacement_lines": {"type": "array", "items": {"type": "string"}}},
        "required": ["start_line", "end_line", "replacement_lines"], "additionalProperties": False}}},
    "required": ["draft", "plan", "base_sha256", "edits"], "additionalProperties": False}


def apply(reply, source):
    value = json.loads(reply)
    if (not isinstance(value, dict) or set(value) != set(SCHEMA["required"])
            or not all(isinstance(value[k], str) and value[k].strip() for k in ("draft", "plan", "base_sha256"))
            or value["base_sha256"] != sha(source)):
        raise ValueError("indexed patch requires exact base_sha256 and nonempty draft/plan")
    if not isinstance(value["edits"], list) or not 1 <= len(value["edits"]) <= 4:
        raise ValueError("indexed patch requires 1..4 edits")
    lines = source.splitlines(keepends=True)
    spans = []
    for edit in value["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"start_line", "end_line", "replacement_lines"}:
            raise ValueError("indexed edit requires start_line/end_line/replacement_lines")
        start, end, replacement = edit["start_line"], edit["end_line"], edit["replacement_lines"]
        if not (type(start) is int and type(end) is int and 1 <= start <= end <= len(lines)):
            raise ValueError("line range must be integer, 1-based, inclusive and within the original source")
        if (not isinstance(replacement, list) or len(replacement) > 200
                or not all(isinstance(line, str) and "\r" not in line for line in replacement)):
            raise ValueError("replacement_lines requires at most200 strings without carriage returns")
        # Exact representation decoding: embedded LF is retained, never invented/indented.
        new = "\n".join(replacement) + ("\n" if replacement else "")
        spans.append((start - 1, end, new))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("indexed edits must not overlap; all ranges refer to the original base")
    result = lines[:]
    for start, end, new in reversed(spans):
        result[start:end] = [new]
    code = "".join(result)
    if code == source:
        raise ValueError("indexed transaction changes no source bytes")
    source_sha = admit(code)
    return {"draft": value["draft"], "plan": value["plan"], "code": code, "source_sha256": source_sha,
            "transaction": {"base_sha256": sha(source), "result_sha256": source_sha, "spans": [
                {"start_line": a + 1, "end_line": b, "replacement_sha256": sha(c),
                 "unchanged": "".join(lines[a:b]) == c} for a, b, c in spans]}}


def assignment(arm, history):
    parent = operational.anchor(history) if arm != "rewrite" else history[-1] if history else None
    if arm == "span_repair" and parent is not None:
        fallback = bool(history and history[-1].get("mode") == "span_patch"
                        and history[-1]["evaluation"]["status"] == "patch_error")
        return ("full_source_fallback" if fallback else "span_patch"), parent
    return ("patch" if arm == "operational_patch" and parent is not None else "full_source"), parent


def context(task_id, contract, history, capability, mode):
    if mode == "full_source":
        return semantic.baseline_sections(task_id, contract, history), {}
    if mode == "patch":
        return operational.sections(task_id, contract, history, capability), {
            "schema": operational.PATCH_SCHEMA, "schema_name": "source_patch", "role_prefix": operational.ROLE}
    current = operational.anchor(history)
    route = operational.diagnose(current["evaluation"])["route"]
    parts = operational.sections(task_id, contract, history, capability)
    if mode == "full_source_fallback":
        instructions = (FORMAT_RULES + " Your preceding indexed patch was rejected. This consumes the remaining "
                        "repair slot; return the COMPLETE source now, with no patch fields. Preserve the actual "
                        "contract and repair from the trusted evidence. Keep draft/plan under60 words each.\n" +
                        operational.ROUTE_RULES[route])
        return [(name, instructions if name == "patch_protocol" else text, mandatory)
                for name, text, mandatory in parts if name != "transaction"], {}
    if mode != "span_patch":
        raise ValueError("unknown repair mode")
    instructions = ("Return draft, plan, base_sha256 and1..4 edits. Each edit has start_line, end_line and "
                    "replacement_lines. Line ranges are1-based INCLUSIVE in the ORIGINAL displayed source. "
                    "Replace that entire range with your exact replacement strings joined by LF and a final LF; "
                    "empty array deletes it. Embedded LF in a string is retained verbatim. Preserve indentation; "
                    "never include line-number prefixes in replacement code. All ranges apply atomically against "
                    "the supplied SHA and must not overlap. Unchanged spans are allowed alongside a real edit; "
                    "the transaction must change source. No tools or complete-code response. On patch rejection, "
                    "the remaining slot switches to full-source recovery without adding calls. "
                    "Keep draft/plan under60 words each.\n" + operational.ROUTE_RULES[route])
    numbered = "Current source with1-based line numbers (prefix is NOT code):\n" + "\n".join(
        f"{i:04d}|{line}" for i, line in enumerate(current["code"].splitlines(), 1))
    return [(name, instructions if name == "patch_protocol" else numbered if name == "source" else text, mandatory)
            for name, text, mandatory in parts], {"schema": SCHEMA, "schema_name": "indexed_source_patch", "role_prefix": ROLE}
