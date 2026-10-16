"""Indexed edits preserve authored bytes, enforce source identity and bound fallback."""
import json

import pytest

from kernel_agent import span_repair
from kernel_agent.protocol import sha

SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def reply(edits, base=SOURCE):
    return json.dumps({"draft": "model draft", "plan": "model plan", "base_sha256": sha(base), "edits": edits})


def test_exact_ranges_allow_repeated_source_lines_and_atomic_offsets():
    source = SOURCE.replace("    pass", "    pass\n    pass")
    edits = [{"start_line": 4, "end_line": 4, "replacement_lines": ["    # added", "    pass"]},
             {"start_line": 5, "end_line": 5, "replacement_lines": ["    # second", "    pass"]}]
    value = span_repair.apply(reply(edits, source), source)
    assert value["code"] == SOURCE.replace("    pass", "    # added\n    pass\n    # second\n    pass")
    assert value["transaction"]["result_sha256"] == sha(value["code"])
    assert value["transaction"]["spans"][1]["start_line"] == 5


def test_lossless_embedded_lf_and_unchanged_subedit_do_not_invent_code():
    edits = [{"start_line": 1, "end_line": 1, "replacement_lines": ["import triton"]},
             {"start_line": 4, "end_line": 4, "replacement_lines": ["    # exact\n    pass"]}]
    value = span_repair.apply(reply(edits), SOURCE)
    assert value["code"] == SOURCE.replace("    pass", "    # exact\n    pass")
    assert value["transaction"]["spans"][0]["unchanged"]


@pytest.mark.parametrize("edit", [
    {"start_line": 0, "end_line": 1, "replacement_lines": []},
    {"start_line": 1, "end_line": 99, "replacement_lines": []},
    {"start_line": True, "end_line": 1, "replacement_lines": []},
    {"start_line": 4, "end_line": 3, "replacement_lines": []},
    {"start_line": 4, "end_line": 4, "replacement_lines": ["bad python : ???"]},
    {"start_line": 4, "end_line": 4, "replacement_lines": ["    pass\r"]},
    {"start_line": 4, "end_line": 4, "replacement_lines": "text"},
    {"start_line": 4, "end_line": 4, "replacement_lines": ["    pass"]},
])
def test_rejects_invalid_ranges_bytes_source_and_whole_noops(edit):
    with pytest.raises((ValueError, SyntaxError)):
        span_repair.apply(reply([edit]), SOURCE)


def test_rejects_stale_sha_and_overlapping_edits_atomically():
    edit = {"start_line": 4, "end_line": 4, "replacement_lines": ["    # added", "    pass"]}
    with pytest.raises(ValueError, match="exact base"):
        span_repair.apply(reply([edit], SOURCE + "\n"), SOURCE)
    with pytest.raises(ValueError, match="overlap"):
        span_repair.apply(reply([edit, edit]), SOURCE)


def test_fallback_is_existing_second_slot_only_and_keeps_admitted_source():
    start = {"code": SOURCE, "source_sha256": sha(SOURCE), "evaluation": {"status": "compile_error"}}
    rejected = {"mode": "span_patch", "evaluation": {"status": "patch_error", "message": "bad range"}}
    assert span_repair.assignment("span_repair", [start])[0] == "span_patch"
    assert span_repair.assignment("span_repair", [start, rejected]) == ("full_source_fallback", start)
    assert span_repair.assignment("operational_patch", [start, rejected]) == ("patch", start)
    rejected["evaluation"]["status"] = "numeric_error"
    assert span_repair.assignment("span_repair", [start, rejected])[0] == "span_patch"
    assert span_repair.assignment("span_repair", []) == ("full_source", None)
