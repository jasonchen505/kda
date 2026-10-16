import json

import pytest

from kernel_agent.line_protocol import proposal


def test_line_protocol_preserves_indentation_blank_lines_and_literal_backslashes():
    lines = ["import triton", "", "def run(x):", "    label = 'literal\\n'", "    return x"]
    parsed = proposal(json.dumps({"draft": "d", "plan": "p", "code_lines": lines}))
    assert parsed["code"].splitlines() == lines
    assert "    label = 'literal\\n'" in parsed["code"]


@pytest.mark.parametrize("fields", [
    {"code_lines": ["def run(x):\n    return x"]},
    {"code_lines": [1]}, {"code_lines": []},
    {"code_lines": ["x"], "code": "other"}, {"code": "x"},
])
def test_malformed_or_ambiguous_lines_are_not_silently_repaired(fields):
    with pytest.raises(ValueError):
        proposal(json.dumps({"draft": "d", "plan": "p", **fields}))
