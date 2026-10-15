import asyncio
import json
from pathlib import Path

import pytest
from hmz.sdk import fakes

from kernel_agent.protocol import sha


@pytest.mark.parametrize("source_format", ["string", "lines"])
def test_repair_receives_rejected_source_and_line_indentation_is_preserved(source_format):
    bad = "import torchimport triton\n"
    good = "import triton\n@triton.jit\ndef k():\n    pass\ndef run(x):\n    return x\n"

    def reply(source):
        fields = {"code": source} if source_format == "string" else {"code_lines": source.splitlines()}
        return json.dumps({"draft": "d", "plan": "p", **fields})

    builder = fakes.FakeAgentDriver(reply=[reply(bad), reply(good)])
    passed = {"status": "correct", "source_sha256": sha(good), "checks_passed": 28,
              "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in (1, 2, 3)]}
    calls = []

    def evaluate(command, env):
        calls.append(command)
        return 0, json.dumps(passed), ""

    result = asyncio.run(fakes.run_fake(str(Path(__file__).resolve().parents[2] / "flows/kernel_loop.py"),
                         "add", agents={"builder": builder}, local=fakes.FakeEnvDriver(run=evaluate),
                         params={"task_id": "vector_add", "mode": "reference_guided", "run_id": "fixture",
                                 "proposal_format": source_format, "evaluator": ["python", "evaluate.py"]}))
    assert result["correct"] and result["attempts"][0]["evaluation"]["status"] == "protocol_error"
    assert result["attempts"][0]["proposed_code"] == bad
    assert bad.strip() in builder.prompts[1]
    assert result["attempts"][1]["code"] == good and len(calls) == 2
    if source_format == "lines":
        assert "code_lines" in builder.prompts[0] and "code_lines" in builder.prompts[1]
