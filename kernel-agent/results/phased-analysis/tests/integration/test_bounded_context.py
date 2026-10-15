import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.protocol import sha


def test_repair_uses_fresh_session_and_omits_unusable_history():
    source = "import triton\n@triton.jit\ndef k():\n    pass\ndef run(x):\n    return x\n"
    revised = source + "\n# revision\n"
    first = json.dumps({"draft": "draft", "plan": "plan", "code": source}) + "HISTORY_JUNK" * 1000
    second = json.dumps({"draft": "draft", "plan": "plan", "code": revised})
    builder = fakes.FakeAgentDriver(reply=[first, second])
    passed = {"status": "correct", "source_sha256": sha(revised), "checks_passed": 28,
              "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in (1, 2, 3)]}
    results = iter([{"status": "numeric_error", "message": "bad tail"}, passed, passed])
    environment = fakes.FakeEnvDriver(run=lambda command, env: (0, json.dumps(next(results)), ""))
    result = asyncio.run(fakes.run_fake(str(Path(__file__).resolve().parents[2] / "flows/kernel_loop.py"),
                         "add", agents={"builder": builder}, local=environment,
                         params={"task_id": "vector_add", "mode": "reference_guided", "run_id": "fixture",
                                 "evaluator": ["python", "evaluate.py"]}))
    assert result["promoted"] and len(builder.sessions) == 2
    assert "HISTORY_JUNK" not in builder.prompts[1]
    assert "bad tail" in builder.prompts[1] and source in builder.prompts[1]
