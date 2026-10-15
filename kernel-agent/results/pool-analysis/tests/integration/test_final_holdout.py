import asyncio
import json
from pathlib import Path

import pytest
from hmz.sdk import fakes

from kernel_agent.protocol import sha


@pytest.mark.parametrize("final_kind", ["numeric_error", "correct_but_slow"])
def test_final_failure_overrules_search_promotion_without_feedback(final_kind):
    source = "import triton\n@triton.jit\ndef k():\n    pass\ndef run(x):\n    return x\n"
    model = fakes.FakeAgentDriver(reply=json.dumps({"draft": "draft", "plan": "plan", "code": source}))
    final = ({"status": "numeric_error", "message": "final-only shape fails"} if final_kind == "numeric_error" else
             {"status": "correct", "source_sha256": sha(source), "checks_passed": 28,
              "timings": [{"shape": [n], "compiled_us": 1.01, "candidate_us": 1.} for n in (5, 6, 7)]})
    results = iter([{"status": "correct", "source_sha256": sha(source), "checks_passed": 28,
                     "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in (1, 2, 3)]},
                    final])
    environment = fakes.FakeEnvDriver(run=lambda command, env: (0, json.dumps(next(results)), ""))
    result = asyncio.run(fakes.run_fake(str(Path(__file__).resolve().parents[2] / "flows/kernel_loop.py"),
                         "add", agents={"builder": model}, local=environment,
                         params={"task_id": "vector_add", "mode": "evidence_loop", "run_id": "fixture",
                                 "evaluator": ["python", "evaluate.py"]}))
    assert result["search_promoted"] and not result["promoted"]
    assert result["correct"] == (final_kind == "correct_but_slow")
    assert len(model.prompts) == 1
