import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.protocol import sha


def test_api_reference_context_is_visible_to_the_real_flow_generator():
    source = "import triton\n@triton.jit\ndef k():\n    pass\ndef run(x):\n    return x\n"
    builder = fakes.FakeAgentDriver(reply=json.dumps({"draft": "draft", "plan": "plan", "code": source}))
    result = {"status": "correct", "source_sha256": sha(source), "checks_passed": 28,
              "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in (1, 2, 3)]}
    environment = fakes.FakeEnvDriver(run=lambda command, env: (0, json.dumps(result), ""))
    outcome = asyncio.run(fakes.run_fake(str(Path(__file__).resolve().parents[2] / "flows/kernel_loop.py"),
                           "add", agents={"builder": builder}, local=environment,
                           params={"task_id": "vector_add", "mode": "api_guided", "run_id": "fixture",
                                   "evaluator": ["python", "evaluate.py"]}))
    assert "pass torch tensors directly" in builder.prompts[0]
    assert outcome["initial_reference_context"]["sources"]["launch"].endswith("01-vector-add.py")
    assert outcome["promoted"]
