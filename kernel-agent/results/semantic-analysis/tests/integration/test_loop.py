import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.protocol import sha

FLOW = Path(__file__).resolve().parents[2] / "flows/kernel_loop.py"
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def reply(code=SOURCE):
    return json.dumps({"draft": "draft", "plan": "plan", "code": code})


def good(code=SOURCE):
    return {"status": "correct", "source_sha256": sha(code), "checks_passed": 28,
            "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in (1, 2, 3)]}


def run(replies, evaluations, mode="evidence_loop"):
    outcomes = iter(evaluations)
    final = next((row for row in reversed(evaluations) if row.get("status") == "correct"), good())
    builder = fakes.FakeAgentDriver(reply=replies)
    environment = fakes.FakeEnvDriver(run=lambda command, env: (0, json.dumps(next(outcomes, final)), ""))
    result = asyncio.run(fakes.run_fake(str(FLOW), "vector add", agents={"builder": builder}, local=environment,
                         params={"task_id": "vector_add", "mode": mode, "run_id": "fixture",
                                 "evaluator": ["python", "evaluate.py"]}))
    return result, builder


def test_numeric_failure_reaches_model_and_revised_code_is_selected():
    revised = SOURCE + "\n# revised\n"
    result, builder = run([reply(), reply(revised)],
                          [{"status": "numeric_error", "message": "odd shape masking"}, good(revised)])
    assert result["promoted"] and len(result["attempts"]) == 2
    assert "odd shape masking" in builder.prompts[1]
    assert result["selected_sha256"] == sha(revised)
    assert result["attempts"][1]["parent_sha256"] == sha(SOURCE)
    assert len(builder.sessions) == 1


def test_baseline_stops_after_one_failed_turn():
    result, builder = run([reply()], [{"status": "numeric_error"}], mode="kda_prompt")
    assert not result["correct"] and len(builder.prompts) == 1


def test_duplicate_candidate_stops_without_another_gpu_evaluation():
    result, _ = run([reply(), reply()], [{"status": "numeric_error"}])
    assert result["terminal"] == "duplicate" and not result["promoted"]


def test_protocol_failure_can_be_repaired():
    result, builder = run(["claimed success", reply()], [good()])
    assert result["promoted"] and "protocol_error" in builder.prompts[1]


def test_stale_source_evidence_is_rejected():
    measurement = good()
    measurement["source_sha256"] = "unrelated code"
    result, _ = run([reply()], [measurement], mode="kda_prompt")
    assert not result["promoted"] and result["attempts"][0]["evaluation"]["status"] == "stale_evidence"


def test_correct_slow_candidate_survives_a_later_failed_optimization():
    measurement = good()
    for row in measurement["timings"]:
        row["candidate_us"] = 3.
    revised = SOURCE + "\n# slower revision\n"
    result, _ = run([reply(), reply(revised), reply(revised)],
                    [measurement, {"status": "numeric_error"}])
    assert result["correct"] and not result["promoted"]
    assert result["selected_sha256"] == sha(SOURCE)
