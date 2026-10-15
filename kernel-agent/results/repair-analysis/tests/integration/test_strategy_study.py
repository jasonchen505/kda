import asyncio
import json
from pathlib import Path

import pytest
from hmz.sdk import fakes

from kernel_agent.protocol import sha

FLOW = Path(__file__).resolve().parents[2] / "flows/kernel_loop.py"
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def reply(source):
    return json.dumps({"draft": "draft", "plan": "plan", "code_lines": source.splitlines()})


def measured(source, ratio, marker=""):
    return {"status": "correct", "source_sha256": sha(source), "checks_passed": 28, "message": marker,
            "timings": [{"shape": [n], "compiled_us": ratio, "candidate_us": 1.} for n in (1, 2, 3)]}


@pytest.mark.parametrize("policy", ["blind", "feedback"])
def test_same_budget_feedback_isolated_and_final_data_cannot_reselect(policy):
    second, third = SOURCE + "# second\n", SOURCE + "# third\n"
    builder = fakes.FakeAgentDriver(reply=[reply(SOURCE), reply(second), reply(third)])
    results = iter([measured(SOURCE, 2., "SEARCH_SENTINEL"), measured(second, .8),
                    {"status": "compile_error", "message": "broken API"},
                    measured(SOURCE, .9, "FINAL_SENTINEL"), measured(second, 3., "FINAL_SENTINEL")])
    commands = []

    def execute(command, env):
        commands.append(command)
        return 0, json.dumps(next(results)), ""

    result = asyncio.run(fakes.run_fake(str(FLOW), "contract", agents={"builder": builder},
                         local=fakes.FakeEnvDriver(run=execute),
                         params={"task_id": "rmsnorm", "mode": "reference_guided", "run_id": "fixture",
                                 "proposal_format": "lines", "study_policy": policy, "evaluator": ["python", "eval.py"]}))
    assert len(builder.prompts) == len(builder.sessions) == 3
    assert result["terminal"] == "fixed_budget_complete" and result["search_promoted"]
    assert not result["promoted"] and result["selected_sha256"] == sha(SOURCE)
    assert SOURCE in builder.prompts[1]
    assert ("SEARCH_SENTINEL" in builder.prompts[1]) == (policy == "feedback")
    assert all("FINAL_SENTINEL" not in p for p in builder.prompts)
    assert len(commands) == 5  # three search sources and two unique correct-source final checks
    assert result["candidate_final_evaluations"][sha(second)]["decision"]["promoted"]
    assert result["candidate_final_evaluations"][sha(third)]["evaluation"]["status"] == "search_rejected"


def test_duplicate_consumes_proposal_opportunity_without_repeat_gpu_measurement():
    builder = fakes.FakeAgentDriver(reply=[reply(SOURCE)] * 3)
    commands = []

    def execute(command, env):
        commands.append(command)
        return 0, json.dumps(measured(SOURCE, 2.)), ""

    result = asyncio.run(fakes.run_fake(str(FLOW), "contract", agents={"builder": builder},
                         local=fakes.FakeEnvDriver(run=execute),
                         params={"task_id": "rmsnorm", "mode": "reference_guided", "run_id": "fixture",
                                 "proposal_format": "lines", "study_policy": "blind", "evaluator": ["python", "eval.py"]}))
    assert len(builder.prompts) == 3 and len(commands) == 2
    assert [a.get("duplicate_of") for a in result["attempts"]] == [None, 1, 1]
    assert len(result["candidate_final_evaluations"]) == 1


@pytest.mark.parametrize("selection_policy", ["gate_first", "geomean"])
def test_promotion_gate_is_not_overridden_by_a_higher_mean(selection_policy):
    second, third = SOURCE + "# eligible\n", SOURCE + "# failed\n"
    builder = fakes.FakeAgentDriver(reply=[reply(SOURCE), reply(second), reply(third)])
    first_search = measured(SOURCE, 1.)
    for row, ratio in zip(first_search["timings"], (.77, 1.8, 1.8), strict=True):
        row["compiled_us"] = ratio
    second_search = measured(second, 1.3)
    chosen = second if selection_policy == "gate_first" else SOURCE
    other = SOURCE if chosen == second else second
    results = iter([first_search, second_search, {"status": "numeric_error"},
                    measured(chosen, 1.2 if chosen == second else .92, "FINAL_ONLY"),
                    measured(other, 1.2 if other == second else .92, "FINAL_ONLY")])
    result = asyncio.run(fakes.run_fake(str(FLOW), "contract", agents={"builder": builder},
                         local=fakes.FakeEnvDriver(run=lambda command, env: (0, json.dumps(next(results)), "")),
                         params={"task_id": "silu_mul", "mode": "reference_guided", "run_id": "fixture",
                                 "proposal_format": "lines", "study_policy": "feedback",
                                 "selection_policy": selection_policy, "evaluator": ["python", "eval.py"]}))
    assert result["selected_sha256"] == sha(chosen)
    assert result["promoted"] == (selection_policy == "gate_first")
    assert all("FINAL_ONLY" not in prompt for prompt in builder.prompts)
    assert not result["candidate_final_evaluations"][sha(third)]["decision"]["correct"]
