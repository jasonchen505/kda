import asyncio
import copy
import json
from pathlib import Path

import pytest
from hmz.sdk import fakes

from kernel_agent.campaign import digest
from kernel_agent.deferred import TimingQueue
from kernel_agent.protocol import sha

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def measured(code, ratio):
    return {"status": "correct", "source_sha256": sha(code), "checks_passed": 28,
            "timings": [{"shape": [n], "compiled_us": ratio, "candidate_us": 1.} for n in (1, 2, 3)],
            "measurement_condition": "shared_observed",
            "resource_audit": {"policy": "shared_observed", "gpu_index": 4,
                               "before": {"free_mib": 10000}, "after": {"pids": [123]}, "samples": []}}


def generated(policy="feedback"):
    sources = (SOURCE, SOURCE + "# revised\n", SOURCE)
    builder = fakes.FakeAgentDriver(reply=[json.dumps({"draft": "fixture", "plan": "fixture",
                                                      "code_lines": s.splitlines()}) for s in sources])
    outcomes = iter([measured(sources[0], 1.3), measured(sources[1], 1.1)])
    commands = []

    def execute(command, env):
        commands.append(command)
        return 0, json.dumps(next(outcomes)), ""

    run = asyncio.run(fakes.run_fake(str(ROOT / "flows/kernel_loop.py"), "contract", agents={"builder": builder},
                      local=fakes.FakeEnvDriver(run=execute),
                      params={"task_id": "softmax", "mode": "reference_guided", "run_id": "phase-fixture",
                              "proposal_format": "lines", "study_policy": policy, "evaluation_mode": "search_only",
                              "evaluator": ["python", "shared.py"]}))
    return run, builder, commands


def plan(run):
    return {"generation_result": run, "generation_result_sha256": digest(run),
            "runtime": {"gpu_index": 4}, "measurement_condition": "shared_observed"}


@pytest.mark.parametrize("policy", ["blind", "feedback"])
def test_online_feedback_commits_search_without_final_data_access(policy):
    run, builder, commands = generated(policy)
    assert len(builder.prompts) == 3 and len(commands) == 2
    assert all("--final" not in command for command in commands)
    assert ("Independent evaluator status: correct" in builder.prompts[1]) == (policy == "feedback")
    assert run["measurement_state"] == "final_pending"
    assert run["selected_sha256"] == sha(SOURCE)
    assert run["final_evaluation"] is None and run["candidate_final_evaluations"] == {}
    assert not run["correct"] and not run["promoted"]  # No final verdict claimed at the phase boundary.


def test_final_interruption_reuses_committed_search_and_retains_final_veto(tmp_path):
    run, _, _ = generated()
    frozen, path = plan(run), tmp_path / "queue.json"
    with TimingQueue(path, frozen) as queue:
        assert queue.state["selection_committed"]
        assert all(row["status"] == "measured" for row in queue.state["search"])

        def crash(candidate, final):
            assert final  # Search must never be rerun in the finalization phase.
            raise RuntimeError("interruption after selection")

        with pytest.raises(RuntimeError):
            queue.run(crash)
    calls = []
    with TimingQueue(path, frozen) as queue:
        result = queue.run(lambda candidate, final: calls.append(final) or measured(candidate["code"], .8))
        assert result["correct"] and not result["promoted"]
        assert result["selected_sha256"] == run["selected_sha256"]
        assert result["final"]["observations"][0]["status"] == "interrupted_evaluation"
    assert calls == [True]
    with TimingQueue(path, frozen) as queue:
        queue.run(lambda *_: pytest.fail("complete queues cannot rerun work"))


def test_imported_search_needs_original_selection_and_resource_evidence(tmp_path):
    run, _, _ = generated()
    changed = copy.deepcopy(run)
    changed["selected_sha256"] = sha(SOURCE + "# revised\n")
    with pytest.raises(ValueError, match="committed selection"):
        with TimingQueue(tmp_path / "changed.json", plan(changed)):
            pass
    changed = copy.deepcopy(run)
    changed["attempts"][0]["evaluation"].pop("resource_audit")
    with pytest.raises(ValueError, match="boundary evidence"):
        with TimingQueue(tmp_path / "missing.json", plan(changed)):
            pass
