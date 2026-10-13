import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest
from hmz.sdk import fakes

from kernel_agent.campaign import digest
from kernel_agent.deferred import TimingQueue
from kernel_agent.protocol import decision, sha
from scripts import evaluate_shared
from scripts.run_deferred import queue_plan

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def numeric(source):
    return {"status": "numeric_correct", "evaluation_phase": "search_correctness",
            "checks_passed": 28, "source_sha256": sha(source), "timings": None}


def generated(policy="feedback"):
    builder = fakes.FakeAgentDriver(reply=[json.dumps({"draft": "fixture", "plan": "fixture", "code_lines": s.splitlines()})
                                         for s in (SOURCE, SOURCE + "# new\n", SOURCE)])
    outcomes = iter([numeric(SOURCE), numeric(SOURCE + "# new\n")])
    commands = []

    def execute(command, env):
        commands.append(command)
        return 0, json.dumps(next(outcomes)), ""

    result = asyncio.run(fakes.run_fake(str(ROOT / "flows/kernel_loop.py"), "contract", agents={"builder": builder},
                        local=fakes.FakeEnvDriver(run=execute),
                        params={"task_id": "softmax", "mode": "reference_guided", "run_id": "deferred-fixture",
                                "proposal_format": "lines", "study_policy": policy, "evaluation_mode": "deferred",
                                "evaluator": ["python", "shared.py"]}))
    return result, builder, commands


def plan():
    run, _, _ = generated()
    return queue_plan({"plan_sha256": "generation-fixture", "plan": {"runtime": {"gpu_index": 4}, "source_sha256": {}}}, run)


def measured(candidate, ratio):
    return {"status": "correct", "source_sha256": candidate["source_sha256"], "checks_passed": 28,
            "timings": [{"shape": [n], "compiled_us": ratio, "candidate_us": 1.} for n in (1, 2, 3)],
            "resource_audit": {"gpu_index": 4, "before_pids": [], "after_pids": []}}


@pytest.mark.parametrize("policy", ["feedback", "blind"])
def test_shared_generation_never_requests_final_or_timing_and_retains_unique_candidates(policy):
    run, builder, commands = generated(policy)
    assert len(builder.prompts) == 3 and len(commands) == 2
    assert all("--final" not in command for command in commands)
    assert run["final_evaluation"] is None and not run["correct"] and not run["promoted"]
    assert run["measurement_state"] == "timing_pending"
    assert run["timing_candidates"] == [sha(SOURCE), sha(SOURCE + "# new\n")]
    assert ("Independent evaluator status: numeric_correct" in builder.prompts[1]) == (policy == "feedback")
    assert not decision(numeric(SOURCE))["correct"]  # Search correctness cannot impersonate final evidence.


def test_busy_timing_queue_resumes_without_model_calls_and_final_cannot_reselect(tmp_path):
    frozen = plan()
    path = tmp_path / "queue.json"
    with TimingQueue(path, frozen) as queue:
        queue.run(lambda candidate, final: {"status": "resource_busy"})
        assert queue.state["status"] == "awaiting_measurement_resources"
        assert not queue.state["selection_committed"]
    calls = []

    def evaluate(candidate, final):
        calls.append((candidate["source_sha256"], final))
        return measured(candidate, .8 if final else 1.4 if candidate["iteration"] == 1 else 1.2)

    with TimingQueue(path, frozen) as queue:
        result = queue.run(evaluate)
        assert result["status"] == "complete" and result["correct"] and not result["promoted"]
        assert result["selected_sha256"] == sha(SOURCE)  # Final slowdown never selects another source.
        assert len(result["search"][0]["observations"]) == 2  # Busy observation is preserved.
    assert len(calls) == 3
    with TimingQueue(path, frozen) as queue:
        queue.run(lambda *_: pytest.fail("committed measurements must be reused"))


def test_interrupted_pure_evaluation_can_retry_but_selection_is_already_frozen(tmp_path):
    frozen, path = plan(), tmp_path / "queue.json"
    with TimingQueue(path, frozen) as queue:
        def crash(candidate, final):
            if final:
                raise RuntimeError("simulate interruption after search selection")
            return measured(candidate, 1.4 if candidate["iteration"] == 1 else 1.2)
        with pytest.raises(RuntimeError):
            queue.run(crash)
        assert queue.state["selection_committed"] and queue.state["final"]["status"] == "running"
    calls = []
    with TimingQueue(path, frozen) as queue:
        result = queue.run(lambda candidate, final: (calls.append(final) or measured(candidate, 1.3)))
        assert result["promoted"]
        assert result["final"]["observations"][0]["status"] == "interrupted_evaluation"
    assert calls == [True]


def test_source_tampering_and_foreign_timing_evidence_are_rejected(tmp_path):
    frozen = plan()
    changed = copy.deepcopy(frozen)
    changed["generation_result"]["attempts"][0]["code"] += "# manual change\n"
    changed["generation_result_sha256"] = digest(changed["generation_result"])
    with pytest.raises(ValueError, match="source differs"):
        with TimingQueue(tmp_path / "tampered.json", changed):
            pass
    with TimingQueue(tmp_path / "queue.json", frozen) as queue:
        def contended(candidate, final):
            value = measured(candidate, 2.)
            value["resource_audit"]["after_pids"] = [12345]
            return value
        with pytest.raises(ValueError, match="boundary evidence"):
            queue.run(contended)


def test_shared_numeric_child_runs_with_foreign_pids_and_cannot_return_speedups(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_shared, "observe", lambda _: {"free_mib": 8000, "pids": [12345]})
    marker = tmp_path / "ran"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('yes'); "
               "print('{\"status\":\"numeric_correct\",\"timings\":null}')"]
    value = evaluate_shared.evaluate(command, 4)
    assert marker.exists() and value["status"] == "numeric_correct"
    assert value["resource_audit"]["before"]["pids"] == [12345]
    value = evaluate_shared.evaluate([sys.executable, "-c", "print('{\"status\":\"correct\",\"timings\":[]}')"], 4)
    assert value["status"] == "resource_monitor_error"


def test_low_memory_prevents_shared_evaluator_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_shared, "observe", lambda _: {"free_mib": 100})
    marker = tmp_path / "must-not-run"
    value = evaluate_shared.evaluate([sys.executable, "-c", f"open({str(marker)!r},'w').write('ran')"], 4)
    assert value["status"] == "resource_exhausted" and not marker.exists()
