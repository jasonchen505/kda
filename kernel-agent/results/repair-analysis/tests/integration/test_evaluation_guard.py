import asyncio
import json
import sys

from hmz.sdk import fakes

from scripts import evaluate_checked
from scripts.validate_recovery import ROOT, SOURCE


def command():
    # Actual CPU child process; its JSON is synthetic evidence, with no CUDA allocation.
    return [sys.executable, "-c", "print('{\"status\":\"correct\",\"fixture\":true}')"]


def test_busy_gpu_prevents_evaluator_side_effects(monkeypatch, tmp_path):
    sentinel = tmp_path / "must-not-exist"
    monkeypatch.setattr(evaluate_checked, "occupants", lambda gpu: [999999])
    value = evaluate_checked.evaluate([sys.executable, "-c", f"open({str(sentinel)!r},'w').write('started')"], 4)
    assert value["status"] == "resource_busy" and not sentinel.exists()


def test_new_foreign_process_vetoes_original_child_verdict(monkeypatch):
    readings = iter([[], [999999]])
    monkeypatch.setattr(evaluate_checked, "occupants", lambda gpu: next(readings))
    value = evaluate_checked.evaluate(command(), 4)
    assert value["status"] == "resource_contended"
    assert value["discarded_evaluation"] == {"status": "correct", "fixture": True}


def test_uncontended_child_preserves_verdict_and_boundary_evidence(monkeypatch):
    monkeypatch.setattr(evaluate_checked, "occupants", lambda gpu: [])
    value = evaluate_checked.evaluate(command(), 4)
    assert value["status"] == "correct"
    assert value["resource_audit"]["before_pids"] == value["resource_audit"]["after_pids"] == []


def test_monitor_failure_cannot_fall_through_to_evaluation(monkeypatch):
    def unavailable(gpu):
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr(evaluate_checked, "occupants", unavailable)
    value = evaluate_checked.evaluate(command(), 4)
    assert value["status"] == "resource_monitor_error"


def test_resource_failure_stops_real_humanize_flow_without_model_repair():
    reply = json.dumps({"draft": "fixture", "plan": "fixture", "code_lines": SOURCE.splitlines()})
    builder = fakes.FakeAgentDriver(reply=[reply] * 3)
    result = asyncio.run(fakes.run_fake(str(ROOT / "flows/kernel_loop.py"), "contract",
                        agents={"builder": builder},
                        local=fakes.FakeEnvDriver(run=lambda command, env: (0, json.dumps({"status": "resource_busy"}), "")),
                        params={"task_id": "rmsnorm", "mode": "reference_guided", "run_id": "resource-fixture",
                                "proposal_format": "lines", "study_policy": "feedback",
                                "evaluator": ["python", "checked-eval.py"]}))
    assert result["terminal"] == "resource_unavailable" and len(builder.prompts) == 1
    assert not result["correct"] and not result["promoted"]
