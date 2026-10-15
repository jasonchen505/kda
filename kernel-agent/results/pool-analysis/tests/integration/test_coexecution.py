import copy
import json
import sys

import pytest

from kernel_agent.campaign import digest
from kernel_agent.deferred import TimingQueue
from kernel_agent.protocol import sha
from scripts import evaluate_coexecuted, run_campaign
from scripts.summarize_campaign import check_resource_evidence

SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def telemetry():
    return {"free_mib": 12000, "used_mib": 10437, "utilization_percent": 0, "pids": [98765]}


def test_resident_process_does_not_prevent_actual_child_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_coexecuted, "observe", lambda gpu: telemetry())
    marker = tmp_path / "executed"
    command = [sys.executable, "-c", f"open({str(marker)!r},'w').write('yes');print('{{\"status\":\"correct\"}}')"]
    value = evaluate_coexecuted.evaluate(command, 4)
    assert marker.exists() and value["status"] == "correct"
    assert value["measurement_condition"] == "shared_observed"
    assert value["resource_audit"]["before"]["pids"] == [98765]


def test_missing_monitor_evidence_cannot_become_a_shared_speedup(monkeypatch):
    observations = iter([telemetry(), OSError("device query unavailable")])

    def observe(gpu):
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(evaluate_coexecuted, "observe", observe)
    value = evaluate_coexecuted.evaluate([sys.executable, "-c", "print('{\"status\":\"correct\"}')"], 4)
    assert value["status"] == "resource_monitor_error"


def plan():
    numeric = {"status": "numeric_correct", "checks_passed": 28, "timings": None,
               "evaluation_phase": "search_correctness", "source_sha256": sha(SOURCE)}
    run = {"evaluation_mode": "deferred", "final_evaluation": None, "timing_candidates": [sha(SOURCE)],
           "attempts": [{"iteration": 1, "source_sha256": sha(SOURCE), "code": SOURCE, "evaluation": numeric,
                         "reply": json.dumps({"draft": "fixture", "plan": "fixture", "code_lines": SOURCE.splitlines()})}]}
    return {"runtime": {"gpu_index": 4}, "generation_result": run, "generation_result_sha256": digest(run),
            "measurement_condition": "shared_observed"}


def shared_result(candidate, ratio):
    return {"status": "correct", "source_sha256": candidate["source_sha256"], "checks_passed": 28,
            "timings": [{"shape": [n], "candidate_us": 1., "compiled_us": ratio} for n in (1, 2, 3)],
            "measurement_condition": "shared_observed",
            "resource_audit": {"policy": "shared_observed", "gpu_index": 4,
                               "before": telemetry(), "after": telemetry(), "samples": []}}


def test_shared_queue_keeps_final_veto_and_cannot_rewrite_old_policy(tmp_path):
    path, frozen = tmp_path / "queue.json", plan()
    with TimingQueue(path, frozen) as queue:
        result = queue.run(lambda candidate, final: shared_result(candidate, .8 if final else 1.3))
        assert result["correct"] and not result["promoted"]
        assert result["selected_sha256"] == sha(SOURCE)
    changed = copy.deepcopy(frozen)
    changed.pop("measurement_condition")
    with pytest.raises(ValueError, match="plan changed"):
        with TimingQueue(path, changed):
            pass


def test_shared_result_cannot_be_imported_as_uncontended_evidence(tmp_path):
    frozen = plan()
    frozen.pop("measurement_condition")
    with TimingQueue(tmp_path / "queue.json", frozen) as queue:
        with pytest.raises(ValueError, match="boundary evidence"):
            queue.run(lambda candidate, final: shared_result(candidate, 1.3))


def test_shared_analysis_requires_explicit_condition_and_preserves_strict_old_mode():
    frozen = {"runtime": {"gpu_index": 4}, "generation": {"measurement_condition": "shared_observed"}}
    value = shared_result({"source_sha256": sha(SOURCE)}, 1.3)
    check_resource_evidence(value, frozen)
    with pytest.raises(ValueError, match="uncontended boundary"):
        check_resource_evidence(value, {"runtime": {"gpu_index": 4}, "generation": {}})


def test_shared_readiness_accepts_resident_memory_when_model_is_idle(monkeypatch):
    class Metrics:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n"

    service = {"port": 8005, "pid": 123}
    frozen = {"runtime": {"service": service, "gpu_python": "fixture", "evaluator_runtime": {}, "gpu_index": 4},
              "source_sha256": {}, "generation": {"measurement_condition": "shared_observed"}}
    monkeypatch.setattr(run_campaign, "source_hashes", lambda: {})
    monkeypatch.setattr(run_campaign, "evaluator_identity", lambda _: {})
    monkeypatch.setattr(run_campaign, "service_identity", lambda *_: service)
    monkeypatch.setattr(run_campaign.urllib.request, "urlopen", lambda *_args, **_kwargs: Metrics())
    monkeypatch.setattr(evaluate_coexecuted, "observe", lambda _: telemetry())
    run_campaign.readiness(frozen)
    monkeypatch.setattr(evaluate_coexecuted, "observe", lambda _: {"free_mib": 100})
    with pytest.raises(RuntimeError, match="less than 1024"):
        run_campaign.readiness(frozen)
