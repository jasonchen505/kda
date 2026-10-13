import io

import pytest

from scripts import run_campaign


def test_frozen_design_is_balanced_reproducible_and_stochastic():
    first = run_campaign.make_plan({"runtime": "fixture"}, {"source": "fixture"})
    assert first == run_campaign.make_plan({"runtime": "fixture"}, {"source": "fixture"})
    assert len(first["jobs"]) == len({j["id"] for j in first["jobs"]}) == 36
    for task in run_campaign.TASKS:
        for seed in run_campaign.SEEDS:
            assert {j["study_policy"] for j in first["jobs"] if j["task_id"] == task and j["seed"] == seed} == {"blind", "feedback"}
    assert first["generation"]["temperature"] > 0
    assert first["campaign_budget"]["output_tokens"] == 36 * 20000


@pytest.mark.parametrize("tasks,seeds", [(["unknown"], [1]), (["softmax", "softmax"], [1]),
                                       (["softmax"], [1, 1]), (["softmax"], [])])
def test_invalid_task_or_seed_design_rejected(tasks, seeds):
    with pytest.raises(ValueError):
        run_campaign.make_plan({}, {}, tasks=tasks, seeds=seeds)


@pytest.mark.parametrize("running,waiting,memory,blocked", [(0, 0, 0, False), (1, 0, 0, True),
                                                          (0, 1, 0, True), (0, 0, 10000, True)])
def test_resource_guard_never_needs_a_model_completion(monkeypatch, running, waiting, memory, blocked):
    service = {"port": 8005, "pid": 123}
    plan = {"runtime": {"service": service, "gpu_index": 6}, "source_sha256": {"fixture": "sha"}}
    monkeypatch.setattr(run_campaign, "source_hashes", lambda: plan["source_sha256"])
    monkeypatch.setattr(run_campaign, "service_identity", lambda port, pid: service)
    metrics = f'vllm:num_requests_running{{model_name="base"}} {running}\nvllm:num_requests_waiting{{model_name="base"}} {waiting}\n'
    calls = []

    def read_metrics(url, timeout):
        calls.append(url)
        return io.BytesIO(metrics.encode())

    monkeypatch.setattr(run_campaign.urllib.request, "urlopen", read_metrics)
    monkeypatch.setattr(run_campaign.subprocess, "check_output", lambda *a, **k: str(memory))
    if blocked:
        with pytest.raises(RuntimeError):
            run_campaign.readiness(plan)
    else:
        run_campaign.readiness(plan)
    assert calls == ["http://127.0.0.1:8005/metrics"]
