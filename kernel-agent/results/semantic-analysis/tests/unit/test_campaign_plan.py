import io
import json
import os
import subprocess
import venv

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
    evaluator = {"invocation": "fixture-python"}
    plan = {"runtime": {"service": service, "gpu_index": 6, "gpu_python": "fixture-python",
                        "evaluator_runtime": evaluator}, "source_sha256": {"fixture": "sha"}}
    monkeypatch.setattr(run_campaign, "source_hashes", lambda: plan["source_sha256"])
    monkeypatch.setattr(run_campaign, "service_identity", lambda port, pid: service)
    monkeypatch.setattr(run_campaign, "evaluator_identity", lambda executable: evaluator)
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


def test_symlinked_virtualenv_interpreter_is_verified_and_preserved(tmp_path):
    # CPU fixture modules emulate version imports; no CUDA or real Torch installation.
    environment = tmp_path / "evaluation-env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / "bin/python"
    library = subprocess.check_output([str(executable), "-c",
                                      "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
    from pathlib import Path
    for name, version in (("torch", "2.6.0+cu124"), ("triton", "3.2.0")):
        Path(library, name + ".py").write_text(f"__version__ = {json.dumps(version)}\n")
    assert executable.is_symlink()
    identity = run_campaign.evaluator_identity(str(executable))
    assert identity["invocation"] == os.path.abspath(executable)
    assert identity["prefix"] == str(environment)
    assert identity["invocation"] != str(executable.resolve())


def test_changed_evaluator_runtime_blocks_before_service_or_gpu(monkeypatch):
    plan = {"source_sha256": {}, "runtime": {"service": {}, "gpu_python": "env-python",
                                             "evaluator_runtime": {"prefix": "expected"}}}
    monkeypatch.setattr(run_campaign, "source_hashes", lambda: {})
    monkeypatch.setattr(run_campaign, "evaluator_identity", lambda executable: {"prefix": "changed"})
    monkeypatch.setattr(run_campaign, "service_identity", lambda *a: pytest.fail("must stop before requests"))
    with pytest.raises(ValueError, match="evaluation Python environment changed"):
        run_campaign.readiness(plan)


def test_readiness_wait_retries_preflight_only(monkeypatch, tmp_path):
    attempts = []

    def ready(plan):
        attempts.append("preflight")
        if len(attempts) == 1:
            raise RuntimeError("GPU occupied")

    monkeypatch.setattr(run_campaign, "readiness", ready)
    monkeypatch.setattr(run_campaign.time, "sleep", lambda seconds: None)
    run_campaign.wait_until_ready({"scheduling": {"resource_wait_seconds": 30, "poll_seconds": 30}},
                                  {"id": "pending-job"}, tmp_path / "resource.json")
    assert attempts == ["preflight", "preflight"]
    assert not json.loads((tmp_path / "resource.json").read_text())["waiting"]


def test_source_drift_is_fatal_instead_of_waited_away(monkeypatch, tmp_path):
    def drift(plan):
        raise ValueError("sources changed")

    monkeypatch.setattr(run_campaign, "readiness", drift)
    monkeypatch.setattr(run_campaign.time, "sleep", lambda seconds: pytest.fail("must not wait on drift"))
    with pytest.raises(ValueError, match="sources changed"):
        run_campaign.wait_until_ready({"scheduling": {"resource_wait_seconds": 30, "poll_seconds": 30}},
                                      {"id": "pending-job"}, tmp_path / "resource.json")
