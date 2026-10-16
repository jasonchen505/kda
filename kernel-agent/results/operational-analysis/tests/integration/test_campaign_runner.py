import json
import os

from scripts import run_campaign


def test_local_campaign_worker_preserves_frozen_arm_budget_and_proxy_settings(monkeypatch, tmp_path):
    import hmz.sdk

    service = {"port": 8005, "pid": 123}
    plan = run_campaign.make_plan({"service": service, "gpu_index": 6, "gpu_python": "gpu-python"},
                                  {"fixture": "sha"})
    job = plan["jobs"][0]
    captured = []

    class Invocation:
        def run(self):
            return {"task_id": job["task_id"], "study_policy": job["study_policy"],
                    "correct": False, "promoted": False, "attempts": [], "final_evaluation": None}

    class FakeHmz:
        def __init__(self, root):
            assert root == run_campaign.ROOT

        def run(self, flow, task, **options):
            captured.append(options)
            config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
            endpoint = config["provider"]["kernel-local"]["options"]["baseURL"]
            assert endpoint.startswith("http://127.0.0.1:") and endpoint.endswith("/json-lines/v1")
            assert config["enabled_providers"] == ["kernel-local"]
            assert config["agent"]["build"]["temperature"] == 0.2
            assert run_campaign.local_proxy.SEED == job["seed"]
            assert run_campaign.local_proxy.UPSTREAM_MODEL_ID == run_campaign.BASE
            return Invocation()

    monkeypatch.setattr(hmz.sdk, "Hmz", FakeHmz)
    monkeypatch.setattr(run_campaign, "source_hashes", lambda: plan["source_sha256"])
    # Restore environment and proxy globals after the worker's task-scoped changes.
    for key in ("OPENCODE_CONFIG_CONTENT", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_DEFAULT_PLUGINS",
                "CUDA_VISIBLE_DEVICES", "PYTHONPATH"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    for key in ("UPSTREAM", "MODEL_ID", "UPSTREAM_MODEL_ID", "SEED", "TEMPERATURE", "TRACE_DIR"):
        monkeypatch.setattr(run_campaign.local_proxy, key, getattr(run_campaign.local_proxy, key))
    result = run_campaign.execute(plan, job, tmp_path)
    assert result["seed"] == job["seed"] and result["job_id"] == job["id"]
    assert captured[0]["budget"] == {"duration": 900, "output_tokens": 20000, "graceful": False}
    assert captured[0]["params"]["study_policy"] == job["study_policy"]
    assert captured[0]["params"]["selection_policy"] == "gate_first"
    assert captured[0]["params"]["rounds"] == 3
