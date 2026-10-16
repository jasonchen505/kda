"""Complete native-client/patch/controller/auditor wiring with fake HTTP and GPU verdicts."""
import json
import shutil
from pathlib import Path

import pytest

from kernel_agent import model_pool, operational, span_repair
from kernel_agent.campaign import atomic_json, digest
from kernel_agent.protocol import sha
from scripts import run_span_study
from scripts.summarize_span import analyze

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def test_full_paired_study_checks_patch_recovery_budgets_and_raw_reply_binding(monkeypatch, tmp_path):
    endpoint = {"port": 1234, "model": "fake", "fingerprint": "fake"}
    monkeypatch.setattr(model_pool, "validate", lambda e: {"running": 0, "waiting": 0})
    def http(port, path, body=None, **kwargs):
        if path == "/tokenize":
            return {"count": 1000}
        prompt = body["messages"][-1]["content"]
        if body["response_format"]["json_schema"]["schema"] == span_repair.SCHEMA:
            transaction = json.loads(prompt.split("Patch transaction:\n", 1)[1].split("\n\n", 1)[0])
            rejected = body["seed"] == run_span_study.REGRESSION_SEEDS[0] + 1
            reply = json.dumps({"draft": "fake indexed", "plan": "fake fallback", "base_sha256": transaction["base_sha256"],
                                "edits": [{"start_line": 999999 if rejected else 1, "end_line": 999999 if rejected else 1,
                                           "replacement_lines": ["import triton", f"# indexed seed {body['seed']}"]}]})
        elif body["response_format"]["json_schema"]["schema"] == operational.PATCH_SCHEMA:
            transaction = json.loads(prompt.split("Patch transaction:\n", 1)[1].split("\n\n", 1)[0])
            source = prompt.split("Current admitted source (exact):\n", 1)[1].split("\n\nTrusted search-only evidence:", 1)[0]
            first = source.splitlines()[0]
            rejected = body["seed"] == run_span_study.REGRESSION_SEEDS[0] + 1
            reply = json.dumps({"draft": "fake edit", "plan": "fake recovery", "base_sha256": transaction["base_sha256"],
                                "edits": [{"before_lines": ["MISSING-SPAN"] if rejected else [first],
                                           "after_lines": [first, f"# fake patch seed {body['seed']}"]}]})
        else:
            reply = json.dumps({"draft": "fake source", "plan": "fake source", "code_lines": (SOURCE + f"# seed {body['seed']}\n").splitlines()})
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}], "usage": {"completion_tokens": 30}}
    monkeypatch.setattr(model_pool, "http", http)
    plan = run_span_study.make_plan()
    plan["runtime"] = {"evaluation_lock_file": "fake-lock"}
    atomic_json(tmp_path / "plan.json", plan)
    for name in plan["source_sha256"]:
        target = tmp_path / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    study = run_span_study.SpanStudy.__new__(run_span_study.SpanStudy)
    study.directory, study.plan = tmp_path, plan
    counters, finals = {}, []
    def evaluate(task, path, final=False):
        folder = path.parents[1]
        if final:
            committed = json.loads((folder / "generation.json").read_text())
            assert set(committed["arms"]) == set(span_repair.ARMS)
            assert all(a["selected_sha256"] for a in committed["arms"].values())
            finals.append(folder.name)
        before = model_pool.utc()
        v = {"status": "numeric_error" if not final and counters.get(folder.name, 0) == 0 else "correct",
             "source_sha256": sha(path.read_text()), "checks_passed": 28,
             "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in [1, 3, 5]],
             "device_binding": {"verified": True, "expected_gpu_index": 4, "expected_uuid": "GPU-fake",
                                "actual": {"gpu_uuid": "fake", "visible_devices": "GPU-fake", "device_index": 0, "device_count": 1}},
             "resource_audit": {"before": {"observed_utc": before, "gpu_uuid": "GPU-fake"}, "after": {"observed_utc": model_pool.utc()}},
             "cooperative_queue": {"lock_file": "fake-lock", "pid": 1},
             "semantic_probes": [{"id": str(i), "inputs": [{"shape": [2], "values": [2., 3.]}],
                                  "expected": {"shape": [2], "values": [2., 3.]},
                                  "actual": {"shape": [2], "values": [1., 1.]}} for i in [1, 2]]}
        counters[folder.name] = counters.get(folder.name, 0) + 1
        return v
    study.evaluate = evaluate
    state = {"plan_sha256": digest(plan), "blocks": [], "services": {}, "status": "complete",
             "allocations": {"model_calls": 140, "output_tokens": 286720}}
    for block in plan["blocks"]:
        result, _ = study.run_block(block, endpoint)
        assert result["model_calls"] == block["model_call_cap"]
        if block["stratum"] == "regression" and block["seed"] == run_span_study.REGRESSION_SEEDS[0]:
            attempts = result["arms"]["operational_patch"]["attempts"]
            assert attempts[1]["evaluation"]["status"] == "patch_error"
            assert attempts[2]["parent_sha256"] == attempts[0]["source_sha256"]
            assert attempts[2]["evaluation"]["status"] == "correct"
            new = result["arms"]["span_repair"]["attempts"]
            assert new[1]["evaluation"]["status"] == "patch_error"
            assert new[2]["mode"] == "full_source_fallback" and new[2]["evaluation"]["status"] == "correct"
        state["blocks"].append({"block": block, "endpoint": endpoint, "status": "complete", "result_sha256": digest(result)})
    state["state_sha256"] = digest(state)
    atomic_json(tmp_path / "checkpoint.json", state)
    summary = analyze(tmp_path)
    assert summary["resolved"] == 66 and summary["dispatched_requests"] == 140
    assert all(c["correct"] == c["scheduled"] for s in summary["strata"] for c in s["cells"])
    assert {s["stratum"]: s["contrasts"]["span_vs_operational"]["interval"]["clusters"] for s in summary["strata"]} == {"regression": 5, "fresh": 4}
    assert summary["patch_statuses"]["operational_patch"]["patch_error"] == 7
    assert summary["patch_statuses"]["span_repair"]["patch_error"] == 7
    assert len(finals) == 66
    from scripts.export_span import compact
    from scripts.verify_span_summary import verify
    output = tmp_path / "compact.json"
    published = compact(tmp_path, output)
    assert verify(output)["resolved"] == 66
    assert published["arm_mechanisms"]["span_repair"]["full_source_fallbacks"] == 7
    assert published["arm_mechanisms"]["span_repair"]["fallback_search_correct"] == 7
    assert "blocks" not in published and all("reply" not in t and "code" not in t for c in published["cases"] for a in c["arms"].values() for t in a["attempts"])
    assert output.stat().st_size < 1000000
    original = Path.glob
    with monkeypatch.context() as order:
        order.setattr(Path, "glob", lambda p, pattern: iter(reversed(list(original(p, pattern)))))
        assert analyze(tmp_path) == summary
    path = next((tmp_path / "blocks").glob("*/requests/operational_patch-r2.response.json"))
    response = json.loads(path.read_text())
    response["choices"][0]["message"]["content"] = "substituted"
    atomic_json(path, response)
    receipt = path.with_name("operational_patch-r2.json")
    value = json.loads(receipt.read_text())
    value["response_sha256"] = digest(response)
    atomic_json(receipt, value)
    with pytest.raises(ValueError, match="differs from raw response"):
        analyze(tmp_path)
