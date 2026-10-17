"""Complete native-client/patch/controller/auditor wiring with fake HTTP and GPU verdicts."""
import json
import shutil
from pathlib import Path

import pytest

from kernel_agent import model_pool, span_repair, grounded_budget as grounded_repair
from kernel_agent.campaign import atomic_json, digest
from kernel_agent.protocol import sha
from scripts import run_grounded_budget_study
from scripts.summarize_grounded_budget import analyze

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


@pytest.mark.parametrize("force_context_failure", [False, True])
def test_full_paired_study_checks_patch_recovery_budgets_and_raw_reply_binding(monkeypatch, tmp_path, force_context_failure):
    # Portable fake-parent fixture: no private remote source bank is required.
    from scripts.run_repair_study import source_files
    fixture_root = tmp_path / "fixture-project"
    for name in list(source_files()) + ["data/repair-transfer-tasks.json", "data/runtime-triton-api.json",
                                       "docs/grounded-budget-study-plan.md"]:
        destination = fixture_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    bank = json.loads((ROOT / "data/grounded-failure-bank.json").read_text())
    bank["parent_raw_manifest_sha256"] = "synthetic-parent-fixture-not-agent-evidence"
    for case in bank["cases"]:
        parent_source = tmp_path / (case["id"] + ".py")
        parent_attempt = tmp_path / (case["id"] + ".json")
        parent_source.write_text(SOURCE)
        parent_attempt.write_text(json.dumps({"reply": json.dumps({"draft": "fake", "plan": "fake",
                                                                   "code_lines": SOURCE.splitlines()})}))
        case.update(source_path=str(parent_source), attempt_path=str(parent_attempt), source_sha256=sha(SOURCE))
    (fixture_root / "data/grounded-failure-bank.json").write_text(json.dumps(bank))
    monkeypatch.setattr(run_grounded_budget_study, "ROOT", fixture_root)
    endpoint = {"port": 1234, "model": "fake", "fingerprint": "fake"}
    monkeypatch.setattr(model_pool, "validate", lambda e: {"running": 0, "waiting": 0})
    rejecting = [False]
    original_choose = grounded_repair.choose
    def choose(endpoint, path, task, contract, history, capability, mode, arm):
        rejecting[0] = force_context_failure and arm == "grounded" and path.stem.endswith("r1")
        return original_choose(endpoint, path, task, contract, history, capability, mode, arm)
    monkeypatch.setattr(grounded_repair, "choose", choose)
    def http(port, path, body=None, **kwargs):
        if path == "/tokenize":
            return {"count": 9000 if rejecting[0] else 1000}
        prompt = body["messages"][-1]["content"]
        if body["response_format"]["json_schema"]["schema"] == span_repair.SCHEMA:
            transaction = json.loads(prompt.split("Patch transaction:\n", 1)[1].split("\n\n", 1)[0])
            rejected = body["seed"] == run_grounded_budget_study.REGRESSION_SEEDS[0] + 1
            reply = json.dumps({"draft": "fake indexed", "plan": "fake fallback", "base_sha256": transaction["base_sha256"],
                                "edits": [{"start_line": 999999 if rejected else 1, "end_line": 999999 if rejected else 1,
                                           "replacement_lines": ["import triton", f"# indexed seed {body['seed']}"]}]})
        else:
            reply = json.dumps({"draft": "fake source", "plan": "fake source", "code_lines": (SOURCE + f"# seed {body['seed']}\n").splitlines()})
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}], "usage": {"completion_tokens": 30}}
    monkeypatch.setattr(model_pool, "http", http)
    plan = run_grounded_budget_study.make_plan()
    plan["runtime"] = {"evaluation_lock_file": "fake-lock"}
    atomic_json(tmp_path / "plan.json", plan)
    for name in plan["source_sha256"]:
        target = tmp_path / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_root / name, target)
    study = run_grounded_budget_study.GroundedBudgetStudy.__new__(run_grounded_budget_study.GroundedBudgetStudy)
    study.directory, study.plan = tmp_path, plan
    for name, identity in plan["parent_inputs"].items():
        target = tmp_path / "inputs" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(identity["path"], target)
    counters, finals = {}, []
    def evaluate(task, path, final=False):
        folder = path.parents[1]
        if final:
            committed = json.loads((folder / "generation.json").read_text())
            assert set(committed["arms"]) == set(grounded_repair.ARMS)
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
             "allocations": {"model_calls": 24, "output_tokens": 49152}}
    for block in plan["blocks"]:
        result, _ = study.run_block(block, endpoint)
        assert result["model_calls"] == block["model_call_cap"] - int(force_context_failure)
        assert result["infrastructure_stop"] is None
        if force_context_failure:
            assert result["arms"]["grounded"]["attempts"][1]["evaluation"]["status"] == "context_budget_error"
            assert all((tmp_path / "blocks" / block["id"] / "requests" / ("indexed-r" + str(i) + ".response.json")).exists() for i in [1, 2])
        if block["stratum"] == "regression" and block["seed"] == run_grounded_budget_study.REGRESSION_SEEDS[0]:
            for arm in grounded_repair.ARMS:
                attempts = result["arms"][arm]["attempts"]
                blocked = force_context_failure and arm == "grounded"
                assert attempts[1]["evaluation"]["status"] == ("context_budget_error" if blocked else "patch_error")
                assert attempts[2]["mode"] == ("span_patch" if blocked else "full_source_fallback")
                assert attempts[2]["parent_sha256"] == attempts[0]["source_sha256"]
                assert attempts[2]["evaluation"]["status"] == "correct"
                assert ("source_facts" in attempts[1]) == (arm == "grounded" and not blocked)
        state["blocks"].append({"block": block, "endpoint": endpoint, "status": "complete", "result_sha256": digest(result)})
    state["state_sha256"] = digest(state)
    atomic_json(tmp_path / "checkpoint.json", state)
    summary = analyze(tmp_path)
    assert summary["resolved"] == 12 and summary["dispatched_requests"] == (18 if force_context_failure else 24)
    assert all(c["correct"] == c["scheduled"] for s in summary["strata"] for c in s["cells"])
    assert {s["stratum"]: s["contrasts"]["grounded_vs_indexed"]["interval"]["clusters"] for s in summary["strata"]} == {"regression": 3}
    assert summary["patch_statuses"]["indexed"]["patch_error"] == 3
    assert summary["patch_statuses"]["grounded"].get("patch_error", 0) == (0 if force_context_failure else 3)
    assert len(finals) == (12 if force_context_failure else 6)  # Identical fake replies share one final measurement per block.
    from scripts.export_grounded_budget import compact
    from scripts.verify_grounded_budget_summary import verify
    output = tmp_path / "compact.json"
    published = compact(tmp_path, output)
    assert verify(output)["resolved"] == 12
    assert published["arm_mechanisms"]["grounded"]["full_source_fallbacks"] == (0 if force_context_failure else 3)
    assert published["arm_mechanisms"]["grounded"]["fallback_search_correct"] == (0 if force_context_failure else 3)
    assert "blocks" not in published and all("reply" not in t and "code" not in t for c in published["cases"] for a in c["arms"].values() for t in a["attempts"])
    assert output.stat().st_size < 1000000
    original = Path.glob
    with monkeypatch.context() as order:
        order.setattr(Path, "glob", lambda p, pattern: iter(reversed(list(original(p, pattern)))))
        assert analyze(tmp_path) == summary
    wire = next((tmp_path / "blocks").glob("*/requests/grounded-r2.request.json"))
    saved_wire = wire.read_bytes()
    receipt_path = wire.with_name("grounded-r2.json")
    saved_receipt = receipt_path.read_bytes()
    request = json.loads(saved_wire)
    request["messages"][-1]["content"] += "\nForged source fact"
    atomic_json(wire, request)
    import hashlib
    receipt_value = json.loads(saved_receipt)
    receipt_value["request_sha256"] = hashlib.sha256(wire.read_bytes()).hexdigest()
    atomic_json(receipt_path, receipt_value)
    with pytest.raises(ValueError, match="differs from assigned policy"):
        analyze(tmp_path)
    wire.write_bytes(saved_wire)
    receipt_path.write_bytes(saved_receipt)
    path = next((tmp_path / "blocks").glob("*/requests/grounded-r2.response.json"))
    response = json.loads(path.read_text())
    response["choices"][0]["message"]["content"] = "substituted"
    atomic_json(path, response)
    receipt = path.with_name("grounded-r2.json")
    value = json.loads(receipt.read_text())
    value["response_sha256"] = digest(response)
    atomic_json(receipt, value)
    with pytest.raises(ValueError, match="differs from raw response"):
        analyze(tmp_path)
