"""Real controller/client/receipt/auditor wiring with explicit fake HTTP and GPU work."""
import json
import shutil
from pathlib import Path

import pytest

from kernel_agent import model_pool, semantic
from kernel_agent.campaign import atomic_json, digest
from kernel_agent.protocol import sha
from scripts import run_semantic_study
from scripts.summarize_semantic import analyze

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def test_complete_study_has_shared_reviews_equal_budgets_and_tamper_rejection(monkeypatch, tmp_path):
    endpoint = {"port": 1234, "model": "fake", "fingerprint": "fake"}
    monkeypatch.setattr(model_pool, "validate", lambda e: {"running": 0, "waiting": 0})
    def http(port, path, body=None, **kwargs):
        if path == "/tokenize":
            return {"count": 1000}
        if body["response_format"]["json_schema"]["schema"] == semantic.REVIEW_SCHEMA:
            reply = json.dumps({"hypothesis": "untrusted story", "repair_plan": "untrusted plan",
                                "reference_expression": "x0+1", "candidate_expression": "x0"})
        else:
            source = SOURCE + ("# initial\n" if "Previous candidate source:" not in body["messages"][-1]["content"] else "# revision\n")
            reply = json.dumps({"draft": "d", "plan": "p", "code_lines": source.splitlines()})
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}], "usage": {"completion_tokens": 20}}
    monkeypatch.setattr(model_pool, "http", http)
    plan = run_semantic_study.make_plan()
    plan["runtime"] = {"evaluation_lock_file": "fake-gpu4-lock"}
    atomic_json(tmp_path / "plan.json", plan)
    for name in plan["source_sha256"]:
        target = tmp_path / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    study = run_semantic_study.SemanticStudy.__new__(run_semantic_study.SemanticStudy)
    study.directory = tmp_path
    current = {}
    final_calls = []
    def evaluate(task, path, final=False):
        block_path = path.parents[1]
        if final:
            committed = json.loads((block_path / "generation.json").read_text())
            assert set(committed["arms"]) == set(semantic.ARMS)
            assert all(a["selected_sha256"] for a in committed["arms"].values())
            final_calls.append(block_path.name)
        source = path.read_text()
        before = model_pool.utc()
        result = {"status": "numeric_error" if not final and current.get(block_path.name, 0) == 0 else "correct",
                  "checks_passed": 28, "source_sha256": sha(source),
                  "timings": [{"shape": [n], "compiled_us": 2, "candidate_us": 1} for n in [1, 3, 5]],
                  "device_binding": {"verified": True, "expected_gpu_index": 4, "expected_uuid": "GPU-fake",
                                     "actual": {"gpu_uuid": "fake", "visible_devices": "GPU-fake", "device_index": 0, "device_count": 1}},
                  "semantic_probes": [{"id": "p", "inputs": [{"shape": [2], "values": [1., 2.]}],
                                       "expected": {"shape": [2], "values": [2., 3.]},
                                       "actual": {"shape": [2], "values": [1., 2.]}}],
                  "resource_audit": {"before": {"observed_utc": before, "gpu_uuid": "GPU-fake"},
                                     "after": {"observed_utc": model_pool.utc()}},
                  "cooperative_queue": {"lock_file": "fake-gpu4-lock", "pid": 1}}
        current[block_path.name] = current.get(block_path.name, 0) + 1
        return result
    study.evaluate = evaluate
    state = {"plan_sha256": digest(plan), "blocks": [], "services": {}, "allocations": {"model_calls": 118, "output_tokens": 118 * 2048},
             "status": "complete"}
    for block in plan["blocks"]:
        value, _ = study.run_block(block, endpoint)
        assert value["model_calls"] == block["model_call_cap"]
        assert value["review"]["validation"]["reference"]["valid"]
        assert all(a["model_opportunities"][0] == "review" for name, a in value["arms"].items() if name != "rewrite")
        state["blocks"].append({"block": block, "endpoint": endpoint, "status": "complete", "result_sha256": digest(value)})
    state["state_sha256"] = digest(state)
    atomic_json(tmp_path / "checkpoint.json", state)
    summary = analyze(tmp_path)
    assert summary["resolved"] == 66 and summary["dispatched_requests"] == 118
    assert len(final_calls) == 22  # Every arm selects the same successful revision in this fake fixture.
    assert [s["contrasts"]["grounded_vs_rewrite"]["interval"]["clusters"] for s in summary["strata"]] == [5, 8]
    original_glob = Path.glob
    def reversed_glob(path, pattern):
        return iter(reversed(list(original_glob(path, pattern))))
    with monkeypatch.context() as order:
        order.setattr(Path, "glob", reversed_glob)
        assert analyze(tmp_path) == summary  # Filesystem order must not change the exported report.
    response_path = next((tmp_path / "blocks").glob("*/requests/review.response.json"))
    response = json.loads(response_path.read_text())
    response["choices"][0]["message"]["content"] = "substituted"
    atomic_json(response_path, response)
    receipt_path = response_path.with_name("review.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["response_sha256"] = digest(response)
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="review reply changed"):
        analyze(tmp_path)


def test_reviewer_only_sees_search_evidence_and_complete_source():
    start = {"code": SOURCE, "evaluation": {"status": "numeric_error", "message": "search failure",
             "semantic_probes": [], "final_result": "FORBIDDEN_FINAL_DATA", "timings": "UNNEEDED_PERFORMANCE_DATA"}}
    context = "\n".join(p[1] for p in semantic.reviewer_sections("TASK_CONTRACT", start))
    assert SOURCE in context and "TASK_CONTRACT" in context and "search failure" in context
    assert "FORBIDDEN_FINAL_DATA" not in context and "UNNEEDED_PERFORMANCE_DATA" not in context
