"""Real control boundaries: shared starts, context packing, uncertainty and scheduling."""
import json
import time

import pytest

from kernel_agent import model_pool, repair
from kernel_agent.protocol import sha


def endpoint(port=1234, fingerprint="a"):
    return {"port": port, "model": "qwen", "fingerprint": fingerprint}


def good(source="source"):
    return {"source_sha256": sha(source), "code": source, "evaluation": {
        "status": "correct", "checks_passed": 28,
        "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in [1, 3, 5]]}}


def test_both_repairs_share_previous_source_but_only_treatment_sees_witness():
    row = {"code": "COMPLETE_SOURCE", "source_sha256": "failed", "evaluation": {
        "status": "numeric_error", "message": "ordinary mismatch", "counterexample": {"marker": "TINY_WITNESS"}}}
    raw = repair.sections("transpose", "contract", "baseline", [row], "baseline")
    new = repair.sections("transpose", "contract", "baseline", [row], "counterexample")
    assert "COMPLETE_SOURCE" in "\n".join(p[1] for p in raw)
    assert "COMPLETE_SOURCE" in "\n".join(p[1] for p in new)
    assert "TINY_WITNESS" not in "\n".join(p[1] for p in raw)
    assert "TINY_WITNESS" in "\n".join(p[1] for p in new)


def test_bad_repair_cannot_replace_correct_candidate_and_anchor_is_available():
    first = good()
    bad = {"code": "failed source", "source_sha256": "bad", "evaluation": {"status": "numeric_error"}}
    assert repair.select([first, bad]) is first
    sections = repair.sections("row_sum", "contract", "contract", [first, bad], "counterexample")
    assert any(p[0] == "correct_anchor" and first["code"] in p[1] for p in sections)


def test_pool_uses_new_idle_replica_then_observed_fast_replica():
    a, b = endpoint(), endpoint(1235, "b")
    assert model_pool.choose([a, b], set(), {"a": {"assigned": 1, "ewma_seconds": 1}}) == b
    stats = {"a": {"assigned": 1, "ewma_seconds": 1}, "b": {"assigned": 1, "ewma_seconds": 100}}
    assert model_pool.choose([a, b], set(), stats) == a
    assert model_pool.choose([a, b], {"a"}, stats) == b


def setup_http(monkeypatch, reply_error=False):
    calls = []
    monkeypatch.setattr(model_pool, "validate", lambda e: {"running": 0, "waiting": 0})
    def http(port, path, body=None, **kwargs):
        if path == "/tokenize":
            return {"count": sum(len(m["content"]) for m in body["messages"])}
        calls.append(body)
        if reply_error:
            raise ConnectionResetError("response unknown")
        return {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}
    monkeypatch.setattr(model_pool, "http", http)
    return calls


def test_context_packer_drops_whole_optional_blocks_not_source(monkeypatch, tmp_path):
    calls = setup_http(monkeypatch)
    client = model_pool.Client(endpoint(), tmp_path, time.monotonic() + 60)
    assert client.propose([("source", "SOURCE_INTACT", True), ("example", "x" * 9000, False)], 7, "one") == "answer"
    assert len(calls) == 1 and "SOURCE_INTACT" in calls[0]["messages"][-1]["content"]
    assert "x" * 100 not in calls[0]["messages"][-1]["content"]
    receipt = json.loads((tmp_path / "one.json").read_text())
    assert receipt["omitted_sections"] == ["example"] and receipt["status"] == "completed"


def test_mandatory_overflow_never_dispatches(monkeypatch, tmp_path):
    calls = setup_http(monkeypatch)
    with pytest.raises(model_pool.NotDispatched):
        model_pool.Client(endpoint(), tmp_path, time.monotonic() + 60).propose([("source", "x" * 9000, True)], 7, "big")
    assert calls == []


def test_unknown_response_is_committed_and_same_block_cannot_retry(monkeypatch, tmp_path):
    calls = setup_http(monkeypatch, reply_error=True)
    client = model_pool.Client(endpoint(), tmp_path, time.monotonic() + 60)
    with pytest.raises(model_pool.OutcomeUnknown):
        client.propose([("contract", "c", True)], 7, "unknown")
    with pytest.raises(model_pool.NotDispatched):
        client.propose([("contract", "c", True)], 7, "retry")
    assert len(calls) == 1
    assert json.loads((tmp_path / "unknown.json").read_text())["status"] == "response_unknown"
    assert json.loads((tmp_path / "retry.json").read_text())["status"] == "not_dispatched"


@pytest.mark.parametrize("prefix", [["/bin/vllm", "serve"], ["/bin/python", "-m", "vllm.entrypoints.openai.api_server"]])
def test_discovery_binds_public_gpu_parent_chain_without_reading_environments(monkeypatch, tmp_path, prefix):
    root = sorted(model_pool.ROOTS)[0]
    for pid, parent, args in [(123, 1, [*prefix, root, "--host", "127.0.0.1", "--port", "1234"]),
                               (124, 123, ["engine"])]:
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(("\0".join(args) + "\0").encode())
        (folder / "stat").write_text(f"{pid} (worker name) " + " ".join(["S", str(parent)] + ["0"] * 17 + ["777"]))
    monkeypatch.setattr(model_pool, "describe", lambda pid, port, root, start, gpus:
                        {"pid": pid, "port": port, "root": root, "start": start, "gpus": gpus})
    found = model_pool.discover(tmp_path, [(124, 2), (124, 3)])
    assert len(found) == 1 and found[0]["pid"] == 123 and found[0]["gpus"] == [2, 3]


def test_restarted_process_is_rejected_before_http_dispatch(monkeypatch):
    monkeypatch.setattr(model_pool, "process", lambda pid: ([], 1, "new"))
    with pytest.raises(ValueError, match="restarted"):
        model_pool.validate({"pid": 7, "process_start_ticks": "old"})


def test_factorial_imports_shared_starts_and_commits_all_selection_before_final(monkeypatch, tmp_path):
    from scripts import run_repair_study
    source = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"
    class FakeClient:
        def __init__(self, *args):
            self.calls = 0
        def propose(self, sections, seed, request_id):
            self.calls += 1
            code = source + f"# {request_id}\n"
            return json.dumps({"draft": "d", "plan": "p", "code_lines": code.splitlines()})
    monkeypatch.setattr(model_pool, "Client", FakeClient)
    study = run_repair_study.Study.__new__(run_repair_study.Study)
    study.directory = tmp_path
    block = {"id": "test-block", "task_id": "transpose", "seed": 123}
    final_calls = []
    def evaluate(task, path, final=False):
        if final:
            committed = json.loads((tmp_path / "blocks/test-block/generation.json").read_text())
            assert len(committed["arms"]) == 4 and all(a["final"] is None for a in committed["arms"].values())
            final_calls.append(str(path))
        value = good(path.read_text())["evaluation"]
        value["source_sha256"] = sha(path.read_text())
        return value
    study.evaluate = evaluate
    value, _ = study.run_block(block, endpoint())
    assert value["model_calls"] == 10 and len(final_calls) == 2
    for generation in repair.GENERATION:
        starts = [value["arms"][repair.arm_id(generation, r)]["attempts"][0] for r in repair.REPAIR]
        assert starts[0] == starts[1] == value["initial"][generation]
    assert sum("final_reused_from" in a for a in value["arms"].values()) == 2


def test_offline_audit_binds_response_to_proposal_even_if_response_hash_is_resigned(monkeypatch, tmp_path):
    from scripts import run_repair_study, summarize_repair
    from kernel_agent.campaign import atomic_json, digest
    source = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"
    counter = 0
    monkeypatch.setattr(run_repair_study, "evaluator_identity", lambda p: {"invocation": p})
    monkeypatch.setattr(model_pool, "validate", lambda e: {"running": 0, "waiting": 0})
    def http(port, path, body=None, **kwargs):
        nonlocal counter
        if path == "/tokenize":
            return {"count": 10}
        counter += 1
        content = json.dumps({"draft": "d", "plan": "p", "code_lines": (source + f"# {counter}\n").splitlines()})
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}
    monkeypatch.setattr(model_pool, "http", http)
    study = run_repair_study.Study(tmp_path, "fake-gpu-python")
    eval_counter = 0
    def evaluate(task, path, final=False):
        nonlocal eval_counter
        eval_counter += 2
        before = f"2026-10-15T00:00:{eval_counter:02d}+00:00"
        after = f"2026-10-15T00:00:{eval_counter + 1:02d}+00:00"
        return {**good(path.read_text())["evaluation"], "source_sha256": sha(path.read_text()),
                "resource_audit": {"before": {"observed_utc": before, "gpu_uuid": "GPU-fixture"},
                                   "after": {"observed_utc": after}},
                "device_binding": {"verified": True, "expected_gpu_index": 4, "expected_uuid": "GPU-fixture",
                                   "actual": {"gpu_uuid": "fixture", "visible_devices": "GPU-fixture",
                                              "device_count": 1, "device_index": 0}},
                "cooperative_queue": {"lock_file": str(tmp_path / "gpu4.lock"), "pid": eval_counter,
                                      "scope": "serializes this project's workers only"}}
    study.evaluate = evaluate
    item = study.state["blocks"][0]
    item.update(status="running", endpoint=endpoint())
    result, _ = study.run_block(item["block"], endpoint())
    item.update(status="complete", result_sha256=digest(result))
    study.save()
    summary = summarize_repair.analyze(tmp_path)
    assert summary["resolved"] == 4 and summary["dispatched_requests"] == 10
    response_path = next((tmp_path / "blocks" / item["block"]["id"] / "requests").glob("initial-*.response.json"))
    value = json.loads(response_path.read_text())
    value["choices"][0]["message"]["content"] = "substituted model answer"
    atomic_json(response_path, value)
    receipt_path = response_path.with_name(response_path.name.replace(".response", ""))
    receipt = json.loads(receipt_path.read_text())
    receipt["response_sha256"] = digest(value)
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="reply differs from model response"):
        summarize_repair.analyze(tmp_path)
