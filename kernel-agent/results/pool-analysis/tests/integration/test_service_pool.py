import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_agent import services
from kernel_agent.resource_lock import acquire
from kernel_agent.transport import PROPOSER_PREFIX
from scripts import pool_proxy
from scripts import service_gateway
from scripts.run_pool import worker_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("port", [18118, 54575, 60953])
def test_every_seed_pair_stays_on_one_service_and_uses_the_common_protocol(port):
    plan = worker_plan({"service": {"port": port}}, {}, port)
    assert len(plan["jobs"]) == 8
    assert plan["generation"]["common_context_cap"] == 8192
    assert plan["generation"]["max_tokens_per_request"] == 2048
    for task in services.SERVICES[port]["families"]:
        for seed in plan["seeds"]:
            pair = [j for j in plan["jobs"] if (j["task_id"], j["seed"]) == (task, seed)]
            assert {j["study_policy"] for j in pair} == {"blind", "feedback"}
            assert {j["service_port"] for j in pair} == {port}


@pytest.mark.parametrize("metrics", ["", "vllm:num_requests_running{} 0", "vllm:num_requests_running{} nan\nvllm:num_requests_waiting{} 0"])
def test_missing_or_invalid_load_cannot_be_treated_as_idle(metrics):
    with pytest.raises(ValueError):
        services.parse_load(metrics)


def test_waiting_by_reason_is_not_double_counted_and_busy_service_is_not_admitted(monkeypatch):
    metrics = 'vllm:num_requests_running{a="b"} 0\nvllm:num_requests_waiting{a="b"} 2\nvllm:num_requests_waiting_by_reason{reason="capacity"} 2'
    assert services.parse_load(metrics) == {"running": 0, "waiting": 2}
    monkeypatch.setattr(services, "load", lambda _: {"running": 1, "waiting": 0})
    with pytest.raises(RuntimeError, match="no request dispatched"):
        services.wait_idle(54575, timeout=0)


def handler(server, *, stream=False):
    obj = pool_proxy.Handler.__new__(pool_proxy.Handler)
    obj.server, obj.path = server, "/json-lines/v1/chat/completions"
    body = json.dumps({"model": "kernel-qwen-local-27b", "stream": stream, "max_tokens": 4096,
                       "messages": [{"role": "system", "content": PROPOSER_PREFIX}], "tools": [{"name": "shell"}]})
    obj.rfile, obj.wfile = io.BytesIO(body.encode()), io.BytesIO()
    obj.headers = {"Content-Length": str(len(body.encode()))}
    obj.responses = []
    obj.send_error = lambda code, *args: obj.responses.append(code)
    obj.send_response = lambda code: obj.responses.append(code)
    obj.send_header = lambda *args: None
    obj.end_headers = lambda: None
    return obj


def proxy_server(tmp_path, monkeypatch, count=100):
    service = {"port": 54575, "base_model": "steamserve-qwen38"}
    server = SimpleNamespace(plan={"runtime": {"service": service}, "generation": {"temperature": .2}},
                             job={"id": "fixture", "seed": 31}, trace_dir=tmp_path,
                             request_lock=threading.Lock(), quarantined=False)
    monkeypatch.setattr(pool_proxy, "identity", lambda _: service)
    monkeypatch.setattr(pool_proxy, "wait_idle", lambda _: {"running": 0, "waiting": 0})
    monkeypatch.setattr(pool_proxy, "load", lambda _: {"running": 0, "waiting": 0})
    monkeypatch.setattr(pool_proxy, "request_json", lambda *args: {"count": count})
    return server


def test_context_overflow_never_dispatches_or_silently_truncates_a_prompt(monkeypatch, tmp_path):
    server = proxy_server(tmp_path, monkeypatch, count=7000)
    monkeypatch.setattr(pool_proxy.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("must not dispatch"))
    request = handler(server)
    request.do_POST()
    assert request.responses == [409] and not list(tmp_path.glob("*.json"))
    value = json.loads(next((tmp_path / "admission").glob("*.json")).read_text())
    assert value["status"] == "not_dispatched" and "context" in value["error"]


def test_uncertain_stream_quarantines_the_job_without_replaying_upstream(monkeypatch, tmp_path):
    server = proxy_server(tmp_path, monkeypatch)
    calls = []

    def upstream(*args, **kwargs):
        calls.append(1)
        response = io.BytesIO(b'data: {"partial":true}\n\n')
        response.status, response.headers = 200, {"Content-Type": "text/event-stream"}
        return response

    monkeypatch.setattr(pool_proxy.urllib.request, "urlopen", upstream)
    handler(server, stream=True).do_POST()
    handler(server, stream=True).do_POST()
    assert calls == [1] and server.quarantined
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "admission").glob("*.json")]
    assert {r["status"] for r in receipts} == {"response_unknown", "not_dispatched"}


def test_own_concurrent_requests_are_serialized_and_bound_to_the_real_alias(monkeypatch, tmp_path):
    server = proxy_server(tmp_path, monkeypatch)
    active, peak, bodies = 0, 0, []

    class Response(io.BytesIO):
        status, headers = 200, {"Content-Type": "application/json"}

        def __enter__(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(.02)
            return self

        def __exit__(self, *args):
            nonlocal active
            active -= 1
            self.close()

    def upstream(request, **kwargs):
        bodies.append(json.loads(request.data))
        return Response(b'{"choices":[]}')

    monkeypatch.setattr(pool_proxy.urllib.request, "urlopen", upstream)
    threads = [threading.Thread(target=handler(server).do_POST) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert peak == 1 and len(bodies) == 2
    assert all(b["model"] == "steamserve-qwen38" and b["max_tokens"] == 2048 and not b.get("tools") for b in bodies)
    assert all(json.loads(p.read_text())["status"] == "completed" for p in (tmp_path / "admission").glob("*.json"))


def test_gpu_child_keeps_the_lock_after_its_parent_is_killed(tmp_path):
    lock, ready, stop = (tmp_path / n for n in ("gpu.lock", "ready.json", "stop"))
    script = '''import json,signal,subprocess,sys
from pathlib import Path
from kernel_agent.resource_lock import acquire
with acquire(sys.argv[1]) as (lock,waited):
 child=subprocess.Popen([sys.executable,'-c',"import time,sys; from pathlib import Path; end=time.monotonic()+20; "+
  "\\nwhile not Path(sys.argv[1]).exists() and time.monotonic()<end: time.sleep(.05)",sys.argv[3]],pass_fds=(lock.fileno(),))
 Path(sys.argv[2]).write_text(json.dumps({'child':child.pid}))
 signal.pause()
'''
    proc = subprocess.Popen([sys.executable, "-c", script, str(lock), str(ready), str(stop)],
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(.02)
        proc.kill()
        assert proc.wait(timeout=3) == -9
        with pytest.raises(TimeoutError):
            with acquire(lock, timeout=.1):
                pass
        stop.touch()
        with acquire(lock, timeout=3):
            pass
    finally:
        stop.touch()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def test_endpoint_handoff_rejects_a_changed_model_declaration(monkeypatch):
    values = iter([{"data": [{"id": "base", "root": "different-weights", "parent": None, "max_model_len": 8192}]},
                   {"version": "0.25.1"}])
    monkeypatch.setattr(service_gateway.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(next(values)).encode()))
    with pytest.raises(ValueError, match="declaration"):
        service_gateway.verify_target({"base_model": "base", "model_root": "recorded-weights",
                                       "max_model_len": 8192, "vllm_version": "0.25.1"}, 56825)


def test_gateway_preserves_stream_bytes_and_does_not_forward_unlisted_paths():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import urllib.error
    import urllib.request

    calls = []
    payload = b'data: {"choices":[{"delta":{"content":"fixture"}}]}\n\ndata: [DONE]\n\n'

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), service_gateway.Handler)
    gateway.target = upstream.server_port
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, gateway)]
    for thread in threads:
        thread.start()
    try:
        url = f"http://127.0.0.1:{gateway.server_port}"
        with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=b"{}"), timeout=3) as response:
            assert response.read() == payload
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/private", timeout=3)
        assert error.value.code == 404 and calls == ["/v1/chat/completions"]
    finally:
        for server in (gateway, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
