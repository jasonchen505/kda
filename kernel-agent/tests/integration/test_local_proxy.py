import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from scripts import local_proxy


def test_existing_service_receives_only_explicit_base_model_and_seed(monkeypatch, tmp_path):
    received = []
    real_urlopen = urllib.request.urlopen

    def upstream(request, timeout):
        if isinstance(request, urllib.request.Request) and request.full_url.startswith("http://127.0.0.1:8016/"):
            received.append(json.loads(request.data))
            response = io.BytesIO(b'{"choices":[]}')
            response.status, response.headers = 200, {"Content-Type": "application/json"}
            return response
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(local_proxy, "MODEL_ID", "kernel-qwen-local-27b")
    monkeypatch.setattr(local_proxy, "UPSTREAM_MODEL_ID", "qwen38-27b-fp8")
    monkeypatch.setattr(local_proxy, "UPSTREAM", "http://127.0.0.1:8016")
    monkeypatch.setattr(local_proxy, "SEED", 20261012)
    monkeypatch.setattr(local_proxy, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", upstream)
    server = ThreadingHTTPServer(("127.0.0.1", 0), local_proxy.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/json-lines/v1/chat/completions"
    try:
        for model in ("expert-sft-actions-v1", "kernel-qwen-local-27b"):
            request = urllib.request.Request(endpoint, json.dumps({"model": model,
                "messages": [{"role": "system", "content": "You are a GPU kernel source-code proposer."}],
                "tools": [{"type": "function"}], "max_tokens": 8000}).encode(),
                {"Content-Type": "application/json"})
            if model == "expert-sft-actions-v1":
                with pytest.raises(urllib.error.HTTPError, match="400"):
                    real_urlopen(request, timeout=2)
            else:
                with real_urlopen(request, timeout=2) as response:
                    assert response.status == 200
        assert len(received) == 1
        body = received[0]
        assert body["model"] == "qwen38-27b-fp8" and body["seed"] == 20261012
        assert body["max_tokens"] == 4096 and body["temperature"] == 0 and "tools" not in body
        assert body["response_format"]["json_schema"]["schema"]["properties"]["code_lines"]["type"] == "array"
        trace = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert trace["model"] == "qwen38-27b-fp8" and trace["proxy_model_id"] == "kernel-qwen-local-27b"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
