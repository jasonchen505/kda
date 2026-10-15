"""Explicitly authorized loopback services; no foreign process or filesystem access."""
import json
import math
import time
import urllib.request

SERVICES = {
    18118: {"model": "qwen38-27b-fp8", "gpus": "5,6", "families": ["vector_add", "row_sum"],
            "access": "owned", "root": "/data/home/chenyizhou/my-research-project-v11/artifacts/p1/models/Qwen3.8-27B-FP8"},
    54575: {"model": "steamserve-qwen38", "gpus": "0,1", "families": ["softmax", "rmsnorm"],
            "access": "user_authorized_shared_endpoint", "root": "/data/home/yikaifu/steamspec/models/Qwen_Qwen3.8-27B-FP8"},
    60953: {"model": "steamserve-qwen38", "gpus": "2,3", "families": ["transpose", "silu_mul"],
            "access": "user_authorized_shared_endpoint", "root": "/data/home/yikaifu/steamspec/models/Qwen_Qwen3.8-27B-FP8"},
}


def request_json(port, path, body=None):
    if port not in SERVICES:
        raise ValueError("service is outside the user's explicit allowlist")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def identity(port):
    expected = SERVICES[port]
    models = request_json(port, "/v1/models")["data"]
    base = next((m for m in models if m["id"] == expected["model"]), None)
    if (not base or base.get("parent") is not None or base.get("root") != expected["root"]
            or base.get("max_model_len", 0) < 8192):
        raise ValueError("authorized service changed its base-model declaration")
    return {"port": port, "base_model": base["id"], "model_root": base["root"],
            "generation_gpus": expected["gpus"], "gpu_mapping_scope": "user-declared physical indices",
            "access": expected["access"], "max_model_len": base["max_model_len"],
            "vllm_version": request_json(port, "/version")["version"], "adapter_requested": None,
            "weight_identity_scope": "HTTP model declaration only; cross-root weight equivalence unverified"}


def parse_load(metrics):
    values = {"running": [], "waiting": []}
    for line in metrics.splitlines():
        for name in values:
            if line.startswith(f"vllm:num_requests_{name}{{") or line.startswith(f"vllm:num_requests_{name} "):
                value = float(line.rsplit(" ", 1)[1])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid service load")
                values[name].append(value)
    if any(not rows for rows in values.values()):
        raise ValueError("missing service load metrics")
    return {key: sum(rows) for key, rows in values.items()}


def load(port):
    if port not in SERVICES:
        raise ValueError("service is outside the user's explicit allowlist")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        return parse_load(response.read().decode())


def wait_idle(port, timeout=90):
    started = time.monotonic()
    while True:
        value = load(port)
        if not any(value.values()):
            return {**value, "wait_seconds": time.monotonic() - started}
        if time.monotonic() - started >= timeout:
            raise RuntimeError("authorized service is busy; no request dispatched")
        time.sleep(min(1, timeout - (time.monotonic() - started)))
