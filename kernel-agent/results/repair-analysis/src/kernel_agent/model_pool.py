"""Discover explicitly authorized Qwen replicas and retain per-process HTTP evidence."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from kernel_agent.campaign import atomic_json, digest
from kernel_agent.services import parse_load
from kernel_agent.transport import LINE_SCHEMA, PROPOSER_PREFIX

ROOTS = {
    "/data/home/chenyizhou/my-research-project-v11/artifacts/p1/models/Qwen3.8-27B-FP8",
    "/data/home/yikaifu/steamspec/models/Qwen_Qwen3.8-27B-FP8",
}


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def http(port, path, body=None, timeout=5):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     json.dumps(body).encode() if body is not None else None,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode() if path == "/metrics" else json.loads(data)


def process(pid, proc=Path("/proc")):
    # Only public process arguments/statistics. Never read environ, weights or logs.
    args = (proc / str(pid) / "cmdline").read_bytes().decode().strip("\0").split("\0")
    stat = (proc / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    return args, int(stat[1]), stat[19]


def gpu_processes():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], text=True)
    indices = {row.split(",")[0].strip(): int(row.split(",")[1]) for row in rows.splitlines()}
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    return [(int(row.split(",")[0]), indices[row.split(",")[1].strip()]) for row in rows.splitlines() if row.strip()]


def describe(pid, port, root, start, gpus):
    models = http(port, "/v1/models")["data"]
    model = next((m for m in models if m.get("root") == root and m.get("parent") is None), None)
    if model is None or model.get("max_model_len", 0) < 8192:
        raise ValueError("Qwen declaration or context differs from the allowed family")
    return {"pid": pid, "process_start_ticks": start, "port": port, "root": root, "model": model["id"],
            "native_context": model["max_model_len"], "version": http(port, "/version")["version"],
            "gpus": sorted(set(gpus)), "weight_scope": "HTTP declaration; cross-root weight equality unverified"}


def discover(proc=Path("/proc"), gpu_rows=None):
    candidates = []
    gpu_rows = gpu_processes() if gpu_rows is None else gpu_rows
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            pid = int(path.name)
            args, _, start = process(pid, proc)
            if (not any("vllm" in a for a in args[:4])
                    or not ("serve" in args or "vllm.entrypoints.openai.api_server" in args)):
                continue
            root = next((a for a in args if a in ROOTS), None)
            if root is None or "--port" not in args or args[args.index("--host") + 1] != "127.0.0.1":
                continue
            port = int(args[args.index("--port") + 1])
            gpus = []
            for child, gpu in gpu_rows:
                ancestor = child
                for _ in range(12):
                    if ancestor == pid:
                        gpus.append(gpu)
                        break
                    if ancestor <= 1:
                        break
                    try:
                        ancestor = process(ancestor, proc)[1]
                    except (OSError, ValueError):
                        break
            if not gpus or 4 in gpus:
                continue
            endpoint = describe(pid, port, root, start, gpus)
            endpoint["fingerprint"] = digest(endpoint)
            candidates.append(endpoint)
        except (OSError, ValueError, KeyError, IndexError, StopIteration):
            continue
    return sorted(candidates, key=lambda e: e["port"])


def validate(endpoint):
    if process(endpoint["pid"])[2] != endpoint["process_start_ticks"]:
        raise ValueError("model process restarted; active block is not reassigned")
    actual = describe(endpoint["pid"], endpoint["port"], endpoint["root"],
                      endpoint["process_start_ticks"], endpoint["gpus"])
    if digest(actual) != endpoint["fingerprint"]:
        raise ValueError("model declaration changed")
    return parse_load(http(endpoint["port"], "/metrics"))


def choose(endpoints, active, stats):
    """First use every idle replica; then favor lower observed request latency."""
    eligible = [e for e in endpoints if e["fingerprint"] not in active]
    return min(eligible, key=lambda e: (stats.get(e["fingerprint"], {}).get("assigned", 0) > 0,
                                      stats.get(e["fingerprint"], {}).get("ewma_seconds", 60), e["port"])) if eligible else None


class NotDispatched(RuntimeError):
    pass


class OutcomeUnknown(RuntimeError):
    pass


class Client:
    """Exactly one explicit dispatch attempt per call, with durable uncertainty."""
    def __init__(self, endpoint, directory, deadline):
        self.endpoint, self.directory, self.deadline = endpoint, Path(directory), deadline
        self.directory.mkdir(parents=True, exist_ok=True)
        self.quarantined = False
        self.calls = 0

    def propose(self, sections, seed, request_id):
        receipt = {"request_id": request_id, "created_utc": utc(), "endpoint": self.endpoint,
                   "status": "not_dispatched", "omitted_sections": []}
        path = self.directory / f"{request_id}.json"
        started = time.monotonic()
        try:
            if self.quarantined or time.monotonic() >= self.deadline:
                raise NotDispatched("quarantined block or expired wall-time budget")
            while True:
                load = validate(self.endpoint)
                if not any(load.values()):
                    break
                if time.monotonic() - started > 90 or time.monotonic() >= self.deadline:
                    raise NotDispatched("service busy; no model request sent")
                time.sleep(1)
            receipt["admission"] = load
            kept = list(sections)
            while True:
                messages = [{"role": "system", "content": PROPOSER_PREFIX + " No tools. Follow the supplied contract."},
                            {"role": "user", "content": "\n\n".join(s[1] for s in kept)}]
                count = http(self.endpoint["port"], "/tokenize", {"model": self.endpoint["model"],
                             "messages": messages, "add_generation_prompt": True,
                             "chat_template_kwargs": {"enable_thinking": False}})["count"]
                if count + 2048 <= 8192:
                    break
                removable = next((i for i in range(len(kept) - 1, -1, -1) if not kept[i][2]), None)
                if removable is None:
                    raise NotDispatched("mandatory contract/source/feedback exceed common context; no truncation")
                receipt["omitted_sections"].append(kept.pop(removable)[0])
            body = {"model": self.endpoint["model"], "messages": messages, "temperature": .2, "seed": seed,
                    "max_tokens": 2048, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "kernel_proposal", "strict": True, "schema": LINE_SCHEMA}}}
            # Persist the exact request before crossing the external-call boundary.
            wire = self.directory / f"{request_id}.request.json"
            atomic_json(wire, body)
            receipt.update(status="dispatched", dispatched_utc=utc(), input_tokens=count,
                           request_sha256=hashlib.sha256(wire.read_bytes()).hexdigest())
            atomic_json(path, receipt)
            self.calls += 1
            response = http(self.endpoint["port"], "/v1/chat/completions", body,
                            timeout=max(1, min(300, self.deadline - time.monotonic())))
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("missing content in response")
            atomic_json(self.directory / f"{request_id}.response.json", response)
            receipt.update(status="completed", finished_utc=utc(),
                           response_sha256=digest(response), usage=response.get("usage"),
                           finish_reason=response["choices"][0].get("finish_reason"))
            return content
        except Exception as error:
            if receipt["status"] == "dispatched":
                self.quarantined = True
                receipt["status"] = "response_unknown"
                raise OutcomeUnknown(str(error)) from error
            raise NotDispatched(str(error)) from error
        finally:
            receipt["finished_utc"] = utc()
            receipt["elapsed_seconds"] = time.monotonic() - started
            atomic_json(path, receipt)
