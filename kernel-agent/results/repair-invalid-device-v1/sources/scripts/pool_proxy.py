"""Serialize own requests and admit them only to idle, explicitly allowed services."""
import hashlib
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from kernel_agent.campaign import atomic_json
from kernel_agent.services import identity, load, request_json, wait_idle
from kernel_agent.transport import normalize


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path not in {"/v1/models", "/json-lines/v1/models"}:
            self.send_error(404)
            return
        body = json.dumps({"object": "list", "data": [{"id": "kernel-qwen-local-27b", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/json-lines/v1/chat/completions":
            self.send_error(404)
            return
        server = self.server
        job, plan = server.job, server.plan
        service = plan["runtime"]["service"]
        port = service["port"]
        request_id = uuid4().hex
        receipt = {"request_id": request_id, "port": port, "job_id": job["id"], "status": "not_dispatched",
                   "created_utc": datetime.now(timezone.utc).isoformat()}
        headers_sent = False
        with server.request_lock:
            try:
                if server.quarantined:
                    raise ValueError("this job's proxy is quarantined after an uncertain upstream response")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("model") != "kernel-qwen-local-27b":
                    raise ValueError("only the configured local model alias is allowed")
                if identity(port) != service:
                    raise ValueError("authorized service identity changed")
                receipt["admission"] = wait_idle(port)
                body.update(model=service["base_model"], temperature=plan["generation"]["temperature"],
                            seed=job["seed"], chat_template_kwargs={"enable_thinking": False})
                body["max_tokens"] = min(body.pop("max_completion_tokens", body.get("max_tokens", 2048)), 2048)
                body = normalize(body, "json-lines")
                tokens = request_json(port, "/tokenize", {"model": body["model"], "messages": body["messages"],
                                      "add_generation_prompt": True, "chat_template_kwargs": body["chat_template_kwargs"]})
                receipt.update(input_tokens=tokens["count"], max_output_tokens=body["max_tokens"], context_cap=8192)
                if tokens["count"] + body["max_tokens"] > 8192:
                    raise ValueError("request exceeds the common 8192-token context; no silent prompt truncation")
                traced = {k: body[k] for k in ("messages", "tools", "tool_choice", "model", "max_tokens", "stream",
                          "temperature", "seed", "response_format", "chat_template_kwargs") if k in body}
                traced.update(policy="json-lines", upstream=f"http://127.0.0.1:{port}",
                              proxy_model_id="kernel-qwen-local-27b", request_id=request_id)
                wire = server.trace_dir / f"{request_id}.json"
                atomic_json(wire, traced)
                receipt.update(status="dispatched", wire_sha256=hashlib.sha256(wire.read_bytes()).hexdigest(),
                               dispatched_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(server.trace_dir / "admission" / f"{request_id}.json", receipt)
                request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
                done = not body.get("stream", False)
                with urllib.request.urlopen(request, timeout=300) as response:
                    self.send_response(response.status)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                    self.end_headers()
                    headers_sent = True
                    while line := response.readline():
                        done |= line.strip() == b"data: [DONE]"
                        self.wfile.write(line)
                        self.wfile.flush()
                if not done:
                    raise RuntimeError("upstream stream ended without its completion marker")
                receipt.update(status="completed", load_after=load(port))
            except Exception as error:  # noqa: BLE001 -- external uncertainty must not trigger another model request
                if receipt["status"] == "dispatched":
                    server.quarantined = True
                    receipt["status"] = "response_unknown"
                receipt["error"] = str(error)[-2000:]
                if not headers_sent:
                    self.send_error(409, "local request not admitted or outcome uncertain; no proxy retry")
            finally:
                receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(server.trace_dir / "admission" / f"{request_id}.json", receipt)


@contextmanager
def serve(plan, job, trace_dir):
    trace_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.plan, server.job, server.trace_dir = plan, job, trace_dir
    server.request_lock, server.quarantined = threading.Lock(), False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/json-lines/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
