"""Normalize the local chat-completions protocol; never route to a paid provider."""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from kernel_agent.transport import POLICIES, normalize

UPSTREAM = "http://127.0.0.1:18117"
MODEL_ID = "kernel-qwen3-4b"
TRACE_DIR = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path not in ["/v1/models", *[f"/{policy}/v1/models" for policy in POLICIES]]:
            self.send_error(404)
            return
        body = json.dumps({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        paths = {f"/{policy}/v1/chat/completions": policy for policy in POLICIES}
        paths["/v1/chat/completions"] = "passthrough"
        if self.path not in paths:
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") != MODEL_ID:
            self.send_error(400, "only the local base model is allowed")
            return
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["max_tokens"] = min(body.pop("max_completion_tokens", body.get("max_tokens", 4096)), 4096)
        body["temperature"] = 0
        body["seed"] = 20261011
        policy = paths[self.path]
        tools_before = len(body.get("tools", []))
        try:
            body = normalize(body, policy)
        except ValueError as error:
            self.send_error(400, str(error))
            return
        if TRACE_DIR is not None:
            # Never log HTTP headers or credentials. Only this task's model protocol.
            traced = {key: body[key] for key in ("messages", "tools", "tool_choice", "model",
                       "max_tokens", "stream", "temperature", "seed", "response_format") if key in body}
            traced.update(policy=policy, tools_before=tools_before)
            (TRACE_DIR / f"{uuid4().hex}.json").write_text(json.dumps(traced, indent=2))
        request = urllib.request.Request(UPSTREAM + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.end_headers()
                while line := response.readline():
                    self.wfile.write(line)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except urllib.error.HTTPError as error:
            self.send_response(error.code)
            self.end_headers()
            self.wfile.write(error.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument("--model-id", choices=("kernel-qwen3-4b", "kernel-qwen-local-27b"), default=MODEL_ID)
    args = parser.parse_args()
    endpoint = urllib.parse.urlparse(args.upstream)
    if (endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost", "::1")
            or endpoint.path not in ("", "/") or endpoint.username or endpoint.password):
        parser.error("upstream must be a loopback-only local HTTP model server")
    UPSTREAM = args.upstream.rstrip("/")
    MODEL_ID = args.model_id
    TRACE_DIR = args.trace_dir
    if TRACE_DIR is not None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", 18116), Handler).serve_forever()
