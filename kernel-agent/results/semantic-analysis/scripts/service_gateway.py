"""Restore logical loopback endpoints after a recorded, declaration-matched restart."""
import argparse
import hashlib
import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROUTES = {54575: 56825, 60953: 43619}
PATHS = {"/v1/models", "/version", "/metrics", "/tokenize", "/v1/chat/completions"}


def verify_target(service, target):
    with urllib.request.urlopen(f"http://127.0.0.1:{target}/v1/models", timeout=5) as response:
        models = json.load(response)["data"]
    with urllib.request.urlopen(f"http://127.0.0.1:{target}/version", timeout=5) as response:
        version = json.load(response)["version"]
    base = next((m for m in models if m["id"] == service["base_model"]), None)
    if (not base or base.get("root") != service["model_root"] or base.get("parent") is not None
            or base.get("max_model_len") != service["max_model_len"] or version != service["vllm_version"]):
        raise ValueError("replacement endpoint changed its recorded model declaration")
    return {"base_model": base["id"], "model_root": base["root"], "vllm_version": version,
            "max_model_len": base["max_model_len"], "parent": None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def forward(self):
        if self.path not in PATHS:
            self.send_error(404)
            return
        data = None
        if self.command == "POST":
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                self.send_error(413)
                return
            data = self.rfile.read(length)
        request = urllib.request.Request(f"http://127.0.0.1:{self.server.target}{self.path}", data=data,
                                         headers={"Content-Type": "application/json"}, method=self.command)
        sent = False
        try:
            try:
                response = urllib.request.urlopen(request, timeout=300)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.end_headers()
                sent = True
                while chunk := response.readline():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError:
            if not sent:
                self.send_error(502, "recorded upstream unavailable; gateway does not retry")

    do_GET = forward
    do_POST = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    checks = []
    for worker in plan["workers"]:
        if worker["port"] in ROUTES:
            checks.append({"logical_port": worker["port"], "target_port": ROUTES[worker["port"]],
                           "generation_gpus": worker["service"]["generation_gpus"],
                           "declaration": verify_target(worker["service"], ROUTES[worker["port"]])})
    servers = []
    try:
        for port, target in ROUTES.items():
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            server.target = target
            servers.append(server)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "routes": checks,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "HTTP declarations and physical GPU mapping matched after process restart; weight equality and cache continuity unverified; no request retry, model restart or foreign filesystem access"}, indent=2) + "\n")
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Event().wait()
    finally:
        for server in servers:
            server.server_close()


if __name__ == "__main__":
    main()
