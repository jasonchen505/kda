"""Run a small real OpenCode/local-Qwen turn with the updated Humanize workspace API."""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from hmz.sdk import Hmz

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--model-id", choices=("qwen38-27b-fp8", "kernel-qwen3-4b"), default="qwen38-27b-fp8")
    parser.add_argument("--output", default="results/humanize-current-smoke.json")
    args = parser.parse_args()
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError(target)
    proc = Path(f"/proc/{args.pid}")
    if proc.stat().st_uid != os.getuid():
        raise ValueError("local service must belong to this user")
    argv = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    if argv[argv.index("--port") + 1] != str(args.port) or argv[argv.index("--host") + 1] != "127.0.0.1":
        raise ValueError("PID is not the specified loopback service")
    endpoint = f"http://127.0.0.1:{args.port}"
    models = json.load(urllib.request.urlopen(endpoint + "/v1/models", timeout=5))
    base = next(m for m in models["data"] if m["id"] == args.model_id and m["parent"] is None)
    metrics = urllib.request.urlopen(endpoint + "/metrics", timeout=5).read().decode()
    queue = {line.split("{")[0]: float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
             if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))}
    if len(queue) != 2 or any(queue.values()):
        raise ValueError(f"service is busy: {queue}")
    config_name = "data/opencode-local.json" if args.model_id == "kernel-qwen3-4b" else "data/opencode-qwen27.json"
    config = json.loads((ROOT / config_name).read_text())
    provider = config["provider"]["kernel-local"]
    provider["options"]["baseURL"] = endpoint + "/v1"
    model = provider["models"].pop(next(iter(provider["models"])))
    model["limit"]["output"] = 512
    provider["models"][base["id"]] = model
    config["model"] = config["small_model"] = "kernel-local/" + base["id"]
    config["agent"]["build"]["prompt"] = "Return the exact JSON requested. Do not call any tools."
    os.environ.update(OPENCODE_CONFIG_CONTENT=json.dumps(config), OPENCODE_DISABLE_AUTOUPDATE="true",
                      OPENCODE_DISABLE_DEFAULT_PLUGINS="true", PYTHONPATH=str(ROOT / "src"))
    started = datetime.now(UTC).isoformat()
    reply = Hmz(ROOT).run(str(ROOT / "flows/smoke.py"),
                         'Do not call tools. Reply with exactly {"local_kernel_agent":"ready"}.',
                         agents={"builder": "opencode/kernel-local/" + base["id"]},
                         budget={"duration": 180, "output_tokens": 1024, "graceful": False}).run()
    # Some CLIs surround an otherwise exact JSON reply with a Markdown fence.
    cleaned = reply.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1])
    if json.loads(cleaned) != {"local_kernel_agent": "ready"}:
        raise ValueError(f"unexpected smoke reply: {reply!r}")
    target.write_text(json.dumps({"status": "passed", "started_utc": started,
                                 "completed_utc": datetime.now(UTC).isoformat(), "reply": reply,
                                 "python": sys.executable, "endpoint": endpoint, "base_model_id": base["id"],
                                 "adapter_requested": None, "initial_queue": queue,
                                 "effective_config": config,
                                 "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                                   for name in ("flows/smoke.py", "src/kernel_agent/humanize_api.py",
                                                                "scripts/check_updated_humanize.py")},
                                 "scope": "real SDK/CLI/local-model integration smoke, not a kernel trial"}, indent=2) + "\n")
    print(json.dumps({"status": "passed", "reply": reply, "output": str(target)}))


if __name__ == "__main__":
    main()
