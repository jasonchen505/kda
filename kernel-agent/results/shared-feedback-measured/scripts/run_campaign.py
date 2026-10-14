"""Prepare or resume a frozen local-model campaign without replaying uncertain calls."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from kernel_agent.campaign import Campaign, atomic_json, digest
from scripts import local_proxy

TASKS = ("vector_add", "softmax", "rmsnorm", "transpose", "row_sum", "silu_mul")
SEEDS = (20261021, 20261022, 20261023)
BASE = "qwen38-27b-fp8"
ALIAS = "kernel-qwen-local-27b"


def source_hashes():
    paths = ["flows/kernel_loop.py", "scripts/run_campaign.py", "scripts/local_proxy.py",
             "scripts/evaluate_candidate.py", "scripts/evaluate_checked.py", "docs/measurement-guard.md",
             "scripts/evaluate_shared.py", "scripts/run_deferred.py",
             "docs/deferred-evaluation.md",
             "scripts/evaluate_coexecuted.py", "scripts/finish_deferred.py",
             "scripts/run_shared_campaign.py", "docs/shared-study-plan.md",
             "data/tasks.json", "data/opencode-qwen27.json",
             "results/qwen27-asset-manifest.json", "docs/multiseed-study-plan.md", "pyproject.toml", "uv.lock"]
    paths += [str(p.relative_to(ROOT)) for folder in ("src/kernel_agent", "references")
              for p in sorted((ROOT / folder).rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(set(paths))}


def local_models(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        return json.load(response)


def service_identity(port, pid):
    """Verify an existing owned loopback service; never start/kill a model process."""
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid():
        raise ValueError("model service must belong to this user")
    argv = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    if ("--port" not in argv or argv[argv.index("--port") + 1] != str(port)
            or "--host" not in argv or argv[argv.index("--host") + 1] != "127.0.0.1"):
        raise ValueError("PID must be the specified loopback service")
    base = next((m for m in local_models(port)["data"] if m["id"] == BASE), None)
    asset = json.loads((ROOT / "results/qwen27-asset-manifest.json").read_text())
    if not base or base.get("parent") is not None or base.get("root") != asset["directory"]:
        raise ValueError("expected the recorded cached base model, without an adapter")
    directory = Path(asset["directory"])
    for name, expected in asset["metadata_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"model metadata changed: {name}")
    for shard in asset["weight_shards"]:
        stat = (directory / shard["name"]).stat()
        if stat.st_size != shard["bytes"] or stat.st_mtime_ns != shard["mtime_ns"]:
            raise ValueError(f"cached weight inventory changed: {shard['name']}")
    env = dict(item.split("=", 1) for item in proc.joinpath("environ").read_bytes().decode().split("\0")
               if "=" in item)
    # Only retain device visibility, not the process environment or credentials.
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not visible or not all(p.isdecimal() for p in visible.split(",")):
        raise ValueError("model GPU visibility must be explicit physical indices")
    version = subprocess.check_output([str(Path(argv[0]).with_name("python")), "-c",
                                      "import importlib.metadata; print(importlib.metadata.version('vllm'))"],
                                     text=True, timeout=20).strip()
    return {"port": port, "pid": pid, "process_start_ticks": proc.joinpath("stat").read_text().split()[21],
            "command_sha256": digest(argv), "base_model": BASE, "model_root": base["root"],
            "generation_gpus": visible, "vllm_version": version, "adapter_requested": None,
            "weight_identity_scope": "metadata hashes plus shard sizes/mtimes; full weight hashes unverified"}


def runtime_identity(args):
    from hmz.flows import Agent
    from hmz.sdk import Hmz

    service = service_identity(args.port, args.pid)
    if str(args.gpu_index) in service["generation_gpus"].split(","):
        raise ValueError("evaluation GPU must be separate from model GPUs")
    paths = {"Agent": Path(inspect.getfile(Agent)), "Hmz": Path(inspect.getfile(Hmz))}
    evaluator = evaluator_identity(args.gpu_python)
    return {"service": service, "hmz_version": importlib.metadata.version("hmz"),
            "hmz_public_source_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
            "opencode_version": subprocess.check_output(["opencode", "--version"], text=True, timeout=20).strip(),
            "gpu_index": args.gpu_index, "gpu_python": evaluator["invocation"],
            "evaluator_runtime": evaluator, "workspace": str(ROOT)}


def evaluator_identity(executable):
    # Resolving bin/python's symlink discards the virtualenv and its installed packages.
    # Normalize only the invocation path; verify that exact path in a fresh process.
    invocation = os.path.abspath(os.path.expanduser(executable))
    captured = json.loads(subprocess.check_output([invocation, "-c",
                         "import json,sys,torch,triton; print(json.dumps({'python':sys.version.split()[0],"
                         "'prefix':sys.prefix,'executable':sys.executable,'torch':torch.__version__,"
                         "'triton':triton.__version__}))"], text=True, timeout=30))
    if (captured["torch"], captured["triton"]) != ("2.6.0+cu124", "3.2.0"):
        raise ValueError("evaluation requires the recorded Torch 2.6.0+cu124 / Triton 3.2.0 runtime")
    return {"invocation": invocation, **captured}


def make_plan(runtime, sources, *, tasks=TASKS, seeds=SEEDS, resource_wait_seconds=0):
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError("tasks must be unique members of the six contracted families")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be nonempty and unique")
    if type(resource_wait_seconds) is not int or not 0 <= resource_wait_seconds <= 86400:
        raise ValueError("resource wait must be integer seconds between 0 and 86400")
    jobs = []
    for task in tasks:
        for seed in seeds:
            arms = ["blind", "feedback"]
            random.Random(f"kernel-agent-v1:{task}:{seed}").shuffle(arms)
            jobs.extend({"id": f"{task}-s{seed}-{arm}", "task_id": task, "seed": seed,
                         "study_policy": arm} for arm in arms)
    random.Random(20261020).shuffle(jobs)
    return {"schema_version": 1, "study_kind": "multiseed_feedback", "jobs": jobs,
            "runtime": runtime, "source_sha256": sources, "seeds": list(seeds), "tasks": list(tasks),
            "scheduling": {"resource_wait_seconds": resource_wait_seconds, "poll_seconds": 30},
            "per_job_budget": {"duration_seconds": 900, "output_tokens": 20000},
            "campaign_budget": {"duration_seconds": 900 * len(jobs), "output_tokens": 20000 * len(jobs)},
            "generation": {"temperature": 0.2, "max_tokens_per_request": 4096, "rounds": 3,
                           "selection_policy": "gate_first", "proposal_format": "lines",
                           "mode": "reference_guided", "adapter_requested": None,
                           "evaluation_guard": "pre_post_foreign_cuda_pid_check"},
            "scope": "development-task replication, not held-out generalization; task-clustered descriptive contrasts"}


def readiness(plan):
    runtime = plan["runtime"]
    service = runtime["service"]
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("campaign sources changed; use a new campaign instead of resuming")
    if evaluator_identity(runtime["gpu_python"]) != runtime["evaluator_runtime"]:
        raise ValueError("evaluation Python environment changed")
    if service_identity(service["port"], service["pid"]) != service:
        raise ValueError("model service identity changed")
    with urllib.request.urlopen(f"http://127.0.0.1:{service['port']}/metrics", timeout=5) as response:
        metrics = response.read().decode()
    counts = {"running": [], "waiting": []}
    for line in metrics.splitlines():
        for name in counts:
            if line.startswith(f"vllm:num_requests_{name}" ) and not line.startswith("#"):
                counts[name].append(float(line.rsplit(" ", 1)[1]))
    if any(not values or any(v != 0 for v in values) for values in counts.values()):
        raise RuntimeError("local model service has active or queued requests")
    if plan.get("generation", {}).get("measurement_condition") == "shared_observed":
        from scripts.evaluate_coexecuted import observe
        if observe(runtime["gpu_index"])["free_mib"] < 1024:
            raise RuntimeError("evaluation GPU has less than 1024 MiB free")
        return
    used = subprocess.check_output(["nvidia-smi", "-i", str(runtime["gpu_index"]),
                                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   text=True, timeout=10).strip()
    if int(used) >= 256:
        raise RuntimeError(f"evaluation GPU {runtime['gpu_index']} is occupied ({used} MiB)")


def wait_until_ready(plan, job, path):
    """Retry only resource readiness, never a failed/unknown model job; no allocation yet."""
    deadline = time.monotonic() + plan["scheduling"]["resource_wait_seconds"]
    while True:
        try:
            readiness(plan)
            atomic_json(path, {"waiting": False, "job_id": job["id"]})
            return
        except RuntimeError as error:
            remaining = deadline - time.monotonic()
            atomic_json(path, {"waiting": True, "job_id": job["id"], "reason": str(error),
                               "remaining_seconds": max(0, int(remaining)),
                               "observed_utc": datetime.now(timezone.utc).isoformat()})
            if remaining <= 0:
                raise
            print(json.dumps({"waiting_for_resources": str(error), "job_id": job["id"]}), flush=True)
            time.sleep(min(plan["scheduling"]["poll_seconds"], remaining))


@contextmanager
def proxy(plan, job, trace_dir):
    local_proxy.UPSTREAM = f"http://127.0.0.1:{plan['runtime']['service']['port']}"
    local_proxy.MODEL_ID, local_proxy.UPSTREAM_MODEL_ID = ALIAS, BASE
    local_proxy.SEED = job["seed"]
    local_proxy.TEMPERATURE = plan["generation"]["temperature"]
    local_proxy.TRACE_DIR = trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    server = local_proxy.ThreadingHTTPServer(("127.0.0.1", 0), local_proxy.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/json-lines/v1"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def execute(plan, job, trace_root):
    from hmz.sdk import Hmz

    fixture = json.loads((ROOT / "data/tasks.json").read_text())[job["task_id"]]
    config = json.loads((ROOT / "data/opencode-qwen27.json").read_text())
    generation = plan["generation"]
    run_id = f"campaign-{digest(plan)[:12]}-{job['id']}"
    os.environ.update(OPENCODE_DISABLE_AUTOUPDATE="true", OPENCODE_DISABLE_DEFAULT_PLUGINS="true",
                      CUDA_VISIBLE_DEVICES=str(plan["runtime"]["gpu_index"]),
                      PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT))
    with proxy(plan, job, trace_root / job["id"]) as endpoint:
        config["provider"]["kernel-local"]["options"]["baseURL"] = endpoint
        config["agent"]["build"]["temperature"] = generation["temperature"]
        os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        print(json.dumps({"starting": job["id"]}), flush=True)
        budget = plan["per_job_budget"]
        deferred = generation.get("evaluation_mode") == "deferred"
        evaluator_script = "evaluate_shared.py" if deferred else "evaluate_checked.py"
        if generation.get("measurement_condition") == "shared_observed":
            evaluator_script = "evaluate_coexecuted.py"
        result = Hmz(ROOT).run(str(ROOT / "flows/kernel_loop.py"),
                              fixture["contract"] + "\nVisible examples: " + json.dumps(fixture["examples"]),
                              agents={"builder": f"opencode/kernel-local/{ALIAS}"},
                              budget={"duration": budget["duration_seconds"], "output_tokens": budget["output_tokens"],
                                      "graceful": False},
                              params={"task_id": job["task_id"], "study_policy": job["study_policy"],
                                      "run_id": run_id, "mode": generation["mode"], "rounds": generation["rounds"],
                                      "proposal_format": generation["proposal_format"],
                                      "selection_policy": generation["selection_policy"],
                                      "evaluation_mode": "deferred" if deferred else "integrated",
                                      "evaluator": [plan["runtime"]["gpu_python"],
                                                    str(ROOT / "scripts" / evaluator_script), "--gpu-index",
                                                    str(plan["runtime"]["gpu_index"])]}).run()
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("measured sources changed during a job")
    result.update(seed=job["seed"], transport_policy="json-lines", job_id=job["id"])
    return result


def freeze_sources(checkpoint, sources):
    for name, sha in sources.items():
        target = checkpoint.parent / "sources" / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise ValueError(f"frozen campaign source differs: {name}")
        else:
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError(f"source changed while preparing: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="freeze the plan; no model/GPU requests")
    action.add_argument("--run", action="store_true", help="resume pending jobs under a single ownership lock")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/multiseed-v1/checkpoint.json")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, default=6)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    parser.add_argument("--max-new-runs", type=int, help="stop at a task boundary after N new jobs")
    parser.add_argument("--resource-wait-seconds", type=int, default=0,
                        help="bounded readiness waiting per pending job; never retries a model job")
    args = parser.parse_args()
    plan = make_plan(runtime_identity(args), source_hashes(), resource_wait_seconds=args.resource_wait_seconds)
    with Campaign(args.checkpoint, plan) as campaign:
        freeze_sources(args.checkpoint, plan["source_sha256"])
        if args.run:
            campaign.run(lambda job: execute(plan, job, args.checkpoint.parent / "wire"),
                         ready=lambda job: wait_until_ready(plan, job, args.checkpoint.parent / "resource-wait.json"),
                         max_new_jobs=args.max_new_runs)
        summary = campaign.summary()
        atomic_json(args.checkpoint.parent / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    if args.run and summary["pending"]:
        return 2  # Resource-blocked/paused/budget-limited campaigns are not completed experiments.
    return 0


if __name__ == "__main__":
    sys.exit(main())
