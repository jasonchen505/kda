"""Prepare a service-blocked comparison and run three isolated endpoint workers."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import Campaign, atomic_json, digest  # noqa: E402
from kernel_agent.services import SERVICES, identity  # noqa: E402
from scripts import run_campaign  # noqa: E402

SEEDS = [20261027, 20261028]


def worker_plan(runtime, sources, port):
    plan = run_campaign.make_plan(runtime, sources, tasks=SERVICES[port]["families"], seeds=SEEDS)
    plan.update(study_kind="service_blocked_feedback", scope="two seeds per development task; paired within service")
    plan["generation"].update(evaluation_mode="search_only", measurement_condition="shared_observed",
                              evaluation_guard="cooperatively_serialized_shared_observed", pool_protocol=True,
                              max_tokens_per_request=2048, common_context_cap=8192, evaluator_timeout_seconds=400)
    for job in plan["jobs"]:
        job["service_port"] = port
    return plan


def prepare(directory, args):
    sources = run_campaign.source_hashes()
    common = run_campaign.runtime_identity(args)
    workers = []
    for port in SERVICES:
        folder = directory / "workers" / f"p{port}"
        workspace = folder / "workspace"
        runtime = {**common, "service": identity(port), "workspace": str(workspace),
                   "gpu_lock_path": str(directory / "gpu4.lock")}
        plan = worker_plan(runtime, sources, port)
        for name, expected in sources.items():
            target = workspace / name
            value = (ROOT / name).read_bytes()
            if target.exists() and target.read_bytes() != value:
                raise ValueError(f"worker source differs: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
        plan_path = folder / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("worker plan changed; use a fresh study directory")
        atomic_json(plan_path, plan)
        with Campaign(folder / "generation.json", plan):
            run_campaign.freeze_sources(folder / "generation.json", sources)
        workers.append({"port": port, "directory": str(folder), "plan_sha256": digest(plan),
                        "service": runtime["service"], "jobs": plan["jobs"]})
    plan = {"schema_version": 1, "study_kind": "service_blocked_feedback", "workers": workers,
            "source_sha256": sources, "seeds": SEEDS, "scheduled": 24,
            "budget_allocation": {"duration_seconds": 21600, "output_tokens": 480000},
            "scope": "six development families, two seeds, two arms; service-blocked descriptive replication; weights across roots unverified"}
    path = directory / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("pool plan changed")
    atomic_json(path, plan)
    run_campaign.freeze_sources(directory / "manifest.json", sources)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pid", type=int, required=True, help="owned port-18118 PID; foreign services are HTTP-only")
    parser.add_argument("--port", type=int, default=18118)
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    args = parser.parse_args()
    directory = args.directory.absolute()
    plan = prepare(directory, args)
    if args.prepare:
        print(json.dumps({"status": "prepared", "scheduled": plan["scheduled"], "workers": 3, "model_calls": 0}))
        return 0
    children = []
    for worker in plan["workers"]:
        folder = Path(worker["directory"])
        log = (folder / "worker.log").open("a")
        env = {**os.environ, "PYTHONPATH": str(folder / "workspace/src")}
        child = subprocess.Popen([sys.executable, str(folder / "workspace/scripts/pool_worker.py"),
                                  "--directory", str(folder)], cwd=folder / "workspace", env=env,
                                 stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        log.close()
        children.append((worker["port"], child))
    exits = {str(port): child.wait(timeout=10800) for port, child in children}
    atomic_json(directory / "workers-exit.json", exits)
    print(json.dumps({"worker_exits": exits}))
    return 0 if all(code == 0 for code in exits.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
