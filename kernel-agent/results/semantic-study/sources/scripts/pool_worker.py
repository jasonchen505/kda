"""One process and workspace per service; finalization never replays model work."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import Campaign, atomic_json, digest  # noqa: E402
from kernel_agent.deferred import TimingQueue, candidates  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from kernel_agent.services import identity, wait_idle  # noqa: E402
from scripts import run_campaign  # noqa: E402


def ready(plan):
    if run_campaign.source_hashes() != plan["source_sha256"]:
        raise ValueError("worker source changed")
    runtime = plan["runtime"]
    if identity(runtime["service"]["port"]) != runtime["service"]:
        raise ValueError("service declaration changed")
    if run_campaign.evaluator_identity(runtime["gpu_python"]) != runtime["evaluator_runtime"]:
        raise ValueError("evaluation environment changed")
    wait_idle(runtime["service"]["port"])


def finalize(plan, row, directory):
    run = row["result"]
    runtime = plan["runtime"]
    folder = directory / "final" / row["job"]["id"]
    folder.mkdir(parents=True, exist_ok=True)
    for candidate in candidates(run):
        path = folder / f"{candidate['source_sha256']}.py"
        if path.exists() and path.read_bytes() != candidate["code"].encode():
            raise ValueError("queued source changed")
        path.write_bytes(candidate["code"].encode())
    queue_plan = {"schema_version": 1, "measurement_condition": "shared_observed", "runtime": runtime,
                  "source_sha256": plan["source_sha256"], "generation_plan_sha256": digest(plan),
                  "generation_result": run, "generation_result_sha256": row["result_sha256"]}

    def evaluate(candidate, final):
        if not final:
            raise ValueError("online-search measurements must be imported, never rerun")
        path = folder / f"{candidate['source_sha256']}.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["source_sha256"]:
            raise ValueError("candidate changed before final evaluation")
        command = [runtime["gpu_python"], str(ROOT / "scripts/evaluate_serial.py"), "--gpu-index", "4",
                   "--lock-file", runtime["gpu_lock_path"], run["task_id"], str(path), "--final"]
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=400)
            return json.loads(child.stdout) if child.returncode == 0 else {
                "status": "evaluator_error", "message": (child.stdout + child.stderr)[-4000:]}
        except subprocess.TimeoutExpired:
            return {"status": "evaluator_timeout"}

    with TimingQueue(folder / "checkpoint.json", queue_plan) as queue:
        return queue.run(evaluate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.absolute()
    plan = json.loads((directory / "plan.json").read_text())
    port = plan["runtime"]["service"]["port"]
    os.environ.update(CUDA_VISIBLE_DEVICES="4", PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT))
    with acquire(directory.parents[1] / "leases" / f"port-{port}.lock", timeout=0):
        with Campaign(directory / "generation.json", plan) as campaign:
            while True:
                before = campaign.summary()["terminal"]
                campaign.run(lambda job: run_campaign.execute(plan, job, directory / "wire"),
                             ready=lambda job: ready(plan), max_new_jobs=1)
                finals = []
                for row in campaign.state["jobs"]:
                    if row["status"] == "completed":
                        measured = finalize(plan, row, directory)
                        finals.append({"job_id": row["job"]["id"], "status": measured["status"],
                                       "correct": measured.get("correct"), "promoted": measured.get("promoted")})
                atomic_json(directory / "final-summary.json", finals)
                summary = campaign.summary()
                print(json.dumps({"port": port, "generation": summary, "finals": finals}), flush=True)
                if not summary["pending"] or summary["terminal"] == before:
                    break
    return 0 if not summary["pending"] and all(r["status"] == "complete" for r in finals) else 2


if __name__ == "__main__":
    sys.exit(main())
