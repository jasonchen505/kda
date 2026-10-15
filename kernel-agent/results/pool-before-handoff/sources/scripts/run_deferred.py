"""Generate with shared numerical feedback; resume performance queues independently."""
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
from scripts import evaluate_shared, run_campaign  # noqa: E402


def ready(plan):
    runtime = plan["runtime"]
    if run_campaign.source_hashes() != plan["source_sha256"]:
        raise ValueError("generation sources changed")
    if run_campaign.service_identity(runtime["service"]["port"], runtime["service"]["pid"]) != runtime["service"]:
        raise ValueError("model service changed")
    if run_campaign.evaluator_identity(runtime["gpu_python"]) != runtime["evaluator_runtime"]:
        raise ValueError("numerical evaluator runtime changed")
    observation = evaluate_shared.observe(runtime["gpu_index"])
    if observation["free_mib"] < 1024:
        raise RuntimeError("shared GPU has less than 1024 MiB free")


def queue_plan(state, run):
    return {"schema_version": 1, "kind": "deferred_timing", "generation_plan_sha256": state["plan_sha256"],
            "generation_result": run, "generation_result_sha256": digest(run),
            "runtime": state["plan"]["runtime"], "source_sha256": state["plan"]["source_sha256"],
            "selection": "gate_first_search_only_then_final_veto", "scope": "phase-split engineering pilot, not feedback efficacy"}


def materialize(directory, state):
    for row in state["jobs"]:
        if row["status"] != "completed":
            continue
        run = row["result"]
        queue_dir = directory / "timing" / row["job"]["id"]
        for candidate in candidates(run):
            path = queue_dir / f"{candidate['source_sha256']}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != candidate["code"]:
                raise ValueError("queued candidate file changed")
            path.write_text(candidate["code"])
        with TimingQueue(queue_dir / "checkpoint.json", queue_plan(state, run)):
            pass


def measure(directory):
    state = json.loads((directory / "generation.json").read_text())
    materialize(directory, state)
    runtime = state["plan"]["runtime"]
    if run_campaign.source_hashes() != state["plan"]["source_sha256"]:
        raise ValueError("measurement source changed; use the frozen project/runtime")
    if run_campaign.evaluator_identity(runtime["gpu_python"]) != runtime["evaluator_runtime"]:
        raise ValueError("measurement Python environment changed")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(runtime["gpu_index"]), PYTHONPATH=str(ROOT / "src"))
    for path in sorted((directory / "timing").glob("*/checkpoint.json")):
        stored = json.loads(path.read_text())

        def evaluator(candidate, final):
            source = path.parent / f"{candidate['source_sha256']}.py"
            if hashlib.sha256(source.read_bytes()).hexdigest() != candidate["source_sha256"]:
                raise ValueError("queued source mutated before evaluation")
            command = [runtime["gpu_python"], str(ROOT / "scripts/evaluate_checked.py"), "--gpu-index",
                       str(runtime["gpu_index"]), stored["plan"]["generation_result"]["task_id"], str(source)]
            if final:
                command.append("--final")
            try:
                child = subprocess.run(command, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired:
                return {"status": "evaluator_timeout", "message": "queued evaluator exceeded timeout"}
            return json.loads(child.stdout) if child.returncode == 0 else {
                "status": "evaluator_error", "message": (child.stdout + child.stderr)[-4000:]}

        with TimingQueue(path, stored["plan"]) as queue:
            result = queue.run(evaluator)
            print(json.dumps({"queue": path.parent.name, "status": result["status"],
                              "correct": result.get("correct"), "promoted": result.get("promoted")}))
            if result["status"] != "complete":
                return 2
    return 0 if state["status"] == "complete" else 2


def summary(directory):
    state = json.loads((directory / "generation.json").read_text())
    queues = [json.loads(p.read_text()) for p in sorted((directory / "timing").glob("*/checkpoint.json"))]
    rows = []
    for row in state["jobs"]:
        result = row.get("result", {})
        queues_for_job = [q for q in queues if q["plan"]["generation_result"]["job_id"] == row["job"]["id"]]
        queue = queues_for_job[0] if queues_for_job else None
        attempts = result.get("attempts", [])
        rows.append({"job": row["job"], "generation_status": row["status"], "terminal": result.get("terminal"),
                     "proposals": len(attempts), "numerically_eligible_sources": len(result.get("timing_candidates", [])),
                     "shared_evaluations": [a["evaluation"].get("resource_audit") for a in attempts
                                            if a["evaluation"].get("resource_audit")],
                     "reported_output_tokens": result.get("usage", {}).get("output_tokens"),
                     "timing_status": queue["status"] if queue else "not_queued", "correct": queue.get("correct") if queue else None,
                     "promoted": queue.get("promoted") if queue else None,
                     "selected_sha256": queue["selected_sha256"] if queue else None})
    value = {"schema_version": 1, "kind": "deferred_engineering_pilot", "plan_sha256": state["plan_sha256"],
             "generation_status": state["status"], "runtime": state["plan"]["runtime"],
             "scheduled": len(state["jobs"]), "jobs": rows, "budget_allocations": state["reservations"],
             "scope": "Shared numerical validation and a separate durable timing queue. "
                      "No speedup or feedback benefit before independent queued measurements; not pooled with integrated studies."}
    atomic_json(directory / "summary.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "measure", "summary"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    parser.add_argument("--task", choices=run_campaign.TASKS, default="softmax")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    directory = args.directory.absolute()
    exit_code = 0
    if args.action == "generate":
        if not args.pid:
            parser.error("generate requires the owned model service PID")
        plan = run_campaign.make_plan(run_campaign.runtime_identity(args), run_campaign.source_hashes(),
                                      tasks=[args.task], seeds=[args.seed])
        plan.update(study_kind="deferred_correctness_pilot", scope="phase separation; no efficacy claim")
        plan["generation"].update(evaluation_mode="deferred", evaluation_guard="shared_search_correctness")
        with Campaign(directory / "generation.json", plan) as campaign:
            run_campaign.freeze_sources(directory / "generation.json", plan["source_sha256"])
            campaign.run(lambda job: run_campaign.execute(plan, job, directory / "wire"), ready=lambda job: ready(plan))
            materialize(directory, campaign.state)
            if campaign.summary()["pending"]:
                exit_code = 2
    elif args.action == "measure":
        exit_code = measure(directory)
    print(json.dumps(summary(directory), indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
