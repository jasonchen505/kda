"""Complete committed generation under a separately frozen shared measurement plan."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent.deferred import TimingQueue, candidates  # noqa: E402
from scripts.run_campaign import evaluator_identity, freeze_sources, source_hashes  # noqa: E402


def origin(path, sources):
    state = json.loads(path.read_text())
    if (state["state_sha256"] != digest({k: v for k, v in state.items() if k != "state_sha256"})
            or state["plan_sha256"] != digest(state["plan"])):
        raise ValueError("origin generation integrity mismatch")
    if [row["job"] for row in state["jobs"]] != state["plan"]["jobs"]:
        raise ValueError("origin schedule changed")
    for name, expected in state["plan"]["source_sha256"].items():
        if hashlib.sha256((sources / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"origin frozen source changed: {name}")
    for row in state["jobs"]:
        if row["status"] != "completed" or digest(row["result"]) != row["result_sha256"]:
            raise ValueError("only committed generation jobs can be replayed for measurement")
        run = row["result"]
        if (run["task_id"], run["study_policy"], run["seed"], run["job_id"]) != (
                row["job"]["task_id"], row["job"]["study_policy"], row["job"]["seed"], row["job"]["id"]):
            raise ValueError("origin result/job mismatch")
        candidates(run)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--generation-sources", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="commit queues and selection without GPU/model calls")
    args = parser.parse_args()
    source = args.generation_sources or args.generation.parent / "sources"
    state = origin(args.generation, source)
    directory = args.directory.absolute()
    runtime = {**state["plan"]["runtime"], "workspace": str(ROOT)}
    if evaluator_identity(runtime["gpu_python"]) != runtime["evaluator_runtime"]:
        raise ValueError("GPU evaluation environment changed")
    manifest = source_hashes()
    freeze_sources(directory / "manifest.json", manifest)
    plan = {"schema_version": 1, "measurement_condition": "shared_observed", "runtime": runtime,
            "origin_checkpoint_sha256": state["state_sha256"], "origin_plan_sha256": state["plan_sha256"],
            "origin_source_sha256": state["plan"]["source_sha256"], "source_sha256": manifest,
            "generation_result_sha256": {r["job"]["id"]: r["result_sha256"] for r in state["jobs"]},
            "scope": "same committed proposals, new observed-sharing measurements; zero new model calls"}
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("shared measurement plan changed")
    atomic_json(plan_path, plan)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(runtime["gpu_index"]), PYTHONPATH=str(ROOT / "src"))
    rows, complete = [], True
    for row in state["jobs"]:
        run = row["result"]
        folder = directory / "queues" / row["job"]["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for candidate in candidates(run):
            path = folder / f"{candidate['source_sha256']}.py"
            if path.exists() and path.read_text() != candidate["code"]:
                raise ValueError("queued source changed")
            path.write_text(candidate["code"])
        queue_plan = {**plan, "generation_result": run, "generation_result_sha256": digest(run)}

        def evaluate(candidate, final):
            path = folder / f"{candidate['source_sha256']}.py"
            if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["source_sha256"]:
                raise ValueError("queued source changed before measurement")
            command = [runtime["gpu_python"], str(ROOT / "scripts/evaluate_coexecuted.py"), "--gpu-index",
                       str(runtime["gpu_index"]), run["task_id"], str(path)]
            if final:
                command.append("--final")
            try:
                child = subprocess.run(command, capture_output=True, text=True, timeout=200)
                return json.loads(child.stdout) if child.returncode == 0 else {
                    "status": "evaluator_error", "message": (child.stdout + child.stderr)[-4000:]}
            except subprocess.TimeoutExpired:
                return {"status": "evaluator_timeout"}

        with TimingQueue(folder / "checkpoint.json", queue_plan) as queue:
            result = queue.state if args.prepare else queue.run(evaluate)
            complete &= result["status"] == "complete"
            rows.append({"job": row["job"], "status": result["status"], "correct": result.get("correct"),
                         "promoted": result.get("promoted"), "selected_sha256": result["selected_sha256"],
                         "search": result["search"], "final": result["final"]})
    if source_hashes() != manifest:
        raise ValueError("measurement sources changed during execution")
    summary = {"schema_version": 1, "status": "complete" if complete else "incomplete",
               "measurement_condition": "shared_observed", "plan": plan, "plan_sha256": digest(plan),
               "jobs": rows, "new_model_requests": 0,
               "origin_proposals": sum(len(r["result"]["attempts"]) for r in state["jobs"]),
               "origin_output_tokens": sum(r["result"]["usage"]["output_tokens"] for r in state["jobs"]),
               "scope": "conditional shared-environment speedups; not pooled with isolated studies or independent new generation"}
    atomic_json(directory / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "final_correct": sum(bool(r["correct"]) for r in rows),
                      "promoted": sum(bool(r["promoted"]) for r in rows), "new_model_requests": 0}))
    return 0 if args.prepare or complete else 2


if __name__ == "__main__":
    sys.exit(main())
