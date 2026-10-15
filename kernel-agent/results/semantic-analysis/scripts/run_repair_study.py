"""Run a prospective factorial study through a dynamically balanced local Qwen pool."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent import line_protocol, model_pool, repair  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.run_campaign import evaluator_identity  # noqa: E402

TASKS = ["vector_add", "softmax", "rmsnorm", "transpose", "row_sum", "silu_mul", "row_mean", "layer_norm"]
SEEDS = [20261101, 20261102]
INFRA = {"resource_busy", "resource_contended", "resource_monitor_error", "resource_exhausted", "evaluator_timeout", "device_mismatch"}


def source_files():
    paths = [ROOT / n for n in ["data/tasks.json", "pyproject.toml", "uv.lock", "docs/repair-study-plan.md"]]
    for name in ["src", "scripts", "references"]:
        paths.extend(p for p in (ROOT / name).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.name != ".DS_Store" and p.suffix not in {".pyc", ".log"})
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def make_plan():
    blocks = [{"id": f"{t}-s{s}", "task_id": t, "seed": s} for t in TASKS for s in SEEDS]
    random.Random(20261030).shuffle(blocks)
    return {"schema_version": 2, "kind": "contract_counterexample_factorial", "blocks": blocks,
            "tasks": TASKS, "seeds": SEEDS, "generation": list(repair.GENERATION), "repair": list(repair.REPAIR),
            "arms_per_block": 4, "scheduled_arms": 64, "max_model_calls": 160,
            "context_limit": 8192, "max_output_tokens": 2048, "temperature": .2,
            "calls_per_block": 10, "model_phase_deadline_seconds": 1800, "allocated_output_tokens": 327680,
            "source_sha256": source_files(), "allowed_model_roots": sorted(model_pool.ROOTS),
            "evaluation_gpu": 4, "physical_device_proof_required": True,
            "scope": "within-process paired factorial after explicit UUID binding; v1 invalid pre-run retained separately; no weight-equivalence claim"}


class Study:
    def __init__(self, directory, gpu_python, *, plan=None):
        self.directory, self.gpu_python = Path(directory).absolute(), gpu_python
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.plan = make_plan() if plan is None else plan
        self.plan["runtime"] = {"evaluation_lock_file": str(self.directory / "gpu4.lock"),
                                "evaluator": evaluator_identity(gpu_python)}
        path = self.directory / "plan.json"
        if path.exists() and json.loads(path.read_text()) != self.plan:
            raise ValueError("immutable study source/plan changed")
        atomic_json(path, self.plan)
        for name, expected in self.plan["source_sha256"].items():
            target = self.directory / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("frozen source changed")
            shutil.copyfile(ROOT / name, target)
        state_path = self.directory / "checkpoint.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            assert self.state["plan_sha256"] == digest(self.plan)
            assert self.state["state_sha256"] == digest({k: v for k, v in self.state.items() if k != "state_sha256"})
            for block in self.state["blocks"]:
                if block["status"] == "running":
                    # Unknown model work is not replayed. This study is not session-resumable.
                    block.update(status="interrupted_unknown", error="supervisor interrupted; no automatic replay")
        else:
            self.state = {"plan_sha256": digest(self.plan), "blocks": [{"block": b, "status": "pending"}
                          for b in self.plan["blocks"]], "allocations": {"model_calls": 0, "output_tokens": 0},
                          "services": {}, "dispatches": [], "status": "prepared"}
        self.save()

    def save(self):
        with self.mutex:
            self.state.pop("state_sha256", None)
            self.state["updated_utc"] = model_pool.utc()
            self.state["state_sha256"] = digest(self.state)
            atomic_json(self.directory / "checkpoint.json", self.state)

    def evaluate(self, task, code_path, final=False):
        command = [self.gpu_python, str(ROOT / "scripts/evaluate_serial.py"), "--gpu-index", "4",
                   "--lock-file", str(self.directory / "gpu4.lock"), "--require-device-proof", task, str(code_path)]
        command.append("--final" if final else "--diagnostics")
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=400)
            return json.loads(result.stdout) if result.returncode == 0 else {
                "status": "evaluator_error", "message": (result.stdout + result.stderr)[-2500:]}
        except (subprocess.SubprocessError, ValueError) as error:
            return {"status": "evaluator_error", "message": str(error)[-2500:]}

    def run_block(self, block, endpoint):
        folder = self.directory / "blocks" / block["id"]
        folder.mkdir(parents=True, exist_ok=True)
        client = model_pool.Client(endpoint, folder / "requests", time.monotonic() + 1800)
        task = json.loads((ROOT / "data/tasks.json").read_text())[block["task_id"]]
        result = {"block": block, "endpoint": endpoint, "initial": {}, "arms": {}, "started_utc": model_pool.utc()}
        seen = {}
        stopped = None

        def propose(generation, policy, history, call_id, seed):
            nonlocal stopped
            row = {"request_id": call_id, "parent_sha256": history[-1].get("source_sha256") if history else None}
            if stopped:
                row["evaluation"] = {"status": "not_run_infrastructure", "message": stopped}
                return row
            try:
                reply = client.propose(repair.sections(block["task_id"], task["contract"], generation, history, policy), seed, call_id)
                row["reply"] = reply
                proposed = line_protocol.proposal(reply)
                row.update(draft=proposed["draft"], plan=proposed["plan"], proposed_code=proposed["code"])
                source_sha = admit(proposed["code"])
                row.update(source_sha256=source_sha, code=proposed["code"])
                code_path = folder / "candidates" / f"{source_sha}.py"
                code_path.parent.mkdir(exist_ok=True)
                code_path.write_text(proposed["code"])
                if source_sha in seen:
                    row.update(duplicate_of=seen[source_sha][0], evaluation=seen[source_sha][1])
                else:
                    row["evaluation"] = self.evaluate(block["task_id"], code_path)
                    seen[source_sha] = (call_id, row["evaluation"])
                if row["evaluation"]["status"] in INFRA:
                    stopped = row["evaluation"]["status"]
            except (model_pool.OutcomeUnknown, model_pool.NotDispatched) as error:
                row["evaluation"] = {"status": "model_infrastructure_error", "message": str(error)[-1500:]}
                stopped = str(error)
            except (ValueError, SyntaxError, TypeError) as error:
                row["evaluation"] = {"status": "protocol_error", "message": str(error)}
            atomic_json(folder / "attempts" / f"{call_id}.json", row)
            return row

        generation_order = list(repair.GENERATION)
        random.Random(block["seed"]).shuffle(generation_order)
        for generation in generation_order:
            result["initial"][generation] = propose(generation, "baseline", [], f"initial-{generation}", block["seed"])
        # Interleave the two repair arms by round; both import the identical initial attempt.
        for generation in generation_order:
            histories = {p: [result["initial"][generation]] for p in repair.REPAIR}
            for round_index in range(2):
                order = list(repair.REPAIR)
                random.Random(f"{block['seed']}:{generation}:{round_index}").shuffle(order)
                for policy in order:
                    history = histories[policy]
                    row = propose(generation, policy, history, f"{generation}-{policy}-r{round_index + 1}",
                                  block["seed"] + round_index + 1)
                    history.append(row)
            for policy, history in histories.items():
                best = repair.select(history)
                result["arms"][repair.arm_id(generation, policy)] = {
                    "generation": generation, "repair": policy, "attempts": history,
                    "selected_sha256": best["source_sha256"] if best else None,
                    "search_correct": best is not None, "selection_committed_utc": model_pool.utc(),
                    "final": None, "correct": False, "promoted": False}
        # Commit every search decision before accessing any final-data result.
        result.update(generation_committed_utc=model_pool.utc(), model_calls=client.calls,
                      infrastructure_stop=stopped)
        atomic_json(folder / "generation.json", result)
        final_cache = {}
        for name, arm in result["arms"].items():
            if arm["selected_sha256"]:
                source_sha = arm["selected_sha256"]
                if source_sha in final_cache:
                    origin, value = final_cache[source_sha]
                    arm["final_reused_from"] = origin
                else:
                    value = self.evaluate(block["task_id"], folder / "candidates" / f"{source_sha}.py", final=True)
                    final_cache[source_sha] = (name, value)
                arm["final"] = value
                verdict = decision(value)
                arm.update(correct=verdict["correct"], promoted=verdict["promoted"], final_decision=verdict)
            atomic_json(folder / "final" / f"{name}.json", arm)
        result["finished_utc"] = model_pool.utc()
        atomic_json(folder / "result.json", result)
        durations = []
        for p in (folder / "requests").glob("*.json"):
            receipt = json.loads(p.read_text())
            if receipt.get("status") == "completed":
                durations.append(receipt["elapsed_seconds"])
        return result, sum(durations) / len(durations) if durations else 60

    def run(self):
        futures, active = {}, set()
        deadline = time.monotonic() + 10800
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            while time.monotonic() < deadline:
                for future in list(futures):
                    if not future.done():
                        continue
                    row, endpoint = futures.pop(future)
                    active.remove(endpoint["fingerprint"])
                    try:
                        value, elapsed = future.result()
                        row.update(status="complete", result_sha256=digest(value))
                        stats = self.state["services"][endpoint["fingerprint"]]
                        stats["completed"] = stats.get("completed", 0) + 1
                        stats["ewma_seconds"] = .5 * stats.get("ewma_seconds", elapsed) + .5 * elapsed
                    except Exception as error:  # noqa: BLE001 -- preserve failed block denominator
                        row.update(status="failed", error=str(error)[-3000:])
                    self.save()
                    print(json.dumps({"block": row["block"]["id"], "status": row["status"]}), flush=True)
                pending = [b for b in self.state["blocks"] if b["status"] == "pending"]
                if not pending and not futures:
                    self.state["status"] = "complete"
                    self.save()
                    return
                endpoints = []
                for endpoint in model_pool.discover():
                    try:
                        if not any(model_pool.validate(endpoint).values()):
                            endpoints.append(endpoint)
                    except (OSError, ValueError):
                        continue
                while pending and len(futures) < 8:
                    endpoint = model_pool.choose(endpoints, active, self.state["services"])
                    if endpoint is None:
                        break
                    row = pending.pop(0)
                    key = endpoint["fingerprint"]
                    stats = self.state["services"].setdefault(key, {"endpoint": endpoint, "assigned": 0})
                    stats["assigned"] += 1
                    row.update(status="running", endpoint=endpoint, started_utc=model_pool.utc())
                    self.state["dispatches"].append({"block_id": row["block"]["id"], "endpoint": endpoint,
                        "utc": model_pool.utc(), "predicted_call_seconds": stats.get("ewma_seconds", 60),
                        "policy": "unused idle replica first, then latency EWMA; one active block per process"})
                    calls = row["block"].get("model_call_cap", self.plan["calls_per_block"])
                    self.state["allocations"]["model_calls"] += calls
                    self.state["allocations"]["output_tokens"] += calls * self.plan["max_output_tokens"]
                    self.state["status"] = "running"
                    self.save()  # allocation and identity commit before any model request
                    active.add(key)
                    futures[executor.submit(self.run_block, row["block"], endpoint)] = (row, endpoint)
                time.sleep(2)
        self.state["status"] = "deadline_expired"
        self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    with acquire(args.directory / "supervisor.lock", timeout=0):
        study = Study(args.directory, args.gpu_python)
        if args.prepare:
            print(json.dumps({"status": "prepared", "blocks": 16, "arms": 64, "max_calls": 160}))
        else:
            study.run()


if __name__ == "__main__":
    main()
