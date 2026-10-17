"""Paired indexed repair with deterministic source-grounded structural facts."""
import argparse
import os
import subprocess
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent import line_protocol, model_pool, operational, repair, span_repair, grounded_budget as grounded_repair  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.run_repair_study import Study, source_files  # noqa: E402
from scripts.run_semantic_study import SemanticStudy  # noqa: E402
from scripts.summarize_semantic import INFRA  # noqa: E402

REGRESSION_SEEDS = [20261221, 20261222]
FRESH_SEED = 20261213
FRESH_TASKS = ["column_sum", "l2_normalize", "affine_silu", "scale_transpose",
               "transpose", "row_mean", "rmsnorm", "silu_mul"]


def task_specs():
    return {**json.loads((ROOT / "data/tasks.json").read_text()),
            **json.loads((ROOT / "data/repair-transfer-tasks.json").read_text())}


def make_plan():
    bank = json.loads((ROOT / "data/grounded-failure-bank.json").read_text())
    assert len(bank["cases"]) == 3
    parent_inputs = {}
    for case in bank["cases"]:
        for suffix, field in ((".py", "source_path"), (".json", "attempt_path")):
            path = Path(case[field])
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            if suffix == ".py":
                assert expected == case["source_sha256"]
            parent_inputs[case["id"] + suffix] = {"path": str(path), "sha256": expected}
    blocks = [{"id": f"reg-{c['id']}-s{s}", "task_id": c["task_id"], "seed": s,
               "stratum": "regression", "parent_id": c["id"], "model_call_cap": 4}
              for c in bank["cases"] for s in REGRESSION_SEEDS]
    random.Random(20261220).shuffle(blocks)
    sources = source_files()
    for name in ["docs/grounded-budget-study-plan.md", "data/grounded-failure-bank.json",
                 "data/repair-transfer-tasks.json", "data/runtime-triton-api.json"]:
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {"schema_version": 1, "kind": "budget_aware_source_grounded_repair", "blocks": blocks,
            "arms": list(grounded_repair.ARMS), "scheduled_arms": 12, "calls_per_block": 4,
            "max_model_calls": 24, "allocated_output_tokens": 49152, "context_limit": 8192,
            "max_output_tokens": 2048, "temperature": .2, "evaluation_gpu": 4,
            "physical_device_proof_required": True, "source_sha256": sources,
            "parent_inputs": parent_inputs, "regression_seeds": REGRESSION_SEEDS,
            "fresh_seed": None, "fresh_tasks": [],
            "strata": {"regression": {"blocks": 6, "families": 3}},
            "scope": "Separate post-v1 controller correction cohort: all3 prior sources x2 NEW seeds; no replacement or pooling with v1; no fresh-generation claim"}


class GroundedBudgetStudy(Study):
    def __init__(self, directory, gpu_python, *, plan=None):
        super().__init__(directory, gpu_python, plan=make_plan() if plan is None else plan)
        for name, identity in self.plan["parent_inputs"].items():
            target = self.directory / "inputs" / name
            target.parent.mkdir(exist_ok=True)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != identity["sha256"]:
                raise ValueError("frozen parent input changed")
            shutil.copyfile(identity["path"], target)


    evaluate = SemanticStudy.evaluate

    def verify_current_sources(self):
        for name, expected in self.plan["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("executed canonical source differs from frozen plan: " + name)

    def run_block(self, block, endpoint):
        self.verify_current_sources()
        folder = self.directory / "blocks" / block["id"]
        folder.mkdir(parents=True, exist_ok=True)
        client = model_pool.Client(endpoint, folder / "requests", time.monotonic() + 1800)
        capability = json.loads((ROOT / "data/runtime-triton-api.json").read_text())
        contract = task_specs()[block["task_id"]]["contract"]
        result = {"block": block, "endpoint": endpoint, "started_utc": model_pool.utc(), "arms": {}}
        seen, stopped = {}, None

        def evaluate_source(row, code):
            nonlocal stopped
            source_sha = admit(code)
            row.update(code=code, source_sha256=source_sha)
            source = folder / "candidates" / f"{source_sha}.py"
            source.parent.mkdir(exist_ok=True)
            source.write_text(code)
            row["api_audit"] = operational.api_audit(code, capability)
            if row["api_audit"]["status"] == "api_error":
                row["evaluation"] = {**row["api_audit"], "source_sha256": source_sha}
            elif source_sha in seen:
                row.update(duplicate_of=seen[source_sha][0], evaluation=seen[source_sha][1])
            else:
                row["evaluation"] = self.evaluate(block["task_id"], source)
                seen[source_sha] = (row["request_id"], row["evaluation"])
            if row["evaluation"]["status"] in INFRA:
                stopped = row["evaluation"]["status"]

        def call(call_id, history, arm, seed):
            nonlocal stopped
            mode, parent = grounded_repair.assignment(arm, history)
            patching = mode in {"patch", "span_patch"}
            row = {"request_id": call_id, "mode": mode,
                   "parent_sha256": parent.get("source_sha256") if parent else None}
            if stopped:
                row["evaluation"] = {"status": "not_run_infrastructure", "message": stopped}
            else:
                try:
                    parts, settings, budget = grounded_repair.choose(endpoint,
                        folder / "context-checks" / (call_id + ".json"), block["task_id"], contract,
                        history, capability, mode, arm)
                    row["context_budget"] = budget
                    if arm == "grounded" and parent and budget["selected_view"] != "baseline":
                        row["source_facts"] = grounded_repair.packet(parent["code"], budget["selected_view"])
                    if mode != "full_source":
                        row["diagnosis"] = operational.diagnose(parent["evaluation"])
                    assert client.calls < block["model_call_cap"], "call allocation exceeded"
                    row["reply"] = client.propose(parts, seed, call_id, **settings)
                    proposed = (span_repair.apply(row["reply"], parent["code"]) if mode == "span_patch" else
                                operational.apply(row["reply"], parent["code"]) if patching else
                                line_protocol.proposal(row["reply"]))
                    row.update(draft=proposed["draft"], plan=proposed["plan"], proposed_code=proposed["code"])
                    if patching:
                        row["transaction"] = proposed["transaction"]
                    evaluate_source(row, proposed["code"])
                except grounded_repair.ContextBudgetError as error:
                    row["context_budget"] = error.receipt
                    row["evaluation"] = {"status": "context_budget_error", "message": str(error)}
                except (model_pool.OutcomeUnknown, model_pool.NotDispatched) as error:
                    rp = folder / "context-checks" / (call_id + ".json")
                    if rp.exists():
                        row["context_budget"] = json.loads(rp.read_text())
                    if isinstance(error, model_pool.NotDispatched) and "common context" in str(error):
                        row["evaluation"] = {"status": "context_budget_error", "message": str(error)}
                    else:
                        stopped = str(error)
                        row["evaluation"] = {"status": "model_infrastructure_error", "message": stopped}
                except (ValueError, SyntaxError, TypeError, KeyError) as error:
                    row["evaluation"] = {"status": "patch_error" if patching else "protocol_error", "message": str(error)}
            atomic_json(folder / "attempts" / f"{call_id}.json", row)
            return row

        if block["stratum"] == "regression":
            bank = json.loads((ROOT / "data/grounded-failure-bank.json").read_text())
            parent = next(c for c in bank["cases"] if c["id"] == block["parent_id"])
            atomic_json(folder / "parent.json", parent)
            code = (self.directory / "inputs" / (parent["id"] + ".py")).read_text()
            origin = json.loads((self.directory / "inputs" / (parent["id"] + ".json")).read_text())
            assert hashlib.sha256(code.encode()).hexdigest() == parent["source_sha256"]
            start = {"request_id": "imported", "mode": "imported", "parent_sha256": None, "imported": True,
                     "origin": parent["id"], "reply": origin["reply"]}
            evaluate_source(start, code)
            atomic_json(folder / "attempts/imported.json", start)
        else:
            start = call("initial", [], "indexed", block["seed"])
        result["initial"] = start
        histories = {arm: [start] for arm in grounded_repair.ARMS}
        for round_index in range(2):
            order = list(grounded_repair.ARMS)
            random.Random(f"{block['seed']}:{round_index}").shuffle(order)
            for arm in order:
                history = histories[arm]
                history.append(call(f"{arm}-r{round_index+1}", history, arm, block["seed"] + round_index + 1))
        for arm, history in histories.items():
            best = repair.select(history)
            result["arms"][arm] = {"attempts": history, "selected_sha256": best["source_sha256"] if best else None,
                                   "model_opportunities": [f"{arm}-r1", f"{arm}-r2"],
                                   "correct": False, "promoted": False, "final": None}
        result.update(model_calls=client.calls, infrastructure_stop=stopped, generation_committed_utc=model_pool.utc())
        atomic_json(folder / "generation.json", result)
        finals = {}
        for arm, value in result["arms"].items():
            selected = value["selected_sha256"]
            if selected:
                if selected in finals:
                    origin, evaluation = finals[selected]
                    value["final_reused_from"] = origin
                else:
                    evaluation = self.evaluate(block["task_id"], folder / "candidates" / f"{selected}.py", final=True)
                    finals[selected] = (arm, evaluation)
                verdict = decision(evaluation)
                value.update(final=evaluation, final_decision=verdict, correct=verdict["correct"], promoted=verdict["promoted"])
            atomic_json(folder / "final" / f"{arm}.json", value)
        result["finished_utc"] = model_pool.utc()
        atomic_json(folder / "result.json", result)
        receipts = [json.loads(p.read_text()) for p in sorted((folder / "requests").glob("*.json"))
                    if not p.name.endswith((".request.json", ".response.json"))]
        elapsed = [r["elapsed_seconds"] for r in receipts if r["status"] == "completed"]
        self.verify_current_sources()
        return result, sum(elapsed) / len(elapsed) if elapsed else 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kernel-agent-storage/runtimes/gpu/bin/python")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.background:
        if args.directory.exists():
            raise ValueError("background launch requires a NEW experiment directory; no replay")
        endpoints = [{"endpoint": e, "load": model_pool.validate(e)} for e in model_pool.discover()]
        if len(endpoints) < 3:
            raise ValueError("three authorized Qwen services must be discovered before this study")
        study = GroundedBudgetStudy(args.directory, args.gpu_python, plan=make_plan())
        launcher = Path("/data/home/chenyizhou/kernel-agent-storage/bin/run.py")
        command = [sys.executable, str(launcher), "run_grounded_budget_study.py", "--directory", str(study.directory),
                   "--gpu-python", args.gpu_python]
        with (study.directory / "supervisor.log").open("wb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        receipt = {"started_utc": model_pool.utc(), "pid": child.pid, "command": command,
                   "directory": str(study.directory), "source_root": str(ROOT), "plan_sha256": study.state["plan_sha256"],
                   "scheduled_outcomes": 12, "physical_call_cap": 24, "services": endpoints}
        atomic_json(study.directory / "launch.json", receipt)
        print(json.dumps(receipt))
        return
    with acquire(args.directory / "supervisor.lock", timeout=0):
        study = GroundedBudgetStudy(args.directory, args.gpu_python, plan=make_plan())
        if args.prepare:
            print(json.dumps({"status": "prepared", "plan_sha256": study.state["plan_sha256"]}))
            return
        receipt = {"started_utc": model_pool.utc()}
        try:
            study.run()
            receipt.update(status=study.state["status"], exit_code=0)
        except BaseException as error:
            receipt.update(status="supervisor_error", exit_code=1, error=str(error))
            raise
        finally:
            atomic_json(study.directory / "supervisor-exit.json", {**receipt, "finished_utc": model_pool.utc()})


if __name__ == "__main__":
    main()
