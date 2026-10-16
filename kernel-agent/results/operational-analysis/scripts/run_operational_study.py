"""Equal-call repair comparison with evidence-derived obligations and transactional patches."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent import line_protocol, model_pool, operational, repair, semantic  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.run_repair_study import Study, source_files  # noqa: E402
from scripts.run_semantic_study import SemanticStudy, make_plan as parent_plan  # noqa: E402
from scripts.summarize_semantic import INFRA  # noqa: E402

REGRESSION_SEEDS = [20261121, 20261122]
TRANSFER_SEEDS = [20261123, 20261124]


def task_specs():
    return {**json.loads((ROOT / "data/tasks.json").read_text()),
            **json.loads((ROOT / "data/repair-transfer-tasks.json").read_text())}


def make_plan():
    parent_plan()  # Verify the original complete source-bank selection, not repair outcomes.
    bank = json.loads((ROOT / "data/semantic-failure-bank.json").read_text())
    transfer = json.loads((ROOT / "data/repair-transfer-tasks.json").read_text())
    blocks = [{"id": f"reg-{c['id']}-s{s}", "task_id": c["task_id"], "seed": s, "stratum": "regression",
               "parent_id": c["id"], "model_call_cap": 4} for c in bank["cases"] for s in REGRESSION_SEEDS]
    blocks += [{"id": f"new-{t}-s{s}", "task_id": t, "seed": s, "stratum": "transfer", "model_call_cap": 5}
               for t in transfer for s in TRANSFER_SEEDS]
    random.Random(20261120).shuffle(blocks)
    sources = source_files()
    for name in ["docs/operational-study-plan.md", "data/repair-transfer-tasks.json", "data/runtime-triton-api.json",
                 "data/semantic-failure-bank.json", "results/repair-study/summary.json"] + [c["source_path"] for c in bank["cases"]]:
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {"schema_version": 1, "kind": "operational_patch_repair", "blocks": blocks,
            "arms": list(operational.ARMS), "scheduled_arms": 44, "calls_per_block": 4, "max_model_calls": 96,
            "allocated_output_tokens": 196608, "context_limit": 8192, "max_output_tokens": 2048, "temperature": .2,
            "evaluation_gpu": 4, "physical_device_proof_required": True,
            "source_sha256": sources, "parent_plan_sha256": bank["parent_plan_sha256"],
            "regression_seeds": REGRESSION_SEEDS, "transfer_seeds": TRANSFER_SEEDS,
            "strata": {"regression": {"blocks": 14, "families": 5}, "transfer": {"blocks": 8, "families": 4}},
            "scope": "7 known failed parents x2 repair seeds and 4 new task families x2 generation seeds; distinct strata, exploratory"}


class OperationalStudy(Study):
    evaluate = SemanticStudy.evaluate

    def run_block(self, block, endpoint):
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
            parent = operational.anchor(history) if arm == "operational_patch" else history[-1] if history else None
            patching = arm == "operational_patch" and parent is not None
            row = {"request_id": call_id, "mode": "patch" if patching else "full_source",
                   "parent_sha256": parent.get("source_sha256") if parent else None}
            if stopped:
                row["evaluation"] = {"status": "not_run_infrastructure", "message": stopped}
            else:
                try:
                    if patching:
                        parts = operational.sections(block["task_id"], contract, history, capability)
                        row["diagnosis"] = operational.diagnose(parent["evaluation"])
                        settings = {"schema": operational.PATCH_SCHEMA, "schema_name": "source_patch",
                                    "role_prefix": operational.ROLE}
                    else:
                        parts = semantic.baseline_sections(block["task_id"], contract, history)
                        settings = {}
                    assert client.calls < block["model_call_cap"], "call allocation exceeded"
                    row["reply"] = client.propose(parts, seed, call_id, **settings)
                    proposed = operational.apply(row["reply"], parent["code"]) if patching else line_protocol.proposal(row["reply"])
                    row.update(draft=proposed["draft"], plan=proposed["plan"], proposed_code=proposed["code"])
                    if patching:
                        row["transaction"] = proposed["transaction"]
                    evaluate_source(row, proposed["code"])
                except (model_pool.OutcomeUnknown, model_pool.NotDispatched) as error:
                    stopped = str(error)
                    row["evaluation"] = {"status": "model_infrastructure_error", "message": stopped}
                except (ValueError, SyntaxError, TypeError, KeyError) as error:
                    row["evaluation"] = {"status": "patch_error" if patching else "protocol_error", "message": str(error)}
            atomic_json(folder / "attempts" / f"{call_id}.json", row)
            return row

        if block["stratum"] == "regression":
            bank = json.loads((ROOT / "data/semantic-failure-bank.json").read_text())
            parent = next(c for c in bank["cases"] if c["id"] == block["parent_id"])
            atomic_json(folder / "parent.json", parent)
            code = (ROOT / parent["source_path"]).read_text()
            assert hashlib.sha256(code.encode()).hexdigest() == parent["source_sha256"]
            start = {"request_id": "imported", "mode": "imported", "parent_sha256": None, "imported": True,
                     "origin": parent["id"], "reply": parent["initial"]["reply"]}
            evaluate_source(start, code)
            atomic_json(folder / "attempts/imported.json", start)
        else:
            start = call("initial", [], "rewrite", block["seed"])
        result["initial"] = start
        histories = {arm: [start] for arm in operational.ARMS}
        for round_index in range(2):
            order = list(operational.ARMS)
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
        return result, sum(elapsed) / len(elapsed) if elapsed else 60


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    args = parser.parse_args()
    with acquire(args.directory / "supervisor.lock", timeout=0):
        OperationalStudy(args.directory, args.gpu_python, plan=make_plan()).run()
