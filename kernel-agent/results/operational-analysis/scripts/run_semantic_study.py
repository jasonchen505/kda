"""Paired regression bank and fresh-start evaluation of grounded semantic repair."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent import line_protocol, model_pool, repair, semantic  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.run_repair_study import INFRA, TASKS, Study, source_files  # noqa: E402

REGRESSION_SEEDS = [20261111, 20261112]
FRESH_SEED = 20261113


def make_plan():
    bank = json.loads((ROOT / "data/semantic-failure-bank.json").read_text())
    parent_path = ROOT / "results/repair-study/summary.json"
    parent_summary = json.loads(parent_path.read_text())
    expected = {b["block"]["id"] + "-" + g for b in parent_summary["blocks"]
                for g, a in b["result"]["initial"].items() if a.get("source_sha256") and
                a["evaluation"]["status"] in {"numeric_error", "compile_error", "contract_error", "runtime_error", "policy_error"}}
    assert {c["id"] for c in bank["cases"]} == expected and len(expected) == 7
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == bank["parent_summary_sha256"]
    blocks = [{"id": f"reg-{case['id']}-s{seed}", "task_id": case["task_id"], "seed": seed,
               "stratum": "regression", "parent_id": case["id"], "model_call_cap": 5}
              for case in bank["cases"] for seed in REGRESSION_SEEDS]
    blocks.extend({"id": f"fresh-{t}-s{FRESH_SEED}", "task_id": t, "seed": FRESH_SEED,
                   "stratum": "fresh", "model_call_cap": 6} for t in TASKS)
    random.Random(20261110).shuffle(blocks)
    sources = source_files()
    for name in ["docs/semantic-study-plan.md", "data/semantic-failure-bank.json", "results/repair-study/summary.json"] + [c["source_path"] for c in bank["cases"]]:
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {"schema_version": 1, "kind": "grounded_semantic_repair", "blocks": blocks,
            "arms": list(semantic.ARMS), "scheduled_arms": 66, "calls_per_block": 5,
            "max_model_calls": 118, "allocated_output_tokens": 118 * 2048,
            "context_limit": 8192, "max_output_tokens": 2048, "temperature": .2,
            "evaluation_gpu": 4, "physical_device_proof_required": True,
            "source_sha256": sources, "parent_plan_sha256": bank["parent_plan_sha256"],
            "regression_seeds": REGRESSION_SEEDS, "fresh_seed": FRESH_SEED,
            "model_phase_deadline_seconds": 1800,
            "scope": "Known failed-source regression (7 parents, 5 families, 2 seeds) and separately reported fresh starts (8 families, 1 seed); no held-out benchmark claim"}


class SemanticStudy(Study):
    def evaluate(self, task, source, final=False):
        import subprocess
        command = [self.gpu_python, str(ROOT / "scripts/evaluate_serial.py"), "--gpu-index", "4",
                   "--lock-file", str(self.directory / "gpu4.lock"), "--require-device-proof", task, str(source)]
        command.extend(["--final"] if final else ["--semantic-probes"])
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=400)
            return json.loads(result.stdout) if result.returncode == 0 else {
                "status": "evaluator_error", "message": (result.stdout + result.stderr)[-2000:]}
        except (subprocess.SubprocessError, ValueError) as error:
            return {"status": "evaluator_error", "message": str(error)[-2000:]}

    def run_block(self, block, endpoint):
        folder = self.directory / "blocks" / block["id"]
        folder.mkdir(parents=True, exist_ok=True)
        client = model_pool.Client(endpoint, folder / "requests", time.monotonic() + 1800)
        contract = json.loads((ROOT / "data/tasks.json").read_text())[block["task_id"]]["contract"]
        result = {"block": block, "endpoint": endpoint, "started_utc": model_pool.utc(), "arms": {}}
        seen, stopped = {}, None

        def evaluate_source(row, code):
            nonlocal stopped
            source_sha = admit(code)
            row.update(code=code, source_sha256=source_sha)
            path = folder / "candidates" / f"{source_sha}.py"
            path.parent.mkdir(exist_ok=True)
            path.write_text(code)
            if source_sha in seen:
                row.update(duplicate_of=seen[source_sha][0], evaluation=seen[source_sha][1])
            else:
                row["evaluation"] = self.evaluate(block["task_id"], path)
                seen[source_sha] = (row["request_id"], row["evaluation"])
            if row["evaluation"]["status"] in INFRA | {"evaluator_error"}:
                stopped = row["evaluation"]["status"]

        def code_call(call_id, parts, parent, seed):
            nonlocal stopped
            row = {"request_id": call_id, "parent_sha256": parent.get("source_sha256") if parent else None}
            if stopped:
                row["evaluation"] = {"status": "not_run_infrastructure", "message": stopped}
            else:
                try:
                    assert client.calls < block["model_call_cap"], "block dispatch cap exceeded"
                    row["reply"] = client.propose(parts, seed, call_id)
                    proposed = line_protocol.proposal(row["reply"])
                    row.update(draft=proposed["draft"], plan=proposed["plan"], proposed_code=proposed["code"])
                    evaluate_source(row, proposed["code"])
                except (model_pool.OutcomeUnknown, model_pool.NotDispatched) as error:
                    stopped = str(error)
                    row["evaluation"] = {"status": "model_infrastructure_error", "message": stopped}
                except (ValueError, SyntaxError, TypeError) as error:
                    row["evaluation"] = {"status": "protocol_error", "message": str(error)}
            atomic_json(folder / "attempts" / f"{call_id}.json", row)
            return row

        if block["stratum"] == "regression":
            bank = json.loads((ROOT / "data/semantic-failure-bank.json").read_text())
            parent = next(c for c in bank["cases"] if c["id"] == block["parent_id"])
            atomic_json(folder / "parent.json", parent)
            code = (ROOT / parent["source_path"]).read_text()
            assert hashlib.sha256(code.encode()).hexdigest() == parent["source_sha256"]
            start = {"request_id": "imported", "parent_sha256": None, "origin": parent["id"],
                     "reply": parent["initial"]["reply"], "draft": parent["initial"].get("draft"),
                     "plan": parent["initial"].get("plan"), "imported": True}
            evaluate_source(start, code)
            atomic_json(folder / "attempts/imported.json", start)
        else:
            start = code_call("initial", semantic.baseline_sections(block["task_id"], contract), None, block["seed"])
        result["initial"] = start
        histories = {arm: [start] for arm in semantic.ARMS}
        review = {"request_id": "review", "review": {}, "validation": {
            "status": "not_run", "reference": {"valid": False}, "candidate": {"valid": False}}}
        first_order = ["rewrite", "review"]
        random.Random(block["seed"]).shuffle(first_order)
        for action in first_order:
            if action == "rewrite":
                history = histories["rewrite"]
                history.append(code_call("rewrite-r1", semantic.baseline_sections(block["task_id"], contract, history),
                                         start, block["seed"] + 1))
            elif stopped:
                review["error"] = "not_run_infrastructure: " + stopped
            else:
                try:
                    assert client.calls < block["model_call_cap"]
                    reply = client.propose(semantic.reviewer_sections(contract, start), block["seed"] + 1,
                                           "review", schema=semantic.REVIEW_SCHEMA, schema_name="semantic_review",
                                           role_prefix="Independent mathematical reviewer. Return JSON, not kernel source.")
                    review["reply"] = reply
                    review["review"] = json.loads(reply)
                    review["validation"] = semantic.validate_review(review["review"], start["evaluation"].get("semantic_probes", []))
                except (model_pool.OutcomeUnknown, model_pool.NotDispatched) as error:
                    stopped = str(error)
                    review["error"] = stopped
                except (ValueError, TypeError) as error:
                    review["error"] = str(error)
                    review["review"] = {}
                    review["validation"] = semantic.validate_review({}, [])
        result["review"] = review
        atomic_json(folder / "review.json", review)
        second_order = list(semantic.ARMS)
        random.Random(block["seed"] + 29).shuffle(second_order)
        for arm in second_order:
            history = histories[arm]
            if arm == "rewrite":
                parts = semantic.baseline_sections(block["task_id"], contract, history)
            else:
                parts = semantic.writer_sections(block["task_id"], contract, start, review["review"],
                                                  review["validation"], grounded=arm == "grounded")
            history.append(code_call(f"{arm}-r2", parts, history[-1], block["seed"] + 2))
            best = repair.select(history)
            result["arms"][arm] = {"attempts": history, "selected_sha256": best["source_sha256"] if best else None,
                                   "model_opportunities": ["rewrite-r1", "rewrite-r2"] if arm == "rewrite" else ["review", f"{arm}-r2"],
                                   "correct": False, "promoted": False, "final": None}
        result.update(model_calls=client.calls, infrastructure_stop=stopped,
                      generation_committed_utc=model_pool.utc())
        atomic_json(folder / "generation.json", result)
        final_cache = {}
        for arm, value in result["arms"].items():
            selected = value["selected_sha256"]
            if selected:
                if selected in final_cache:
                    origin, evaluation = final_cache[selected]
                    value["final_reused_from"] = origin
                else:
                    evaluation = self.evaluate(block["task_id"], folder / "candidates" / f"{selected}.py", final=True)
                    final_cache[selected] = (arm, evaluation)
                verdict = decision(evaluation)
                value.update(final=evaluation, final_decision=verdict,
                             correct=verdict["correct"], promoted=verdict["promoted"])
            atomic_json(folder / "final" / f"{arm}.json", value)
        result["finished_utc"] = model_pool.utc()
        atomic_json(folder / "result.json", result)
        durations = [json.loads(p.read_text())["elapsed_seconds"] for p in (folder / "requests").glob("*.json")
                     if p.name not in {"."} and json.loads(p.read_text()).get("status") == "completed"]
        return result, sum(durations) / len(durations) if durations else 60


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    args = parser.parse_args()
    with acquire(args.directory / "supervisor.lock", timeout=0):
        SemanticStudy(args.directory, args.gpu_python, plan=make_plan()).run()
