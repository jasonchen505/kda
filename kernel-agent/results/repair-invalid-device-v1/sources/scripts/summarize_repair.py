"""Verify shared starts, unchanged selections and factorial study outcomes offline."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.protocol import decision, sha  # noqa: E402
from kernel_agent.line_protocol import proposal  # noqa: E402
from kernel_agent.repair import GENERATION, REPAIR, arm_id, select  # noqa: E402
from scripts.summarize_campaign import bootstrap_interval, digest, require, same_decision  # noqa: E402


def analyze(folder):
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    state = json.loads((folder / "checkpoint.json").read_text())
    require(state["plan_sha256"] == digest(plan) and state["state_sha256"] ==
            digest({k: v for k, v in state.items() if k != "state_sha256"}), "repair ledger changed")
    for name, expected in plan["source_sha256"].items():
        require(hashlib.sha256((folder / "sources" / name).read_bytes()).hexdigest() == expected,
                "repair frozen source changed")
    require(len(state["blocks"]) == 16 and {r["block"]["id"] for r in state["blocks"]} ==
            {b["id"] for b in plan["blocks"]}, "repair block denominator changed")
    rows, initial, receipts, timeline, gpu_seen, blocks = [], [], [], [], set(), []
    rescue_pairs = []
    counters = Counter()

    def gpu_window(evaluation, block_id, phase):
        audit = evaluation.get("resource_audit", {})
        if "before" not in audit or "after" not in audit:
            return
        queue = evaluation.get("cooperative_queue", {})
        require(queue.get("lock_file") == plan["runtime"]["evaluation_lock_file"]
                and queue.get("scope") == "serializes this project's workers only", "missing cooperative GPU lock")
        key = (audit["before"]["observed_utc"], audit["after"]["observed_utc"], queue.get("pid"))
        if key not in gpu_seen:
            gpu_seen.add(key)
            timeline.append({"lane": "GPU4", "block": block_id, "phase": phase,
                             "start_utc": key[0], "end_utc": key[1]})

    for item in state["blocks"]:
        block = item["block"]
        path = folder / "blocks" / block["id"]
        by_request = {}
        for receipt_path in sorted((path / "requests").glob("*.json")):
            value = json.loads(receipt_path.read_text())
            if "status" not in value:
                continue
            receipts.append(value)
            by_request[value["request_id"]] = value
            if value.get("dispatched_utc"):
                body_path = path / "requests" / f"{value['request_id']}.request.json"
                require(hashlib.sha256(body_path.read_bytes()).hexdigest() == value["request_sha256"], "repair request changed")
                body = json.loads(body_path.read_text())
                require(body["max_tokens"] == 2048 and body["temperature"] == .2 and not body.get("tools")
                        and body["stream"] is False and body["chat_template_kwargs"] == {"enable_thinking": False}
                        and body["model"] == value["endpoint"]["model"] and value["input_tokens"] + 2048 <= 8192
                        and not any(value["admission"].values()), "repair request bypassed common protocol")
                require(value["endpoint"] == item["endpoint"], "active repair block migrated between processes")
                expected_seed = block["seed"] + (int(value["request_id"][-1]) if "-r" in value["request_id"] else 0)
                require(body["seed"] == expected_seed, "repair request seed changed")
            if value["status"] == "completed":
                response = json.loads((path / "requests" / f"{value['request_id']}.response.json").read_text())
                require(digest(response) == value["response_sha256"], "repair response changed")
                timeline.append({"lane": f":{value['endpoint']['port']}", "block": block["id"], "phase": "model",
                                 "start_utc": value["dispatched_utc"], "end_utc": value["finished_utc"]})
        result_path = path / "result.json"
        value = None
        if item["status"] == "complete":
            value = json.loads(result_path.read_text())
            require(digest(value) == item["result_sha256"], "repair result changed")
            committed = json.loads((path / "generation.json").read_text())
            require(value["endpoint"] == item["endpoint"] == committed["endpoint"], "repair endpoint changed")
            for generation in GENERATION:
                start = value["initial"][generation]
                initial.append({"block": block["id"], "task_id": block["task_id"], "generation": generation,
                                "correct": decision(start["evaluation"])["correct"], "attempt": start})
                for policy in REPAIR:
                    name = arm_id(generation, policy)
                    arm = value["arms"][name]
                    prior = committed["arms"][name]
                    require(arm["attempts"][0] == start == committed["initial"][generation], "repair arms did not share exact initial evidence")
                    require(arm["attempts"] == prior["attempts"] and arm["selected_sha256"] == prior["selected_sha256"],
                            "repair final data changed search or selection")
                    require(json.loads((path / "final" / f"{name}.json").read_text()) == arm, "repair final artifact changed")
                    chosen = select(arm["attempts"])
                    require(arm["selected_sha256"] == (chosen["source_sha256"] if chosen else None), "repair selection differs from search")
                    for index, attempt in enumerate(arm["attempts"]):
                        require(attempt["parent_sha256"] == (arm["attempts"][index - 1].get("source_sha256") if index else None),
                                "repair parent lineage changed")
                        counters[attempt["evaluation"]["status"]] += 1
                        if "reply" in attempt:
                            receipt = by_request[attempt["request_id"]]
                            require(receipt["status"] == "completed", "reply without a completed receipt")
                            response = json.loads((path / "requests" / f"{attempt['request_id']}.response.json").read_text())
                            require(response["choices"][0]["message"]["content"] == attempt["reply"], "reply differs from model response")
                        if attempt.get("source_sha256"):
                            require(sha(attempt["code"]) == attempt["source_sha256"] ==
                                    sha((path / "candidates" / f"{attempt['source_sha256']}.py").read_text()), "repair candidate changed")
                            require(proposal(attempt["reply"])["code"] == attempt["code"], "repair source was manually changed")
                        gpu_window(attempt["evaluation"], block["id"], "search")
                    verdict = decision(arm["final"] or {})
                    require((arm["correct"], arm["promoted"]) == (verdict["correct"], verdict["promoted"]), "repair final headline conflicts")
                    if arm["final"]:
                        if verdict["correct"]:
                            require(arm["final"]["source_sha256"] == arm["selected_sha256"], "repair final source changed")
                        require(same_decision(verdict, arm["final_decision"]), "repair final decision changed")
                        gpu_window(arm["final"], block["id"], "final")
                    rows.append({"block": block["id"], "task_id": block["task_id"], "seed": block["seed"],
                                 "generation": generation, "repair": policy, "status": "complete",
                                 "initial_correct": decision(start["evaluation"])["correct"],
                                 "correct": arm["correct"], "promoted": arm["promoted"],
                                 "selected_sha256": arm["selected_sha256"]})
                if not decision(start["evaluation"])["correct"]:
                    arms = [value["arms"][arm_id(generation, p)] for p in REPAIR]
                    observed = all(by_request.get(a["request_id"], {}).get("status") == "completed"
                                   for arm in arms for a in arm["attempts"][1:])
                    rescue_pairs.append({"block": block["id"], "task_id": block["task_id"], "generation": generation,
                                         "fully_observed": observed, "baseline": arms[0]["correct"],
                                         "counterexample": arms[1]["correct"]})
        else:
            for generation in GENERATION:
                for policy in REPAIR:
                    rows.append({"block": block["id"], "task_id": block["task_id"], "seed": block["seed"],
                                 "generation": generation, "repair": policy, "status": item["status"],
                                 "initial_correct": False, "correct": False, "promoted": False})
        blocks.append({"block": block, "status": item["status"], "endpoint": item.get("endpoint"), "result": value})
    resolved = sum(r["status"] in {"complete", "failed", "interrupted_unknown"} for r in rows)
    cells = []
    for generation in GENERATION:
        for policy in REPAIR:
            group = [r for r in rows if (r["generation"], r["repair"]) == (generation, policy)]
            cells.append({"generation": generation, "repair": policy, "scheduled": 16,
                          "resolved": sum(r["status"] in {"complete", "failed", "interrupted_unknown"} for r in group),
                          "correct": sum(r["correct"] for r in group), "promoted": sum(r["promoted"] for r in group),
                          "initial_failed": sum(not r["initial_correct"] for r in group if r["status"] == "complete"),
                          "rescued": sum(not r["initial_correct"] and r["correct"] for r in group if r["status"] == "complete")})
    generation_counts = [{"generation": g, "scheduled": 16,
                          "observed": sum(i["generation"] == g and "reply" in i["attempt"] for i in initial),
                          "correct": sum(i["generation"] == g and i["correct"] for i in initial)} for g in GENERATION]
    contrasts = {}
    if resolved == 64:
        for kind in ["generation", "repair"]:
            differences = []
            for task in plan["tasks"]:
                group = [r for r in rows if r["task_id"] == task]
                control, treatment = (GENERATION if kind == "generation" else REPAIR)
                differences.append((sum(r["correct"] for r in group if r[kind] == treatment)
                                    - sum(r["correct"] for r in group if r[kind] == control)) / 4)
            contrasts[kind] = {"mean_difference": sum(differences) / len(differences),
                               "exploratory_interval": bootstrap_interval(differences)}
        diffs = [(sum(i["correct"] for i in initial if i["task_id"] == t and i["generation"] == "contract")
                  - sum(i["correct"] for i in initial if i["task_id"] == t and i["generation"] == "baseline")) / 2
                 for t in plan["tasks"]]
        contrasts["first_generation"] = {"mean_difference": sum(diffs) / len(diffs),
                                          "exploratory_interval": bootstrap_interval(diffs)}
    events = sorted([(t["start_utc"], 1, t["lane"]) for t in timeline] +
                    [(t["end_utc"], -1, t["lane"]) for t in timeline], key=lambda x: (x[0], x[1]))
    active, peak_model, gpu, peak_gpu = Counter(), 0, 0, 0
    for _, delta, lane in events:
        if lane == "GPU4":
            gpu += delta
            peak_gpu = max(peak_gpu, gpu)
        else:
            active[lane] += delta
            peak_model = max(peak_model, sum(v > 0 for v in active.values()))
    require(peak_gpu <= 1, "repair GPU windows overlapped")
    return {"status": "complete" if resolved == 64 else "incomplete", "plan_sha256": digest(plan), "scheduled": 64,
            "resolved": resolved, "initial": generation_counts, "cells": cells, "contrasts": contrasts,
            "blocks": blocks, "rows": rows, "services": list(state["services"].values()), "timeline": timeline,
            "request_statuses": dict(Counter(r["status"] for r in receipts)),
            "dispatched_requests": sum(bool(r.get("dispatched_utc")) for r in receipts),
            "reported_output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in receipts),
            "usage_missing_dispatched": sum(bool(r.get("dispatched_utc")) and r.get("usage") is None for r in receipts),
            "repair_rescue_pairs": rescue_pairs,
            "peak_model_ports": peak_model, "independent_gpu_windows": len(gpu_seen), "peak_gpu_windows": peak_gpu,
            "scope": plan["scope"], "shared_initial_binding_verified": True, "final_selection_binding_verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
