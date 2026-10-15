"""Read-only analysis of endpoint-blocked generation and durable final verdicts."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.protocol import decision, sha  # noqa: E402
from scripts.summarize_campaign import (  # noqa: E402
    audit_state, bootstrap_interval, check_resource_evidence, digest, require, same_decision,
)


def analyze(folder):
    folder = Path(folder)
    pool = json.loads((folder / "plan.json").read_text())
    rows, admissions, intervals, worker_rows, partial_proposals = [], [], [], [], 0
    timeline, gpu_intervals, gpu_seen = [], [], set()

    def gpu_window(value, job, phase, lock_path):
        audit, queue = value.get("resource_audit", {}), value.get("cooperative_queue", {})
        if decision(value)["correct"]:
            require(queue.get("lock_file") == lock_path, "pool success bypassed cooperative GPU serialization")
        start, end = audit.get("before", {}).get("observed_utc"), audit.get("after", {}).get("observed_utc")
        key = (queue.get("pid"), start, end)
        if not start or not end or key in gpu_seen:
            return
        require(queue.get("lock_file") == lock_path and start <= end, "pool GPU interval identity changed")
        gpu_seen.add(key)
        gpu_intervals.extend([(start, 1), (end, -1)])
        timeline.append({"lane": "GPU4", "job_id": job["id"], "phase": phase,
                         "start_utc": start, "end_utc": end, "status": value.get("status"),
                         "wait_seconds": queue["wait_seconds"]})
    for worker in pool["workers"]:
        port = worker["port"]
        path = folder / "workers" / f"p{port}"
        state = json.loads((path / "generation.json").read_text())
        plan = audit_state(state, folder / "sources")
        require(state["plan_sha256"] == worker["plan_sha256"] and plan["source_sha256"] == pool["source_sha256"],
                "pool worker provenance changed")
        require(plan["generation"]["pool_protocol"] and plan["generation"]["evaluation_mode"] == "search_only"
                and plan["generation"]["max_tokens_per_request"] == 2048
                and plan["generation"]["common_context_cap"] == 8192, "pool protocol changed")
        worker_rows.append({"port": port, "service": worker["service"],
                            "generation_statuses": dict(Counter(r["status"] for r in state["jobs"]))})
        for item in state["jobs"]:
            job, run = item["job"], item.get("result")
            require(job["service_port"] == port and plan["runtime"]["service"] == worker["service"],
                    "pool job routed to another service")
            row = {"job": job, "generation_status": item["status"], "terminal": run.get("terminal") if run else None,
                   "error": item.get("error"), "proposals": len(run["attempts"]) if run else None,
                   "model_replies": sum("reply" in a for a in run["attempts"]) if run else None,
                   "admitted_candidates": sum(bool(a.get("source_sha256")) for a in run["attempts"]) if run else None,
                   "unique_sources": len({a["source_sha256"] for a in run["attempts"] if a.get("source_sha256")}) if run else None,
                   "output_tokens": run.get("usage", {}).get("output_tokens") if run else None,
                   "final_status": "not_queued", "resolved": item["status"] in {"failed", "interrupted_unknown"},
                   "correct": False if item["status"] in {"failed", "interrupted_unknown"} else None,
                   "promoted": False if item["status"] in {"failed", "interrupted_unknown"} else None,
                   "selected_sha256": run.get("selected_sha256") if run else None}
            row["first_source_sha256"] = run["attempts"][0].get("source_sha256") if run and run["attempts"] else None
            if not run and item["status"] in {"failed", "interrupted_unknown"}:
                run_id = f"campaign-{state['plan_sha256'][:12]}-{job['id']}"
                trace = path / "runs" / run_id / "trace.json"
                attempts = json.loads(trace.read_text()) if trace.exists() else []
                partial_proposals += len(attempts)
                for attempt in attempts:
                    if attempt.get("source_sha256"):
                        from kernel_agent.line_protocol import proposal
                        require(sha(proposal(attempt["reply"])["code"]) == attempt["source_sha256"]
                                == sha((trace.parent / f"candidate-{attempt['iteration']}.py").read_text()),
                                "pool uncommitted source changed")
            if run:
                require(run["measurement_state"] == "final_pending" and run["final_evaluation"] is None
                        and not run["candidate_final_evaluations"], "pool generation accessed final data")
                for attempt in run["attempts"]:
                    if attempt.get("source_sha256"):
                        source = path / "runs" / run["run_id"] / f"candidate-{attempt['iteration']}.py"
                        require(sha(source.read_text()) == attempt["source_sha256"], "pool candidate file changed")
                    if not attempt.get("duplicate_of"):
                        gpu_window(attempt["evaluation"], job, "gpu_search", plan["runtime"]["gpu_lock_path"])
                queue_path = path / "final" / job["id"] / "checkpoint.json"
                if queue_path.exists():
                    queue = json.loads(queue_path.read_text())
                    require(queue["state_sha256"] == digest({k: v for k, v in queue.items() if k != "state_sha256"})
                            and queue["plan_sha256"] == digest(queue["plan"]), "pool final queue integrity mismatch")
                    require(queue["plan"]["generation_result"] == run
                            and queue["plan"]["generation_result_sha256"] == item["result_sha256"]
                            and queue["plan"]["generation_plan_sha256"] == state["plan_sha256"]
                            and queue["selected_sha256"] == run["selected_sha256"] and queue["selection_committed"],
                            "pool final selection changed")
                    original = {}
                    for attempt in run["attempts"]:
                        if attempt.get("decision", {}).get("correct"):
                            original.setdefault(attempt["source_sha256"], attempt["evaluation"])
                    require([r["source_sha256"] for r in queue["search"]] == run["timing_candidates"],
                            "pool imported schedule changed")
                    for measured in queue["search"]:
                        require(measured["status"] == "measured" and measured["evaluation"] == original[measured["source_sha256"]]
                                and len(measured["observations"]) == 1
                                and measured["observations"][0]["status"] == "imported_committed_search",
                                "pool search was changed or repeated")
                        require(sha((queue_path.parent / f"{measured['source_sha256']}.py").read_text()) == measured["source_sha256"],
                                "pool queued source changed")
                    row.update(final_status=queue["status"], resolved=queue["status"] == "complete",
                               correct=queue.get("correct"), promoted=queue.get("promoted"),
                               imported_searches=len(queue["search"]), final=queue["final"])
                    row["outcome_type"] = ("no_eligible_candidate" if queue["status"] == "complete" and
                                           queue["selected_sha256"] is None else "final_correct" if queue.get("correct")
                                           else "final_rejected" if queue["status"] == "complete" else "final_pending")
                    if queue["final"]["status"] == "measured":
                        value = queue["final"]["evaluation"]
                        verdict = decision(value)
                        require(same_decision(verdict, queue["final"]["decision"])
                                and (queue["correct"], queue["promoted"]) == (verdict["correct"], verdict["promoted"]),
                                "pool final verdict changed")
                        if verdict["correct"]:
                            require(value["source_sha256"] == run["selected_sha256"], "pool final source changed")
                            check_resource_evidence(value, plan)
                    for observation in queue["final"]["observations"]:
                        if observation.get("evaluation"):
                            gpu_window(observation["evaluation"], job, "gpu_final", plan["runtime"]["gpu_lock_path"])
            for receipt_path in sorted((path / "wire" / job["id"] / "admission").glob("*.json")):
                receipt = json.loads(receipt_path.read_text())
                require((receipt["port"], receipt["job_id"]) == (port, job["id"]), "pool admission belongs to another job")
                dispatched = receipt.get("dispatched_utc") is not None
                if dispatched:
                    wire = path / "wire" / job["id"] / f"{receipt['request_id']}.json"
                    require(hashlib.sha256(wire.read_bytes()).hexdigest() == receipt["wire_sha256"], "pool request bytes changed")
                    body = json.loads(wire.read_text())
                    require(body["model"] == worker["service"]["base_model"] and body["seed"] == job["seed"]
                            and body["temperature"] == .2 and body["max_tokens"] <= 2048
                            and body["chat_template_kwargs"] == {"enable_thinking": False}
                            and receipt["input_tokens"] + receipt["max_output_tokens"] <= 8192
                            and not any(receipt["admission"][k] for k in ("running", "waiting")),
                            "pool request bypassed service, context or idle admission")
                    proposer = bool(body.get("response_format"))
                    require(not proposer or not body.get("tools"), "pool proposer had tool access")
                    if proposer and receipt["status"] == "completed":
                        intervals.append((receipt["dispatched_utc"], 1, port))
                        intervals.append((receipt["finished_utc"], -1, port))
                        timeline.append({"lane": f":{port}", "job_id": job["id"], "phase": "model_proposal",
                                         "start_utc": receipt["dispatched_utc"], "end_utc": receipt["finished_utc"]})
                    receipt = {**receipt, "proposer": proposer}
                admissions.append(receipt)
            rows.append(row)
    require(len(rows) == pool["scheduled"] == 24 and len({r["job"]["id"] for r in rows}) == 24,
            "pool denominator changed")
    handoff = None
    handoff_path = folder / "handoff/gateway-receipt.json"
    new_job_ids = set()
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text())
        before = json.loads((folder / "handoff/before.json").read_text())
        require(hashlib.sha256((folder / "handoff/service_gateway.py").read_bytes()).hexdigest() == handoff["source_sha256"],
                "handoff gateway source changed")
        for worker in pool["workers"]:
            name = f"p{worker['port']}"
            state = json.loads((folder / "workers" / name / "generation.json").read_text())
            by_id = {row["job"]["id"]: row for row in state["jobs"]}
            for job_id, expected in before[name]["committed_results"].items():
                require(by_id[job_id].get("result_sha256") == expected, "handoff replayed or changed committed generation")
            new_job_ids.update(before[name]["pending"])
            require(all(state["reservations"][k] >= v for k, v in before[name]["reservations"].items()),
                    "handoff reset budget reservations")
        for name, expected in before["request_file_sha256"].items():
            require(hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, "handoff changed original request evidence")
        for route in handoff["routes"]:
            expected = next(w["service"] for w in pool["workers"] if w["port"] == route["logical_port"])
            require(all(route["declaration"][k] == expected[k] for k in
                        ("base_model", "model_root", "max_model_len", "vllm_version")), "handoff model declaration differs")
        handoff = {**handoff, "preserved_jobs": 24 - len(new_job_ids), "pending_jobs_at_handoff": len(new_job_ids),
                   "old_request_files_preserved": len(before["request_file_sha256"])}
    for row in rows:
        row["service_epoch"] = "after_restart" if row["job"]["id"] in new_job_ids else "original"
    complete = all(r["resolved"] for r in rows)
    arms = [{"policy": policy, "scheduled": len(group := [r for r in rows if r["job"]["study_policy"] == policy]),
             "resolved": sum(r["resolved"] for r in group), "correct": sum(bool(r["correct"]) for r in group),
             "promoted": sum(bool(r["promoted"]) for r in group),
             "reported_output_tokens": sum(r["output_tokens"] or 0 for r in group),
             "usage_missing_jobs": sum(r["output_tokens"] is None for r in group)} for policy in ("blind", "feedback")]
    tasks, contrasts, pairs = [], {}, []
    for task in sorted({r["job"]["task_id"] for r in rows}):
        group = [r for r in rows if r["job"]["task_id"] == task]
        require(len(group) == 4 and len({r["job"]["service_port"] for r in group}) == 1,
                "task was not blocked within one service")
        tasks.append({"task_id": task, "service_port": group[0]["job"]["service_port"],
                      "arms": {p: {"correct": sum(bool(r["correct"]) for r in group if r["job"]["study_policy"] == p),
                                   "promoted": sum(bool(r["promoted"]) for r in group if r["job"]["study_policy"] == p)}
                               for p in ("blind", "feedback")}})
        for seed in pool["seeds"]:
            pair = [r for r in group if r["job"]["seed"] == seed]
            require(len(pair) == 2 and {r["job"]["study_policy"] for r in pair} == {"blind", "feedback"},
                    "pool pair changed")
            pairs.append({"task_id": task, "seed": seed, "service_port": pair[0]["job"]["service_port"],
                          "resolved": all(r["resolved"] for r in pair),
                          "crosses_restart": len({r["service_epoch"] for r in pair}) > 1,
                          "first_source_matches": pair[0]["first_source_sha256"] == pair[1]["first_source_sha256"]
                          if all(r["first_source_sha256"] for r in pair) else None})
    if complete and not handoff:
        for key in ("correct", "promoted"):
            differences = [(t["arms"]["feedback"][key] - t["arms"]["blind"][key]) / 2 for t in tasks]
            contrasts[key] = {"macro_task_difference": sum(differences) / 6,
                              "exploratory_interval": bootstrap_interval(differences)}
    live, peak = set(), 0
    for _, delta, port in sorted(intervals, key=lambda x: (x[0], x[1])):
        live.add(port) if delta > 0 else live.discard(port)
        peak = max(peak, len(live))
    active_gpu, peak_gpu = 0, 0
    for _, delta in sorted(gpu_intervals, key=lambda x: (x[0], x[1])):
        active_gpu += delta
        peak_gpu = max(peak_gpu, active_gpu)
    require(peak_gpu <= 1, "this project's GPU evaluation windows overlap")
    return {"schema_version": 1, "captured_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "incomplete", "plan_sha256": digest(pool), "scheduled": 24,
            "resolved": sum(r["resolved"] for r in rows), "rows": rows, "arms": arms, "tasks": tasks,
            "workers": worker_rows, "contrasts": contrasts, "pairs": pairs,
            "committed_proposals": sum(r["proposals"] or 0 for r in rows), "uncommitted_trace_proposals": partial_proposals,
            "committed_model_replies": sum(r["model_replies"] or 0 for r in rows),
            "admitted_candidate_files": sum(r["admitted_candidates"] or 0 for r in rows),
            "unique_sources_per_job_sum": sum(r["unique_sources"] or 0 for r in rows),
            "outcome_types": dict(Counter(r.get("outcome_type", r["generation_status"]) for r in rows)),
            "admission_statuses": dict(Counter(r["status"] for r in admissions)),
            "handoff": handoff, "cross_restart_pairs": sum(p["crosses_restart"] for p in pairs),
            "pooled_contrasts_suppressed": bool(handoff),
            "dispatched_proposer_requests": sum(bool(r.get("proposer")) for r in admissions),
            "dispatched_auxiliary_requests": sum("dispatched_utc" in r and not r.get("proposer") for r in admissions),
            "peak_overlapping_proposal_ports": peak,
            "peak_overlapping_gpu_windows": peak_gpu, "independent_gpu_windows": len(gpu_seen), "timeline": timeline,
            "concurrency_scope": "overlap of completed proposer proxy intervals, not a measured end-to-end speedup",
            "scope": pool["scope"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
