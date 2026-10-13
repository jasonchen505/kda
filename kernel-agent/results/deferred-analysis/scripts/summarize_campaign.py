"""Offline task/seed analysis of immutable campaign checkpoints, including incomplete runs."""
import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_agent.protocol import decision, prefers, sha  # noqa: E402
from kernel_agent.line_protocol import proposal  # noqa: E402

TERMINAL = {"completed", "failed", "interrupted_unknown"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def same_decision(actual, recorded):
    # libm can differ by one ULP between Linux and macOS; flags remain exact.
    return actual.keys() == recorded.keys() and all(
        math.isclose(value, recorded[key], rel_tol=1e-12, abs_tol=1e-12)
        if type(value) is float and type(recorded[key]) in (float, int) else value == recorded[key]
        for key, value in actual.items())


def audit_state(state, sources=None):
    require(state["state_sha256"] == digest({k: v for k, v in state.items() if k != "state_sha256"}),
            "checkpoint hash mismatch")
    plan = state["plan"]
    require(state["plan_sha256"] == digest(plan), "plan hash mismatch")
    require([r["job"] for r in state["jobs"]] == plan["jobs"], "schedule differs from frozen plan")
    for index, event in enumerate(state["events"]):
        require(event["sequence"] == index and event["previous_sha256"] ==
                (state["events"][index - 1]["sha256"] if index else None), "event chain mismatch")
        require(event["sha256"] == digest({k: v for k, v in event.items() if k != "sha256"}),
                "event hash mismatch")
    started = sum(r["status"] != "pending" for r in state["jobs"])
    require(state["reservations"] == {k: started * v for k, v in plan["per_job_budget"].items()},
            "allocation ledger mismatch")
    require(all(v <= plan["campaign_budget"][k] for k, v in state["reservations"].items()),
            "campaign over-allocation")
    if sources:
        for name, expected in plan["source_sha256"].items():
            require(hashlib.sha256((sources / name).read_bytes()).hexdigest() == expected,
                    f"frozen source mismatch: {name}")
    for row in state["jobs"]:
        require(row["status"] in {"pending", "running", *TERMINAL}, "unknown job state")
        if row["status"] != "completed":
            continue
        run, job = row["result"], row["job"]
        require(digest(run) == row["result_sha256"], "committed result hash mismatch")
        require((run["task_id"], run["study_policy"], run["seed"], run["job_id"]) ==
                (job["task_id"], job["study_policy"], job["seed"], job["id"]), "result/job mismatch")
        best = None
        for attempt in run["attempts"]:
            if attempt.get("source_sha256"):
                code = proposal(attempt["reply"])["code"]
                require(code == attempt["code"] and sha(code) == attempt["source_sha256"],
                        "model source changed")
            verdict = decision(attempt["evaluation"])
            if attempt.get("decision") is not None:
                require(same_decision(verdict, attempt["decision"]), "search decision mismatch")
            if verdict["correct"]:
                if plan["generation"].get("evaluation_guard"):
                    occupancy = attempt["evaluation"].get("resource_audit") or {}
                    require(occupancy.get("gpu_index") == plan["runtime"]["gpu_index"]
                            and occupancy.get("before_pids") == occupancy.get("after_pids") == [],
                            "guarded search evidence missing uncontended boundary checks")
                require(attempt["evaluation"].get("source_sha256") == attempt.get("source_sha256"),
                        "search result not bound to source")
                if prefers(verdict, best["decision"] if best else None, plan["generation"]["selection_policy"]):
                    best = {"source_sha256": attempt["source_sha256"], "decision": verdict}
        require(run.get("selected_sha256") == (best["source_sha256"] if best else None),
                "search-selected source changed")
        final = decision(run.get("final_evaluation") or {})
        require(type(run["correct"]) is bool and type(run["promoted"]) is bool
                and (run["correct"], run["promoted"]) == (final["correct"], final["promoted"]),
                "headline differs from final verdict")
        if final["correct"]:
            require(run["final_evaluation"]["source_sha256"] == run["selected_sha256"],
                    "final result not bound to selected source")
            if plan["generation"].get("evaluation_guard"):
                occupancy = run["final_evaluation"].get("resource_audit") or {}
                require(occupancy.get("gpu_index") == plan["runtime"]["gpu_index"]
                        and occupancy.get("before_pids") == occupancy.get("after_pids") == [],
                        "guarded final evidence missing uncontended boundary checks")
    return plan


def bootstrap_interval(values):
    """Exact task-cluster bootstrap for at most six tasks; descriptive, not a causal CI."""
    require(0 < len(values) <= 6, "expected one to six task clusters")
    distribution = sorted(sum(sample) / len(values) for sample in itertools.product(values, repeat=len(values)))
    return {"low": distribution[int(.025 * (len(distribution) - 1))],
            "high": distribution[int(.975 * (len(distribution) - 1))],
            "clusters": len(values), "resamples": len(distribution),
            "method": "exact percentile task-cluster bootstrap; exploratory, development-task set"}


def summarize(state, sources=None):
    plan = audit_state(state, sources)
    jobs = state["jobs"]
    statuses = Counter(r["status"] for r in jobs)
    complete = all(r["status"] in TERMINAL for r in jobs)
    arms, task_rows, pairs = [], [], []

    def metrics(group):
        done = [r for r in group if r["status"] == "completed"]
        terminal = [r for r in group if r["status"] in TERMINAL]
        known = [r["result"].get("usage", {}).get("output_tokens") for r in done]
        valid = [n for n in known if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
        actual = [r["result"] for r in done]
        failures = Counter(r["status"] for r in terminal if r["status"] != "completed")
        failures.update(r.get("final_evaluation", {}).get("status", "no_final_evidence")
                        if r.get("final_evaluation") else "no_final_evidence" for r in actual if not r["correct"])
        return {"scheduled": len(group), "started": len(group) - sum(r["status"] == "pending" for r in group),
                "terminal": len(terminal), "result_available": len(done),
                "pending": sum(r["status"] == "pending" for r in group),
                "running": sum(r["status"] == "running" for r in group),
                "correct": sum(r["correct"] for r in actual), "promoted": sum(r["promoted"] for r in actual),
                "correct_per_scheduled": sum(r["correct"] for r in actual) / len(group)
                if len(terminal) == len(group) else None,
                "promoted_per_scheduled": sum(r["promoted"] for r in actual) / len(group)
                if len(terminal) == len(group) else None,
                "reported_output_tokens": sum(valid), "usage_missing_jobs": len(group) - len(valid),
                "reported_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in actual),
                "proposals": sum(len(r["attempts"]) for r in actual),
                "unique_source_sha_per_job_sum": sum(len({a["source_sha256"] for a in r["attempts"]
                                                          if a.get("source_sha256")}) for r in actual),
                "duplicate_proposals": sum(bool(a.get("duplicate_of")) for r in actual for a in r["attempts"]),
                "terminal_reasons": dict(Counter(r.get("terminal", "unknown") for r in actual)),
                "resource_interrupted_jobs": sum(r.get("terminal") == "resource_unavailable" for r in actual),
                "resource_affected_jobs": sum(r.get("terminal") == "resource_unavailable" or
                                              (r.get("final_evaluation") or {}).get("status") in
                                              {"resource_busy", "resource_contended", "resource_monitor_error",
                                               "resource_exhausted", "evaluator_timeout"} for r in actual),
                "failures": dict(failures)}

    for policy in ("blind", "feedback"):
        arms.append({"policy": policy, **metrics([r for r in jobs if r["job"]["study_policy"] == policy])})
    for task in plan["tasks"]:
        task_arms = {policy: metrics([r for r in jobs if r["job"]["task_id"] == task
                                     and r["job"]["study_policy"] == policy]) for policy in ("blind", "feedback")}
        task_rows.append({"task_id": task, "arms": task_arms})
        for seed in plan["seeds"]:
            pair = {r["job"]["study_policy"]: r for r in jobs if r["job"]["task_id"] == task
                    and r["job"]["seed"] == seed}
            require(set(pair) == {"blind", "feedback"}, "incomplete pair in plan")
            resolved = all(r["status"] in TERMINAL for r in pair.values())
            values = {policy: {key: bool(row.get("result", {}).get(key)) for key in ("correct", "promoted")}
                      for policy, row in pair.items()} if resolved else None
            pairs.append({"task_id": task, "seed": seed, "resolved": resolved,
                          "statuses": {p: r["status"] for p, r in pair.items()}, "outcomes": values,
                          "first_source_matches": (pair["blind"]["result"]["attempts"][0].get("source_sha256") ==
                                                   pair["feedback"]["result"]["attempts"][0].get("source_sha256"))
                          if all(r.get("result", {}).get("attempts") and r["result"]["attempts"][0].get("source_sha256")
                                 for r in pair.values()) else None,
                          "feedback_minus_blind": {k: int(values["feedback"][k]) - int(values["blind"][k])
                                                   for k in ("correct", "promoted")} if resolved else None})
    contrasts = None
    if complete:
        contrasts = {}
        for outcome in ("correct", "promoted"):
            values = [r["arms"]["feedback"][outcome + "_per_scheduled"] -
                      r["arms"]["blind"][outcome + "_per_scheduled"] for r in task_rows]
            contrasts[outcome] = {"macro_task_difference": sum(values) / len(values),
                                  "task_differences": dict(zip(plan["tasks"], values)),
                                  "exploratory_interval": bootstrap_interval(values)}
    return {"schema_version": 1, "status": "complete" if complete else "incomplete",
            "execution_status": "running" if statuses["running"] else state["status"],
            "captured_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": state["plan_sha256"],
            "checkpoint_sha256": state["state_sha256"], "scheduled": len(jobs), "statuses": dict(statuses),
            "terminal": sum(statuses[s] for s in TERMINAL), "arms": arms, "tasks": task_rows, "pairs": pairs,
            "resolved_pairs": sum(p["resolved"] for p in pairs), "contrasts": contrasts,
            "budget_allocations": state["reservations"], "runtime": plan["runtime"],
            "generation": plan["generation"],
            "scope": "Reference-assisted development-task replication; dependent proposals are not trials. "
                     "Pending/running jobs have no verdict; failed/unknown terminal jobs stay in the denominator. "
                     "Task-cluster intervals appear only after all scheduled jobs terminate. "
                     "Paired seeds do not imply matched initial candidates or causal attribution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--resource-state", type=Path, help="optional read-only readiness observation sidecar")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.checkpoint.read_text()), args.sources)
    if args.resource_state and args.resource_state.exists():
        result["resource_wait"] = json.loads(args.resource_state.read_text())
        if result["resource_wait"].get("waiting") and not result["statuses"].get("running"):
            result["execution_status"] = ("resource_wait_expired" if
                                          result["resource_wait"].get("remaining_seconds") == 0 else
                                          "waiting_for_resources")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "terminal": result["terminal"],
                      "scheduled": result["scheduled"], "arms": result["arms"]}, indent=2))


if __name__ == "__main__":
    main()
