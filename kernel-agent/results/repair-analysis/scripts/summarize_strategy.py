"""Summarize candidate-selection studies without treating audits as new trials."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_agent.protocol import prefers


def selection(run, which):
    attempts = run["attempts"]
    if which in ("gate_first", "best"):
        best = None
        for attempt in attempts:
            if prefers(attempt.get("decision", {}), best["decision"] if best else None,
                       policy="gate_first" if which == "gate_first" else "geomean"):
                best = attempt
        digest = best.get("source_sha256") if best else None
    else:
        digest = attempts[0 if which == "first" else -1].get("source_sha256") if attempts else None
    recorded = run.get("candidate_final_evaluations", {}).get(digest, {})
    return {"source_sha256": digest, "correct": bool(recorded.get("decision", {}).get("correct")),
            "promoted": bool(recorded.get("decision", {}).get("promoted")),
            "geomean_speedup": recorded.get("decision", {}).get("geomean_speedup"),
            "worst_speedup": recorded.get("decision", {}).get("worst_speedup"),
            "reason": recorded.get("evaluation", {}).get("status", "no_admitted_candidate")}


def summarize(study):
    rows = []
    for run in study["runs"]:
        leaders = {"best": None, "gate_first": None}
        prefixes = []
        for attempt in run["attempts"]:
            for which, policy in (("best", "geomean"), ("gate_first", "gate_first")):
                if prefers(attempt.get("decision", {}), leaders[which]["decision"] if leaders[which] else None, policy):
                    leaders[which] = attempt
            digests = {key: value.get("source_sha256") if value else None for key, value in leaders.items()}
            prefixes.append({"iteration": attempt["iteration"], **digests,
                             "diverged": digests["best"] != digests["gate_first"]})
        rows.append({"task": run["task_id"], "split": run.get("task_split", "unlabeled"), "policy": run["study_policy"],
                     "terminal": run["terminal"], "correct": run["correct"], "promoted": run["promoted"],
                     "proposals": len(run["attempts"]),
                     "unique_sources": len({a["source_sha256"] for a in run["attempts"] if a.get("source_sha256")}),
                     "duplicate_proposals": sum(bool(a.get("duplicate_of")) for a in run["attempts"]),
                     "output_tokens": run.get("usage", {}).get("output_tokens"),
                     "elapsed_seconds": run.get("elapsed_seconds"),
                     "search_prefix_rankings": prefixes,
                     "selections": {which: selection(run, which) for which in ("first", "last", "best", "gate_first")}})
    conditions = []
    for policy in study["metadata"]["study_policies"]:
        group = [r for r in rows if r["policy"] == policy]
        conditions.append({"policy": policy, "planned_runs": sum(p == policy for _, p in study["schedule"]),
                           "completed_runs": len(group), "final_correct": sum(r["correct"] for r in group),
                           "final_promoted": sum(r["promoted"] for r in group),
                           "prospective_correct": sum(r["correct"] for r in group if r["split"] == "prospective"),
                           "prospective_completed": sum(r["split"] == "prospective" for r in group),
                           "selection_audit": {which: {"correct": sum(r["selections"][which]["correct"] for r in group),
                                                       "promoted": sum(r["selections"][which]["promoted"] for r in group)}
                                               for which in ("first", "last", "best", "gate_first")}})
    selection_policy = study.get("metadata", {}).get("selection_policy", "geomean")
    scope = ("one run per task/policy; gate_first is a post-hoc search-only counterfactual on existing final audits, "
             "not a fresh run; original final results and selection never change"
             if selection_policy == "geomean" else
             "one fresh run per task/policy using gate_first selection; first/last/geomean audits share generated artifacts "
             "and are not independent trials")
    return {"status": study["status"], "conditions": conditions, "runs": rows, "scope": scope}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/agent-strategy-study.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.input)
    source = source if source.is_absolute() else ROOT / source
    target = (Path(args.output) if args.output else source.with_name(
        "agent-strategy-summary.json" if source.name == "agent-strategy-study.json" else f"{source.stem}-summary.json"))
    target = target if target.is_absolute() else ROOT / target
    result = summarize(json.loads(source.read_text()))
    result["study_file"] = str(source.relative_to(ROOT))
    result["study_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["conditions"], indent=2))
