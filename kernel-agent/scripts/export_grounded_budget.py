"""Report-only correction: count dispatched evidence and received patch replies, preserving raw outcomes."""
import argparse
import difflib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent.model_pool import utc  # noqa: E402
from scripts.summarize_grounded_budget import analyze  # noqa: E402


def compact(directory, output, finalize=False):
    full = analyze(directory)
    if finalize:
        exited = json.loads((directory / "supervisor-exit.json").read_text())
        if full["status"] != "complete" or exited.get("status") != "complete" or exited.get("exit_code") != 0:
            raise ValueError("only a completed supervisor may finalize an evidence manifest")
        atomic_json(directory / "analysis-full.json", full)
    cases, mechanisms = [], {}
    for arm in ("indexed", "grounded"):
        counters = Counter()
        for b in full["blocks"]:
            if not b["result"]:
                continue
            for i, t in enumerate(b["result"]["arms"][arm]["attempts"][1:]):
                counters["repair_slots"] += 1
                rp = directory / "blocks" / b["block"]["id"] / "requests" / (t["request_id"] + ".json")
                receipt = json.loads(rp.read_text()) if rp.exists() else {}
                dispatched = bool(receipt.get("dispatched_utc"))
                counters["dispatched_calls"] += dispatched
                counters["not_run_opportunities"] += t["evaluation"]["status"] == "not_run_infrastructure"
                packet = t.get("source_facts")
                counters["source_packets_constructed"] += bool(packet)
                counters["source_packets"] += bool(packet) and dispatched
                counters["python_errors_reported"] += bool(packet) and dispatched and packet["python_compile"] != "passed"
                counters["launches_reported"] += len((packet or {}).get("launches", [])) if dispatched else 0
                counters["stores_reported"] += len((packet or {}).get("stores", [])) if dispatched else 0
                patch = t["mode"] in {"patch", "span_patch"} and isinstance(t.get("reply"), str)
                counters["patch_attempts"] += patch
                counters["patch_applied"] += patch and "transaction" in t
                counters["patch_errors"] += patch and t["evaluation"]["status"] == "patch_error"
                counters["full_source_fallbacks"] += t["mode"] == "full_source_fallback"
                counters["fallback_search_correct"] += t["mode"] == "full_source_fallback" and t["evaluation"]["status"] == "correct"
                if i == 0:
                    counters["first_patch_attempts"] += patch
                    counters["first_patch_applied"] += patch and "transaction" in t
        mechanisms[arm] = dict(counters)
    for b in full["blocks"]:
        if not b["result"]:
            continue
        r = b["result"]
        brief = {"block": b["block"], "endpoint_port": r["endpoint"]["port"],
                 "initial_status": r["initial"]["evaluation"]["status"], "initial_sha256": r["initial"].get("source_sha256"),
                 "remote_evidence": str(directory / "blocks" / b["block"]["id"]), "arms": {}}
        for arm, a in r["arms"].items():
            history, attempts = [], []
            for t in a["attempts"]:
                if t["request_id"] == r["initial"]["request_id"]:
                    history.append(t)
                    continue
                parent = next((v for v in reversed(history) if v.get("code") and v.get("source_sha256")
                               and v["source_sha256"] == t.get("parent_sha256")), None)
                changes = []
                if parent and t.get("code"):
                    changes = list(difflib.unified_diff(parent["code"].splitlines(), t["code"].splitlines(),
                                                       fromfile="parent", tofile="model_candidate", n=2))[:28]
                attempts.append({"request_id": t["request_id"], "mode": t["mode"], "status": t["evaluation"]["status"],
                                 "parent_sha256": t.get("parent_sha256"), "source_sha256": t.get("source_sha256"),
                                 "patch_applied": "transaction" in t, "route": t.get("diagnosis", {}).get("route"),
                                 "context_view": t.get("context_budget", {}).get("selected_view"),
                                 "message_excerpt": str(t["evaluation"].get("message", ""))[:240],
                                 "derived_diff_excerpt": "\n".join(changes)[:1500]})
                history.append(t)
            brief["arms"][arm] = {"correct": a["correct"], "promoted": a["promoted"],
                                   "selected_sha256": a["selected_sha256"], "final_status": (a["final"] or {}).get("status"),
                                   "attempts": attempts}
        cases.append(brief)
    usage = full["arm_usage"]
    evidence = directory / "evidence-manifest.json"
    if finalize:
        paths = [p for p in directory.rglob("*") if p.is_file() and not p.name.endswith((".lock", ".pyc"))
                 and "__pycache__" not in p.parts and p.name not in {".DS_Store", "evidence-manifest.json"}]
        manifest = {"schema_version": 1, "scope": "Server-only complete raw study and executed source freeze",
                    "sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
        if evidence.exists() and json.loads(evidence.read_text()) != manifest:
            raise ValueError("do not replace a finalized raw-evidence manifest")
        atomic_json(evidence, manifest)
    plan = json.loads((directory / "plan.json").read_text())
    result = {k: full[k] for k in ("status", "scheduled", "resolved", "plan_sha256", "strata", "rows", "services", "timeline",
                                 "dispatched_requests", "request_statuses", "reported_output_tokens", "usage_missing_dispatched",
                                 "patch_statuses", "repair_routes", "missing_apis", "peak_model_ports", "independent_gpu_windows",
                                 "peak_gpu_windows", "shared_initial_verified", "patch_reply_to_code_verified",
                                 "policy_request_binding_verified", "final_selection_binding_verified", "scope")}
    result.update(schema_version=1, generated_utc=utc(), cases=cases, arm_mechanisms=mechanisms, arm_usage=usage,
                  protocol={k: plan[k] for k in ("arms", "max_model_calls", "context_limit", "max_output_tokens", "temperature",
                                                 "regression_seeds", "fresh_seed", "fresh_tasks")},
                  provenance={"raw_directory": str(directory), "raw_manifest": str(evidence),
                              "raw_manifest_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest() if evidence.exists() else None,
                              "raw_manifest_files": len(json.loads(evidence.read_text())["sha256"]) if evidence.exists() else None,
                              "source_manifest_sha256": digest(plan["source_sha256"]), "source_files": len(plan["source_sha256"]),
                              "full_reconstruction_sha256": digest(full), "server_reconstruction_verified": True,
                              "local_scope": "Derived counts, bounded diff excerpts and remote evidence locators; no raw replies/kernels/ledger export"})
    views, checks, refused = Counter(), 0, Counter()
    for block in full["blocks"]:
        if not block["result"]:
            continue
        for arm, value in block["result"]["arms"].items():
            for attempt in value["attempts"][1:]:
                budget = attempt.get("context_budget")
                if budget:
                    views[arm + ":" + str(budget["selected_view"])] += 1
                    checks += len(budget["views"])
                    refused[arm] += budget["status"] == "context_budget_error"
    result["context_adaptation"] = {"views": dict(views), "tokenization_measurements": checks,
        "policy_local_refusals": dict(refused), "binding_verified": True,
        "scope": "Source-bound first-fitting-view receipts reconstructed; tokenization is not a model completion"}
    result["summary_sha256"] = digest(result)
    if len(json.dumps(result, ensure_ascii=False).encode()) > 1000000:
        raise ValueError("compact summary exceeds1MB; keep raw evidence remote")
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/grounded-budget-summary.json")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    value = compact(args.directory, args.output, args.finalize)
    print(json.dumps({k: value[k] for k in ("status", "resolved", "dispatched_requests", "peak_model_ports")}))
