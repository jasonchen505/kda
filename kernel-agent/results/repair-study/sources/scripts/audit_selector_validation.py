"""Verify a completed selector replication against actual files and local request traces."""
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kernel_agent.line_protocol import proposal
from kernel_agent.protocol import prefers
from kernel_agent.transport import PROPOSER_PREFIX
from scripts.summarize_strategy import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "results/agent-selector-validation.json"
    study = json.loads(source.read_text())
    assert study["status"] == "complete", "only completed campaigns can be audited"
    assert study["metadata"]["selection_policy"] == "gate_first"
    assert len(study["runs"]) == len(study["schedule"]) == 3
    assert digest(ROOT / "results/agent-strategy-study.json") == study["metadata"]["parent_study_sha256"]
    frozen = ROOT / "results/study8-measured"
    for name, expected in study["metadata"]["source_sha256"].items():
        actual = ROOT / name
        assert digest(actual) == expected, name
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == expected, name
        else:
            shutil.copyfile(actual, target)
    candidates, rows = {}, []
    for run in study["runs"]:
        best = None
        for attempt in run["attempts"]:
            if prefers(attempt.get("decision", {}), best["decision"] if best else None):
                best = attempt
            sha = attempt.get("source_sha256")
            if not sha:
                continue
            code = attempt["code"]
            assert proposal(attempt["reply"])["code"] == code, "proposed source was altered"
            name = f"runs/{run['run_id']}/candidate-{attempt['iteration']}.py"
            actual = ROOT / name
            assert digest(actual) == hashlib.sha256(code.encode()).hexdigest() == sha, name
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual, target)
            candidates[name] = sha
            final = run.get("candidate_final_evaluations", {}).get(sha, {})
            if attempt.get("decision", {}).get("correct"):
                assert final, "search-correct candidate lacks final audit"
                if final["evaluation"].get("status") == "correct":
                    assert final["evaluation"]["source_sha256"] == sha
            else:
                assert final["evaluation"]["status"] == "search_rejected"
        selected = best["source_sha256"] if best else None
        assert run.get("selected_sha256") == selected, "actual choice differs from gate-first search ranking"
        if selected:
            final = run["candidate_final_evaluations"][selected]
            assert run["final_evaluation"] == final["evaluation"]
            assert run["final_decision"] == final["decision"]
            assert run["correct"] == final["decision"]["correct"]
            assert run["promoted"] == final["decision"]["promoted"]
        rows.append({"task": run["task_id"], "selected_sha256": selected,
                     "correct": run["correct"], "promoted": run["promoted"], "ranking_matches": True})
    traces = [json.loads(p.read_text()) for p in sorted((ROOT / "runs/selector-validation-wire").glob("*.json"))]
    upstream = study["metadata"]["generation_service"]
    assert traces, "request traces missing"
    for trace in traces:
        assert trace["upstream"] == upstream["endpoint"]
        assert trace["model"] == upstream["base_model_id"]
        assert trace["seed"] == study["metadata"]["seed"]
        assert trace["temperature"] == 0 and trace["max_tokens"] <= 4096
    proposer = [t for t in traces if any(m.get("role") == "system"
                and isinstance(m.get("content"), str) and m["content"].startswith(PROPOSER_PREFIX)
                for m in t["messages"])]
    for trace in proposer:
        assert not trace.get("tools") and "tool_choice" not in trace
        assert trace["response_format"]["json_schema"]["schema"]["properties"]["code_lines"]["type"] == "array"
    assert len(proposer) == sum(len(r["attempts"]) for r in study["runs"])
    summary = summarize(study)
    for row, summarized in zip(rows, summary["runs"], strict=True):
        choices = summarized["selections"]
        row["geomean_selected_sha256"] = choices["best"]["source_sha256"]
        row["ranking_diverged"] = choices["best"]["source_sha256"] != choices["gate_first"]["source_sha256"]
        row["search_prefix_rankings"] = summarized["search_prefix_rankings"]
        row["prefix_ranking_diverged"] = any(p["diverged"] for p in summarized["search_prefix_rankings"])
    audit = {"verified_utc": datetime.now(UTC).isoformat(), "study_sha256": digest(source),
             "parent_study_sha256": study["metadata"]["parent_study_sha256"],
             "measured_files_verified": len(study["metadata"]["source_sha256"]),
             "candidate_files_verified": len(candidates), "candidate_sha256": candidates,
             "selection_audit": rows, "conditions": summary["conditions"],
             "request_audit": {"proposer": len(proposer), "background": len(traces) - len(proposer),
                               "base_model_only": True, "seed": study["metadata"]["seed"],
                               "tools_removed_for_proposer": True, "line_schema": True},
             "scope": "fresh integration replication on prior task families; changed inference runtime; "
                      "search-only rule audits share candidate sources, no matched causal comparison"}
    analysis_names = ("scripts/summarize_strategy.py", "scripts/audit_selector_validation.py",
                      "scripts/build_report.py", "reports/template.html")
    audit["analysis_source_sha256"] = {name: digest(ROOT / name) for name in analysis_names}
    for name in analysis_names:
        target = ROOT / "results/study8-analysis" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    target = ROOT / "results/selector-validation-delivery-audit.json"
    target.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("study_sha256", "measured_files_verified",
                     "candidate_files_verified", "request_audit", "selection_audit")}, indent=2))


if __name__ == "__main__":
    main()
