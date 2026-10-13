"""Check published frozen evidence without a model, CUDA, or a running service."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kernel_agent.line_protocol import proposal
from kernel_agent.protocol import decision, prefers
from kernel_agent.transport import PROPOSER_PREFIX
from scripts.summarize_campaign import audit_state


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_run(run, frozen):
    best = None
    candidates = 0
    for attempt in run["attempts"]:
        sha = attempt.get("source_sha256")
        if not sha:
            continue
        code = proposal(attempt["reply"])["code"]
        require(code == attempt["code"], "model proposal was altered")
        path = frozen / "runs" / run["run_id"] / f"candidate-{attempt['iteration']}.py"
        require(digest(path) == hashlib.sha256(code.encode()).hexdigest() == sha,
                f"candidate SHA mismatch: {path}")
        evaluation = attempt["evaluation"]
        require(evaluation["source_sha256"] == sha, "search evidence bound to wrong source")
        require(decision(evaluation) == attempt["decision"], "search decision mismatch")
        if prefers(attempt["decision"], best["decision"] if best else None):
            best = attempt
        final = run["candidate_final_evaluations"][sha]
        require(final["evaluation"]["source_sha256"] == sha, "final evidence bound to wrong source")
        require(decision(final["evaluation"]) == final["decision"], "final decision mismatch")
        candidates += 1
    require(best is not None and best["source_sha256"] == run["selected_sha256"],
            "actual output differs from search-only gate-first selection")
    final = run["candidate_final_evaluations"][best["source_sha256"]]
    require(run["final_evaluation"] == final["evaluation"] and run["final_decision"] == final["decision"],
            "selected final evidence mismatch")
    require(run["correct"] == final["decision"]["correct"] and run["promoted"] == final["decision"]["promoted"],
            "headline result differs from evaluator verdict")
    return candidates


def check_recovery(evidence, root):
    require(evidence["status"] == "passed", "recovery validation failed")
    require(evidence["evidence_sha256"] == object_digest({k: v for k, v in evidence.items()
                                                        if k != "evidence_sha256"}), "recovery evidence changed")
    for name, expected in evidence["source_sha256"].items():
        require(digest(root / "results" / evidence["snapshot_directory"] / name) == expected,
                f"frozen recovery source changed: {name}")
    for state in (evidence["before"], evidence["after"]):
        require(state["state_sha256"] == object_digest({k: v for k, v in state.items() if k != "state_sha256"}),
                "checkpoint integrity changed")
        require(state["plan_sha256"] == object_digest(state["plan"]), "checkpoint plan changed")
        for index, event in enumerate(state["events"]):
            require(event["sequence"] == index and event["previous_sha256"] ==
                    (state["events"][index - 1]["sha256"] if index else None), "checkpoint event chain changed")
            require(event["sha256"] == object_digest({k: v for k, v in event.items() if k != "sha256"}),
                    "checkpoint event hash changed")
        for row in state["jobs"]:
            if row["status"] == "completed":
                require(row["result_sha256"] == object_digest(row["result"]), "committed recovery result changed")
                result = row["result"]
                require(result["task_id"] == row["job"]["task_id"]
                        and result["study_policy"] == row["job"]["study_policy"], "recovery result job changed")
                final = decision(result["final_evaluation"])
                require((result["correct"], result["promoted"]) == (final["correct"], final["promoted"]),
                        "recovery headline changed")
                selected = result["selected_sha256"]
                require(result["final_evaluation"]["source_sha256"] == selected
                        and any(hashlib.sha256(a.get("code", "").encode()).hexdigest() == selected
                                for a in result["attempts"]), "recovery final SHA changed")
        allocated = sum(r["status"] != "pending" for r in state["jobs"])
        require(state["reservations"] == {k: allocated * v for k, v in state["plan"]["per_job_budget"].items()},
                "recovery allocations changed")
    require(evidence["before"]["plan"] == evidence["after"]["plan"], "recovery changed the plan")
    require([r["status"] for r in evidence["before"]["jobs"]] == ["completed", "running", "pending"]
            and [r["status"] for r in evidence["after"]["jobs"]] == ["completed", "interrupted_unknown", "completed"],
            "recovery transition changed")
    require(evidence["before"]["jobs"][0]["result_sha256"] == evidence["after"]["jobs"][0]["result_sha256"],
            "completed recovery result not reused")
    require(evidence["worker_calls"] == [{"job_id": n} for n in "ABC"]
            and evidence["killed_process_returncode"] == -9, "fault execution trace changed")
    require(evidence["summary"] == evidence["repeated_resume_summary"]
            and evidence["summary"]["scheduled"] == evidence["summary"]["terminal"] == 3
            and evidence["summary"]["interrupted_unknown"] == 1, "recovery denominator changed")
    return {"status": "verified", "signal": "SIGKILL", "jobs": 3, "unknown_outcomes": 1,
            "scope": "CPU fault injection using fake model/GPU drivers; no new GPU success counts"}


def check_campaigns(root):
    pilot_path = root / "results/multiseed-gpu4-pilot-checkpoint.json"
    if not pilot_path.exists():
        return None
    pilot = json.loads(pilot_path.read_text())
    frozen = root / "results/multiseed-gpu4-pilot-measured"
    audit_state(pilot, frozen)
    completed = [row for row in pilot["jobs"] if row["status"] == "completed"]
    require(len(completed) == 1, "pilot denominator changed")
    run = completed[0]["result"]
    for attempt in run["attempts"]:
        require(digest(frozen / "runs" / run["run_id"] / f"candidate-{attempt['iteration']}.py") ==
                attempt["source_sha256"], "pilot candidate file changed")
    traces = [json.loads(p.read_text()) for p in (root / "evidence/multiseed-gpu4-pilot-requests").rglob("*.json")]
    require(len(traces) == 6, "pilot request denominator changed")
    proposer = []
    for trace in traces:
        require(trace["model"] == pilot["plan"]["runtime"]["service"]["base_model"]
                and trace["seed"] == completed[0]["job"]["seed"] and trace["temperature"] == .2,
                "pilot request model/settings changed")
        if trace.get("response_format"):
            require(not trace.get("tools")
                    and trace["response_format"]["json_schema"]["schema"]["properties"]["code_lines"]["type"] == "array",
                    "pilot proposer schema changed")
            proposer.append(trace)
    require(len(proposer) == 3, "pilot proposer count changed")
    qualification = json.loads((root / "results/multiseed-gpu4-pilot-qualification.json").read_text())
    require(qualification["clean_performance_qualified"] is False
            and qualification["process_owned_by_this_user"] is False
            and qualification["job_started_utc"] < qualification["process_started_utc"] < qualification["job_committed_utc"],
            "pilot interference qualification changed")
    current = json.loads((root / "results/agent-multiseed-checkpoint.json").read_text())
    audit_state(current, root / "results/study9-measured")
    require(current["plan"]["generation"]["evaluation_guard"] == "pre_post_foreign_cuda_pid_check",
            "guarded campaign lost measurement qualification")
    return {"pilot_completed": len(completed), "pilot_proposer_requests": len(proposer),
            "pilot_clean_performance_qualified": False,
            "guarded_scheduled": len(current["jobs"]),
            "guarded_terminal": sum(r["status"] in {"completed", "failed", "interrupted_unknown"} for r in current["jobs"]),
            "scope": "separate integration pilot and guarded campaign snapshot; no pooling with historical GPU scores"}


def verify(root=ROOT):
    raw = root / "results/agent-selector-validation.json"
    study = json.loads(raw.read_text())
    audit = json.loads((root / "results/selector-validation-delivery-audit.json").read_text())
    require(digest(raw) == audit["study_sha256"], "raw study SHA mismatch")
    require(study["status"] == "complete", "campaign incomplete")
    require(digest(root / "results/agent-strategy-study.json") == study["metadata"]["parent_study_sha256"],
            "original strategy study changed")
    require(study["metadata"]["selection_policy"] == "gate_first", "unexpected selection policy")
    frozen = root / "results/study8-measured"
    for name, expected in study["metadata"]["source_sha256"].items():
        require(digest(frozen / name) == expected, f"frozen measured file changed: {name}")
    for name, expected in audit["analysis_source_sha256"].items():
        require(digest(root / "results/study8-analysis" / name) == expected,
                f"frozen analysis file changed: {name}")
    require(len(study["runs"]) == len(study["schedule"]) == 3, "run denominator changed")
    require([(r["task_id"], r["study_policy"]) for r in study["runs"]]
            == [tuple(item) for item in study["schedule"]], "run schedule mismatch")
    candidates = sum(check_run(run, frozen) for run in study["runs"])
    traces = [json.loads(path.read_text()) for path in sorted((root / "evidence/selector-requests").glob("*.json"))]
    proposer = []
    for trace in traces:
        require(trace["model"] == study["metadata"]["generation_service"]["base_model_id"], "non-base model request")
        require(trace["seed"] == study["metadata"]["seed"] and trace["temperature"] == 0, "request settings changed")
        if any(m.get("role") == "system" and isinstance(m.get("content"), str)
               and m["content"].startswith(PROPOSER_PREFIX) for m in trace["messages"]):
            require(not trace.get("tools") and "tool_choice" not in trace, "proposer tool access present")
            require(trace["response_format"]["json_schema"]["schema"]["properties"]["code_lines"]["type"] == "array",
                    "proposal schema changed")
            proposer.append(trace)
    require(len(traces) == 18 and len(proposer) == candidates == 9, "request/candidate denominator mismatch")
    recovery_path = root / "results/campaign-recovery-validation.json"
    recovery = check_recovery(json.loads(recovery_path.read_text()), root) if recovery_path.exists() else None
    return {"status": "verified", "study_sha256": digest(raw), "recovery_validation": recovery,
            "multiseed_campaigns": check_campaigns(root),
            "measured_files": len(study["metadata"]["source_sha256"]), "candidate_files": candidates,
            "proposer_requests": len(proposer), "background_requests": len(traces) - len(proposer),
            "correct": sum(run["correct"] for run in study["runs"]),
            "promoted": sum(run["promoted"] for run in study["runs"]),
            "scope": "artifact integrity and recorded decisions; no new GPU or model execution"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
