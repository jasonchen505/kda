"""Check published frozen evidence without a model, CUDA, or a running service."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_agent.line_protocol import proposal
from kernel_agent.protocol import decision, prefers
from kernel_agent.transport import PROPOSER_PREFIX


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    return {"status": "verified", "study_sha256": digest(raw),
            "measured_files": len(study["metadata"]["source_sha256"]), "candidate_files": candidates,
            "proposer_requests": len(proposer), "background_requests": len(traces) - len(proposer),
            "correct": sum(run["correct"] for run in study["runs"]),
            "promoted": sum(run["promoted"] for run in study["runs"]),
            "scope": "artifact integrity and recorded decisions; no new GPU or model execution"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
