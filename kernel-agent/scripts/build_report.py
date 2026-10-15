"""Build a self-contained report from measured evidence; never fabricate missing runs."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kernel_agent.line_protocol import proposal as line_proposal
from kernel_agent.protocol import proposal
from scripts.summarize_strategy import summarize

cli = argparse.ArgumentParser()
cli.add_argument("--input", help="Result path relative to the project, overriding automatic latest-study selection")
args = cli.parse_args()
canonical = [ROOT / f"results/{name}" for name in ("agent-selector-validation.json", "agent-strategy-study.json", "agent-pilot-qwen27.json", "agent-pilot-lines.json",
             "agent-pilot-transport.json", "agent-pilot-reference.json", "agent-pilot.json")]
latest = (ROOT / args.input) if args.input else max((p for p in canonical if p.exists()),
          key=lambda p: json.loads(p.read_text()).get("started_utc", ""))
study = json.loads(latest.read_text())
payload = {"study": study, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "validation": "控制器、兼容性与证据测试单独计数；真实模型运行使用独立 GPU 裁决。",
           "upstreams": {"kda": "ef6ce617693ef0782b3ecb9f37e39bbf10226a90",
                         "humanize": "77e3d0658d9dd4c54cd9497e45940593bf6a43ff"},
           "strategy_summary": summarize(study) if study.get("metadata", {}).get("study_kind") in {"strategy_ablation", "selector_validation"} else None,
           "original_strategy_study": json.loads((ROOT / "results/agent-strategy-study.json").read_text()) if (ROOT / "results/agent-strategy-study.json").exists() else None,
           "selector_recheck": json.loads((ROOT / "results/silu-selector-recheck.json").read_text()) if (ROOT / "results/silu-selector-recheck.json").exists() else None,
           "selector_validation_audit": json.loads((ROOT / "results/selector-validation-delivery-audit.json").read_text()) if (ROOT / "results/selector-validation-delivery-audit.json").exists() else None,
           "capacity_study": json.loads((ROOT / "results/agent-pilot-qwen27.json").read_text()) if (ROOT / "results/agent-pilot-qwen27.json").exists() else None,
           "initial_study": json.loads((ROOT / "results/agent-pilot-unguided.json").read_text()) if (ROOT / "results/agent-pilot-unguided.json").exists() else None,
           "api_study": json.loads((ROOT / "results/agent-pilot-api-comparison.json").read_text()) if (ROOT / "results/agent-pilot-api-comparison.json").exists() else None,
           "reference_study": json.loads((ROOT / "results/agent-pilot-reference.json").read_text()) if (ROOT / "results/agent-pilot-reference.json").exists() else None,
           "wire_probe": json.loads((ROOT / "results/phase4-wire-probe.json").read_text()) if (ROOT / "results/phase4-wire-probe.json").exists() else None,
           "schema_preflight": json.loads((ROOT / "results/phase4-schema-preflight.json").read_text()) if (ROOT / "results/phase4-schema-preflight.json").exists() else None,
           "transport_study": json.loads((ROOT / "results/agent-pilot-transport.json").read_text()) if (ROOT / "results/agent-pilot-transport.json").exists() else None,
           "strategy_wire_audit": json.loads((ROOT / "results/strategy-wire-audit.json").read_text()) if (ROOT / "results/strategy-wire-audit.json").exists() else None,
           "wire_audit": json.loads((ROOT / "results/transport-wire-audit.json").read_text()) if (ROOT / "results/transport-wire-audit.json").exists() else None,
           "line_study": json.loads((ROOT / "results/agent-pilot-lines.json").read_text()) if (ROOT / "results/agent-pilot-lines.json").exists() else None,
           "probe": json.loads((ROOT / "results/proposer-probe.json").read_text()) if (ROOT / "results/proposer-probe.json").exists() else None,
           "startup_files": [p.name for p in sorted((ROOT / "results").glob("*failed*"))]}
for key, name in (("recovery_validation", "campaign-recovery-validation.json"),
                  ("multiseed_preflight", "multiseed-preflight.json"),
                  ("multiseed_summary", "agent-multiseed-summary.json"),
                  ("multiseed_checkpoint", "agent-multiseed-checkpoint.json"),
                  ("multiseed_checks", "multiseed-checks.json"),
                  ("multiseed_pilot", "multiseed-gpu4-pilot-checkpoint.json"),
                  ("multiseed_pilot_qualification", "multiseed-gpu4-pilot-qualification.json"),
                  ("deferred_summary", "deferred-evaluation-summary.json"),
                  ("deferred_generation", "deferred-generation-checkpoint.json"),
                  ("delivery_checks", "deferred-checks.json"),
                  ("shared_completion", "shared-completion/summary.json"),
                  ("shared_feedback", "shared-feedback-summary.json"),
                  ("shared_feedback_checkpoint", "shared-feedback-checkpoint.json"),
                  ("coexecution_checks", "coexecution-checks.json"),
                  ("upgrade_checks", "harness-upgrade-checks.json")):
    path = ROOT / "results" / name
    payload[key] = json.loads(path.read_text()) if path.exists() else None
checkpoint = ROOT / "results/agent-multiseed-checkpoint.json"
payload["raw_multiseed_checkpoint"] = checkpoint.read_text() if checkpoint.exists() else None
payload["raw_deferred_evidence"] = json.dumps(
    {"summary": payload["deferred_summary"], "generation": payload["deferred_generation"],
     "shared_completion": payload["shared_completion"]},
    ensure_ascii=False, allow_nan=False)
shared_checkpoint = ROOT / "results/shared-feedback-checkpoint.json"
payload["raw_shared_checkpoint"] = shared_checkpoint.read_text() if shared_checkpoint.exists() else None
payload["shared_failed_traces"] = []
phased = ROOT / "results/phased-recovery"
payload["phased"] = ({key: json.loads((phased / path).read_text()) for key, path in (
    ("generation", "generation.json"), ("final", "final/summary.json"),
    ("receipt", "supervisor-exit.json"), ("kill", "final/kill-receipt.json"),
    ("launch", "service-launch.json"), ("initial_exit", "initial-supervisor-exit.json"))} if phased.exists() else None)
payload["phased_checks"] = (json.loads((ROOT / "results/phased-checks.json").read_text())
                            if (ROOT / "results/phased-checks.json").exists() else None)
payload["raw_phased_generation"] = (phased / "generation.json").read_text() if phased.exists() else None
if shared_checkpoint.exists():
    checkpoint = json.loads(shared_checkpoint.read_text())
    for row in checkpoint["jobs"]:
        if row["status"] not in {"failed", "interrupted_unknown"}:
            continue
        run_id = f"campaign-{checkpoint['plan_sha256'][:12]}-{row['job']['id']}"
        trace = ROOT / "results/shared-feedback-measured/runs" / run_id / "trace.json"
        payload["shared_failed_traces"].append({"job": row["job"], "status": row["status"],
                                               "error": row.get("error"), "run_id": run_id,
                                               "trace_attempts": json.loads(trace.read_text()) if trace.exists() else []})
metrics = {}
for run in study["runs"]:
    replies = parsed = admitted = 0
    for attempt in run["attempts"]:
        if "reply" in attempt:
            replies += 1
            parser = line_proposal if run.get("proposal_format") == "lines" else proposal
            try:
                parser(attempt["reply"])
                parsed += 1
            except ValueError:
                pass
        admitted += bool(attempt.get("source_sha256"))
    metrics[run["run_id"]] = {"replies": replies, "parsed": parsed, "admitted": admitted}
payload["proposal_metrics"] = metrics
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
html = (ROOT / "reports/template.html").read_text().replace("@@DATA@@", encoded)
target = ROOT / "reports/kernel-agent-report.html"
target.write_text(html)
print(target)
