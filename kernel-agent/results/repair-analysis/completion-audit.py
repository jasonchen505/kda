"""Post-run audit; no new model calls, CUDA work, tuning or trial replacement."""
import datetime
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / "src")]
from scripts.summarize_repair import analyze  # noqa: E402


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


folder = root / "results/repair-study"
plan, state = load(folder / "plan.json"), load(folder / "checkpoint.json")
summary = analyze(folder)
require(summary == load(folder / "summary.json"), "summary is not reproducible")
require(summary["status"] == state["status"] == "complete" and summary["resolved"] == 64,
        "formal study is incomplete")
require(plan["seeds"] == [20261101, 20261102] and len(plan["tasks"]) == 8,
        "formal task/seed denominator changed")
require(state["allocations"] == {"model_calls": 160, "output_tokens": 327680},
        "formal reserved budget changed")
require(summary["dispatched_requests"] <= 160, "formal dispatch cap exceeded")

statuses, omissions, finish_reasons, probes = Counter(), Counter(), Counter(), Counter()
dispatched = completed = unknown = opportunities = duplicates = 0
final_sources = 0
proxy_seconds = 0.0
gpu_uuid = None
infra_blocks = []
observed_blocks = []
infra_statuses = {"resource_busy", "resource_contended", "resource_monitor_error", "resource_exhausted",
                  "evaluator_timeout", "device_mismatch", "evaluator_error", "model_infrastructure_error",
                  "not_run_infrastructure"}
for block in summary["blocks"]:
    require(block["status"] == "complete", "a block is not sealed")
    result = block["result"]
    path = folder / "blocks" / block["block"]["id"]
    unique = {}
    for arm in result["arms"].values():
        require(len(arm["attempts"]) == 3, "repair opportunity count changed")
        for attempt in arm["attempts"]:
            key = attempt["request_id"]
            require(key not in unique or unique[key] == attempt, "shared initial record changed")
            unique[key] = attempt
        final = arm["final"]
        if final:
            final_sources += "final_reused_from" not in arm
            window = final.get("resource_audit", {}).get("before")
            if window:
                require(datetime.datetime.fromisoformat(window["observed_utc"]) >=
                        datetime.datetime.fromisoformat(result["generation_committed_utc"]),
                        "final-data execution preceded committed search decisions")
    require(len(unique) == 10, "block does not have ten unique opportunities")
    infra = sorted({a['evaluation']['status'] for a in unique.values()
                    if a['evaluation']['status'] in infra_statuses}
                   | {(a.get('final') or {}).get('status') for a in result['arms'].values()
                      if (a.get('final') or {}).get('status') in infra_statuses})
    if infra or result.get('infrastructure_stop'):
        infra_blocks.append({'block': block['block']['id'], 'statuses': infra,
                             'stop': result.get('infrastructure_stop')})
    elif all(isinstance(a.get('reply'), str) for a in unique.values()):
        observed_blocks.append(block['block']['id'])
    opportunities += len(unique)
    for attempt in unique.values():
        evaluation = attempt["evaluation"]
        statuses[evaluation["status"]] += 1
        duplicates += bool(attempt.get("duplicate_of"))
        raw = path / "attempts" / f"{attempt['request_id']}.json"
        if raw.exists():
            require(load(raw) == attempt, "attempt artifact disagrees with committed history")
        binding = evaluation.get("device_binding")
        if binding and binding.get("verified"):
            current = binding["expected_uuid"]
            require(gpu_uuid in (None, current), "physical evaluation GPU changed")
            gpu_uuid = current
        witness = evaluation.get("counterexample")
        if witness:
            expected, actual = witness.get("expected", {}), witness.get("actual", {})
            if witness.get("probe_error") or not actual:
                category = "probe_unavailable"
            elif actual.get("nonfinite") or actual.get("shape") != expected.get("shape"):
                category = "contradiction_observed"
            elif "values" in actual and "values" in expected:
                atol, rtol = ((.015, .015) if "bfloat16" in expected.get("dtype", "")
                              else (2e-5, 2e-4))
                bad = len(actual["values"]) != len(expected["values"]) or any(
                    not math.isfinite(a) or abs(a - e) > atol + rtol * abs(e)
                    for a, e in zip(actual["values"], expected["values"]))
                category = "contradiction_observed" if bad else "tiny_probe_passed"
            else:
                category = "probe_unavailable"
            probes[category] += 1
    calls_in_block = 0
    for receipt_path in (path / "requests").glob("*.json"):
        receipt = load(receipt_path)
        if "status" not in receipt:
            continue
        omissions.update(receipt.get("omitted_sections", []))
        if receipt.get("dispatched_utc"):
            calls_in_block += 1
            dispatched += 1
            completed += receipt["status"] == "completed"
            unknown += receipt["status"] != "completed"
            if receipt["status"] == "completed":
                finish_reasons[str(receipt.get("finish_reason"))] += 1
                proxy_seconds += receipt["elapsed_seconds"]
            usage = receipt.get("usage")
            require(usage is None or usage.get("completion_tokens", 0) <= 2048,
                    "response exceeded the output ceiling")
    require(result["model_calls"] == calls_in_block <= 10, "block dispatch accounting changed")

require(dispatched == summary["dispatched_requests"] and opportunities == 160,
        "unique opportunity/request accounting disagrees")
require(summary["peak_gpu_windows"] <= 1 and summary["peak_model_ports"] <= min(8, len(summary["services"])),
        "concurrency evidence changed")
for contrast in summary["contrasts"].values():
    interval = contrast["exploratory_interval"]
    require(interval["clusters"] == 8 and interval["resamples"] == 8 ** 8,
            "analysis failed to use eight task clusters")

invalid = root / "results/repair-invalid-device-v1"
exclusion = load(invalid / "exclusion-summary.json")
require(exclusion["dispatched_requests"] == 48, "invalid pre-run dispatches changed")
invalid_receipts = [load(p) for p in (invalid / "blocks").glob("*/requests/*.json")]
invalid_dispatched = [r for r in invalid_receipts if r.get("dispatched_utc")]
require(len(invalid_dispatched) == 48 and sum(r.get("status") != "completed"
        for r in invalid_dispatched) == 3, "invalid unknown-call ledger changed")
policy_name = "src/kernel_agent/repair.py"
require(sha(invalid / "sources" / policy_name) == sha(folder / "sources" / policy_name)
        == sha(root / policy_name), "treatment changed between invalid and formal runs")
require(all(sha(root / name) == expected for name, expected in plan["source_sha256"].items()
            if name != "scripts/summarize_repair.py"), "an executed generation file changed")

value = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "verified", "scope": "Post-run integrity and recorded outcomes only; no new model/GPU work",
    "scheduled_blocks": 16, "scheduled_outcomes": 64, "unique_opportunities": opportunities,
    "formal_dispatched": dispatched, "formal_completed_responses": completed,
    "formal_unconfirmed": unknown, "formal_allocated_calls": 160,
    "invalid_prerun_dispatched": 48, "invalid_prerun_unconfirmed": 3,
    "invalid_prerun_allocated_calls": 60, "iteration_dispatched": 48 + dispatched,
    "iteration_allocated_calls": 220, "invalid_prerun_excluded": True,
    "unchanged_policy_sha256": sha(root / policy_name),
    "independent_final_source_checks": final_sources, "physical_evaluation_gpu_uuid": gpu_uuid,
    "unique_opportunity_statuses": dict(statuses), "duplicate_search_sources": duplicates,
    "omitted_optional_sections": dict(omissions), "finish_reasons": dict(finish_reasons),
    "unique_diagnostic_probes": dict(probes),
    "completed_call_proxy_seconds_sum": proxy_seconds,
    "policy_observation_sensitivity": {
        "infrastructure_blocks": infra_blocks,
        "fully_observed_blocks": observed_blocks,
        "cells": [{"generation": cell['generation'], "repair": cell['repair'],
                   "correct": sum(r['correct'] for r in summary['rows']
                                  if r['block'] in observed_blocks and r['generation'] == cell['generation']
                                  and r['repair'] == cell['repair']),
                   "denominator": len(observed_blocks)} for cell in summary['cells']],
        "scope": "Post-hoc complete-block descriptive counts only; primary 16-per-cell results retain every infrastructure failure. No replacement trial or new efficacy claim.",
    },
    "analysis_only_correction": {
        "reason": "Historical interval helper accepts at most six clusters; prospective design requires eight.",
        "decision_time": "before any sealed formal v2 block; recorded in repair-analysis-correction.json",
        "method": "Exact weighted multisets, same percentile definition, representing all 8**8 ordered draws",
        "executed_analyzer_sha256": plan["source_sha256"]["scripts/summarize_repair.py"],
        "delivered_analyzer_sha256": sha(root / "scripts/summarize_repair.py"),
        "model_generation_unchanged": True,
    },
    "plan_sha256": summary["plan_sha256"],
    "checkpoint_file_sha256": sha(folder / "checkpoint.json"),
}
(root / "results/repair-completion-audit.json").write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(value, indent=2))
