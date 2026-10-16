"""Offline, hash-bound audit and separately clustered semantic-repair comparisons."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent import line_protocol, repair, semantic  # noqa: E402
from kernel_agent.protocol import decision, sha  # noqa: E402
from kernel_agent.transport import LINE_SCHEMA  # noqa: E402
from scripts.summarize_campaign import digest, require, same_decision  # noqa: E402
from scripts.summarize_repair import task_cluster_interval  # noqa: E402

INFRA = {"resource_busy", "resource_contended", "resource_monitor_error", "resource_exhausted", "evaluator_timeout",
         "device_mismatch", "evaluator_error", "model_infrastructure_error", "not_run_infrastructure"}


def analyze(folder):
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    state = json.loads((folder / "checkpoint.json").read_text())
    require(state["plan_sha256"] == digest(plan) and state["state_sha256"] ==
            digest({k: v for k, v in state.items() if k != "state_sha256"}), "semantic ledger changed")
    require([r["block"] for r in state["blocks"]] == plan["blocks"] and len(plan["blocks"]) == 22,
            "semantic planned denominator changed")
    for name, expected in plan["source_sha256"].items():
        require(hashlib.sha256((folder / "sources" / name).read_bytes()).hexdigest() == expected,
                "semantic frozen source changed: " + name)
    allocations = sum(r["block"]["model_call_cap"] for r in state["blocks"] if r["status"] != "pending")
    require(state["allocations"] == {"model_calls": allocations, "output_tokens": allocations * 2048}, "semantic budget reset")
    rows, blocks, receipts, timeline, windows, reviews = [], [], [], [], set(), []
    def gpu(evaluation, block_id, phase, committed=None):
        if evaluation.get("status") in INFRA or evaluation.get("status") == "protocol_error":
            return
        binding = evaluation.get("device_binding", {})
        audit = evaluation.get("resource_audit", {})
        actual = binding.get("actual", {})
        require(binding.get("verified") is True and binding.get("expected_gpu_index") == 4
                and actual.get("device_count") == 1 and actual.get("device_index") == 0
                and actual.get("visible_devices") == binding.get("expected_uuid") == audit.get("before", {}).get("gpu_uuid")
                and str(actual.get("gpu_uuid", "")).lower().removeprefix("gpu-") ==
                str(binding.get("expected_uuid", "")).lower().removeprefix("gpu-"), "semantic physical GPU proof missing")
        queue = evaluation.get("cooperative_queue", {})
        require(queue.get("lock_file") == plan["runtime"]["evaluation_lock_file"], "semantic cooperative GPU lock changed")
        before, after = audit["before"]["observed_utc"], audit["after"]["observed_utc"]
        if committed:
            require(before >= committed, "final data accessed before all search selections committed")
        key = before, after, queue.get("pid")
        if key not in windows:
            windows.add(key)
            timeline.append({"lane": "GPU4", "block": block_id, "phase": phase, "start_utc": before, "end_utc": after})

    for item in state["blocks"]:
        b, value = item["block"], None
        path = folder / "blocks" / b["id"]
        by_id = {}
        for p in (path / "requests").glob("*.json"):
            if p.name.endswith((".request.json", ".response.json")):
                continue
            receipt = json.loads(p.read_text())
            by_id[receipt["request_id"]] = receipt
            receipts.append(receipt)
            if receipt.get("dispatched_utc"):
                wire = path / "requests" / f"{receipt['request_id']}.request.json"
                require(hashlib.sha256(wire.read_bytes()).hexdigest() == receipt["request_sha256"], "semantic request changed")
                body = json.loads(wire.read_text())
                call = receipt["request_id"]
                seed = b["seed"] + (0 if call == "initial" else 1 if call in {"review", "rewrite-r1"} else 2)
                require(body["seed"] == seed and body["max_tokens"] == 2048 and body["temperature"] == .2
                        and body["stream"] is False and not body.get("tools")
                        and body["chat_template_kwargs"] == {"enable_thinking": False}
                        and receipt["input_tokens"] + 2048 <= 8192 and not any(receipt["admission"].values())
                        and receipt["endpoint"] == item["endpoint"] and body["model"] == item["endpoint"]["model"],
                        "semantic request settings/process changed")
                expected_schema = semantic.REVIEW_SCHEMA if call == "review" else LINE_SCHEMA
                require(body["response_format"]["json_schema"]["schema"] == expected_schema, "semantic response schema changed")
            if receipt["status"] == "completed":
                response = json.loads((path / "requests" / f"{receipt['request_id']}.response.json").read_text())
                require(digest(response) == receipt["response_sha256"], "semantic response changed")
                timeline.append({"lane": ":" + str(receipt["endpoint"]["port"]), "block": b["id"], "phase": "model",
                                 "start_utc": receipt["dispatched_utc"], "end_utc": receipt["finished_utc"]})
        if item["status"] == "complete":
            value = json.loads((path / "result.json").read_text())
            committed = json.loads((path / "generation.json").read_text())
            require(digest(value) == item["result_sha256"] and value["endpoint"] == item["endpoint"], "semantic result changed")
            require(value["initial"] == committed["initial"] and value["review"] == committed["review"]
                    == json.loads((path / "review.json").read_text()), "semantic common start/review changed")
            start, review = value["initial"], value["review"]
            require(value["model_calls"] == sum(bool(r.get("dispatched_utc")) for r in by_id.values())
                    <= b["model_call_cap"], "semantic call budget exceeded")
            if start.get("imported"):
                parent = json.loads((path / "parent.json").read_text())
                bank = json.loads((folder / "sources/data/semantic-failure-bank.json").read_text())
                require(parent == next(c for c in bank["cases"] if c["id"] == b["parent_id"])
                        and start["source_sha256"] == parent["source_sha256"]
                        and start["reply"] == parent["initial"]["reply"], "semantic imported parent changed")
            if "reply" in review:
                receipt = by_id["review"]
                response = json.loads((path / "requests/review.response.json").read_text())
                require(receipt["status"] == "completed" and response["choices"][0]["message"]["content"] == review["reply"],
                        "semantic review reply changed")
                try:
                    parsed = json.loads(review["reply"])
                except ValueError:
                    parsed = {}
                require(parsed == review["review"] and semantic.validate_review(parsed, start["evaluation"].get("semantic_probes", []))
                        == review["validation"], "semantic validation does not reproduce")
            reviews.append({"block": b["id"], "stratum": b["stratum"], "validation": review["validation"]})
            unique = {}
            for arm, a in value["arms"].items():
                prior = committed["arms"][arm]
                require(a["attempts"] == prior["attempts"] and a["selected_sha256"] == prior["selected_sha256"]
                        and a["attempts"][0] == start, "semantic arms do not share start or final changed search")
                require(a == json.loads((path / "final" / f"{arm}.json").read_text()), "semantic final artifact changed")
                require(a["model_opportunities"] == (["rewrite-r1", "rewrite-r2"] if arm == "rewrite" else ["review", f"{arm}-r2"]),
                        "semantic arm did not get exactly two repair opportunities")
                selected = repair.select(a["attempts"])
                require(a["selected_sha256"] == (selected["source_sha256"] if selected else None), "semantic selector changed")
                for i, attempt in enumerate(a["attempts"]):
                    require(attempt["parent_sha256"] == (a["attempts"][i - 1].get("source_sha256") if i else None),
                            "semantic source lineage changed")
                    unique[attempt["request_id"]] = attempt
                    if attempt.get("source_sha256"):
                        require(sha(attempt["code"]) == attempt["source_sha256"] == hashlib.sha256(
                            (path / "candidates" / f"{attempt['source_sha256']}.py").read_bytes()).hexdigest()
                            and line_protocol.proposal(attempt["reply"])["code"] == attempt["code"], "semantic candidate was edited")
                    if "reply" in attempt and not attempt.get("imported"):
                        receipt = by_id[attempt["request_id"]]
                        response = json.loads((path / "requests" / f"{attempt['request_id']}.response.json").read_text())
                        require(receipt["status"] == "completed" and response["choices"][0]["message"]["content"] == attempt["reply"],
                                "semantic proposal differs from response")
                    gpu(attempt["evaluation"], b["id"], "search")
                final = decision(a["final"] or {})
                require((a["correct"], a["promoted"]) == (final["correct"], final["promoted"]), "semantic headline changed")
                if a["final"]:
                    require(same_decision(final, a["final_decision"]), "semantic final decision changed")
                    if final["correct"]:
                        require(a["final"]["source_sha256"] == a["selected_sha256"], "semantic final source changed")
                    gpu(a["final"], b["id"], "final", value["generation_committed_utc"])
                rows.append({"block": b["id"], "stratum": b["stratum"], "task_id": b["task_id"], "arm": arm,
                             "status": "complete", "correct": a["correct"], "promoted": a["promoted"],
                             "initial_correct": decision(start["evaluation"])["correct"],
                             "infrastructure": bool(value["infrastructure_stop"]) or any(
                                 x["evaluation"]["status"] in INFRA for x in a["attempts"]) or (a["final"] or {}).get("status") in INFRA,
                             "fully_observed": all(by_id.get(k, {}).get("status") == "completed" for k in a["model_opportunities"])})
            contract = json.loads((folder / "sources/data/tasks.json").read_text())[b["task_id"]]["contract"]
            for call, receipt in by_id.items():
                if not receipt.get("dispatched_utc"):
                    continue
                if call == "review":
                    parts = semantic.reviewer_sections(contract, start)
                elif call == "initial":
                    parts = semantic.baseline_sections(b["task_id"], contract)
                elif call.startswith("rewrite-"):
                    history = [start] if call.endswith("r1") else [start, unique["rewrite-r1"]]
                    parts = semantic.baseline_sections(b["task_id"], contract, history)
                else:
                    parts = semantic.writer_sections(b["task_id"], contract, start, review["review"], review["validation"], call.startswith("grounded-"))
                omitted = receipt["omitted_sections"]
                require(all(not mandatory for name, _, mandatory in parts if name in omitted), "semantic mandatory context removed")
                expected = "\n\n".join(text for name, text, _ in parts if name not in omitted)
                body = json.loads((path / "requests" / f"{call}.request.json").read_text())
                require(body["messages"][-1]["content"] == expected, "semantic request differs from assigned policy")
        else:
            for arm in semantic.ARMS:
                rows.append({"block": b["id"], "stratum": b["stratum"], "task_id": b["task_id"], "arm": arm,
                             "status": item["status"], "correct": False, "promoted": False,
                             "initial_correct": False, "infrastructure": item["status"] in {"failed", "interrupted_unknown"},
                             "fully_observed": False})
        blocks.append({"block": b, "status": item["status"], "endpoint": item.get("endpoint"), "result": value})
    resolved = sum(r["status"] in {"complete", "failed", "interrupted_unknown"} for r in rows)
    strata = []
    for label, denominator in [("regression", 14), ("fresh", 8)]:
        selected = [r for r in rows if r["stratum"] == label]
        cells = [{"arm": arm, "scheduled": denominator,
                  "resolved": sum(r["status"] in {"complete", "failed", "interrupted_unknown"} for r in selected if r["arm"] == arm),
                  "correct": sum(r["correct"] for r in selected if r["arm"] == arm),
                  "promoted": sum(r["promoted"] for r in selected if r["arm"] == arm),
                  "initial_failed_observed": sum(not r["initial_correct"] and r["fully_observed"] and not r["infrastructure"] for r in selected if r["arm"] == arm),
                  "rescued_observed": sum(not r["initial_correct"] and r["fully_observed"] and not r["infrastructure"] and r["correct"] for r in selected if r["arm"] == arm)} for arm in semantic.ARMS]
        contrasts = {}
        if resolved == 66:
            for control, treatment in [("rewrite", "critic"), ("rewrite", "grounded"), ("critic", "grounded")]:
                differences = []
                for task in sorted({r["task_id"] for r in selected}):
                    group = [r for r in selected if r["task_id"] == task]
                    n = sum(r["arm"] == control for r in group)
                    differences.append((sum(r["correct"] for r in group if r["arm"] == treatment) -
                                        sum(r["correct"] for r in group if r["arm"] == control)) / n)
                contrasts[treatment + "_vs_" + control] = {"mean_difference": sum(differences) / len(differences),
                                                          "interval": task_cluster_interval(differences)}
        strata.append({"stratum": label, "cells": cells, "contrasts": contrasts})
    timeline.sort(key=lambda t: (t["start_utc"], t["end_utc"], t["lane"], t["block"], t["phase"]))
    events = sorted([(t["start_utc"], 1, t["lane"]) for t in timeline] + [(t["end_utc"], -1, t["lane"]) for t in timeline],
                    key=lambda t: (t[0], t[1]))
    active, peak_model, current_gpu, peak_gpu = Counter(), 0, 0, 0
    for _, delta, lane in events:
        if lane == "GPU4":
            current_gpu += delta
            peak_gpu = max(peak_gpu, current_gpu)
        else:
            active[lane] += delta
            peak_model = max(peak_model, sum(v > 0 for v in active.values()))
    require(peak_gpu <= 1, "semantic GPU windows overlapped")
    return {"status": "complete" if resolved == 66 else "incomplete", "scheduled": 66, "resolved": resolved,
            "plan_sha256": digest(plan), "strata": strata, "rows": rows, "blocks": blocks, "reviews": reviews,
            "services": list(state["services"].values()), "timeline": timeline,
            "request_statuses": dict(Counter(r["status"] for r in receipts)),
            "dispatched_requests": sum(bool(r.get("dispatched_utc")) for r in receipts),
            "reported_output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in receipts),
            "usage_missing_dispatched": sum(bool(r.get("dispatched_utc")) and r.get("usage") is None for r in receipts),
            "peak_model_ports": peak_model, "independent_gpu_windows": len(windows), "peak_gpu_windows": peak_gpu,
            "shared_initial_and_review_verified": True, "policy_request_binding_verified": True,
            "final_selection_binding_verified": True, "scope": plan["scope"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
