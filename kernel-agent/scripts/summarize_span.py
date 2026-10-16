"""Reconstruct indexed-patch study from server-only raw evidence; no model/GPU calls."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent import line_protocol, operational, repair, span_repair  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.transport import LINE_SCHEMA, PROPOSER_PREFIX  # noqa: E402
from scripts.summarize_campaign import digest, require, same_decision  # noqa: E402
from scripts.summarize_repair import task_cluster_interval  # noqa: E402
from scripts.summarize_semantic import INFRA  # noqa: E402

STATIC_FAILURES = {"protocol_error", "patch_error", "api_error"}


def analyze(folder):
    folder = Path(folder)
    def load(p):
        return json.loads(p.read_text())
    plan, state = load(folder / "plan.json"), load(folder / "checkpoint.json")
    require(state["plan_sha256"] == digest(plan) and state["state_sha256"] == digest({k: v for k, v in state.items()
            if k != "state_sha256"}), "operational ledger changed")
    require([b["block"] for b in state["blocks"]] == plan["blocks"] and len(plan["blocks"]) == 22
            and plan["arms"] == list(span_repair.ARMS) and plan["max_model_calls"] == 140,
            "operational denominators changed")
    for name, expected in plan["source_sha256"].items():
        require(hashlib.sha256((folder / "sources" / name).read_bytes()).hexdigest() == expected,
                "operational executed-source hash changed: " + name)
    allocated = sum(b["block"]["model_call_cap"] for b in state["blocks"] if b["status"] != "pending")
    require(state["allocations"] == {"model_calls": allocated, "output_tokens": 2048 * allocated}, "operational budget reset")
    sources = folder / "sources"
    specs = {**load(sources / "data/tasks.json"), **load(sources / "data/repair-transfer-tasks.json")}
    capability = load(sources / "data/runtime-triton-api.json")
    bank = load(sources / "data/semantic-failure-bank.json")
    rows, blocks, receipts, timeline, windows = [], [], [], [], set()
    patch_counts, api_counts, code_statuses, routes = Counter(), Counter(), Counter(), Counter()

    def gpu(value, block_id, phase, committed=None):
        if value.get("status") in INFRA | STATIC_FAILURES:
            return
        binding, audit = value.get("device_binding", {}), value.get("resource_audit", {})
        actual = binding.get("actual", {})
        require(binding.get("verified") is True and binding.get("expected_gpu_index") == 4
                and actual.get("device_count") == 1 and actual.get("device_index") == 0
                and actual.get("visible_devices") == binding.get("expected_uuid") == audit.get("before", {}).get("gpu_uuid")
                and str(actual.get("gpu_uuid", "")).lower().removeprefix("gpu-") ==
                str(binding.get("expected_uuid", "")).lower().removeprefix("gpu-"), "operational physical GPU proof missing")
        queue = value.get("cooperative_queue", {})
        require(queue.get("lock_file") == plan["runtime"]["evaluation_lock_file"], "operational GPU lock changed")
        before, after = audit["before"]["observed_utc"], audit["after"]["observed_utc"]
        require(not committed or before >= committed, "operational final accessed before both selections")
        key = before, after, queue.get("pid")
        if key not in windows:
            windows.add(key)
            timeline.append({"lane": "GPU4", "block": block_id, "phase": phase, "start_utc": before, "end_utc": after})

    for item in state["blocks"]:
        b, result = item["block"], None
        path = folder / "blocks" / b["id"]
        by_id = {}
        for p in (sorted((path / "requests").glob("*.json")) if item["status"] != "pending" else []):
            if p.name.endswith((".request.json", ".response.json")) or p.name.startswith("."):
                continue
            r = load(p)
            require("request_id" in r and p.name == r["request_id"] + ".json" and r["request_id"] not in by_id,
                    "unrecognized or duplicate request receipt")
            by_id[r["request_id"]] = r
            receipts.append(r)
            if r.get("dispatched_utc"):
                wire = path / "requests" / (r["request_id"] + ".request.json")
                require(hashlib.sha256(wire.read_bytes()).hexdigest() == r["request_sha256"], "operational wire request changed")
                body = load(wire)
                call = r["request_id"]
                seed = b["seed"] + (0 if call == "initial" else 1 if call.endswith("r1") else 2)
                require(body["seed"] == seed and body["max_tokens"] == 2048 and body["temperature"] == .2
                        and body["stream"] is False and not body.get("tools")
                        and body["chat_template_kwargs"] == {"enable_thinking": False}
                        and r["input_tokens"] + 2048 <= 8192 and not any(r["admission"].values())
                        and r["endpoint"] == item["endpoint"] and body["model"] == item["endpoint"]["model"],
                        "operational call settings/process changed")
            if r["status"] == "completed":
                response = load(path / "requests" / (r["request_id"] + ".response.json"))
                require(digest(response) == r["response_sha256"], "operational response changed")
                timeline.append({"lane": ":" + str(r["endpoint"]["port"]), "block": b["id"], "phase": "model",
                                 "start_utc": r["dispatched_utc"], "end_utc": r["finished_utc"]})
        if item["status"] == "complete":
            result, committed = load(path / "result.json"), load(path / "generation.json")
            require(digest(result) == item["result_sha256"] and result["endpoint"] == item["endpoint"]
                    and result["initial"] == committed["initial"], "operational committed result changed")
            require(result["model_calls"] == sum(bool(r.get("dispatched_utc")) for r in by_id.values())
                    <= b["model_call_cap"], "operational dispatch accounting changed")
            start = result["initial"]
            if b["stratum"] == "regression":
                parent = load(path / "parent.json")
                require(parent == next(c for c in bank["cases"] if c["id"] == b["parent_id"])
                        and start.get("imported") and start["reply"] == parent["initial"]["reply"]
                        and start["source_sha256"] == parent["source_sha256"], "operational imported source changed")
            unique = {}
            for arm, value in result["arms"].items():
                prior = committed["arms"][arm]
                require(value["attempts"] == prior["attempts"] and value["selected_sha256"] == prior["selected_sha256"]
                        and value["attempts"][0] == start and value == load(path / "final" / (arm + ".json")),
                        "operational shared start or final selection changed")
                require(value["model_opportunities"] == [f"{arm}-r1", f"{arm}-r2"] and len(value["attempts"]) == 3,
                        "operational arm opportunity count changed")
                selected = repair.select(value["attempts"])
                require(value["selected_sha256"] == (selected["source_sha256"] if selected else None), "operational selector changed")
                for index, attempt in enumerate(value["attempts"]):
                    history = value["attempts"][:index]
                    mode, parent = span_repair.assignment(arm, history)
                    patching = index > 0 and mode in {"patch", "span_patch"}
                    require(attempt["parent_sha256"] == (parent.get("source_sha256") if parent else None), "operational lineage changed")
                    require(attempt["mode"] == ("imported" if attempt.get("imported") else mode),
                            "operational patch/recovery mode changed")
                    unique[attempt["request_id"]] = attempt
                    if "reply" in attempt:
                        if not attempt.get("imported"):
                            r = by_id[attempt["request_id"]]
                            response = load(path / "requests" / (attempt["request_id"] + ".response.json"))
                            require(r["status"] == "completed" and response["choices"][0]["message"]["content"] == attempt["reply"],
                                    "operational proposal differs from raw response")
                        try:
                            decoded = (span_repair.apply(attempt["reply"], parent["code"]) if mode == "span_patch" and patching else
                                       operational.apply(attempt["reply"], parent["code"]) if patching else
                                       line_protocol.proposal(attempt["reply"]))
                            admit(decoded["code"])
                        except (ValueError, SyntaxError, TypeError, KeyError):
                            require(attempt["evaluation"]["status"] in {"patch_error", "protocol_error"}, "unrecorded source rejection")
                        else:
                            require(attempt.get("code") == decoded["code"] and attempt.get("source_sha256") == admit(decoded["code"]),
                                    "model-authored code or patch application changed")
                            if patching:
                                require(attempt["transaction"] == decoded["transaction"], "operational patch transaction altered")
                    if attempt.get("source_sha256"):
                        require(hashlib.sha256((path / "candidates" / (attempt["source_sha256"] + ".py")).read_bytes()).hexdigest()
                                == attempt["source_sha256"], "operational exact source file changed")
                        expected_api = operational.api_audit(attempt["code"], capability)
                        require(expected_api == attempt["api_audit"], "operational installed API audit changed")
                        if expected_api["status"] == "api_error":
                            require(attempt["evaluation"] == {**expected_api, "source_sha256": attempt["source_sha256"]},
                                    "static API rejection replaced by GPU evidence")
                    gpu(attempt["evaluation"], b["id"], "search")
                    if not attempt.get("imported") and attempt["request_id"] in by_id:
                        r = by_id[attempt["request_id"]]
                        if r.get("dispatched_utc"):
                            parts, settings = span_repair.context(b["task_id"], specs[b["task_id"]]["contract"], history, capability, mode)
                            omitted = r["omitted_sections"]
                            require(all(not mandatory for name, _, mandatory in parts if name in omitted), "mandatory operational context removed")
                            expected = "\n\n".join(text for name, text, _ in parts if name not in omitted)
                            body = load(path / "requests" / (attempt["request_id"] + ".request.json"))
                            schema = settings.get("schema", LINE_SCHEMA)
                            prefix = settings.get("role_prefix", PROPOSER_PREFIX)
                            require(body["messages"] == [{"role": "system", "content": prefix + " No tools. Follow the supplied contract."},
                                    {"role": "user", "content": expected}] and body["response_format"]["json_schema"]["schema"] == schema,
                                    "operational request differs from assigned policy")
                            if mode != "full_source":
                                require(attempt["diagnosis"] == operational.diagnose(parent["evaluation"]), "repair obligation changed")
                verdict = decision(value["final"] or {})
                require((value["correct"], value["promoted"]) == (verdict["correct"], verdict["promoted"]), "operational headline changed")
                if value["final"]:
                    require(same_decision(verdict, value["final_decision"]) and (not verdict["correct"] or value["final"].get("source_sha256") == value["selected_sha256"]),
                            "operational final source/decision changed")
                    gpu(value["final"], b["id"], "final", result["generation_committed_utc"])
                infra = bool(result["infrastructure_stop"]) or any(a["evaluation"]["status"] in INFRA for a in value["attempts"]) or (value["final"] or {}).get("status") in INFRA
                rows.append({"block": b["id"], "stratum": b["stratum"], "task_id": b["task_id"], "arm": arm,
                             "status": "complete", "correct": value["correct"], "promoted": value["promoted"],
                             "initial_correct": decision(start["evaluation"])["correct"], "infrastructure": infra,
                             "fully_observed": all(by_id.get(k, {}).get("status") == "completed" for k in value["model_opportunities"])})
            for attempt in unique.values():
                code_statuses[attempt["evaluation"]["status"]] += 1
                if attempt["mode"] in {"patch", "span_patch"}:
                    patch_counts[(attempt["request_id"].rsplit("-r", 1)[0], "applied" if "transaction" in attempt else attempt["evaluation"]["status"])] += 1
                    if "diagnosis" in attempt:
                        routes[attempt["diagnosis"]["route"]] += 1
                api_counts.update(attempt.get("api_audit", {}).get("missing_language_symbols", []))
        else:
            for arm in span_repair.ARMS:
                rows.append({"block": b["id"], "stratum": b["stratum"], "task_id": b["task_id"], "arm": arm,
                             "status": item["status"], "correct": False, "promoted": False, "initial_correct": False,
                             "infrastructure": item["status"] in {"failed", "interrupted_unknown"}, "fully_observed": False})
        blocks.append({"block": b, "status": item["status"], "endpoint": item.get("endpoint"), "result": result})
    terminal = {"complete", "failed", "interrupted_unknown"}
    resolved = sum(r["status"] in terminal for r in rows)
    strata = []
    for label, declared in plan["strata"].items():
        selected = [r for r in rows if r["stratum"] == label]
        cells = [{"arm": arm, "scheduled": declared["blocks"],
                  "resolved": sum(r["status"] in terminal for r in selected if r["arm"] == arm),
                  "correct": sum(r["correct"] for r in selected if r["arm"] == arm),
                  "promoted": sum(r["promoted"] for r in selected if r["arm"] == arm),
                  "initial_failed_observed": sum(not r["initial_correct"] and r["fully_observed"] and not r["infrastructure"] for r in selected if r["arm"] == arm),
                  "rescued_observed": sum(not r["initial_correct"] and r["fully_observed"] and not r["infrastructure"] and r["correct"] for r in selected if r["arm"] == arm)} for arm in span_repair.ARMS]
        contrasts = {}
        if resolved == 66:
            for key, new, baseline in [("span_vs_operational", "span_repair", "operational_patch"),
                                       ("span_vs_rewrite", "span_repair", "rewrite"),
                                       ("operational_vs_rewrite", "operational_patch", "rewrite")]:
                diffs = []
                for task in sorted({r["task_id"] for r in selected}):
                    group = [r for r in selected if r["task_id"] == task]
                    n = sum(r["arm"] == baseline for r in group)
                    diffs.append((sum(r["correct"] for r in group if r["arm"] == new) -
                                  sum(r["correct"] for r in group if r["arm"] == baseline)) / n)
                require(len(diffs) == declared["families"], "indexed study task clusters changed")
                contrasts[key] = {"mean_difference": sum(diffs) / len(diffs), "interval": task_cluster_interval(diffs)}
        strata.append({"stratum": label, "cells": cells, "contrasts": contrasts})
    timeline.sort(key=lambda t: (t["start_utc"], t["end_utc"], t["lane"], t["block"], t["phase"]))
    active, model_peak, gpu_count, gpu_peak = Counter(), 0, 0, 0
    for _, delta, lane in sorted([(t["start_utc"], 1, t["lane"]) for t in timeline] + [(t["end_utc"], -1, t["lane"]) for t in timeline], key=lambda t: (t[0], t[1])):
        if lane == "GPU4":
            gpu_count += delta
            gpu_peak = max(gpu_peak, gpu_count)
        else:
            active[lane] += delta
            model_peak = max(model_peak, sum(v > 0 for v in active.values()))
    require(gpu_peak <= 1, "operational GPU windows overlap")
    usage = {}
    for receipt in receipts:
        arm = "initial" if receipt["request_id"] == "initial" else receipt["request_id"].rsplit("-r", 1)[0]
        value = usage.setdefault(arm, {"dispatched": 0, "reported_output_tokens": 0})
        value["dispatched"] += bool(receipt.get("dispatched_utc"))
        value["reported_output_tokens"] += (receipt.get("usage") or {}).get("completion_tokens", 0)
    return {"status": "complete" if resolved == 66 else "incomplete", "scheduled": 66, "resolved": resolved,
            "plan_sha256": digest(plan), "strata": strata, "rows": rows, "blocks": blocks,
            "services": list(state["services"].values()), "timeline": timeline,
            "dispatched_requests": sum(bool(r.get("dispatched_utc")) for r in receipts),
            "request_statuses": dict(Counter(r["status"] for r in receipts)),
            "reported_output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in receipts),
            "usage_missing_dispatched": sum(bool(r.get("dispatched_utc")) and r.get("usage") is None for r in receipts),
            "arm_usage": usage, "patch_statuses": {arm: {status: count for (a, status), count in patch_counts.items() if a == arm}
                               for arm in span_repair.ARMS}, "repair_routes": dict(routes), "missing_apis": dict(api_counts),
            "code_opportunity_statuses": dict(code_statuses), "peak_model_ports": model_peak,
            "independent_gpu_windows": len(windows), "peak_gpu_windows": gpu_peak,
            "shared_initial_verified": True, "patch_reply_to_code_verified": True,
            "policy_request_binding_verified": True, "final_selection_binding_verified": True, "scope": plan["scope"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
