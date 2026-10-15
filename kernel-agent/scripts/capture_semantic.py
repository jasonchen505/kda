"""Capture atomic semantic-study ledgers and bound model/evaluator evidence."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from summarize_semantic import analyze


def capture(origin, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = (origin / "checkpoint.json").read_bytes()
    state = json.loads(raw)
    (output / "checkpoint.json").write_bytes(raw)
    shutil.copyfile(origin / "plan.json", output / "plan.json")
    shutil.copytree(origin / "sources", output / "sources", dirs_exist_ok=True)
    for item in state["blocks"]:
        source, target = origin / "blocks" / item["block"]["id"], output / "blocks" / item["block"]["id"]
        for p in (source / "requests").glob("*.json"):
            if p.name.endswith((".request.json", ".response.json")) or p.name.startswith("."):
                continue
            raw_receipt = p.read_bytes()
            receipt = json.loads(raw_receipt)
            (target / "requests").mkdir(parents=True, exist_ok=True)
            if receipt.get("dispatched_utc"):
                wire = source / "requests" / f"{receipt['request_id']}.request.json"
                require_hash = hashlib.sha256(wire.read_bytes()).hexdigest()
                if require_hash != receipt["request_sha256"]:
                    raise ValueError("semantic request changed during capture")
                shutil.copyfile(wire, target / "requests" / wire.name)
            if receipt["status"] == "completed":
                response = source / "requests" / f"{receipt['request_id']}.response.json"
                shutil.copyfile(response, target / "requests" / response.name)
            (target / "requests" / p.name).write_bytes(raw_receipt)
        if item["status"] == "complete":
            target.mkdir(parents=True, exist_ok=True)
            for name in ["generation.json", "result.json", "review.json", "parent.json"]:
                if (source / name).exists():
                    shutil.copyfile(source / name, target / name)
            for name in ["attempts", "candidates", "final"]:
                if (source / name).exists():
                    shutil.copytree(source / name, target / name, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    summary = analyze(output)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.directory, args.output)
    print(json.dumps({k: result[k] for k in ["status", "resolved", "dispatched_requests", "peak_model_ports"]}))
