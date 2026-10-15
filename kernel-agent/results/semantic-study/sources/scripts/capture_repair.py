"""Export atomic study checkpoints and only hash-bound request/response records."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from summarize_repair import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    origin, output = args.directory, args.output
    output.mkdir(parents=True, exist_ok=True)
    raw = (origin / "checkpoint.json").read_bytes()
    state = json.loads(raw)
    (output / "checkpoint.json").write_bytes(raw)
    shutil.copyfile(origin / "plan.json", output / "plan.json")
    shutil.copytree(origin / "sources", output / "sources", dirs_exist_ok=True)
    for item in state["blocks"]:
        source = origin / "blocks" / item["block"]["id"]
        target = output / "blocks" / item["block"]["id"]
        for path in (source / "requests").glob("*.json"):
            if path.name.endswith((".request.json", ".response.json")) or path.name.startswith("."):
                continue
            raw_receipt = path.read_bytes()
            receipt = json.loads(raw_receipt)
            (target / "requests").mkdir(parents=True, exist_ok=True)
            if receipt.get("dispatched_utc"):
                wire = source / "requests" / f"{receipt['request_id']}.request.json"
                body = wire.read_bytes()
                if hashlib.sha256(body).hexdigest() != receipt["request_sha256"]:
                    raise ValueError("request changed during capture")
                (target / "requests" / wire.name).write_bytes(body)
            if receipt["status"] == "completed":
                response = source / "requests" / f"{receipt['request_id']}.response.json"
                shutil.copyfile(response, target / "requests" / response.name)
            (target / "requests" / path.name).write_bytes(raw_receipt)
        if item["status"] == "complete":
            for name in ["generation.json", "result.json"]:
                shutil.copyfile(source / name, target / name)
            for name in ["attempts", "candidates", "final"]:
                if (source / name).exists():
                    shutil.copytree(source / name, target / name, dirs_exist_ok=True)
    summary = analyze(output)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "resolved": summary["resolved"],
                      "calls": summary["dispatched_requests"], "peak_ports": summary["peak_model_ports"]}))


if __name__ == "__main__":
    main()
