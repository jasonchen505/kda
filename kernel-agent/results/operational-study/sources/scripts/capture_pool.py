"""Capture atomic worker states, frozen sources and unedited service-pool evidence."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.summarize_pool import analyze  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.directory / "plan.json", output / "plan.json")
    if (args.directory / "workers-exit.json").exists():
        shutil.copyfile(args.directory / "workers-exit.json", output / "workers-exit.json")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", ".DS_Store", "checkpoint.lock")
    shutil.copytree(args.directory / "sources", output / "sources", dirs_exist_ok=True, ignore=ignore)
    if (args.directory / "handoff").exists():
        (output / "handoff").mkdir(exist_ok=True)
        for path in (args.directory / "handoff").iterdir():
            if path.suffix in {".json", ".py"}:
                shutil.copyfile(path, output / "handoff" / path.name)
    plan = json.loads((output / "plan.json").read_text())
    for worker in plan["workers"]:
        original = Path(worker["directory"])
        target = output / "workers" / f"p{worker['port']}"
        target.mkdir(parents=True, exist_ok=True)
        raw = (original / "generation.json").read_bytes()
        state = json.loads(raw)
        (target / "generation.json").write_bytes(raw)
        shutil.copyfile(original / "plan.json", target / "plan.json")
        for receipt_path in sorted((original / "wire").glob("*/admission/*.json")):
            raw_receipt = receipt_path.read_bytes()
            receipt = json.loads(raw_receipt)
            destination = target / receipt_path.relative_to(original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if receipt.get("dispatched_utc"):
                wire = receipt_path.parent.parent / receipt_path.name
                raw_wire = wire.read_bytes()
                if hashlib.sha256(raw_wire).hexdigest() != receipt["wire_sha256"]:
                    raise ValueError("request changed while capturing")
                (destination.parent.parent / wire.name).write_bytes(raw_wire)
            destination.write_bytes(raw_receipt)
        for queue_path in sorted((original / "final").glob("*/checkpoint.json")):
            raw_queue = queue_path.read_bytes()
            queue = json.loads(raw_queue)
            destination = target / queue_path.relative_to(original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw_queue)
            for row in queue["search"]:
                name = f"{row['source_sha256']}.py"
                shutil.copyfile(queue_path.parent / name, destination.parent / name)
        for row in state["jobs"]:
            if row["status"] not in {"completed", "failed", "interrupted_unknown"}:
                continue
            run_id = row.get("result", {}).get("run_id", f"campaign-{state['plan_sha256'][:12]}-{row['job']['id']}")
            run = Path(state["plan"]["runtime"]["workspace"]) / "runs" / run_id
            if run.exists():
                shutil.copytree(run, target / "runs" / run_id, dirs_exist_ok=True, ignore=ignore)
    summary = analyze(output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "resolved": summary["resolved"],
                      "scheduled": 24, "peak_concurrent_ports": summary["peak_overlapping_proposal_ports"]}))


if __name__ == "__main__":
    main()
