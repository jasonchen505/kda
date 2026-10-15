"""Export a completed engineering pilot without touching its recorded evidence."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-directory", type=Path, required=True)
    parser.add_argument("--final-directory", type=Path, required=True)
    parser.add_argument("--service-launch", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path)
    parser.add_argument("--initial-final-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.generation_directory / "generation.json").read_bytes()
    state = json.loads(raw)
    receipt_path = args.completion_receipt or args.generation_directory / "supervisor-exit.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "complete" or any(row["status"] != "completed" for row in state["jobs"]):
        raise ValueError("only export a fully completed engineering pilot")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "generation.json").write_bytes(raw)
    for name in ("supervisor.py", "search-summary.json"):
        shutil.copyfile(args.generation_directory / name, args.output / name)
    shutil.copyfile(receipt_path, args.output / "supervisor-exit.json")
    if args.completion_receipt:
        shutil.copyfile(args.generation_directory / "supervisor-exit.json", args.output / "initial-supervisor-exit.json")
        shutil.copyfile(args.completion_receipt.parent / "supervisor.py", args.output / "completion-supervisor.py")
    ignore = shutil.ignore_patterns("checkpoint.lock", "generation.json.lock", "__pycache__", "*.pyc", "*.log", ".DS_Store")
    for name in ("sources", "wire"):
        shutil.copytree(args.generation_directory / name, args.output / name, ignore=ignore)
    for row in state["jobs"]:
        run = row["result"]
        shutil.copytree(Path(state["plan"]["runtime"]["workspace"]) / "runs" / run["run_id"],
                        args.output / "runs" / run["run_id"], ignore=ignore)
    shutil.copytree(args.final_directory, args.output / "final", ignore=ignore)
    if args.initial_final_directory:
        shutil.copytree(args.initial_final_directory, args.output / "initial-final-preparation", ignore=ignore)
    shutil.copyfile(args.service_launch, args.output / "service-launch.json")
    print(json.dumps({"status": "exported", "jobs": len(state["jobs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
