"""Export a self-contained offline snapshot of committed shared-study evidence."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.summarize_campaign import summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    raw_checkpoint = args.checkpoint.read_text()
    state = json.loads(raw_checkpoint)
    sources = args.checkpoint.parent / "sources"
    analysis = summarize(state, sources)
    output = args.output_root
    frozen = output / "results/shared-feedback-measured"
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sources, frozen, dirs_exist_ok=True)
    workspace = Path(state["plan"]["runtime"]["workspace"])
    for row in state["jobs"]:
        if row["status"] not in {"completed", "failed", "interrupted_unknown"}:
            continue
        job_id = row["job"]["id"]
        run_id = row.get("result", {}).get("run_id", f"campaign-{state['plan_sha256'][:12]}-{job_id}")
        run = workspace / "runs" / run_id
        if run.exists():
            shutil.copytree(run, frozen / "runs" / run_id, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", ".DS_Store"))
        wire = args.checkpoint.parent / "wire" / job_id
        if wire.exists():
            shutil.copytree(wire, output / "evidence/shared-feedback-requests" / job_id, dirs_exist_ok=True)
    (output / "results/shared-feedback-checkpoint.json").write_text(
        raw_checkpoint)  # Export the same atomic read used for analysis, preserving its bytes.
    (output / "results/shared-feedback-summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": analysis["status"], "terminal": analysis["terminal"],
                      "scheduled": analysis["scheduled"], "measurement_condition": analysis["measurement_condition"]}))


if __name__ == "__main__":
    main()
