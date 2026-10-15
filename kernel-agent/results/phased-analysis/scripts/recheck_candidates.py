"""Repeat final evaluations of frozen candidate files without invoking the model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from kernel_agent.protocol import decision, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python"


def output_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def evaluate(task_id: str, candidate: Path, evaluator_python: str, devices: str, timeout: int) -> dict:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = devices
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [evaluator_python, str(ROOT / "scripts/evaluate_candidate.py"), task_id, str(candidate), "--final"],
        cwd=ROOT,
        capture_output=True,
        env=environment,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        return {"status": "evaluator_error", "exit_code": completed.returncode,
                "message": (completed.stderr or completed.stdout)[-4000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {"status": "evaluator_error", "message": f"invalid evaluator JSON: {error}"}


def summarize(evaluations: list[dict]) -> dict:
    decisions = [decision(item) for item in evaluations]
    valid = [item for item in decisions if item["correct"]]
    return {
        "runs": len(evaluations),
        "correct": sum(item["correct"] for item in decisions),
        "promoted": sum(item["promoted"] for item in decisions),
        "geomean_speedups": [item["geomean_speedup"] for item in valid],
        "worst_speedups": [item["worst_speedup"] for item in valid],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("candidates", nargs="+", help="frozen candidate files to compare")
    parser.add_argument("--repeat", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--cuda-visible-devices", default="6")
    parser.add_argument("--evaluator-python", default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target = output_path(args.output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {target}")
    records = []
    for raw_candidate in args.candidates:
        candidate = output_path(raw_candidate)
        source = candidate.read_text()
        expected = sha(source)
        evaluations = []
        for _ in range(args.repeat):
            measured = evaluate(args.task_id, candidate, args.evaluator_python,
                                args.cuda_visible_devices, args.timeout)
            if measured.get("status") == "correct" and measured.get("source_sha256") != expected:
                measured = {"status": "stale_evidence", "message": "candidate SHA mismatch"}
            evaluations.append(measured)
        records.append({"candidate": str(candidate.relative_to(ROOT)), "source_sha256": expected,
                        "evaluations": evaluations, "summary": summarize(evaluations)})
    evidence = {
        "schema_version": 1,
        "study_kind": "fixed_candidate_recheck",
        "executed_utc": datetime.now(UTC).isoformat(),
        "task_id": args.task_id,
        "repeat": args.repeat,
        "cuda_visible_devices": args.cuda_visible_devices,
        "scope": ("fresh evaluator processes for fixed existing sources; no model generation, "
                  "no candidate reselection, and no replacement of the original campaign"),
        "candidates": records,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps([{key: item["summary"][key] for key in ("runs", "correct", "promoted")}
                      for item in records], indent=2))


if __name__ == "__main__":
    main()
