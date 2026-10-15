"""Serialize this project's GPU work while retaining observed-sharing telemetry."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.evaluate_coexecuted import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("task")
    parser.add_argument("source")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / "scripts/evaluate_candidate.py"), args.task, args.source]
    if args.final:
        command.append("--final")
    if args.diagnostics:
        command.append("--diagnostics")
    try:
        with acquire(args.lock_file) as (lock, waited):
            result = evaluate(command, args.gpu_index, pass_fds=(lock.fileno(),))
            result["cooperative_queue"] = {"lock_file": str(args.lock_file), "wait_seconds": waited,
                                           "pid": os.getpid(), "scope": "serializes this project's workers only"}
    except TimeoutError as error:
        result = {"status": "resource_busy", "message": str(error)}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
