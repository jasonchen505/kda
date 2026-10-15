"""Refuse GPU timings when foreign CUDA processes are seen before/after evaluation."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def occupants(gpu):
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True, timeout=5)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not line.isdecimal() for line in lines):
        raise ValueError("unparseable CUDA PID inventory")
    return sorted({int(line) for line in lines})


def evaluate(command, gpu):
    audit = {"gpu_index": gpu, "method": "foreign CUDA PID checks before and after the evaluator",
             "scope": "boundary observations, not continuous monitoring or an exclusive hardware lease"}
    try:
        audit["before_utc"] = datetime.now(timezone.utc).isoformat()
        audit["before_pids"] = occupants(gpu)
        if audit["before_pids"]:
            return {"status": "resource_busy", "message": "GPU occupied; evaluator was not started",
                    "resource_audit": audit}
        result = subprocess.run(command, capture_output=True, text=True, timeout=165)
        audit["after_utc"] = datetime.now(timezone.utc).isoformat()
        audit["after_pids"] = occupants(gpu)
        if audit["after_pids"]:
            return {"status": "resource_contended", "message": "GPU interference observed; timings cannot qualify",
                    "resource_audit": audit,
                    "discarded_evaluation": json.loads(result.stdout) if result.returncode == 0 else
                    {"status": "evaluator_error", "message": (result.stdout + result.stderr)[-4000:]}}
        if result.returncode:
            return {"status": "evaluator_error", "message": (result.stdout + result.stderr)[-4000:],
                    "resource_audit": audit}
        value = json.loads(result.stdout)
        value["resource_audit"] = audit
        return value
    except subprocess.TimeoutExpired:
        return {"status": "evaluator_timeout", "message": "evaluator/occupancy query exceeded its timeout",
                "resource_audit": audit}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return {"status": "resource_monitor_error", "message": str(error)[-2000:], "resource_audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("task")
    parser.add_argument("source")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("evaluate_candidate.py")), args.task, args.source]
    if args.final:
        command.append("--final")
    print(json.dumps(evaluate(command, args.gpu_index), allow_nan=False))


if __name__ == "__main__":
    main()
