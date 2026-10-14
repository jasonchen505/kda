"""Search numerical checks may share a GPU; no performance verdict is produced."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def observe(gpu):
    values = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=5).strip().split(",")
    free, used, utilization = map(int, values)
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=5)
    pids = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not p.isdecimal() for p in pids):
        raise ValueError("unparseable CUDA process inventory")
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "free_mib": free,
            "used_mib": used, "utilization_percent": utilization, "pids": sorted(set(map(int, pids)))}


def evaluate(command, gpu):
    audit = {"gpu_index": gpu, "policy": "shared_search_correctness", "minimum_free_mib": 1024,
             "scope": "occupancy observations; no timing, isolation or speedup claim"}
    try:
        audit["before"] = observe(gpu)
        if audit["before"]["free_mib"] < audit["minimum_free_mib"]:
            return {"status": "resource_exhausted", "message": "insufficient free GPU memory",
                    "resource_audit": audit}
        child = subprocess.run(command, capture_output=True, text=True, timeout=165)
        audit["after"] = observe(gpu)
        if child.returncode:
            return {"status": "evaluator_error", "message": (child.stdout + child.stderr)[-4000:],
                    "resource_audit": audit}
        result = json.loads(child.stdout)
        if result.get("status") == "correct" or result.get("timings") is not None:
            raise ValueError("shared evaluator cannot return timing evidence")
        result["resource_audit"] = audit
        return result
    except subprocess.TimeoutExpired:
        return {"status": "evaluator_timeout", "message": "shared numerical check timed out",
                "resource_audit": audit}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return {"status": "resource_monitor_error", "message": str(error)[-2000:], "resource_audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("task")
    parser.add_argument("source")
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("evaluate_candidate.py")), args.task,
               args.source, "--correctness-only"]
    print(json.dumps(evaluate(command, args.gpu_index), allow_nan=False))


if __name__ == "__main__":
    main()
