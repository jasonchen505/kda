"""Measure under observed GPU sharing; never label the result as isolated speedup."""
import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


def observe(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu),
         "--query-gpu=memory.free,memory.used,utilization.gpu,clocks.sm,power.draw",
         "--format=csv,noheader,nounits"], text=True, timeout=5).strip().split(",")
    free, used, utilization, clock = map(int, output[:4])
    power = float(output[4])
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=5)
    pids = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not p.isdecimal() for p in pids):
        raise ValueError("unparseable CUDA PID inventory")
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "free_mib": free,
            "used_mib": used, "utilization_percent": utilization, "sm_clock_mhz": clock,
            "power_watts": power, "pids": sorted(set(map(int, pids)))}


def evaluate(command, gpu, *, pass_fds=()):
    audit = {"gpu_index": gpu, "policy": "shared_observed", "minimum_free_mib": 1024,
             "sample_period_seconds": .5, "samples": [],
             "scope": "observed coexecution; utilization includes evaluator work; no isolation or known-load claim"}
    stop, errors = threading.Event(), []
    thread = None
    try:
        audit["before"] = observe(gpu)
        if audit["before"]["free_mib"] < 1024:
            return {"status": "resource_exhausted", "resource_audit": audit}

        def sample():
            while not stop.wait(.5):
                try:
                    audit["samples"].append(observe(gpu))
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    errors.append(str(error)[-2000:])
                    return

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        child = subprocess.run(command, capture_output=True, text=True, timeout=165, pass_fds=pass_fds)
        stop.set()
        thread.join(timeout=12)
        audit["after"] = observe(gpu)
        if errors or thread.is_alive():
            return {"status": "resource_monitor_error", "message": errors or ["sampler did not finish"],
                    "resource_audit": audit}
        if child.returncode:
            return {"status": "evaluator_error", "message": (child.stdout + child.stderr)[-4000:],
                    "resource_audit": audit}
        value = json.loads(child.stdout)
        value.update(measurement_condition="shared_observed", resource_audit=audit)
        return value
    except subprocess.TimeoutExpired:
        return {"status": "evaluator_timeout", "resource_audit": audit}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return {"status": "resource_monitor_error", "message": str(error)[-2000:], "resource_audit": audit}
    finally:
        stop.set()
        if thread:
            thread.join(timeout=12)


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
