"""Kill a finalization worker before its child launch, then inspect durable recovery."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.deferred import TimingQueue  # noqa: E402


def child(path, marker):
    stored = json.loads(path.read_text())
    with TimingQueue(path, stored["plan"]) as queue:
        def interrupted(candidate, final):
            if not final:
                raise ValueError("committed search was unexpectedly replayed")
            atomic_json(marker, {"phase": "final_before_child_launch", "pid": os.getpid(),
                                 "selected_sha256": candidate["source_sha256"],
                                 "model_calls": 0, "search_evaluator_calls": 0, "final_child_calls": 0})
            signal.pause()
            raise RuntimeError("fault injection worker must be killed")
        queue.run(interrupted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.queue, args.receipt)
        return
    before = json.loads(args.queue.read_text())
    if (before["status"] != "final_pending" or not before["selection_committed"] or
            before["final"]["status"] != "queued" or
            any(row["status"] != "measured" for row in before["search"])):
        raise ValueError("fault injection requires a newly prepared online-search queue")
    marker = args.receipt.with_suffix(".barrier.json")
    if marker.exists() or args.receipt.exists():
        raise ValueError("fault injection is one-shot; use a new receipt path")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--queue", str(args.queue),
                             "--receipt", str(marker), "--child"])
    try:
        deadline = time.monotonic() + 20
        while not marker.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("fault injection worker failed to reach committed selection")
            time.sleep(.05)
        proc.kill()
        code = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    after = json.loads(args.queue.read_text())
    if code != -signal.SIGKILL or after["final"]["status"] != "running":
        raise ValueError("kill did not preserve the in-flight final evaluation state")
    if before["search"] != after["search"] or before["selected_sha256"] != after["selected_sha256"]:
        raise ValueError("search evidence or selection changed during finalization")
    # Reopening acquires the OS-released lock and validates all imported evidence.
    with TimingQueue(args.queue, after["plan"]):
        pass
    receipt = {"status": "kill_verified_resume_pending", "exit_code": code,
               "queue": str(args.queue), "barrier": json.loads(marker.read_text()),
               "before_state_sha256": before["state_sha256"], "after_state_sha256": after["state_sha256"],
               "scope": "real SIGKILL after durable selection, before evaluator child; not an in-flight GPU/model crash"}
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
