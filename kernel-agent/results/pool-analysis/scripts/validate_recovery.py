"""CPU fault injection: real SIGKILL + actual Humanize flow with fake model/GPU drivers."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import Campaign, atomic_json, digest
from kernel_agent.protocol import sha

SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def fixture_plan():
    return {"study_kind": "cpu_recovery_fault_injection",
            "jobs": [{"id": name, "task_id": "rmsnorm", "study_policy": "feedback"} for name in "ABC"],
            "per_job_budget": {"duration_seconds": 900, "output_tokens": 20000},
            "campaign_budget": {"duration_seconds": 2700, "output_tokens": 60000},
            "scope": "synthetic model/GPU drivers; these are not measured kernel successes"}


def fake_flow(job):
    from hmz.sdk import fakes

    sources = [SOURCE + f"# {job['id']} proposal {i}\n" for i in range(3)]
    responses = [json.dumps({"draft": "fixture", "plan": "fixture", "code_lines": source.splitlines()})
                 for source in sources]
    evaluations = iter(sources + [sources[0], sources[1], sources[2]])

    def evaluate(command, env):
        source = next(evaluations)
        result = {"status": "correct", "checks_passed": 28, "source_sha256": sha(source),
                  "timings": [{"shape": [n], "compiled_us": 1.2, "candidate_us": 1} for n in (1, 2, 3)]}
        return 0, json.dumps(result), ""

    return asyncio.run(fakes.run_fake(str(ROOT / "flows/kernel_loop.py"), "synthetic contract",
                       agents={"builder": fakes.FakeAgentDriver(reply=responses)},
                       local=fakes.FakeEnvDriver(run=evaluate),
                       params={"task_id": job["task_id"], "study_policy": job["study_policy"],
                               "mode": "reference_guided", "proposal_format": "lines", "rounds": 3,
                               "selection_policy": "gate_first", "run_id": f"recovery-{job['id']}",
                               "evaluator": ["python", "synthetic-evaluator.py"]}))


def child(directory):
    def worker(job):
        with (directory / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"job_id": job["id"]}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if job["id"] == "B":
            print("EXTERNAL_CALL_IN_FLIGHT", flush=True)
            signal.pause()  # Parent sends SIGKILL after reading the durable running checkpoint.
            raise AssertionError("the uncertain worker must not return")
        return fake_flow(job)

    with Campaign(directory / "checkpoint.json", fixture_plan()) as campaign:
        summary = campaign.run(worker)
    print(json.dumps(summary), flush=True)


def validate(directory):
    """Return auditable fault evidence; never touch a model service or allocate CUDA."""
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(directory)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while True:
            timeout = max(0, deadline - time.monotonic())
            if not timeout or not select.select([process.stdout], [], [], timeout)[0]:
                raise RuntimeError("fault driver did not reach its external-call boundary")
            line = process.stdout.readline()
            if line.strip() == "EXTERNAL_CALL_IN_FLIGHT":
                break
            if not line:
                raise RuntimeError("fault driver exited early: " + process.stderr.read()[-2000:])
        before = json.loads((directory / "checkpoint.json").read_text())
        assert [r["status"] for r in before["jobs"]] == ["completed", "running", "pending"]
        assert before["reservations"]["output_tokens"] == 40000
        process.kill()
        _, errors = process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL, errors
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    resumed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    summary, again = json.loads(resumed.stdout), json.loads(repeated.stdout)
    after = json.loads((directory / "checkpoint.json").read_text())
    calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
    assert calls == [{"job_id": n} for n in "ABC"]  # A reused; B never retried; C starts exactly once.
    assert summary == again and summary["terminal"] == summary["scheduled"] == 3
    assert summary["interrupted_unknown"] == 1 and summary["correct"] == 2
    assert summary["reservations"] == fixture_plan()["campaign_budget"]
    assert [r["status"] for r in after["jobs"]] == ["completed", "interrupted_unknown", "completed"]
    assert before["jobs"][0]["result_sha256"] == after["jobs"][0]["result_sha256"]
    return {"status": "passed", "scope": "CPU synthetic model/GPU drivers; not new GPU/model trials",
            "humanize_version": importlib.metadata.version("hmz"), "signal": "SIGKILL",
            "killed_process_returncode": process.returncode, "worker_calls": calls,
            "before": before, "after": after, "summary": summary, "repeated_resume_summary": again,
            "assertions": {"committed_task_reused": True, "uncertain_task_not_replayed": True,
                           "unknown_outcome_kept_in_denominator": True, "allocations_not_reset": True,
                           "remaining_task_completed": True, "repeated_resume_idempotent": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/campaign-recovery-validation.json")
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    if args.output.exists():
        raise RuntimeError("refusing to overwrite recovery evidence; use a new output path")
    with tempfile.TemporaryDirectory(prefix="kernel-agent-recovery-") as temporary:
        evidence = validate(Path(temporary))
    names = ["src/kernel_agent/campaign.py", "src/kernel_agent/humanize_api.py", "src/kernel_agent/protocol.py",
             "src/kernel_agent/line_protocol.py", "src/kernel_agent/knowledge.py", "src/kernel_agent/references.py",
             "scripts/validate_recovery.py", "flows/kernel_loop.py", "tests/unit/test_campaign.py",
             "tests/integration/test_campaign_recovery.py"]
    snapshot = args.output.parent / "recovery-measured"
    hashes = {}
    for name in names:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError("refusing to alter an existing measured recovery snapshot")
        target.write_bytes(data)
    evidence.update(schema_version=1, captured_utc=datetime.now(UTC).isoformat(),
                    source_sha256=hashes, snapshot_directory="recovery-measured")
    evidence["evidence_sha256"] = digest(evidence)
    atomic_json(args.output, evidence)
    print(json.dumps({"status": evidence["status"], "output": str(args.output),
                      "assertions": evidence["assertions"]}, indent=2))


if __name__ == "__main__":
    main()
