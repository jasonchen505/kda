"""Durable task-boundary scheduling; uncertain external calls are never auto-replayed."""
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from kernel_agent.protocol import decision

TERMINAL = {"completed", "failed", "interrupted_unknown"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic_json(path, value):
    """The old or new complete checkpoint survives; never an in-place partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_result(result, job):
    """A coordinator cannot replace the inner controller's independent final verdict."""
    if result.get("task_id") != job["task_id"] or result.get("study_policy") != job["study_policy"]:
        raise ValueError("worker result belongs to another job")
    if "seed" in job and (result.get("seed") != job["seed"] or result.get("job_id") != job["id"]):
        raise ValueError("worker result seed/job identity mismatch")
    actual = decision(result.get("final_evaluation") or {})
    if (any(type(result.get(k)) is not bool for k in ("correct", "promoted"))
            or (result.get("correct"), result.get("promoted")) != (actual["correct"], actual["promoted"])):
        raise ValueError("worker headline conflicts with final evaluation")
    if actual["correct"]:
        selected = result.get("selected_sha256")
        matching = [a for a in result.get("attempts", []) if a.get("source_sha256") == selected]
        if not selected or not matching or result["final_evaluation"].get("source_sha256") != selected:
            raise ValueError("final evaluation is not bound to the selected proposal")
        if hashlib.sha256(matching[0].get("code", "").encode()).hexdigest() != selected:
            raise ValueError("selected proposal SHA mismatch")
    return result


class Campaign:
    """One immutable plan, one filesystem owner, conservative per-job budget allocations."""

    def __init__(self, path, plan):
        self.path, self.plan = Path(path), json.loads(encoded(plan))
        plan = self.plan
        jobs = plan["jobs"]
        if not jobs or len({j["id"] for j in jobs}) != len(jobs):
            raise ValueError("schedule needs nonempty unique job IDs")
        for budgets in (plan["per_job_budget"], plan["campaign_budget"]):
            if set(budgets) != {"duration_seconds", "output_tokens"} or any(
                    type(v) is not int for v in budgets.values()):
                raise ValueError("budget allocations must be integer seconds and output tokens")
        if any(plan["per_job_budget"][k] <= 0 for k in ("duration_seconds", "output_tokens")):
            raise ValueError("per-job budgets must be positive")
        if any(plan["campaign_budget"][k] < 0 for k in ("duration_seconds", "output_tokens")):
            raise ValueError("campaign budgets cannot be negative")
        self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                self._verify()
            else:
                self.state = {"schema_version": 1, "status": "prepared", "plan": self.plan,
                              "plan_sha256": digest(self.plan), "created_utc": datetime.now(UTC).isoformat(),
                              "reservations": {"duration_seconds": 0, "output_tokens": 0},
                              "jobs": [{"job": job, "status": "pending"} for job in self.plan["jobs"]],
                              "events": []}
                self._event("prepared")
                self._save()
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def __exit__(self, *args):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        self.lock = None

    def _event(self, kind, job_id=None, **detail):
        events = self.state["events"]
        event = {"sequence": len(events), "utc": datetime.now(UTC).isoformat(), "kind": kind,
                 "job_id": job_id, "previous_sha256": events[-1]["sha256"] if events else None,
                 **detail}
        event["sha256"] = digest(event)
        events.append(event)

    def _save(self):
        # Whole-state digest detects accidental edits, not an adversarial signature.
        self.state.pop("state_sha256", None)
        self.state["state_sha256"] = digest(self.state)
        atomic_json(self.path, self.state)

    def _verify(self):
        value = {k: v for k, v in self.state.items() if k != "state_sha256"}
        if self.state.get("state_sha256") != digest(value):
            raise ValueError("checkpoint integrity mismatch")
        if self.state["plan_sha256"] != digest(self.plan) or self.state["plan"] != self.plan:
            raise ValueError("resume plan or runtime/source provenance changed")
        if [r["job"] for r in self.state["jobs"]] != self.plan["jobs"]:
            raise ValueError("schedule changed")
        for index, event in enumerate(self.state["events"]):
            expected_previous = self.state["events"][index - 1]["sha256"] if index else None
            body = {k: v for k, v in event.items() if k != "sha256"}
            if (event["sequence"] != index or event["previous_sha256"] != expected_previous
                    or event["sha256"] != digest(body)):
                raise ValueError("event chain changed")
        started = 0
        for row in self.state["jobs"]:
            if row["status"] not in {"pending", "running", *TERMINAL}:
                raise ValueError("unknown job status")
            started += row["status"] != "pending"
            if row["status"] == "completed":
                if digest(row["result"]) != row["result_sha256"]:
                    raise ValueError("committed result changed")
                validate_result(row["result"], row["job"])
        if self.state["reservations"] != {k: started * v for k, v in self.plan["per_job_budget"].items()}:
            raise ValueError("budget allocations changed")
        if (sum(r["status"] == "running" for r in self.state["jobs"]) > 1
                or any(v > self.plan["campaign_budget"][k] for k, v in self.state["reservations"].items())):
            raise ValueError("invalid concurrent execution or over-allocation")

    def run(self, worker, *, ready=None, max_new_jobs=None):
        if self.lock is None:
            raise RuntimeError("campaign execution requires an active ownership lock")
        self._verify()
        if max_new_jobs is not None and max_new_jobs < 0:
            raise ValueError("max_new_jobs cannot be negative")
        # SIGKILL cannot prove whether the external request finished. Keep its reservation
        # and denominator, do not silently replay it or invent a missing answer.
        for row in self.state["jobs"]:
            if row["status"] == "running":
                row["status"] = "interrupted_unknown"
                row["error"] = "previous process ended without a committed result; automatic replay disabled"
                self._event("recovered_unknown", row["job"]["id"])
        self._save()
        started = 0
        for row in self.state["jobs"]:
            if row["status"] != "pending":
                continue
            if max_new_jobs is not None and started >= max_new_jobs:
                self.state["status"] = "paused_at_task_boundary"
                break
            allocation = self.plan["per_job_budget"]
            if any(self.state["reservations"][k] + v > self.plan["campaign_budget"][k]
                   for k, v in allocation.items()):
                self.state["status"] = "allocation_budget_exhausted"
                break
            try:
                if ready:
                    ready(row["job"])
            except Exception as error:
                self.state["status"] = "awaiting_resources"
                self._event("resource_blocked", row["job"]["id"], error=str(error)[-2000:])
                break
            for key, value in allocation.items():
                self.state["reservations"][key] += value
            row["status"] = "running"
            self._event("started", row["job"]["id"], reserved=allocation)
            self._save()  # Persist reservation before making any external request.
            try:
                result = validate_result(json.loads(encoded(worker(row["job"]))), row["job"])
                row.update(status="completed", result=result, result_sha256=digest(result))
                self._event("committed", row["job"]["id"], result_sha256=row["result_sha256"])
            except Exception as error:
                row.update(status="failed", error=str(error)[-4000:])
                self._event("failed", row["job"]["id"], error=row["error"])
            self._save()
            started += 1
        if all(row["status"] in TERMINAL for row in self.state["jobs"]):
            self.state["status"] = "complete"
        self._save()
        return self.summary()

    def summary(self):
        rows = self.state["jobs"]
        results = [r["result"] for r in rows if r["status"] == "completed"]
        return {"status": self.state["status"], "scheduled": len(rows),
                "terminal": sum(r["status"] in TERMINAL for r in rows),
                "pending": sum(r["status"] == "pending" for r in rows),
                "failed": sum(r["status"] == "failed" for r in rows),
                "interrupted_unknown": sum(r["status"] == "interrupted_unknown" for r in rows),
                "correct": sum(r["correct"] for r in results), "promoted": sum(r["promoted"] for r in results),
                "reservations": dict(self.state["reservations"]), "plan_sha256": self.state["plan_sha256"]}
