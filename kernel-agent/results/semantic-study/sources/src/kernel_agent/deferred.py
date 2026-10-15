"""A durable timing queue for unedited, numerically eligible model proposals."""
import fcntl
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from kernel_agent.campaign import atomic_json, digest
from kernel_agent.line_protocol import proposal
from kernel_agent.protocol import decision, numeric_decision, prefers, sha

RESOURCE_FAILURES = {"resource_busy", "resource_contended", "resource_monitor_error",
                     "resource_exhausted", "evaluator_timeout"}


def candidates(run):
    search_only = run.get("evaluation_mode") == "search_only"
    if run.get("evaluation_mode") not in {"deferred", "search_only"} or run.get("final_evaluation") is not None:
        raise ValueError("queue requires phase-separated generation without final-data access")
    if search_only and (run.get("candidate_final_evaluations") or run.get("measurement_state") != "final_pending"):
        raise ValueError("online search must commit before final-data access")
    items = {}
    for attempt in run["attempts"]:
        if not (decision(attempt["evaluation"])["correct"] if search_only else
                numeric_decision(attempt["evaluation"])["timing_eligible"]):
            continue
        code, digest_ = proposal(attempt["reply"])["code"], attempt["source_sha256"]
        if code != attempt["code"] or sha(code) != digest_ or attempt["evaluation"].get("source_sha256") != digest_:
            raise ValueError("eligible source differs from model/numerical evidence")
        items.setdefault(digest_, {"source_sha256": digest_, "code": code, "iteration": attempt["iteration"]})
        if search_only and digest_ in items and "evaluation" not in items[digest_]:
            items[digest_]["evaluation"] = attempt["evaluation"]
    if list(items) != run["timing_candidates"]:
        raise ValueError("eligible candidate schedule changed")
    return list(items.values())


class TimingQueue:
    """Model calls are never replayed here. Only pure evaluation can be retried."""

    def __init__(self, path, plan):
        self.path = Path(path)
        self.plan = json.loads(json.dumps(plan, allow_nan=False))
        self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                self.verify()
            else:
                eligible = candidates(self.plan["generation_result"])
                self.state = {"schema_version": 1, "plan": self.plan, "plan_sha256": digest(self.plan),
                              "created_utc": datetime.now(UTC).isoformat(), "status": "queued",
                              "search": [{"source_sha256": c["source_sha256"], "status": "queued", "observations": []}
                                         for c in eligible], "selected_sha256": None,
                              "selection_committed": False,
                              "final": {"status": "queued", "observations": []}}
                # Online feedback already paid for these search measurements. Import
                # the exact evidence instead of rerunning search or any model calls.
                for row, candidate in zip(self.state["search"], eligible, strict=True):
                    if "evaluation" in candidate:
                        value = candidate["evaluation"]
                        row.update(status="measured", evaluation=value,
                                   decision=self.checked(value, candidate["source_sha256"]),
                                   observations=[{"status": "imported_committed_search", "evaluation": value}])
                if self.plan["generation_result"].get("evaluation_mode") == "search_only":
                    if self.select() != self.plan["generation_result"].get("selected_sha256"):
                        raise ValueError("committed selection differs from search-only evidence")
                    self.state.update(selection_committed=True, selected_sha256=self.select(), status="final_pending")
                self.save()
                self.verify()
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def __exit__(self, *args):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        self.lock = None

    def save(self):
        self.state.pop("state_sha256", None)
        self.state["state_sha256"] = digest(self.state)
        atomic_json(self.path, self.state)

    def checked(self, result, digest_):
        verdict = decision(result)
        if verdict["correct"]:
            audit = result.get("resource_audit", {})
            shared = self.plan.get("measurement_condition") == "shared_observed"
            valid_condition = (result.get("measurement_condition") == "shared_observed"
                               and audit.get("policy") == "shared_observed"
                               and audit.get("before", {}).get("free_mib", 0) >= 1024
                               and isinstance(audit.get("after", {}).get("pids"), list)
                               and isinstance(audit.get("samples"), list)) if shared else (
                               audit.get("before_pids") == [] and audit.get("after_pids") == [])
            if (result.get("source_sha256") != digest_ or
                    audit.get("gpu_index") != self.plan["runtime"]["gpu_index"] or not valid_condition):
                raise ValueError("timing lacks source binding or uncontended boundary evidence")
        return verdict

    def verify(self):
        if self.state["state_sha256"] != digest({k: v for k, v in self.state.items() if k != "state_sha256"}):
            raise ValueError("timing queue integrity mismatch")
        if self.state["plan"] != self.plan or self.state["plan_sha256"] != digest(self.plan):
            raise ValueError("timing queue plan changed")
        if self.plan["generation_result_sha256"] != digest(self.plan["generation_result"]):
            raise ValueError("generation evidence changed")
        eligible = candidates(self.plan["generation_result"])
        if [r["source_sha256"] for r in self.state["search"]] != [c["source_sha256"] for c in eligible]:
            raise ValueError("timing candidate schedule changed")
        for row in self.state["search"]:
            if row["status"] == "measured":
                actual = self.checked(row["evaluation"], row["source_sha256"])
                recorded = row["decision"]
                if actual.keys() != recorded.keys() or any(
                    not math.isclose(v, recorded[k], rel_tol=1e-12, abs_tol=1e-12)
                    if type(v) is float else v != recorded[k] for k, v in actual.items()):
                    raise ValueError("timing decision changed")
            elif row["status"] not in {"queued", "running"}:
                raise ValueError("unknown timing state")
        if self.state["selection_committed"]:
            if any(row["status"] != "measured" for row in self.state["search"]):
                raise ValueError("selection precedes complete search timings")
            if self.state["selected_sha256"] != self.select():
                raise ValueError("selection differs from search-only evidence")
        if self.plan["generation_result"].get("evaluation_mode") == "search_only":
            expected = {c["source_sha256"]: c["evaluation"] for c in eligible}
            if (not self.state["selection_committed"] or self.state["selected_sha256"] !=
                    self.plan["generation_result"].get("selected_sha256") or any(
                    r.get("evaluation") != expected[r["source_sha256"]] for r in self.state["search"])):
                raise ValueError("imported online-search evidence changed")
        final = self.state["final"]
        if final["status"] == "measured":
            actual = self.checked(final["evaluation"], self.state["selected_sha256"])
            if self.state["status"] == "complete" and (actual["correct"], actual["promoted"]) != (
                    self.state["correct"], self.state["promoted"]):
                raise ValueError("final headline differs from independent verdict")

    def select(self):
        best = None
        for row in self.state["search"]:
            if prefers(row["decision"], best["decision"] if best else None):
                best = row
        return best["source_sha256"] if best else None

    def run(self, evaluator):
        if self.lock is None:
            raise RuntimeError("measurement requires queue ownership")
        self.verify()
        eligible = {c["source_sha256"]: c for c in candidates(self.plan["generation_result"])}

        def measure(row, digest_, final):
            # GPU evaluation is pure and has no model cost. Unknown evaluations may be
            # repeated; observations remain in the ledger, never called model retries.
            if row["status"] == "running":
                row["observations"].append({"status": "interrupted_evaluation", "observed_utc": datetime.now(UTC).isoformat()})
            row["status"] = "running"
            self.save()
            value = evaluator(eligible[digest_], final)
            row["observations"].append({"observed_utc": datetime.now(UTC).isoformat(), "evaluation": value})
            if value.get("status") in RESOURCE_FAILURES:
                row["status"] = "queued"
                self.state["status"] = "awaiting_measurement_resources"
                self.save()
                return False
            row.update(status="measured", evaluation=value, decision=self.checked(value, digest_))
            self.save()
            return True

        for row in self.state["search"]:
            if row["status"] != "measured" and not measure(row, row["source_sha256"], False):
                return self.state
        if not self.state["selection_committed"]:
            self.state.update(selection_committed=True, selected_sha256=self.select(), status="final_pending")
            self.save()  # Freeze search-only selection before any final-data access.
        if self.state["selected_sha256"] is None:
            self.state.update(status="complete", correct=False, promoted=False)
            self.save()
            return self.state
        row = self.state["final"]
        if row["status"] != "measured" and not measure(row, self.state["selected_sha256"], True):
            return self.state
        self.state.update(status="complete", correct=row["decision"]["correct"], promoted=row["decision"]["promoted"])
        self.save()
        return self.state
