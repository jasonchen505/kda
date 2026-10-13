import copy
import hashlib
import json

import pytest

from kernel_agent.campaign import Campaign, atomic_json, digest, validate_result


def plan(count=3):
    return {"jobs": [{"id": f"job-{i}", "task_id": "row_sum", "study_policy": "feedback"}
                     for i in range(count)], "per_job_budget": {"duration_seconds": 900, "output_tokens": 20000},
            "campaign_budget": {"duration_seconds": 900 * count, "output_tokens": 20000 * count},
            "sources": {"controller": "frozen"}}


def result(job):
    code = "# synthetic coordinator fixture, not a GPU candidate\n"
    sha = hashlib.sha256(code.encode()).hexdigest()
    return {"task_id": job["task_id"], "study_policy": job["study_policy"],
            "correct": True, "promoted": True, "selected_sha256": sha,
            "attempts": [{"source_sha256": sha, "code": code}],
            "final_evaluation": {"status": "correct", "checks_passed": 28, "source_sha256": sha,
                                 "timings": [{"shape": [n], "compiled_us": 1.2, "candidate_us": 1}
                                             for n in (1, 2, 3)]}}


def test_resume_reuses_committed_jobs_and_budget(tmp_path):
    path, schedule, calls = tmp_path / "checkpoint.json", plan(), []

    def worker(job):
        calls.append(job["id"])
        return result(job)

    with Campaign(path, schedule) as campaign:
        first = campaign.run(worker, max_new_jobs=1)
    assert first["reservations"]["output_tokens"] == 20000 and first["pending"] == 2
    with Campaign(path, schedule) as campaign:
        final = campaign.run(worker)
    with Campaign(path, schedule) as campaign:
        assert campaign.run(worker) == final
    assert calls == ["job-0", "job-1", "job-2"]
    assert final["terminal"] == final["correct"] == 3
    assert final["reservations"] == schedule["campaign_budget"]


def test_resource_block_happens_before_reservation_or_worker(tmp_path):
    def busy(job):
        raise RuntimeError("GPU busy")

    with Campaign(tmp_path / "state.json", plan()) as campaign:
        summary = campaign.run(lambda job: pytest.fail("external call must not start"), ready=busy)
    assert summary["status"] == "awaiting_resources" and summary["pending"] == 3
    assert summary["reservations"] == {"duration_seconds": 0, "output_tokens": 0}


def test_unknown_interruption_is_not_replayed_or_removed(tmp_path):
    path, schedule = tmp_path / "state.json", plan()

    def interrupted(job):
        raise KeyboardInterrupt()

    with Campaign(path, schedule) as campaign, pytest.raises(KeyboardInterrupt):
        campaign.run(interrupted)
    with Campaign(path, schedule) as campaign:
        calls = []
        summary = campaign.run(lambda j: calls.append(j["id"]) or result(j))
    assert calls == ["job-1", "job-2"] and summary["scheduled"] == 3
    assert summary["correct"] == 2 and summary["interrupted_unknown"] == 1
    assert summary["reservations"] == schedule["campaign_budget"]


def test_budget_cannot_be_reset_by_restart(tmp_path):
    path, schedule = tmp_path / "state.json", plan()
    schedule["campaign_budget"]["output_tokens"] = 20000
    with Campaign(path, schedule) as campaign:
        assert campaign.run(result)["status"] == "allocation_budget_exhausted"
    with Campaign(path, schedule) as campaign:
        summary = campaign.run(lambda job: pytest.fail("budget must remain exhausted"))
    assert summary["terminal"] == 1 and summary["pending"] == 2


@pytest.mark.parametrize("mutation", ["plan", "checkpoint", "result", "event"])
def test_resume_rejects_drift(tmp_path, mutation):
    path, schedule = tmp_path / "state.json", plan()
    with Campaign(path, schedule) as campaign:
        campaign.run(result, max_new_jobs=1)
    if mutation == "plan":
        schedule["sources"]["controller"] = "changed"
    else:
        state = json.loads(path.read_text())
        if mutation == "checkpoint":
            state["reservations"]["output_tokens"] = 0
        elif mutation == "result":
            state["jobs"][0]["result"]["promoted"] = False
        else:
            state["events"][0]["kind"] = "edited"
        # For result/event edits, recompute the outer digest to exercise inner checks.
        if mutation != "checkpoint":
            state.pop("state_sha256")
            state["state_sha256"] = digest(state)
        atomic_json(path, state)
    with pytest.raises(ValueError), Campaign(path, schedule):
        pass


def test_single_writer_lock_and_defensive_plan_copy(tmp_path):
    path, schedule = tmp_path / "state.json", plan()
    original = copy.deepcopy(schedule)
    campaign = Campaign(path, schedule)
    schedule["sources"]["controller"] = "caller mutation"
    with campaign:
        assert campaign.plan == original
        with pytest.raises(BlockingIOError), Campaign(path, original):
            pass
    with Campaign(path, original):
        pass


@pytest.mark.parametrize("mutation", ["verdict", "sha", "job", "seed"])
def test_coordinator_rejects_false_headline_or_unbound_source(mutation):
    job = plan()["jobs"][0]
    value = result(job)
    if mutation == "verdict":
        value["final_evaluation"]["status"] = "numeric_error"
    elif mutation == "sha":
        value["attempts"][0]["code"] += "# edited"
    elif mutation == "job":
        value["task_id"] = "other"
    else:
        job["seed"] = 123
        value.update(seed=124, job_id=job["id"])
    with pytest.raises(ValueError):
        validate_result(value, job)


def test_failed_run_stays_in_denominator(tmp_path):
    def failure(job):
        raise RuntimeError("provider timeout")

    with Campaign(tmp_path / "state.json", plan(1)) as campaign:
        summary = campaign.run(failure)
    assert summary["status"] == "complete" and summary["failed"] == summary["scheduled"] == 1
    assert summary["correct"] == 0 and summary["reservations"]["output_tokens"] == 20000


def test_atomic_write_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_json(path, {"old": True})

    def fail(*args):
        raise OSError("injected rename failure")

    monkeypatch.setattr("kernel_agent.campaign.os.replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]
