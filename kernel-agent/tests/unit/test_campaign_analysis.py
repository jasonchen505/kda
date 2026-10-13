import copy

import pytest

from kernel_agent.campaign import Campaign
from scripts.run_campaign import make_plan
from scripts.summarize_campaign import digest, same_decision, summarize
from scripts.validate_recovery import fake_flow


def complete_fixture(tmp_path):
    plan = make_plan({}, {}, tasks=["rmsnorm"], seeds=[1, 2, 3])
    plan["generation"]["evaluation_guard"] = None  # CPU fixtures exercise unguarded synthetic verdicts.

    def worker(job):
        value = fake_flow(job)
        value.update(seed=job["seed"], job_id=job["id"])
        return value

    with Campaign(tmp_path / "state.json", plan) as campaign:
        campaign.run(worker)
        return copy.deepcopy(campaign.state)


def test_complete_analysis_pairs_seeds_and_uses_task_clusters(tmp_path):
    summary = summarize(complete_fixture(tmp_path))
    assert summary["status"] == "complete" and summary["resolved_pairs"] == 3
    assert all(arm["correct_per_scheduled"] == 1 for arm in summary["arms"])
    assert summary["contrasts"]["promoted"]["macro_task_difference"] == 0
    assert summary["contrasts"]["correct"]["exploratory_interval"]["clusters"] == 1


def test_incomplete_campaign_does_not_turn_pending_jobs_into_failures(tmp_path):
    plan = make_plan({}, {}, tasks=["rmsnorm"], seeds=[1])
    with Campaign(tmp_path / "state.json", plan) as campaign:
        state = copy.deepcopy(campaign.state)
    summary = summarize(state)
    assert summary["status"] == "incomplete" and summary["contrasts"] is None
    assert all(arm["correct_per_scheduled"] is None for arm in summary["arms"])
    assert summary["resolved_pairs"] == 0


def test_failed_jobs_are_kept_as_zero_terminal_outcomes(tmp_path):
    plan = make_plan({}, {}, tasks=["rmsnorm"], seeds=[1])

    def failure(job):
        raise RuntimeError("simulated provider timeout")

    with Campaign(tmp_path / "state.json", plan) as campaign:
        campaign.run(failure)
        state = copy.deepcopy(campaign.state)
    summary = summarize(state)
    assert summary["resolved_pairs"] == 1
    assert all(arm["correct_per_scheduled"] == 0 and arm["usage_missing_jobs"] == 1 for arm in summary["arms"])
    assert all(arm["failures"] == {"failed": 1} for arm in summary["arms"])


def test_rehashed_result_cannot_override_independent_final_verdict(tmp_path):
    state = complete_fixture(tmp_path)
    row = state["jobs"][0]
    row["result"]["promoted"] = False
    row["result_sha256"] = digest(row["result"])
    state.pop("state_sha256")
    state["state_sha256"] = digest(state)
    with pytest.raises(ValueError, match="headline differs"):
        summarize(state)


def test_guarded_results_require_occupancy_evidence(tmp_path):
    state = complete_fixture(tmp_path)
    state["plan"]["generation"]["evaluation_guard"] = "pre_post_foreign_cuda_pid_check"
    state["plan"]["runtime"]["gpu_index"] = 4
    state["plan_sha256"] = digest(state["plan"])
    state.pop("state_sha256")
    state["state_sha256"] = digest(state)
    with pytest.raises(ValueError, match="guarded search evidence missing"):
        summarize(state)


def test_cross_platform_one_ulp_difference_does_not_weaken_verdict_checks():
    actual = {"correct": True, "promoted": False, "geomean_speedup": .5363117184387787}
    recorded = {**actual, "geomean_speedup": .5363117184387788}
    assert same_decision(actual, recorded)
    assert not same_decision(actual, {**recorded, "promoted": True})
    assert not same_decision(actual, {**recorded, "geomean_speedup": .54})
