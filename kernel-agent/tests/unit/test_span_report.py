"""A missing initial source has no parent diff; reporting does not alter outcomes."""
import copy
import json

from scripts import export_span_report


def test_first_admitted_source_after_protocol_failure_has_no_parent_diff(monkeypatch, tmp_path):
    block = {"id": "fake-format-case", "task_id": "fake", "stratum": "fresh", "seed": 1}
    initial = {"request_id": "initial", "evaluation": {"status": "protocol_error"}}
    history = [initial, {"request_id": "repair-r1", "mode": "full_source", "parent_sha256": None,
                        "source_sha256": "fake-source", "code": "# fake formatting fixture\n",
                        "evaluation": {"status": "numeric_error"}}]
    arms = {a: {"attempts": history, "correct": False, "promoted": False,
                "selected_sha256": None, "final": None} for a in ["rewrite", "operational_patch", "span_repair"]}
    full = {k: {} for k in ["request_statuses", "patch_statuses", "repair_routes", "missing_apis", "arm_usage"]}
    full.update({k: 0 for k in ["resolved", "dispatched_requests", "reported_output_tokens", "usage_missing_dispatched",
                               "peak_model_ports", "independent_gpu_windows", "peak_gpu_windows"]})
    full.update(status="incomplete", scheduled=66, plan_sha256="fake-plan", strata=[], rows=[], services=[], timeline=[],
                shared_initial_verified=True, patch_reply_to_code_verified=True, policy_request_binding_verified=True,
                final_selection_binding_verified=True, scope="Formatting fixture only; zero actual model/GPU work",
                blocks=[{"block": block, "result": {"block": block, "initial": initial, "arms": arms, "endpoint": {"port": 0}}}])
    snapshot = copy.deepcopy(full)
    plan = {k: 0 for k in ["max_model_calls", "context_limit", "max_output_tokens", "temperature"]}
    plan.update(arms=list(arms), regression_seeds=[], fresh_seeds=[], source_sha256={})
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(export_span_report, "analyze", lambda _: full)
    summary = export_span_report.compact(tmp_path, tmp_path / "compact.json")
    assert full == snapshot
    assert all(not a["attempts"][0]["derived_diff_excerpt"] for a in summary["cases"][0]["arms"].values())
    assert all(a["attempts"][0]["source_sha256"] == "fake-source" for a in summary["cases"][0]["arms"].values())
