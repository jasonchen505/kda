"""Published artifacts must remain sufficient to reproduce recorded selection audits."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_evidence", ROOT / "scripts/verify_evidence.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_published_evidence_is_self_contained():
    result = audit.verify(ROOT)
    assert result["candidate_files"] == 9 and result["promoted"] == 2
    assert result["recovery_validation"]["unknown_outcomes"] == 1
    assert result["shared_feedback_study"]["candidate_files"] == 36
    assert result["shared_feedback_study"]["uncommitted_candidate_files"] == 3


def test_budget_stopped_task_preserves_sources_without_inventing_a_final_result(tmp_path):
    state = json.loads((ROOT / "results/shared-feedback-checkpoint.json").read_text())
    failed = next(row for row in state["jobs"] if row["status"] == "failed")
    assert "result" not in failed and "duration is spent" in failed["error"]
    shutil.copytree(ROOT / "results/shared-feedback-measured", tmp_path / "results/shared-feedback-measured")
    shutil.copy(ROOT / "results/shared-feedback-checkpoint.json", tmp_path / "results/shared-feedback-checkpoint.json")
    shutil.copytree(ROOT / "evidence/shared-feedback-requests", tmp_path / "evidence/shared-feedback-requests")
    run_id = f"campaign-{state['plan_sha256'][:12]}-{failed['job']['id']}"
    source = tmp_path / "results/shared-feedback-measured/runs" / run_id / "candidate-1.py"
    source.write_text(source.read_text() + "\n# changed after the recorded reply\n")
    with pytest.raises(ValueError, match="shared study candidate changed"):
        audit.check_shared_study(tmp_path)


def test_recovery_artifact_cannot_drop_uncertain_job():
    recovery = json.loads((ROOT / "results/campaign-recovery-validation.json").read_text())
    recovery["summary"]["scheduled"] = 2
    with pytest.raises(ValueError, match="recovery evidence changed"):
        audit.check_recovery(recovery, ROOT)


def test_headline_cannot_override_independent_final_verdict():
    study = json.loads((ROOT / "results/agent-selector-validation.json").read_text())
    run = copy.deepcopy(study["runs"][0])
    run["promoted"] = not run["promoted"]
    with pytest.raises(ValueError, match="headline result"):
        audit.check_run(run, ROOT / "results/study8-measured")


def test_search_selection_cannot_be_replaced_by_final_audit():
    study = json.loads((ROOT / "results/agent-selector-validation.json").read_text())
    run = copy.deepcopy(study["runs"][0])
    run["selected_sha256"] = run["attempts"][0]["source_sha256"]
    with pytest.raises(ValueError, match="actual output differs"):
        audit.check_run(run, ROOT / "results/study8-measured")
