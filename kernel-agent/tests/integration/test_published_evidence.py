"""Published artifacts must remain sufficient to reproduce recorded selection audits."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_evidence", ROOT / "scripts/verify_evidence.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_published_evidence_is_self_contained():
    result = audit.verify(ROOT)
    assert result["candidate_files"] == 9 and result["promoted"] == 2


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
