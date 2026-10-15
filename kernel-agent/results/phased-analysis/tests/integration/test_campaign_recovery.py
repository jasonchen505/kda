from scripts.validate_recovery import validate


def test_real_sigkill_preserves_completed_flow_and_unknown_call(tmp_path):
    evidence = validate(tmp_path)
    assert evidence["status"] == "passed" and all(evidence["assertions"].values())
    # This is the actual Humanize kernel flow with fake model/evaluator drivers.
    completed = [row for row in evidence["after"]["jobs"] if row["status"] == "completed"]
    assert all(len(row["result"]["attempts"]) == 3 for row in completed)
    assert all(row["result"]["selection_policy"] == "gate_first" for row in completed)
