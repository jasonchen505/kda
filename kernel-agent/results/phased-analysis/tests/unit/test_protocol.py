import json

import pytest

from kernel_agent.protocol import admit, decision, feedback, proposal

SOURCE = "import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def valid():
    return {"status": "correct", "checks_passed": 28,
            "timings": [{"shape": [n], "compiled_us": 2., "candidate_us": 1.} for n in (1, 2, 3)]}


def test_model_claim_is_not_evidence():
    assert not decision({"promoted": True, "status": "I passed all tests"})["promoted"]


def test_independent_measurements_promote():
    assert decision(valid())["promoted"]


@pytest.mark.parametrize("fault", ["nan", "negative", "zero", "missing", "duplicate", "checks", "slow"])
def test_bad_measurements_cannot_promote(fault):
    result = valid()
    if fault in ("nan", "negative", "zero", "slow"):
        result["timings"][0]["candidate_us"] = {"nan": float("nan"), "negative": -1., "zero": 0., "slow": 10.}[fault]
    elif fault == "missing":
        result["timings"].pop()
    elif fault == "duplicate":
        result["timings"][1]["shape"] = [1]
    else:
        result["checks_passed"] = 1
    assert not decision(result)["promoted"]


def test_parse_json_surrounded_by_cli_text():
    obj = {"draft": "draft", "plan": "plan", "code": SOURCE}
    assert proposal("CLI preface\n```json\n" + json.dumps(obj) + "\n```") == obj


@pytest.mark.parametrize("text", ["I finished", "{}", '{"code":"pass"}', '{"draft":1}'])
def test_missing_proposal_is_rejected(text):
    with pytest.raises(ValueError):
        proposal(text)


@pytest.mark.parametrize("source", ["import os\n" + SOURCE, "open('test')\n" + SOURCE,
                                    "exec('test')\n" + SOURCE, SOURCE.replace("triton.jit", "staticmethod"),
                                    SOURCE.replace("def run", "def wrong")])
def test_obvious_policy_violations_are_rejected(source):
    with pytest.raises(ValueError):
        admit(source)


def test_feedback_retains_actual_failure():
    said = feedback({"status": "numeric_error", "shape": [7, 257], "message": "unmasked last column"})
    assert "masks" in said and "257" in said and "unmasked last column" in said
