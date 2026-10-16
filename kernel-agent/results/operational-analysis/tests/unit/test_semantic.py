import math

import pytest

from kernel_agent import semantic


def snap(shape, values):
    return {"shape": shape, "values": values, "dtype": "torch.float32"}


def probes():
    x, y = [-2., .25, 1.5], [.5, -3., 2.]
    return [{"id": "p", "inputs": [snap([3], x), snap([3], y)],
             "expected": snap([3], [a / (1 + math.exp(-a)) * b for a, b in zip(x, y)]),
             "actual": snap([3], [b / (1 + math.exp(-a)) for a, b in zip(x, y)])}]


def review():
    return {"reference_expression": "x0 * sigmoid(x0) * x1", "candidate_expression": "sigmoid(x0) * x1",
            "hypothesis": "UNTRUSTED_PRECISION_STORY", "repair_plan": "UNTRUSTED_PATCH_PLAN"}


def test_missing_factor_is_checked_against_the_oracle_not_the_model_story():
    value = semantic.validate_review(review(), probes())
    assert value["reference"]["valid"] and value["candidate"]["valid"]
    bad = review() | {"reference_expression": "sigmoid(x0) * x1"}
    assert not semantic.validate_review(bad, probes())["reference"]["valid"]


def test_grounding_filters_narrative_and_rejected_expressions():
    start = {"code": "COMPLETE_ORIGINAL", "evaluation": {"status": "numeric_error", "semantic_probes": probes()}}
    value = review()
    valid = semantic.validate_review(value, probes())
    grounded = "\n".join(p[1] for p in semantic.writer_sections("silu_mul", "CONTRACT", start, value, valid, True))
    raw = "\n".join(p[1] for p in semantic.writer_sections("silu_mul", "CONTRACT", start, value, valid, False))
    assert "UNTRUSTED_PRECISION_STORY" not in grounded and "UNTRUSTED_PATCH_PLAN" not in grounded
    assert "UNTRUSTED_PRECISION_STORY" in raw and "validated_reference_expression" in grounded
    invalid = review() | {"reference_expression": "WRONG_EXPRESSION"}
    context = "\n".join(p[1] for p in semantic.writer_sections(
        "silu_mul", "CONTRACT", start, invalid, semantic.validate_review(invalid, probes()), True))
    assert "WRONG_EXPRESSION" not in context and "validated_reference_expression" not in context
    assert "COMPLETE_ORIGINAL" in context


def test_transpose_reductions_and_trailing_broadcasting():
    inputs = [snap([2, 3], [1., 2., 3., 4., 5., 6.])]
    assert semantic.interpret("transpose(x0)", inputs).values == (1, 4, 2, 5, 3, 6)
    assert semantic.interpret("mean_last(x0, False)", inputs).shape == (2,)
    assert semantic.interpret("x0 - mean_last(x0, True)", inputs).values == (-1, 0, 1, -1, 0, 1)
    softmax = semantic.interpret("exp(x0-max_last(x0,True))/sum_last(exp(x0-max_last(x0,True)),True)", inputs)
    assert sum(softmax.values[:3]) == pytest.approx(1)
    assert semantic.interpret("rsqrt(4)+2", inputs).values == (2.5,)


@pytest.mark.parametrize("expression", ["__import__('os').system('bad')", "x0.__class__", "x0[0]",
                                       "[v for v in x0]", "2**999999", "sum_last(x0, 1)",
                                       "exp(1000000)", "1/0", "x9", "1e999"])
def test_unsupported_or_unbounded_expression_is_rejected(expression):
    result = semantic.check_expression(expression, probes(), "expected")
    assert not result["valid"]


def test_shape_error_cannot_be_hidden_by_equal_flat_values():
    p = {"id": "matrix", "inputs": [snap([2, 3], list(range(6)))], "expected": snap([3, 2], list(range(6)))}
    assert not semantic.check_expression("x0", [p], "expected")["valid"]


def test_no_probe_or_unobservable_candidate_is_not_a_proof():
    assert not semantic.validate_review(review(), [])["reference"]["valid"]
    p = probes()[0]
    p.pop("actual")
    assert not semantic.validate_review(review(), [p])["candidate"]["valid"]
    assert not semantic.validate_review({"other": 1}, probes())["reference"]["valid"]
