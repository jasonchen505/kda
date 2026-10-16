import json

import pytest

from kernel_agent import operational
from kernel_agent.protocol import sha

SOURCE = "import triton\nimport triton.language as tl\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n"


def patch(before, after, source=SOURCE):
    return json.dumps({"draft": "one change", "plan": "exact patch", "base_sha256": sha(source),
                       "edits": [{"before_lines": before, "after_lines": after}]})


def test_transaction_changes_only_exact_model_authored_span():
    p = patch(["    return x"], ["    return x  # preserved"])
    result = operational.apply(p, SOURCE)
    assert result["code"] == SOURCE.replace("    return x\n", "    return x  # preserved\n")
    assert result["transaction"]["base_sha256"] == sha(SOURCE)
    assert result["transaction"]["result_sha256"] == sha(result["code"])
    with pytest.raises(ValueError, match="exact current"):
        operational.apply(p, SOURCE + "\n")


@pytest.mark.parametrize("before,after", [(["missing"], ["new"]), (["return x"], ["return x"]),
    (["    return x"], ["    return x"]), (["    return x"], ["    return x\nprint(1)"]),
    (["    return x"], ["    invalid python @"]), (["import triton"], ["import subprocess"]),
    ([], ["new"]), (["    pass"], "not-lines")])
def test_bad_edit_never_mutates_the_base(before, after):
    with pytest.raises((ValueError, SyntaxError)):
        operational.apply(patch(before, after), SOURCE)
    assert SOURCE.endswith("    return x\n")


def test_ambiguous_and_overlapping_edits_reject_the_whole_transaction():
    base = SOURCE + "# duplicate\n# duplicate\n"
    with pytest.raises(ValueError, match="exactly one"):
        operational.apply(patch(["# duplicate"], ["# different"], base), base)
    value = json.loads(patch(["def run(x):", "    return x"], ["def run(x):", "    return x # one"]))
    value["edits"].append({"before_lines": ["    return x"], "after_lines": ["    return x # two"]})
    with pytest.raises(ValueError, match="overlapping"):
        operational.apply(json.dumps(value), SOURCE)


def test_direct_language_alias_inventory_and_shadowing():
    capability = {"language_symbols": ["exp", "float32"]}
    assert operational.api_audit(SOURCE.replace("    pass", "    z = tl.expm1(1)"), capability)["missing_language_symbols"] == ["expm1"]
    assert operational.api_audit(SOURCE.replace("    pass", "    z = tl.exp(1)"), capability)["status"] == "api_admitted"
    assert operational.api_audit(SOURCE.replace("    pass", "    tl = x\n    z = tl.local_method()"), capability)["status"] == "api_admitted"


def tensor(shape, values):
    return {"shape": shape, "values": values}


def samples():
    return [{"id": "a", "inputs": [tensor([4], [-2, -1, 0, 3])],
             "actual": tensor([4], [.5, 2., 4., 1.]), "expected": tensor([4], [-1., -2., 0., 3.])},
            {"id": "b", "inputs": [tensor([3], [-3, 2, 4])],
             "actual": tensor([3], [1., 2., .25]), "expected": tensor([3], [-3., 4., 1.])}]


def test_relation_is_derived_from_both_probes_and_not_task_name():
    probes = samples()
    assert [r["expected_matches"] for r in operational.relations(probes)] == ["actual * x0"]
    probes[1]["expected"]["values"][0] += .1
    assert not operational.relations(probes)
    assert not operational.relations(samples()[:1])
    probes = samples()
    for p in probes:
        p["actual"] = p["expected"]
    assert not operational.relations(probes)


def test_layout_coordinates_come_from_oracle_values_not_a_task_table():
    probes = []
    for index, (m, n) in enumerate([(2, 3), (3, 4)]):
        values = [float(i + 1 + index) for i in range(m * n)]
        probes.append({"id": str(index), "inputs": [tensor([m, n], values)],
                       "expected": tensor([n, m], [values[i * n + j] for j in range(n) for i in range(m)])})
    movement = operational.movement(probes)
    assert movement[0]["source_expression"] == "x0"
    witness = movement[0]["witnesses"][0]["examples"][1]
    assert witness["output_coordinate"] == [0, 1] and witness["source_coordinate"] == [1, 0]
    assert witness["output_flat"] == 1 and witness["source_flat"] == 3
    assert operational.diagnose({"status": "numeric_error", "semantic_probes": probes})["route"] == "layout"


def test_invalid_patch_retains_admitted_source_and_final_data_do_not_enter_context():
    start = {"request_id": "initial", "code": SOURCE, "source_sha256": sha(SOURCE),
             "evaluation": {"status": "numeric_error", "semantic_probes": samples(),
                            "final_result": "FORBIDDEN_FINAL", "timings": "UNNEEDED_TIMING"}}
    bad = {"request_id": "operational_patch-r1", "evaluation": {"status": "patch_error", "message": "missing span"}}
    assert operational.anchor([start, bad]) is start
    context = "\n".join(s[1] for s in operational.sections("arbitrary-task", "CONTRACT", [start, bad], {"language_symbols": ["exp"]}))
    assert SOURCE in context and "missing span" in context and sha(SOURCE) in context
    assert "actual * x0" in context
    assert "FORBIDDEN_FINAL" not in context and "UNNEEDED_TIMING" not in context
