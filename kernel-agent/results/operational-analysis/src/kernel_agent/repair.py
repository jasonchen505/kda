"""Task-independent contract planning and counterexample-guided bounded repair."""
import json

from kernel_agent.line_protocol import FORMAT_RULES
from kernel_agent.protocol import decision, feedback, prefers
from kernel_agent.references import retrieve_example
from kernel_agent.knowledge import retrieve

GENERATION = ("baseline", "contract")
REPAIR = ("baseline", "counterexample")

CONTRACT_CHECK = (
    "Before writing code, use draft to derive the output shape and contiguous strides, "
    "the relationship between an output coordinate and each input coordinate, and the corresponding "
    "flat pointer offsets. Check two coordinates including an edge on a small non-square or odd-sized input. "
    "Use plan to check padded neutral values, actual reduction width, mask bounds, dtype/cast order and "
    "the output allocation. Derive these from the contract; do not assume input and output have the same layout. "
    "Keep draft/plan under 80 words each. Then implement that derivation."
)


def base(contract):
    return ("Follow KDA contract -> draft -> executable plan -> complete Triton source. "
            "Use Triton 3.2.0, Torch 2.6.0. No tools or PyTorch compute fallback; torch may allocate and read metadata. "
            "Support every contracted odd size, preserve inputs and allocate a fresh contiguous output. "
            "FP32 atol=2e-5 rtol=2e-4; BF16 atol=rtol=0.015. Independent evaluator decides success; "
            "performance gate is geometric mean >=1.05 and worst shape >=0.90 against torch.compile. "
            "Final test data are unavailable to you. Keep draft/plan under 80 words each.\n"
            + FORMAT_RULES + "\nTask contract:\n" + contract)


def sections(task_id, contract, generation, history=None, repair="baseline"):
    """Mandatory sections are never sliced. Optional references may be dropped by tokenization."""
    history = history or []
    parts = [("contract", base(contract), True)]
    if not history:
        if generation == "contract":
            parts.append(("contract_check", CONTRACT_CHECK, True))
    else:
        last = history[-1]
        source = last.get("code", last.get("proposed_code", ""))
        result = last["evaluation"]
        if repair == "baseline":
            # Historical raw-error repair: latest source and ordinary failure text.
            ordinary = {k: v for k, v in result.items() if k in {
                "status", "message", "shape", "dtype", "scale", "checks_passed", "timings", "traceback"}}
            parts.extend([("previous_source", "Previous candidate source:\n" + source, True),
                          ("feedback", feedback(ordinary), True)])
        else:
            known = [{"sha256": h.get("source_sha256"), "status": h["evaluation"].get("status"),
                      "duplicate_of": h.get("duplicate_of")} for h in history]
            structured = {k: result[k] for k in ("status", "message", "shape", "dtype", "scale", "counterexample") if k in result}
            if result.get("status") == "correct":
                structured["search_timings"] = result["timings"]
            parts.extend([("previous_source", "Previous complete source:\n" + source, True),
                          ("evidence", "Search-only evidence:\n" + json.dumps(structured, ensure_ascii=False), True),
                          ("repair_check", CONTRACT_CHECK + "\nRepair one evidence-supported cause. In draft state the "
                           "failure hypothesis and the specific observed contradiction. Verify the coordinate arithmetic "
                           "against the supplied tiny input/output witness before emitting code. Avoid repeating rejected "
                           "source hashes. If search passed, preserve its semantics while optimizing the slowest shape. "
                           "Do not claim a test ran, change the evaluator, or infer final results.", True),
                          ("failure_memory", "Candidate memory:\n" + json.dumps(known), False)])
            best = select(history)
            if best and best.get("source_sha256") != last.get("source_sha256"):
                parts.append(("correct_anchor", "Previously validated correct source (retain its semantics):\n"
                              + best["code"], False))
    parts.append(("api_notes", retrieve({"status": "task_contract"}, task_id)["context"], False))
    parts.append(("official_example", retrieve_example(task_id)["context"], False))
    return parts


def select(attempts):
    best = None
    for attempt in attempts:
        verdict = decision(attempt["evaluation"])
        if prefers(verdict, decision(best["evaluation"]) if best else None):
            best = attempt
    return best


def arm_id(generation, repair):
    return f"g-{generation}_r-{repair}"
