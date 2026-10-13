"""Real model-driven KDA adaptation with independent evidence and bounded repair."""
import json
import time
from typing import Literal

from hmz.flows import (
    Agent,
    AgentCollection,
    EnvCollection,
    FilesEnvMixin,
    FlowParams,
    LocalEnv,
    Permission,
    PermissionKind,
    ShellEnvMixin,
    flow,
)
from pydantic import Field

from kernel_agent import line_protocol
from kernel_agent import humanize_api
from kernel_agent.knowledge import retrieve
from kernel_agent.protocol import admit, decision, feedback, numeric_decision, prefers, proposal
from kernel_agent.references import retrieve_example


class Builder(Agent):
    """The model proposes source; the controller owns artifact writes and evaluation."""
    _permission = Permission(local=PermissionKind.READ, user=PermissionKind.NONE,
                             system=PermissionKind.NONE, online=PermissionKind.ALL)


class Agents(AgentCollection):
    builder: Builder


class Workspace(LocalEnv, FilesEnvMixin, ShellEnvMixin):
    """Separate trusted evaluator process and immutable candidate artifacts."""


class Envs(EnvCollection):
    workspace: Workspace


class Params(FlowParams):
    task_id: str
    mode: Literal["kda_prompt", "evidence_loop", "api_guided", "reference_guided"]
    run_id: str
    evaluator: list[str]
    rounds: int = Field(default=3, ge=1, le=3)
    proposal_format: Literal["string", "lines"] = "string"
    study_policy: Literal["legacy", "blind", "feedback"] = "legacy"
    selection_policy: Literal["gate_first", "geomean"] = "gate_first"
    evaluation_mode: Literal["integrated", "deferred"] = "integrated"


def first_prompt(task):
    return ("You are a kernel design agent in a separate implementation workspace.\n"
            "Follow the KDA basic workflow: inspect the supplied task contract, draft a plan, "
            "make it executable, implement one candidate, and preserve evidence.\n"
            "The controller will write your draft to docs/draft.md and plan to docs/plan.md before "
            "writing code. It independently validates every proposal using a trusted GPU evaluator.\n"
            "Do not call tools, execute commands, or edit files yourself. Propose a complete Triton implementation. "
            "No PyTorch compute fallback. Torch may allocate tensors and read metadata. "
            "Support all contracted shapes, including odd sizes. Use Triton 3.2.0 and Torch 2.6.0.\n"
            "Correctness tolerance: FP32 atol=2e-5 rtol=2e-4; BF16 atol=0.015 rtol=0.015. "
            "Independent promotion requires geomean speedup >=1.05 and worst >=0.90 versus "
            "torch.compile on three undisclosed evaluation shapes.\n"
            "Return a JSON object with exactly three string fields: draft, plan, code. "
            "The code must import torch, triton and triton.language, contain @triton.jit kernel(s), "
            "and expose the contracted run function. Keep draft and plan under 80 words each. "
            "No source files or tests are supplied.\n\n"
            + task)


@flow(agents=Agents, envs=Envs, params=Params)
async def kernel_loop(task, *, agents, envs, params, ctx):
    fixed_study = params.study_policy != "legacy"
    deferred = params.evaluation_mode == "deferred"
    if deferred and not fixed_study:
        raise ValueError("deferred timing requires fixed proposal budgets")
    if fixed_study and (params.mode != "reference_guided" or params.proposal_format != "lines"):
        raise ValueError("strategy study requires the shared reference-guided line protocol")
    workspace = envs["workspace"]
    working = await workspace.derive_subdir(subdir=f"runs/{params.run_id}")
    docs = await working.derive_subdir(subdir="docs")
    session = await humanize_api.spawn(agents["builder"], working)
    prompt = first_prompt(task)
    if deferred:
        prompt += ("\nGeneration uses search numerical checks only, on a potentially shared GPU. "
                   "Performance selection and final checks occur later in a separate queue. "
                   "No timing or final-shape evidence is available during revision.")
    if params.proposal_format == "lines":
        prompt = line_protocol.first_prompt(prompt)
    initial_context = None
    example = retrieve_example(params.task_id) if params.mode == "reference_guided" else None
    if params.mode in ("api_guided", "reference_guided"):
        initial_context = retrieve({"status": "task_contract"}, params.task_id)
        if params.proposal_format == "lines":
            initial_context["context"] = line_protocol.reference_context(initial_context["context"])
        prompt += "\n\n" + initial_context["context"]
    if example:
        prompt += "\n\n" + example["context"]
    shared_prompt = prompt
    records, seen, best = [], {}, None
    terminal = "fixed_budget_complete" if fixed_study else "budget_exhausted"
    limit = 1 if params.mode == "kda_prompt" else params.rounds
    started = time.perf_counter()
    for iteration in range(limit):
        if iteration and params.mode == "reference_guided":
            session = await humanize_api.spawn(agents["builder"], working)
        round_started = time.perf_counter()
        if fixed_study:
            prompt += f"\n\nProposal {iteration + 1} of {limit}. Submit one complete candidate within the same task contract."
        row = {"iteration": iteration + 1, "parent_sha256": records[-1].get("source_sha256") if records else None,
               "prompt": prompt}
        try:
            reply = await humanize_api.run(agents["builder"], prompt, session=session, workspace=working)
            row["reply"] = reply
        except Exception as error:  # noqa: BLE001 -- retain external CLI failures in the study
            row["evaluation"] = {"status": "harness_error", "message": str(error)[-4000:]}
            row["elapsed_seconds"] = time.perf_counter() - round_started
            row["usage"] = session.usage.model_dump(mode="json")
            records.append(row)
            await working.write("trace.json", json.dumps(records, indent=2).encode())
            terminal = "harness_error"
            break
        try:
            proposed = line_protocol.proposal(reply) if params.proposal_format == "lines" else proposal(reply)
            await docs.write(f"draft-{iteration + 1}.md", proposed["draft"].encode())
            await docs.write(f"plan-{iteration + 1}.md", proposed["plan"].encode())
            row["proposed_code"] = proposed["code"]
            await working.write(f"proposed-{iteration + 1}.py.txt", proposed["code"].encode())
            source_sha = admit(proposed["code"])
            row["source_sha256"] = source_sha
            if source_sha in seen:
                if not fixed_study:
                    row["evaluation"] = {"status": "duplicate", "message": "unchanged source; stop wasting inference"}
                    row["elapsed_seconds"] = time.perf_counter() - round_started
                    row["usage"] = session.usage.model_dump(mode="json")
                    records.append(row)
                    await working.write("trace.json", json.dumps(records, indent=2).encode())
                    terminal = "duplicate"
                    break
                row["duplicate_of"] = seen[source_sha]
            else:
                seen[source_sha] = iteration + 1
            filename = f"candidate-{iteration + 1}.py"
            await working.write(filename, proposed["code"].encode())
            row["code"] = proposed["code"]
            if row.get("duplicate_of"):
                result = dict(records[row["duplicate_of"] - 1]["evaluation"])
            else:
                code, out, err = await workspace.exec(
                    [*params.evaluator, params.task_id, str(working.workdir / filename)], timeout=180)
                if code:
                    result = {"status": "evaluator_error", "exit_code": code, "message": (out + err)[-4000:]}
                else:
                    result = json.loads(out)
                    if result.get("status") in {"correct", "numeric_correct"} and result.get("source_sha256") != source_sha:
                        result = {"status": "stale_evidence", "message": "candidate SHA mismatch"}
        except (ValueError, SyntaxError, TypeError) as error:
            result = {"status": "protocol_error", "message": str(error)}
        except Exception as error:  # noqa: BLE001 -- evaluator process errors are evidence
            result = {"status": "evaluator_error", "message": str(error)[-4000:]}
        row["evaluation"] = result
        row["decision"] = numeric_decision(result) if deferred else decision(result)
        row["elapsed_seconds"] = time.perf_counter() - round_started
        row["usage"] = session.usage.model_dump(mode="json")
        records.append(row)
        await working.write("trace.json", json.dumps(records, indent=2).encode())
        if result.get("status") in {"resource_busy", "resource_contended", "resource_monitor_error",
                                    "resource_exhausted", "evaluator_timeout"}:
            terminal = "resource_unavailable"
            break  # Infrastructure failures must not be misrepresented as code-repair feedback.
        if not deferred and prefers(row["decision"], best["decision"] if best else None, params.selection_policy):
            best = row
        if row["decision"]["promoted"] and not fixed_study:
            terminal = "promoted"
            break
        if fixed_study:
            prompt = (shared_prompt + "\n\nPrevious candidate source (if available):\n"
                      + row.get("proposed_code", "")[:12000]
                      + "\nRevise this candidate to improve correctness and speed across contracted shapes.")
            if params.study_policy == "feedback":
                prompt += "\n\n" + line_protocol.feedback(feedback(result))[:3500]
            continue
        prompt = feedback(result)
        if params.proposal_format == "lines":
            prompt = line_protocol.feedback(prompt)
        if params.mode in ("api_guided", "reference_guided"):
            context = retrieve(result, params.task_id)
            if params.proposal_format == "lines":
                context["context"] = line_protocol.reference_context(context["context"])
            row["next_reference_context"] = context
            prompt += "\n\n" + context["context"]
        if params.mode == "reference_guided":
            previous_source = row.get("proposed_code", row.get("code", ""))[:12000]
            base_prompt = first_prompt(task)
            repair_feedback = feedback(result)
            if params.proposal_format == "lines":
                base_prompt = line_protocol.first_prompt(base_prompt)
                repair_feedback = line_protocol.feedback(repair_feedback)
            prompt = (base_prompt + "\n\n" + repair_feedback[:3500] + "\n\n"
                      + context["context"] + "\n\n" + example["context"]
                      + "\nPrevious candidate source (if available):\n" + previous_source)
    async def final_check(candidate):
        try:
            code, out, err = await workspace.exec(
                [*params.evaluator, params.task_id, str(working.workdir / f"candidate-{candidate['iteration']}.py"), "--final"], timeout=180)
            measured = json.loads(out) if code == 0 else {"status": "evaluator_error", "message": err[-2000:]}
            if measured.get("status") == "correct" and measured.get("source_sha256") != candidate["source_sha256"]:
                measured = {"status": "stale_evidence"}
        except Exception as error:  # noqa: BLE001 -- contain failures at the external evaluator boundary
            measured = {"status": "evaluator_error", "message": str(error)[-2000:]}
        return measured

    final_evaluation = None
    final_decision = None
    candidate_finals = {}
    if best is not None:
        final_evaluation = await final_check(best)
        final_decision = decision(final_evaluation)
        candidate_finals[best["source_sha256"]] = {"evaluation": final_evaluation, "decision": final_decision}
    if fixed_study and not deferred:
        # Audit first/last/best on final data without changing the search-selected source.
        for candidate in records:
            digest = candidate.get("source_sha256")
            if digest and digest not in candidate_finals:
                measured = (await final_check(candidate) if candidate.get("decision", {}).get("correct") else
                            {"status": "search_rejected", "search_status": candidate["evaluation"]["status"]})
                candidate_finals[digest] = {"evaluation": measured, "decision": decision(measured)}
    return {"task_id": params.task_id, "mode": params.mode, "run_id": params.run_id,
            "proposal_format": params.proposal_format,
            "study_policy": params.study_policy,
            "selection_policy": params.selection_policy,
            "evaluation_mode": params.evaluation_mode,
            "measurement_state": "timing_pending" if deferred else "finalized",
            "timing_candidates": list(dict.fromkeys(a["source_sha256"] for a in records
                                      if a.get("decision", {}).get("timing_eligible"))) if deferred else [],
            "terminal": terminal, "correct": bool(final_decision and final_decision["correct"]),
            "promoted": bool(final_decision and final_decision["promoted"]),
            "search_correct": best is not None,
            "search_promoted": bool(best and best["decision"]["promoted"]),
            "selected_sha256": best["source_sha256"] if best else None,
            "selected_decision": best["decision"] if best else None, "attempts": records,
            "initial_reference_context": initial_context,
            "retrieved_example": example,
            "final_evaluation": final_evaluation, "final_decision": final_decision,
            "candidate_final_evaluations": candidate_finals,
            "elapsed_seconds": time.perf_counter() - started, "usage": ctx.usage.model_dump(mode="json")}
