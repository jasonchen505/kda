"""Paired previous/guarded staged policies with a common failed-parent adapter."""
import json
from pathlib import Path

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, flow
from pydantic import BaseModel, ConfigDict

from kernel_agent import model_pool, staged
from kernel_agent.campaign import atomic_json
from kernel_agent.protocol import decision
from kernel_agent.staged_gateway import MARKER
from kernel_agent.staged_tools import compact
from kernel_agent import staged_guard_v2 as guard
from kernel_agent.staged_phase_interface import (
    Tools, goal, review_events, recoverable_writer_reply, record_reply_error,
)


class Agents(AgentCollection):
    planner: Agent
    writer: Agent
    reviewer: Agent


class Envs(EnvCollection):
    pass


class Params(FlowParams):
    directory: str
    task_id: str
    contract: str
    arm: str
    gpu_python: str
    lock_file: str
    smoke: bool = False
    policy: str = "previous"
    initial_path: str = ""


class Proposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    draft: str
    plan: str
    code_lines: list[str]


@flow(agents=Agents, envs=Envs, params=Params)
async def staged_kernel(task, *, agents, envs, params, ctx):
    directory = Path(params.directory)
    guarded = params.policy == "guarded"
    tools = (guard.Tools if guarded else Tools)(directory, params.task_id, params.arm, params.gpu_python, params.lock_file)
    if params.initial_path:
        guard.import_parent(tools, Path(params.initial_path))
    calls, stages = [], []

    async def ask(role, parts, schema, stage):
        if len(calls) >= staged.CALL_CAP:
            raise ValueError('hard model opportunity budget exhausted')
        ticket = {'id': f'call-{len(calls)+1:02d}', 'allocated': len(calls)+1,
                  'role': role, 'stage': stage, 'schema': schema.model_json_schema(),
                  'status': 'allocated_before_dispatch', 'started_utc': model_pool.utc()}
        calls.append(ticket)
        atomic_json(directory / 'calls.json', calls)
        atomic_json(directory / 'control.json', ticket)
        prompt = MARKER + json.dumps({'ticket': ticket['id'], 'sections': parts})
        session = await agents[role].spawn()
        try:
            answer = await agents[role].run(prompt, session=session, output_schema=schema)
            ticket.update(status='sdk_completed', reply=answer.model_dump(mode='json'))
            return answer
        except Exception as error:
            if (role == 'writer' or (guarded and role == 'planner')) and recoverable_writer_reply(directory, ticket['id'], schema):
                ticket.update(status='received_reply_error', error=str(error)[-400:])
                record_reply_error(tools, stage, ticket['id'])
                return None
            raise
        finally:
            ticket['finished_utc'] = model_pool.utc()
            atomic_json(directory / 'calls.json', calls)

    def state_parts(stage, remaining=6):
        visible = [{'id': e['id'], 'tool': e['tool'], 'stage': e['stage'],
                    'source_sha256': e['source_sha256'], 'status': e['result'].get('status', 'returned')}
                   for e in tools.events[-6:]]
        # Preserve the latest complete source/verdict and one exact finite witness; don't replay stale tensors.
        parts = [('contract', staged.brief(params.task_id, params.contract, stage, params.arm), True),
                 ('tools', staged.TOOLS, True),
                 ('active_phase_goal', goal(stage), True),
                 ('history_index', 'Recent actual event index (full records remain on disk):\n' + json.dumps(visible), True)]
        if tools.events:
            event = tools.events[-1]
            value = compact(event['result'], probes=False) if event['tool'] == 'evaluate' else event['result']
            parts.append(('latest_tool_result', 'Latest actual host result:\n'+json.dumps(value), True))
        source_parts = staged.source_context(tools.proposals, params.arm == 'staged_transfer')
        if (tools.events and tools.proposals and tools.events[-1]['tool'] == 'read'
                and tools.events[-1]['result'].get('sha256') == tools.proposals[-1]['source_sha256']
                and 'numbered_source' in tools.events[-1]['result']):
            source_parts = [p for p in source_parts if p[0] != 'parent']
        parts += source_parts
        if tools.proposals:
            actual = tools.proposals[-1]['evaluation']
            parts.append(('latest_verdict', 'Last source verdict:\n'+json.dumps(compact(actual, probes=False)), True))
            probe = staged.common_probe(actual)
            if probe:
                parts.append(('common_probe', 'Exact finite search witness (not proof):\n'+json.dumps(probe), True))
        if params.arm == 'staged_transfer' and tools.proposals:
            parts.append(('derived_evidence', json.dumps(staged.visible_result(tools.proposals[-1], True)['evidence']), False))
        if guarded:
            for i, (name, text, required) in enumerate(parts):
                if name == 'parent':
                    source = tools.proposals[-1]['code']
                    parts[i] = (name, 'Exact current source with 1-based line numbers, SHA '+tools.proposals[-1]['source_sha256']+':\n'+'\n'.join(f'{n}: {line}' for n, line in enumerate(source.splitlines(), 1)), required)
            memory = tools.state(stage, remaining)
            references = memory.pop('retained_references')
            parts.append(('progress_state', json.dumps(memory), True))
            parts.append(('reference_memory', json.dumps(references), False))
        return parts

    failure = None
    try:
        if params.smoke:
            if guarded:
                planned = await ask('planner', [('smoke', 'Schema smoke only. Return objective="integration", '
                    'one step tool=read, argument=guide, purpose="inspect capabilities", fallback="stop".', True)], guard.plan_schema(1), 'smoke')
                action = await ask('writer', [('smoke', 'Return read argument=guide, empty code_lines/edits, '
                    'reason="integration only". No kernel work.', True)], guard.action_schema(['read']), 'smoke')
                event = tools.execute(action, 'smoke')
                reading = await ask('reviewer', [('smoke', 'No kernel was implemented or tested. Return accept=false, '
                    'evidence_ids=[], missing="no kernel", next_step="submit", claims=[].', True)], guard.Review, 'smoke')
                atomic_json(directory / 'smoke.json', {'plan': planned.model_dump(), 'action': action.model_dump(),
                    'event': event, 'review': reading.model_dump(), 'sdk': 'real Humanize/LiteLLM'})
                return 'guarded schema smoke'
            answer = await ask('planner', [('smoke', 'Return a JSON object whose text field is exactly '
                'the single word ready. Do not copy this instruction.', True)], staged.Document, 'smoke')
            action = await ask('writer', [('smoke', 'Request the hosted read tool with argument guide. '
                'Use empty code_lines and edits arrays. Reason: integration smoke only.', True)], staged.Action, 'smoke')
            event = tools.execute(action, 'smoke')
            reading = await ask('reviewer', [('smoke', 'No kernel has been implemented or tested. '
                'Reject stage completion: accept=false, evidence_ids=[], missing="no kernel", '
                'next_step="implement". This is a schema integration smoke.', True)], staged.Review, 'smoke')
            atomic_json(directory / 'smoke.json', {'text': answer.text, 'action': action.model_dump(),
                'event': event, 'review': reading.model_dump(), 'sdk': 'real Humanize/LiteLLM'})
            return answer.text
        if params.arm == 'fixed':
            for _ in range(staged.CANDIDATE_CAP):
                parts = staged.fixed_sections(params.task_id, params.contract, tools.proposals)
                if tools.events and tools.events[-1]['tool'] == 'decode_reply':
                    parts.append(('protocol_error', json.dumps(tools.events[-1]['result']), True))
                answer = await ask('writer', parts, Proposal, 'fixed')
                if answer is None:
                    continue
                event = tools.execute({'tool': 'submit', 'code_lines': answer.code_lines,
                                       'edits': [], 'reason': answer.plan, 'argument': ''}, 'fixed')
                if event['result'].get('status') == 'admitted':
                    tools.execute({'tool': 'evaluate', 'argument': event['source_sha256']}, 'fixed')
        else:
            for stage in staged.STAGES:
                docs = directory / 'docs' / stage
                docs.mkdir(parents=True, exist_ok=True)
                if guarded:
                    tools.start_phase()
                base = staged.brief(params.task_id, params.contract, stage, params.arm) + '\n' + goal(stage)
                draft = await ask('planner', state_parts(stage) + [('draft_instruction',
                    'Write only a research/implementation draft under250 words: baseline, risks, ranked '
                    'directions and concrete tool commands to gather evidence. Do not implement code.', True)],
                    staged.Document, stage)
                (docs / 'draft.md').write_text(draft.text)
                limit = 6 if stage == 'research' else 4
                if guarded:
                    structured = await ask('planner', [('contract', base, True), ('tools', staged.TOOLS, True),
                        ('draft', draft.text, True), ('state', json.dumps(tools.state(stage, limit)), True),
                        ('plan_instruction', f'Return a typed executable plan with at most {limit} SINGLE tool actions. '
                         'Use only supplied tool names. Argument latest/latest_correct is resolved by host. '
                         'References are optional; source/evidence already persist. Reserve submit/patch then evaluate '
                         'for every variant. Refinement must profile and actually evaluate a distinct variant; '
                         'negative results are valid. No hardware counters are available; occupancy/launch/bandwidth '
                         'are hypotheses, never observed facts. Last-action review is automatic; finish optional. '
                         'Fallback must fit the SAME remaining actions. Do not implement source.', True)], guard.plan_schema(limit), stage)
                    issues = guard.plan_issues(structured, limit, stage, bool(tools.proposals), tools.compared())
                    guard.save_plan(docs / 'plan-validation.json', structured, issues)
                    plan = staged.Document(text=json.dumps(structured.model_dump(mode='json')) if structured else 'Invalid plan: see validation errors')
                else:
                    structured, issues = None, []
                    plan = await ask('planner', [('contract', base, True), ('draft', draft.text, True),
                        ('plan_instruction', 'Convert the saved draft into an executable plan under300 words. '
                         'For each step give tool choice, evidence and acceptance condition. '
                         'Research has six writer actions; later phases have four. At most two reviews '
                         'per phase and24 total model opportunities including planning/review. No claimed measurements.', True)],
                        staged.Document, stage)
                (docs / 'plan.md').write_text(plan.text)
                phase = {'stage': stage, 'draft_sha256': staged.text_hash(draft.text),
                         'plan_sha256': staged.text_hash(plan.text), 'reviews': [], 'accepted': False}
                stages.append(phase)
                reviews = 0
                writer_limit = 6 if stage == 'research' else 4
                for action_index in range(writer_limit):
                    if guarded and issues:
                        if action_index > 0:
                            failure = {'type': 'PlanContractRejected', 'stage': stage, 'issues': issues}
                            break
                        revised = await ask('planner', state_parts(stage, writer_limit-1) + [
                            ('invalid_plan', plan.text, True), ('contract_errors', json.dumps(issues), True),
                            ('revision', f'Revise within {writer_limit-1} remaining actions. This revision spends an existing '
                             'writer opportunity, not an extra call. Use actual tools and measured capabilities only.', True)], guard.plan_schema(writer_limit-1), stage)
                        structured = revised
                        issues = guard.plan_issues(revised, writer_limit-1, stage, bool(tools.proposals), tools.compared())
                        guard.save_plan(docs / 'plan-revision.json', revised, issues)
                        if revised:
                            plan = staged.Document(text=json.dumps(revised.model_dump(mode='json')))
                            (docs / 'plan.md').write_text(plan.text)
                            phase['plan_sha256'] = staged.text_hash(plan.text)
                        continue
                    parts = state_parts(stage, writer_limit-action_index) + [('executable_plan', plan.text, True),
                        ('remaining', f'Writer action {action_index+1}/{writer_limit}. '
                         f'Candidates used {tools.submissions}/6. Profile calls {tools.profiles}/2. '
                         'You can use source SHA from tool results. No native tools; emit one JSON action.', True)]
                    if phase['reviews']:
                        parts.append(('last_review', json.dumps(phase['reviews'][-1]), True))
                    action = await ask('writer', parts, guard.action_schema(tools.allowed(stage, writer_limit-action_index)) if guarded else staged.Action, stage)
                    event = tools.execute(action, stage) if action is not None else tools.events[-1]
                    if (action is not None and action.tool == 'finish') or action_index == writer_limit-1:
                        refs = guard.review_packet(tools) if guarded else review_events(tools.events)
                        review = await ask('reviewer', [('contract', base, True), ('plan', plan.text, True),
                            ('evidence', json.dumps(refs), True)] + staged.source_context(tools.proposals, False) +
                            [('review_instruction', 'Fresh independent reading. Only actual host events are evidence. '
                              'Cite their exact event ids. Check the current stage acceptance and code. '
                              'Judge ONLY the active_phase_goal in the contract above. Do not import requirements '
                              'from later phases. Every listed host event is real, including failures or no matches. '
                              'Reject untested sources, invented results or incomplete work. '
                              'When not accepted give a concrete next hosted-tool action. '
                              + ('Include typed claims: search_correct cites actual successful evaluate; activity_observed cites profile; '
                                 'search_compared cites two AST-distinct correct sources on three common search shapes. '
                                 'For observations hypothesis must be empty. Use kind hypothesis for unverified attribution. '
                                 'Do not compare instrumented activity times with independent benchmark times. '
                                 'Missing hardware counters alone is not a blocker.' if guarded else ''), True)], guard.Review if guarded else staged.Review, stage)
                        if guarded:
                            passed, reason, claims = guard.gate(stage, tools, review)
                            atomic_json(docs / f'claim-validation-{reviews+1}.json', {'passed': passed, 'reason': reason, 'claims': claims})
                        else:
                            passed, reason = staged.stage_gate(stage, tools.rows, tools.events, review)
                        phase['reviews'].append({'review': review.model_dump(), 'gate_passed': passed,
                                                 'gate_reason': reason, 'after_event': event['id']})
                        reviews += 1
                        phase['accepted'] = passed
                        atomic_json(directory / 'stages.json', stages)
                        if passed or reviews >= 2:
                            break
                atomic_json(directory / 'stages.json', stages)
                if not phase['accepted']:
                    failure = {'type': 'StageNotAccepted', 'stage': stage,
                               'message': 'Bounded phase ended without acceptance; later phases are blocked.'}
                    break
    except Exception as error:  # noqa: BLE001 -- a failed policy never cancels another study arm
        failure = {'type': type(error).__name__, 'message': str(error)[-1600:]}
    best = staged.select(tools.rows)
    selection = {'source_sha256': best['source_sha256'] if best else None,
                 'candidate_id': best['id'] if best else None, 'committed_utc': model_pool.utc(),
                 'rule': 'search gate-first, before any final evaluation'}
    atomic_json(directory / 'selection.json', selection)
    result = {'task_id': params.task_id, 'arm': params.arm, 'policy': params.policy, 'selection': selection,
              'model_opportunities': len(calls), 'submissions': tools.submissions,
              'stages': stages, 'failure': failure, 'final': None,
              'search_decision': decision(best['evaluation']) if best else None}
    atomic_json(directory / 'result.json', result)
    return json.dumps({'selection': selection, 'failure': failure})
