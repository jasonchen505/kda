"""Real Humanize planner/writer/reviewer flow with phase-specific experiment artifacts."""
import json
from pathlib import Path

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, flow

from kernel_agent import model_pool, staged, staged_contract_v4 as c
from kernel_agent.campaign import atomic_json
from kernel_agent.staged_gateway import MARKER
from kernel_agent.staged_guard import import_parent
from kernel_agent.staged_phase_interface import recoverable_writer_reply, record_reply_error
from kernel_agent.structured_recovery import recover


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
    initial_path: str = ''
    smoke: bool = False


@flow(agents=Agents, envs=Envs, params=Params)
async def staged_kernel(task, *, agents, envs, params, ctx):
    from kernel_agent.staged_contract_v4 import Tools as ContractTools
    directory = Path(params.directory)
    tools = ContractTools(directory, params.task_id, params.arm, params.gpu_python, params.lock_file)
    if params.initial_path:
        import_parent(tools, Path(params.initial_path))
    calls, stages = [], []

    async def ask(role, parts, schema, stage):
        if len(calls) >= staged.CALL_CAP:
            raise ValueError('24 total model opportunities exhausted; no hidden retry')
        ticket = {'id': f'call-{len(calls)+1:02d}', 'allocated': len(calls)+1, 'role': role,
                  'stage': stage, 'schema': schema.model_json_schema(),
                  'status': 'allocated_before_dispatch', 'started_utc': model_pool.utc()}
        calls.append(ticket)
        atomic_json(directory/'calls.json', calls)
        atomic_json(directory/'control.json', ticket)
        session = await agents[role].spawn()
        parts=parts+[('wire_format',c.wire_format(role,schema is staged.Document,schema),True)]
        try:
            reply = await agents[role].run(MARKER+json.dumps({'ticket': ticket['id'], 'sections': parts}),
                                          session=session, output_schema=schema)
            ticket.update(status='sdk_completed', reply=reply.model_dump(mode='json'))
            return reply
        except Exception as error:
            if recoverable_writer_reply(directory, ticket['id'], schema):
                response=json.loads((directory/'requests'/(ticket['id']+'.response.json')).read_text())
                prop=ticket['schema'].get('properties',{}).get('tool',{})
                allowed=prop.get('enum',[prop['const']] if 'const' in prop else [])
                recovered=recover(response['choices'][0]['message']['content'],schema,allowed) if role=='writer' else None
                if recovered:
                    reply,metadata=recovered
                    ticket.update(status='received_prefix_recovered',reply=reply.model_dump(mode='json'),recovery=metadata,
                                  error=str(error)[-400:])
                    return reply
                ticket.update(status='received_reply_error', error=str(error)[-400:])
                record_reply_error(tools, stage, ticket['id'])
                diagnostic=c.diagnose_reply(response['choices'][0]['message']['content'],schema)
                tools.events[-1]['result']['validation_diagnostics']=diagnostic
                tools.events[-1]['result']['message']='Received invalid reply; opportunity spent. '+json.dumps(diagnostic,separators=(',',':'))
                tools.save()
                return None
            raise
        finally:
            ticket['finished_utc'] = model_pool.utc()
            atomic_json(directory/'calls.json', calls)

    def state(stage):
        best = staged.select(tools.rows)
        current = tools.proposals[-1] if tools.proposals else None
        sources = []
        for row in [best, current]:
            if row and not any(r['source_sha256'] == row['source_sha256'] for r in sources):
                sources.append(row)
        packet=tools.packet(stage)
        references=packet.pop('retained_references',[])
        return [('contract', params.contract+'\nRTX4090 Ada, Torch2.6/Triton3.2. '
                 'Compute in Triton; final workloads unavailable. '+c.CAPABILITY, True),
                ('phase', c.GOALS[stage], True),
                ('evidence', json.dumps(packet, separators=(',', ':')), True),
                ('reference_memory',json.dumps(references,separators=(',',':')),False),
                ('source_facts',json.dumps(tools.source_facts(),separators=(',',':')),False),
                *[(f'source_{r["id"]}', 'Immutable source '+r['id']+' SHA '+r['source_sha256']+'\n'+
                   '\n'.join(f'{i}: {s}' for i,s in enumerate(r['code'].splitlines(),1))+
                   '\nActual verdict: '+json.dumps(c.compact_verdict(r['evaluation'], probes=False), separators=(',', ':')), True)
                  for r in sources]]

    failure = None
    try:
        if params.smoke:
            plan = await ask('planner', [('smoke', 'Integration only: hypothesis="none", measurement=search_only, '
                'steps=[{tool:submit_and_evaluate,purpose:integration}], negative_result="stop". No source requested.', True)], c.Plan, 'smoke')
            action = await ask('writer', [('smoke', params.contract+'\n'+c.MODULE_CONTRACT+
                '\nIntegration: request submit_and_evaluate with a complete Triton kernel module defining run(x,y). '
                'No self-test code. argument="". Regression fixture (not an actual previous model call): source argument=latest_correct was rejected; source argument MUST be the empty string. This verifies the real module interface and is excluded from study scores.', True)],
                               c.action_schema(['read', 'submit_and_evaluate']), 'smoke')
            event = tools.execute(action, 'smoke')
            review = await ask('reviewer', [('smoke', 'Integration only, not a full three-stage study. Actual event: '+
                json.dumps({'id':event['id'],'result':c.compact_verdict(event['result'],probes=False)})+
                '. Return accept=false for full-stage completion, cite the actual event; selected_candidate='+
                ('c1' if event['result'].get('status')=='correct' else 'none')+
                ', decision=insufficient, limitation=activity_only_no_causal_attribution, hypotheses=[], reason=integration scope.', True)], c.Review, 'smoke')
            atomic_json(directory/'smoke.json', {'plan': plan.model_dump(), 'event': event, 'review': review.model_dump()})
            return 'integration only'
        for stage in staged.STAGES:
            tools.start_phase()
            docs = directory/'docs'/stage
            docs.mkdir(parents=True, exist_ok=True)
            phase = {'stage': stage, 'accepted': False, 'reviews': []}
            stages.append(phase)
            draft = await ask('planner', state(stage)+[('draft', 'Write a concise experiment draft under120 words. '
                'Separate observations from hypotheses. Choose an actual configuration/code change to test; '
                'a negative result is valid. Do not demand hardware counters or causal isolation.', True)], staged.Document, stage)
            if draft is None:
                raise ValueError('No valid draft received')
            (docs/'draft.md').write_text(draft.text)
            plan = await ask('planner', state(stage)+[('draft', draft.text, True), ('plan',
                'Return up to FOUR atomic actions. submit_and_evaluate submits COMPLETE model-authored code_lines and immediately '
                'runs independent search verification; one submit_and_evaluate consumes one submission and one GPU search. '
                'profile uses the immutable accepted baseline. compare reuses actual cached measurements. '
                'Do not require finish: readiness triggers review automatically. Required phase actions: '+
                {'research':'submit_and_evaluate','profile_refine':'profile then submit_and_evaluate a different executable configuration',
                 'shape_tune':'compare, with submit_and_evaluate first only if two correct variants are not available'}[stage]+
                '. Negative variants do not invalidate an otherwise completed experiment.', True)], c.Plan, stage)
            issues = c.plan_issues(plan, stage, bool(tools.comparison())) if plan else ['No typed plan']
            atomic_json(docs/'plan.json', {'plan': plan.model_dump(mode='json') if plan else None, 'errors': issues})
            # One revision consumes one of the four existing action/model slots.
            offset = 0
            if issues:
                plan = await ask('planner', state(stage)+[('errors', json.dumps(issues), True),
                    ('revision', 'Revise the plan within THREE remaining atomic actions. No extra budget.', True)], c.Plan, stage)
                issues = c.plan_issues(plan, stage, bool(tools.comparison())) if plan else ['No revised plan']
                if plan and len(plan.steps) > 3:
                    issues.append('revision exceeds three remaining actions')
                atomic_json(docs/'plan-revision.json', {'plan': plan.model_dump(mode='json') if plan else None, 'errors': issues})
                offset = 1
            if issues:
                raise ValueError('Plan contract rejected: '+str(issues))
            phase['plan_sha256'] = staged.text_hash(plan.model_dump_json())
            for index in range(offset,4):
                allowed = tools.allowed(stage, 4-index)
                action = await ask('writer', state(stage)+[('plan', plan.model_dump_json(), True),
                    ('action', f'{4-index} atomic actions remain in this phase. Allowed tools: {allowed}. '
                     'submit_and_evaluate emits ONLY a kernel module: imports, Triton JIT helpers and run; no tests, __main__, examples, reference or benchmark. One Python line per code_lines item, preserving indentation; '
                     'change executable configuration/code, not comments. Full source must fit2048 output tokens. '
                     'profile argument=latest_correct; compare argument=""; read=api/example/guide. '
                     'Retain correct baseline after negative results. Never stop merely because counters are absent.', True)],
                    c.action_schema(allowed), stage)
                if action is not None:
                    tools.execute(action, stage)
                if tools.readiness(stage)['ready'] or index == 3:
                    break
            # One evidence-id list replaces duplicated claim/evidence fields. No fabricated
            # observations or host override of reviewer rejection is allowed.
            for attempt in range(2):
                if len(calls) >= staged.CALL_CAP:
                    break
                review = await ask('reviewer', state(stage)+[('review',
                    'Judge ONLY this phase. Readiness enumerates objective requirements and exact event IDs; '
                    'cite ALL required IDs in the single evidence_ids list. Choose the recorded search_winner. '
                    'An activity-only profile with a tested variant, including a slower/failed variant, is a '
                    'completed experiment. Do not require causal isolation, unavailable counters, speedup or '
                    'shape routing in refinement. Shape comparison can retain the generic winner. '
                    'All hypotheses/reason prose remain unverified. Accept only real completed evidence.', True)]+
                    ([('last_rejection', json.dumps(phase['reviews'][-1]), True)] if phase['reviews'] else []),
                    c.review_schema(tools), stage)
                if review is None:
                    passed, errors = False, ['Received malformed reviewer output']
                else:
                    passed, errors = c.gate(stage, tools, review)
                phase['reviews'].append({'review': review.model_dump(mode='json') if review else None,
                    'gate_passed': passed, 'gate_errors': errors, 'readiness': tools.readiness(stage),
                    'after_event': tools.events[-1]['id'] if tools.events else None})
                atomic_json(directory/'stages.json', stages)
                if passed:
                    phase['accepted'] = True
                    break
                if not tools.readiness(stage)['ready']:
                    break
            atomic_json(directory/'stages.json', stages)
            if not phase['accepted']:
                failure = {'type': 'StageNotAccepted', 'stage': stage, 'message': 'Real phase artifact not accepted within budget.'}
                break
    except Exception as error:  # noqa: BLE001 -- failed policy never cancels its paired control
        failure = {'type': type(error).__name__, 'message': str(error)[-1400:]}
    best = staged.select(tools.rows)
    selection = {'source_sha256': best['source_sha256'] if best else None,
                 'candidate_id': best['id'] if best else None, 'committed_utc': model_pool.utc(),
                 'rule': 'search gate-first before final'}
    atomic_json(directory/'selection.json', selection)
    atomic_json(directory/'result.json', {'task_id': params.task_id, 'arm': params.arm, 'policy': 'contract',
        'selection': selection, 'model_opportunities': len(calls), 'submissions': tools.submissions,
        'stages': stages, 'failure': failure, 'final': None})
    return json.dumps({'selection': selection, 'failure': failure})
