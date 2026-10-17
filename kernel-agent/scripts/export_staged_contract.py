"""Read-only reconstruction of wire/source/stage evidence; publish bounded summaries."""
import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent import line_protocol,span_repair,staged,staged_contract as c,staged_guard_v2 as g  # noqa: E402
from kernel_agent.campaign import atomic_json,digest  # noqa: E402
from kernel_agent.protocol import admit,decision  # noqa: E402
from kernel_agent.staged_gateway import MARKER  # noqa: E402
from scripts import export_staged_guarded as old  # noqa: E402

old.guard=g


def load(p):
    return json.loads(p.read_text())


def message_excerpt(value):
    text=str(value)
    return text if len(text)<=340 else text[:100]+' … '+text[-220:]


def prefix(ledger,end,contract):
    obj=old.prefix(ledger,end)
    if contract:
        obj.__class__=c.Tools
        for e in obj.events:
            if e['tool']=='test' and e['result'].get('status')=='semantic_repeat':
                row=next(r for r in reversed(obj.rows) if r['source_sha256']==e['source_sha256'])
                row['evaluation']=e['result']
    return obj


def audit(folder,value,policy,parent):
    contract=policy=='contract'
    tickets=load(folder/'calls.json') if (folder/'calls.json').exists() else []
    ledger=load(folder/'tools.json') if (folder/'tools.json').exists() else {'events':[],'candidates':[],'proposals':[]}
    assert len(tickets)<=24
    for key,cap in [('submissions',6),('gpu_calls',9),('profiles',2)]:
        assert ledger.get(key,0)<=cap
    receipts,intervals=[],[]
    fact_requests=fact_omissions=0
    for t in tickets:
        path=folder/'requests'/(t['id']+'.json')
        if not path.exists():
            continue
        r=load(path)
        receipts.append(r)
        if r['status'] not in {'completed','dispatched','response_unknown'}:
            continue
        wire=path.with_name(t['id']+'.request.json')
        request=load(wire)
        assert hashlib.sha256(wire.read_bytes()).hexdigest()==r['request_sha256']
        assert request['max_tokens']==2048 and r['input_tokens']+2048<=8192
        assert request['response_format']['json_schema']['schema']==t['schema']
        sdk=load(path.with_name(t['id']+'.humanize.json'))
        packet,_=json.JSONDecoder().raw_decode(sdk['messages'][-1]['content'].split(MARKER,1)[1])
        assert len(sdk['messages'])==1 and packet['ticket']==t['id']
        assert request['messages'][1]['content']=='\n\n'.join(x[1] for x in packet['sections'] if x[0] not in r['omitted_sections'])
        if r['status']=='completed':
            for name,content,_ in packet['sections']:
                if name=='source_facts' and content!='null':
                    if name in r['omitted_sections']:
                        fact_omissions+=1
                    else:
                        fact_requests+=1
            reply=load(path.with_name(t['id']+'.response.json'))
            assert digest(reply)==r['response_sha256']
            if t['status']=='sdk_completed':
                assert json.loads(reply['choices'][0]['message']['content'])==t['reply']
            elif t['status']=='received_prefix_recovered':
                from kernel_agent.structured_recovery import recover
                assert t['role']=='writer'
                prop=t['schema']['properties']['tool']
                allowed=prop.get('enum',[prop['const']] if 'const' in prop else [])
                schema=c.action_schema(allowed)
                assert schema.model_json_schema()==t['schema']
                recovered=recover(reply['choices'][0]['message']['content'],schema,allowed)
                assert recovered and recovered[0].model_dump(mode='json')==t['reply'] and recovered[1]==t['recovery']
            intervals += [(r['dispatched_utc'],1),(r['finished_utc'],-1)]
    for r in ledger['proposals']:
        assert staged.text_hash(r['code'])==r['source_sha256'] and Path(r['proposal_path']).read_text()==r['code']
    for r in ledger['candidates']:
        assert admit(r['code'])==r['source_sha256'] and Path(r['path']).read_text()==r['code']
    if parent:
        assert ledger['proposals'][0]['id']=='c0' and ledger['proposals'][0]['code']==parent['code']
        assert ledger['events'][0].get('shared_parent') and ledger['events'][0]['result']==parent['evaluation']
    # One atomic test has two host events but exactly one model reply. Every source is
    # reconstructed from that exact reply; the host never fills in missing kernel code.
    events=[(i,e) for i,e in enumerate(ledger['events']) if not e.get('shared_parent') and e['tool']!='decode_reply']
    cursor=0
    bindings=[]
    unlogged_host_errors=[]
    for t in tickets:
        if t['role']!='writer' or t['status'] not in {'sdk_completed','received_prefix_recovered'}:
            continue
        a=dict(t['reply'].get('action',t['reply']))
        if a['tool']=='submit_and_evaluate':
            a['tool']='test'
        if cursor==len(events) and a['tool']=='read' and a['argument']=='example' and value.get('task_id') in {'gemm','attention'} and value.get('failure',{}).get('type')=='KeyError':
            assert value['failure']['message']==repr(value['task_id'])
            assert t is [x for x in tickets if x['role']=='writer' and x['status']=='sdk_completed'][-1]
            unlogged_host_errors.append({'call':t['id'],'tool':'read','argument':'example',
                'failure':value['failure'],'scope':'Actual terminal host exception before event append; no synthetic event or successful tool result inserted'})
            continue
        index,e=events[cursor]
        cursor+=1
        assert e['tool']==('submit' if a['tool']=='test' else a['tool'])
        if a['tool'] in {'test','submit','patch'} and e['result'].get('status')!='repeat_refused':
            try:
                if a['tool']=='patch':
                    state=prefix(ledger,index,contract)
                    sha=state.resolve('patch',a['argument'])
                    source=next(r['code'] for r in reversed(state.proposals) if r['source_sha256']==sha)
                    code=span_repair.apply(json.dumps({'draft':a['reason'] or 'patch','plan':'evaluate',
                        'base_sha256':sha,'edits':a['edits']}),source)['code']
                else:
                    code=line_protocol.proposal(json.dumps({'draft':a['reason'] or 'source','plan':'evaluate',
                        'code_lines':a['code_lines']}))['code']
            except (ValueError,SyntaxError):
                assert e['result']['status']=='tool_error'
                continue
            assert staged.text_hash(code)==e['source_sha256']
            bindings.append({'call':t['id'],'source_sha256':e['source_sha256']})
            if a['tool']=='test' and e['result'].get('status')=='admitted':
                _,measurement=events[cursor]
                cursor+=1
                assert measurement['tool'] in {'evaluate','test'} and measurement['source_sha256']==e['source_sha256']
    assert cursor==len(events)
    for e in ledger['events']:
        if e['tool']=='evaluate' and not e.get('shared_parent') and e['result'].get('status')=='correct':
            assert e['result']['source_sha256']==e['source_sha256'] and e['result']['device_binding']['verified']
            if value.get('task_id') in {'gemm','attention'}:
                assert sum(e['result']['candidate_jit_dispatches'].values())>0
        if e['tool']=='profile' and e['result'].get('status') in {'profiled','profiled_activity'}:
            matches=[p for p in folder.glob('profile-*/profile.json') if load(p)==e['result']]
            assert matches and e['result']['source_sha256']==e['source_sha256']
            if e['result']['status']=='profiled_activity':
                assert e['result'].get('activity_device_verified') or e['result'].get('device_binding',{}).get('verified')
                trace=load(matches[0].parent/'cuda-activity.json')
                assert any(t.get('cat')=='kernel' for t in trace['traceEvents'])
            else:
                assert e['result']['device_verified']
    phases=[]
    for p in value.get('stages',[]):
        stage=p['stage']
        docs=folder/'docs'/stage
        planners=[t for t in tickets if t['stage']==stage and t['role']=='planner' and t['status']=='sdk_completed']
        if planners:
            assert planners[0]['reply']['text']==(docs/'draft.md').read_text()
        if contract and 'plan_sha256' in p:
            assert c.Plan.model_validate(planners[-1]['reply']).model_dump_json()==c.Plan.model_validate(load(docs/('plan-revision.json' if (docs/'plan-revision.json').exists() else 'plan.json'))['plan']).model_dump_json()
            assert staged.text_hash(c.Plan.model_validate(planners[-1]['reply']).model_dump_json())==p['plan_sha256']
        if not contract:
            for kind in ['draft','plan']:
                assert staged.text_hash((docs/(kind+'.md')).read_text())==p[kind+'_sha256']
        indices=[i for i,e in enumerate(ledger['events']) if e['stage']==stage and not e.get('shared_parent')]
        start=min(indices) if indices else len(ledger['events'])
        before=prefix(ledger,start,contract)
        reviews=[]
        for rr in p['reviews']:
            end=next(i+1 for i,e in enumerate(ledger['events']) if e['id']==rr['after_event']) if rr['after_event'] else 0
            state=prefix(ledger,end,contract)
            state.phase_start=start
            best=staged.select(before.rows)
            state.phase_entry=(best['source_sha256'] if best else None) if contract else (before.proposals[-1]['source_sha256'] if before.proposals else None)
            if contract:
                passed,errors=c.gate(stage,state,c.Review.model_validate(rr['review'])) if rr['review'] else (False,['Received malformed reviewer output'])
                assert passed==rr['gate_passed'] and errors==rr['gate_errors']
                assert state.readiness(stage)==rr['readiness']
            else:
                passed,why,_=g.gate(stage,state,g.Review.model_validate(rr['review']))
                assert passed==rr['gate_passed'] and why==rr['gate_reason']
            reading=copy.deepcopy(rr)
            if reading.get('review'):
                for k in ['missing','next_step','reason']:
                    if k in reading['review']:
                        reading['review'][k]=reading['review'][k][:300]
                if 'hypotheses' in reading['review']:
                    reading['review']['hypotheses']=[h[:220] for h in reading['review']['hypotheses'][:3]]
                for cl in reading['review'].get('claims',[]):
                    cl['hypothesis']=cl['hypothesis'][:220]
            reviews.append(reading)
        assert p['accepted']==any(r['gate_passed'] for r in p['reviews'])
        phases.append({'stage':stage,'accepted':p['accepted'],'reviews':reviews,
            'draft_excerpt':(docs/'draft.md').read_text()[:200] if (docs/'draft.md').exists() else ''})
    assert all(p['accepted'] for p in value.get('stages',[])[:-1])
    best=staged.select(ledger['candidates'])
    assert (best['source_sha256'] if best else None)==value['selection']['source_sha256']
    if value.get('final') and value['final']['status']=='correct':
        assert value['final']['source_sha256']==value['selection']['source_sha256'] and value['final']['device_binding']['verified']
        if value.get('task_id') in {'gemm','attention'}:
            assert sum(value['final']['candidate_jit_dispatches'].values())>0
    counts=collections.Counter(r['status'] for r in receipts)
    final=decision(value['final']) if value.get('final') else {'correct':False,'promoted':False}
    full=len(phases)==3 and all(p['accepted'] for p in phases)
    return {'actual_calls':sum(counts[k] for k in ['completed','dispatched','response_unknown']),
        'unknown_calls':counts['dispatched']+counts['response_unknown'],'not_dispatched':counts['not_dispatched'],
        'source_fact_requests':fact_requests,'source_fact_omissions':fact_omissions,'received_recoveries':sum(t['status']=='received_prefix_recovered' for t in tickets),'opportunities':len(tickets),'input_tokens':sum((r.get('usage') or {}).get('prompt_tokens',0) for r in receipts),
        'output_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in receipts),
        'submissions':ledger.get('submissions',0),'new_searches':ledger.get('gpu_calls',0),'profile_calls':ledger.get('profiles',0),
        'blocked_repeats':sum(e['result'].get('status') in {'repeat_refused','semantic_repeat'} for e in ledger['events']),
        'tool_errors':sum(e['result'].get('status') in {'tool_error','no_matches'} for e in ledger['events']),
        'model_wall_seconds':round(sum(r.get('elapsed_seconds',0) for r in receipts),3),
        'stages':phases,'final_decision':final,'selected_sha256':value['selection']['source_sha256'],
        'full_stage_complete':full,'full_stage_and_final_correct':full and final['correct'],
        'changed_parent_ast':bool(parent and best and g.ast_identity(parent['code'])!=g.ast_identity(best['code'])),
        'unlogged_host_tool_errors':unlogged_host_errors,'source_bindings':bindings,'failure':value.get('failure'),
        'trace':[{'id':e['id'],'stage':e['stage'],'tool':e['tool'],'sha':e['source_sha256'],
            'status':e['result'].get('status','returned'),'reason_hypothesis':e.get('reason','')[:220],
            'message':message_excerpt(e['result'].get('message',''))} for e in ledger['events']]},intervals


def main():
    global c
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--finalize',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    directory=a.directory
    plan,state=load(directory/'plan.json'),load(directory/'checkpoint.json')
    reference_corrected=plan['kind'].startswith('kernel_reference_')
    recovery_corrected=reference_corrected or plan['kind'].startswith('kernel_recovery_')
    arguments_corrected=recovery_corrected or plan['kind'].startswith('kernel_argument_')
    corrected=arguments_corrected or plan['kind'].startswith('kernel_module_')
    if recovery_corrected:
        from kernel_agent import staged_contract_v4
        c=staged_contract_v4
    elif arguments_corrected:
        from kernel_agent import staged_contract_v3
        c=staged_contract_v3
    elif corrected:
        from kernel_agent import staged_contract_v2
        c=staged_contract_v2
    assert digest(plan)==state['plan_sha256']
    assert digest({k:v for k,v in state.items() if k!='state_sha256'})==state['state_sha256']
    for name,sha in plan['source_sha256'].items():
        assert hashlib.sha256((directory/'sources'/name).read_bytes()).hexdigest()==sha
    settlement=load(directory/'settlement.json') if (directory/'settlement.json').exists() else None
    if settlement:
        assert settlement['plan_sha256']==state['plan_sha256']
        assert settlement['original_checkpoint_file_sha256']==hashlib.sha256((directory/'checkpoint.json').read_bytes()).hexdigest()
        assert digest({k:v for k,v in settlement.items() if k!='settlement_sha256'})==settlement['settlement_sha256']
    view=settlement or state
    blocks,intervals=[],[]
    for row in view['blocks']:
        if row['status']!='complete':
            continue
        b=row['block']
        folder=directory/'blocks'/b['id']
        raw=load(folder/'result.json')
        assert digest(raw)==row['result_sha256']
        committed=load(folder/'generation.json')
        assert committed['generation_committed_utc']==raw['generation_committed_utc']
        compact={**b,'status':row['status'],'port':row['endpoint']['port'],'arms':{}}
        parent=load(Path(plan['parents']['parents'][b['task_id']]['path'])) if b['stratum']=='repair' else None
        if parent:
            assert digest(parent)==plan['parents']['parents'][b['task_id']]['parent_digest']
        for arm in plan['arms']:
            value=raw['arms'][arm]
            assert value['selection']['committed_utc']<=raw['generation_committed_utc']<=raw['finished_utc']
            assert committed['arms'][arm]['selection']==value['selection'] and committed['arms'][arm].get('final') is None
            if value.get('final',{}):
                before=value['final'].get('resource_audit',{}).get('before',{})
                if before:
                    assert before['observed_utc']>=raw['generation_committed_utc']
                else:
                    assert value['final']['status']!='correct'
            compact['arms'][arm],spans=audit(folder/arm,value,arm,parent)
            intervals+=spans
        blocks.append(compact)
    active=peak=0
    for _,delta in sorted(intervals,key=lambda x:(x[0],x[1])):
        active+=delta
        peak=max(peak,active)
    summary={'kind':plan['kind'],'status':view['status'],'plan_sha256':state['plan_sha256'],
        'scheduled_outcomes':plan['scheduled_arms'],'completed_blocks':len(blocks),'blocks':blocks,
        'services':state['services'],'peak_completed_request_concurrency':peak,'scope':plan['scope'],
        'source_files':len(plan['source_sha256']),
        'current_sources_match':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items())}
    live=collections.Counter()
    for path in directory.glob('blocks/*/*/requests/call-*.json'):
        if path.stem.removeprefix('call-').isdigit():
            live[load(path)['status']]+=1
    summary['live_request_statuses']=dict(live)
    if settlement:
        summary.update(settlement_sha256=settlement['settlement_sha256'],
                       unfinished_planned_outcomes=2*sum(r['status']!='complete' for r in view['blocks']))
    expected=sum(r['status']=='complete' for r in view['blocks']) if settlement else len(plan['blocks'])
    name=('staged-reference-' if reference_corrected else 'staged-recovery-' if recovery_corrected else 'staged-arguments-' if arguments_corrected else 'staged-module-' if corrected else 'staged-contract-')+plan['cohort']+'-summary.json'
    manifest=directory/'evidence-manifest.json'
    if a.verify_only:
        assert view['status'] in {'complete','stopped_diagnostic'} and len(blocks)==expected
        files=load(manifest)
        assert all(hashlib.sha256((directory/n).read_bytes()).hexdigest()==h for n,h in files.items())
        print(json.dumps({'status':'passed','blocks':len(blocks),'raw_files':len(files),'scope':'Read-only wire/source/stage/final reconstruction; no new model or GPU calls'}))
        return
    analyzer=directory/'analysis-tools'/('export-contract-'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]+'.py')
    analyzer.parent.mkdir(exist_ok=True)
    if not analyzer.exists():
        analyzer.write_bytes(Path(__file__).read_bytes())
    if a.finalize:
        assert view['status'] in {'complete','stopped_diagnostic'} and len(blocks)==expected
        files={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in {'evidence-manifest.json','supervisor.log'} and p.suffix!='.lock'
            and '.hmz' not in p.parts and '.humanize' not in p.parts and '__pycache__' not in p.parts}
        if manifest.exists():
            assert load(manifest)==files
        else:
            atomic_json(manifest,files)
        summary.update(raw_manifest_files=len(files),raw_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    summary['summary_sha256']=digest(summary)
    atomic_json(ROOT/'results'/name,summary)
    print(json.dumps({'status':view['status'],'blocks':len(blocks),'summary_sha256':summary['summary_sha256']}))


if __name__=='__main__':
    main()
