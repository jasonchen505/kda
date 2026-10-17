"""Read-only paired Attention audit; seal once, then verify without mutation."""
import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from kernel_agent import staged_contract_v4  # noqa: E402
from kernel_agent.attention_repair import RepairTools  # noqa: E402
from kernel_agent.campaign import atomic_json,digest  # noqa: E402
from kernel_agent.repair_progress import alpha_identity  # noqa: E402
from kernel_agent.staged_gateway import MARKER  # noqa: E402
from scripts import export_staged_contract as base  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def audit_arm(folder,value,arm,parent):
    base.c=staged_contract_v4
    row,intervals=base.audit(folder,value,'contract',parent)
    ledger=load(folder/'tools.json') if (folder/'tools.json').exists() else {'events':[],'candidates':[],'proposals':[],'gpu_calls':0}
    configuration=load(folder/'configuration.json')
    counters=collections.Counter()
    by_sha={r['source_sha256']:r for r in ledger['candidates']}
    for event in ledger['events']:
        result=event['result']
        status=result.get('status')
        if event.get('shared_parent'):
            continue
        if event['tool']=='evaluate':
            counters[status]+=1
            if status=='repair_no_progress':
                assert arm=='repair'
                current=by_sha[event['source_sha256']]
                prior=by_sha[result['prior_source_sha256']]
                assert alpha_identity(current['code'])==alpha_identity(prior['code'])==result['alpha_identity']
                assert 'device_binding' not in result
            elif status in {'correct','numeric_error','compile_error','runtime_error','contract_error','policy_error'}:
                assert result['device_binding']['verified']
        elif status=='semantic_repeat':
            counters['semantic_repeat']+=1
    assert ledger['gpu_calls']==sum(counters[s] for s in
        ['correct','numeric_error','compile_error','runtime_error','contract_error','policy_error','evaluator_error'])
    compiled=[str(e['result'].get('message','')) for e in ledger['events']
              if e['tool']=='evaluate' and e['result'].get('status')=='compile_error']
    signatures=[re.findall(r'(?m)^(?:[A-Za-z_]\w*\.)?(?:\w*Error|Exception)(?:\(|:).*$',message)
                for message in compiled]
    repeated_compile=sum(n-1 for n in collections.Counter(s[-1] for s in signatures if s).values())
    unclassified_compile=sum(not s for s in signatures)
    feedback_requests=numeric_requests=compiler_requests=0
    feedback_calls=[]
    for ticket in load(folder/'calls.json'):
        path=folder/'requests'/(ticket['id']+'.json')
        if not path.exists() or load(path)['status']!='completed':
            continue
        receipt=load(path)
        request=load(path.with_name(ticket['id']+'.request.json'))
        assert request['seed']==configuration['seed']+ticket['allocated'] and request['temperature']==.2
        assert request['chat_template_kwargs']=={'enable_thinking':False}
        sdk=load(path.with_name(ticket['id']+'.humanize.json'))
        packet,_=json.JSONDecoder().raw_decode(sdk['messages'][-1]['content'].split(MARKER,1)[1])
        sections={name:(content,required) for name,content,required in packet['sections']}
        if 'repair_feedback' not in sections:
            assert arm=='control' or not [e for e in ledger['events']
                if e['tool']=='evaluate' and e['result'].get('status') in {'numeric_error','compile_error','runtime_error'}
                and e['finished']<=datetime.fromisoformat(receipt['dispatched_utc']).timestamp()]
            continue
        assert arm=='repair' and sections['repair_feedback'][1]
        assert 'repair_feedback' not in receipt['omitted_sections']
        value=json.loads(sections['repair_feedback'][0])
        end=sum(e['finished']<=datetime.fromisoformat(receipt['dispatched_utc']).timestamp() for e in ledger['events'])
        state=base.prefix(ledger,end,True)
        state.__class__=RepairTools
        assert value==state.repair_feedback()
        feedback_requests+=1
        numeric_requests+=value['numeric_witness'] is not None
        compiler_requests+=value['compiler_observation'] is not None
        feedback_calls.append({'call':ticket['id'],'failed_event':value['failed_event'],
                               'source_sha256':value['failed_source_sha256'],'status':value['status']})
    first_correct=next((e for e in ledger['events'] if not e.get('shared_parent') and e['tool']=='evaluate' and e['result'].get('status')=='correct'),None)
    first_binding=next((b for b in row['source_bindings'] if first_correct and b['source_sha256']==first_correct['source_sha256']),None)
    row.update(first_correct_call=int(first_binding['call'].split('-')[-1]) if first_binding else None,
               search_correct=any(r['evaluation'].get('status')=='correct' for r in ledger['candidates']),
               outcome_counts=dict(counters),repeated_compile_diagnostics=repeated_compile,unclassified_compile_diagnostics=unclassified_compile,repair_feedback_requests=feedback_requests,
               numeric_feedback_requests=numeric_requests,compiler_feedback_requests=compiler_requests,
               feedback_bindings=feedback_calls,
               numeric_failures=[{'event':e['id'],'source_sha256':e['source_sha256'],
                    'shape':e['result'].get('shape'),'dtype':e['result'].get('dtype'),
                    'causal':e['result'].get('causal'),'checks_passed':e['result'].get('checks_passed'),
                    'failure_pass':e['result'].get('failure_pass'),'numeric_witness':e['result'].get('numeric_witness'),
                    'semantic_relations':e['result'].get('semantic_relations')}
                    for e in ledger['events'] if e['tool']=='evaluate' and e['result'].get('status')=='numeric_error'],
               compiler_failures=[{'event':e['id'],'source_sha256':e['source_sha256'],
                    'message':base.message_excerpt(e['result'].get('message',''))}
                    for e in ledger['events'] if e['tool']=='evaluate' and e['result'].get('status')=='compile_error'])
    return row,intervals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--audit-settled-arms',action='store_true')
    args=parser.parse_args()
    directory=args.directory
    plan=load(directory/'plan.json')
    state=load(directory/'checkpoint.json')
    assert digest(plan)==state['plan_sha256']
    assert digest({k:v for k,v in state.items() if k!='state_sha256'})==state['state_sha256']
    assert all(hashlib.sha256((directory/'sources'/n).read_bytes()).hexdigest()==sha
               for n,sha in plan['source_sha256'].items())
    if args.audit_settled_arms:
        rows=[]
        for entry in state['blocks']:
            block=entry['block']
            parent=load(Path(plan['parents']['parents'][block['parent_key']]['path'])) if block.get('parent_key') else None
            for arm in plan['arms']:
                folder=directory/'blocks'/block['id']/arm
                if not (folder/'result.json').exists():
                    continue
                row,_=audit_arm(folder,load(folder/'result.json'),arm,parent)
                rows.append({'block':block['id'],'arm':arm,'calls':row['actual_calls'],'status':'passed'})
        print(json.dumps({'scope':'Read-only settled-arm audit; no final score or file mutation','arms':rows}))
        return
    blocks,intervals=[],[]
    for entry in state['blocks']:
        if entry['status']!='complete':
            continue
        block=entry['block']
        folder=directory/'blocks'/block['id']
        value=load(folder/'result.json')
        assert digest(value)==entry['result_sha256']
        generation=load(folder/'generation.json')
        parent=None
        if block.get('parent_key'):
            meta=plan['parents']['parents'][block['parent_key']]
            parent=load(Path(meta['path']))
            assert digest(parent)==meta['parent_digest']
        compact={**block,'port':entry['endpoint']['port'],'arms':{}}
        for arm in plan['arms']:
            outcome=value['arms'][arm]
            assert outcome['selection']==generation['arms'][arm]['selection']
            assert generation['arms'][arm]['final'] is None
            assert outcome['selection']['committed_utc']<=value['generation_committed_utc']<=value['finished_utc']
            if outcome.get('final',{}):
                assert outcome['final']['resource_audit']['before']['observed_utc']>=value['generation_committed_utc']
            compact['arms'][arm],spans=audit_arm(folder/arm,outcome,arm,parent)
            intervals+=spans
        blocks.append(compact)
    active=peak=0
    for _,delta in sorted(intervals,key=lambda x:(x[0],x[1])):
        active+=delta
        peak=max(peak,active)
    receipts=collections.Counter(load(p)['status'] for p in directory.glob('blocks/*/*/requests/call-*.json')
                                  if p.stem.removeprefix('call-').isdigit())
    summary={'kind':plan['kind'],'status':state['status'],'plan_sha256':state['plan_sha256'],
             'scheduled_outcomes':plan['scheduled_arms'],'completed_blocks':len(blocks),'blocks':blocks,
             'services':state['services'],'peak_completed_request_concurrency':peak,'scope':plan['scope'],
             'source_files':len(plan['source_sha256']),'live_request_statuses':dict(receipts),
             'current_sources_match':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha
                                         for n,sha in plan['source_sha256'].items())}
    manifest=directory/'evidence-manifest.json'
    if args.verify_only:
        assert state['status']=='complete' and len(blocks)==len(plan['blocks'])
        files=load(manifest)
        assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha for name,sha in files.items())
        print(json.dumps({'status':'passed','blocks':len(blocks),'raw_files':len(files),
                          'scope':'Read-only wire/source/feedback/rename/stage/final audit; no new model/GPU calls'}))
        return
    if manifest.exists():
        raise ValueError('Closed cohort: use --verify-only; do not replace its sealed summary')
    if args.finalize:
        assert state['status']=='complete' and len(blocks)==len(plan['blocks'])
        analyzer=directory/'analysis-tools'/Path(__file__).name
        analyzer.parent.mkdir(exist_ok=True)
        analyzer.write_bytes(Path(__file__).read_bytes())
        files={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(directory.rglob('*')) if p.is_file() and
               p.name not in {'evidence-manifest.json','supervisor.log'} and p.suffix!='.lock' and
               '.hmz' not in p.parts and '.humanize' not in p.parts and '__pycache__' not in p.parts}
        atomic_json(manifest,files)
        summary.update(raw_manifest_files=len(files),raw_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    summary['summary_sha256']=digest(summary)
    atomic_json(ROOT/'results/attention-repair-summary.json',summary)
    print(json.dumps({'status':summary['status'],'blocks':len(blocks),'summary_sha256':summary['summary_sha256']}))


if __name__=='__main__':
    main()
