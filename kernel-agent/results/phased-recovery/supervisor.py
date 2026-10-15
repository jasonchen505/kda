import datetime, hashlib, json, os, subprocess
from pathlib import Path
root=Path('/data/home/chenyizhou/kernel-agent-phased')
folder=root/'runs/phased-v2'
folder.mkdir(parents=True,exist_ok=True)
py='/data/home/chenyizhou/kernel-agent/.venv/bin/python'
env=os.environ.copy()
env['PATH']='/data/home/chenyizhou/.local/bin:'+env['PATH']
env['PYTHONPATH']=str(root/'src')

def run(args,timeout=2100):
    with (folder/'supervisor.log').open('a') as log:
        child=subprocess.run([py,*args],cwd=root,env=env,stdout=log,stderr=log,timeout=timeout)
    if child.returncode:
        raise RuntimeError(f'{args[0]} exit {child.returncode}')

def wire_manifest():
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((folder/'wire').rglob('*.json'))}

receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    run(['scripts/run_phased.py','--pid','2132787','--directory',str(folder)])
    state=json.loads((folder/'generation.json').read_text())
    if any(r['status']!='completed' for r in state['jobs']):
        raise RuntimeError('not all generation results committed; preserve failures without retries')
    gen_sha=hashlib.sha256((folder/'generation.json').read_bytes()).hexdigest()
    wire=wire_manifest()
    final=root/'runs/phased-final-v2'
    base=['scripts/finish_deferred.py','--generation',str(folder/'generation.json'),'--directory',str(final)]
    run([*base,'--prepare'],120)
    first=sorted((final/'queues').glob('*/checkpoint.json'))[0]
    run(['scripts/validate_phased_recovery.py','--queue',str(first),'--receipt',str(final/'kill-receipt.json')],60)
    run(base,600)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (final/'queues').glob('*/checkpoint.json')}
    run(base,120)
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (final/'queues').glob('*/checkpoint.json')}
    if before!=after or wire!=wire_manifest() or gen_sha!=hashlib.sha256((folder/'generation.json').read_bytes()).hexdigest():
        raise RuntimeError('reopening complete queues changed committed evidence')
    summary=json.loads((final/'summary.json').read_text())
    queues=[json.loads(p.read_text()) for p in (final/'queues').glob('*/checkpoint.json')]
    receipt.update(status='complete',generation_sha256=gen_sha,wire_files=len(wire),wire_sha256=wire,
        final_correct=sum(bool(j['correct']) for j in summary['jobs']),final_promoted=sum(bool(j['promoted']) for j in summary['jobs']),
        imported_searches=sum(len(q['search']) for q in queues),
        final_evaluator_calls=sum(sum('evaluation' in o for o in q['final']['observations']) for q in queues),
        interrupted_final_evaluations=sum(sum(o.get('status')=='interrupted_evaluation' for o in q['final']['observations']) for q in queues),
        generation_and_wire_unchanged=True,completed_reopen_unchanged=True,
        new_model_calls_during_finalization=0,new_search_calls_during_finalization=0)
except Exception as error:
    receipt.update(status='failed',error=str(error))
receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(folder/'supervisor-exit.json').write_text(json.dumps(receipt,indent=2))
