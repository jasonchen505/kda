"""Verify the missing real worker/example route on all three Qwen services, excluded from scores."""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent import model_pool  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402

STORE=Path('/data/home/chenyizhou/kernel-agent-storage')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--gpu-python',required=True)
    a=p.parse_args()
    if not a.directory.resolve().is_relative_to(STORE/'experiments'):
        raise ValueError('Raw integration evidence stays remote')
    a.directory.mkdir(exist_ok=False)
    endpoints=model_pool.discover()
    assert len(endpoints)==3
    atomic_json(a.directory/'plan.json',{'endpoints':endpoints,'max_model_calls':9,
        'scope':'Three role calls per service;actual reference read;no scored kernel/noGPU'})
    for name in ['src/kernel_agent/representative_reference_router.py','scripts/staged_reference_worker.py',
                 'scripts/staged_contract_worker.py','flows/staged_reference_smoke.py',
                 'scripts/check_staged_reference_preflight.py','docs/staged-reference-correction.md','data/representative-tasks.json']:
        target=a.directory/'sources'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    def smoke(e):
        d=a.directory/f'sdk-{e["port"]}'
        config={'directory':str(d),'endpoint':e,'seed':2026101840,'task_id':'attention',
            'contract':'Integration only: inspect example reference, do not implement a kernel.',
            'arm':'staged_transfer','policy':'guarded','flow':'staged_reference_smoke.py',
            'gpu_python':a.gpu_python,'lock_file':str(a.directory/'unused.lock')}
        atomic_json(d/'configuration.json',config)
        with (d/'worker.log').open('wb') as log:
            child=subprocess.run([str(STORE/'runtimes/multistage/bin/python'),str(ROOT/'scripts/staged_reference_worker.py'),
                '--configuration',str(d/'configuration.json'),'--smoke'],stdout=log,stderr=subprocess.STDOUT,
                timeout=600,env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT)})
        receipts=[json.loads(f.read_text()) for f in (d/'requests').glob('call-*.json')
                  if f.stem.removeprefix('call-').isdigit()]
        return {'port':e['port'],'exit_code':child.returncode,'smoke':(d/'smoke.json').exists(),
                'confirmed_calls':sum(x['status']=='completed' for x in receipts),
                'unknown_calls':sum(x['status'] in {'dispatched','response_unknown'} for x in receipts)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(smoke,endpoints))
    value={'status':'passed' if all(x['exit_code']==0 and x['smoke'] and x['confirmed_calls']==3 and not x['unknown_calls'] for x in rows) else 'failed',
        'services':rows,'scope':'9integration calls at most;0GPU,no kernel/full-stage score'}
    atomic_json(a.directory/'preflight.json',value)
    print(json.dumps(value))
    if value['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
