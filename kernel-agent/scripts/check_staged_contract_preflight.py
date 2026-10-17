"""Three real SDK schema smokes plus independent oracle math; excluded from scored cohorts."""
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
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.evaluate_coexecuted import evaluate  # noqa: E402

STORE=Path('/data/home/chenyizhou/kernel-agent-storage')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--gpu-python',required=True)
    p.add_argument('--oracle-only',action='store_true')
    a=p.parse_args()
    if not a.directory.resolve().is_relative_to(STORE/'experiments'):
        raise ValueError('Raw preflight stays remote')
    a.directory.mkdir(exist_ok=False)
    for name in ['scripts/validate_representative_math.py','scripts/evaluate_representative.py',
                 'src/kernel_agent/representative.py','src/kernel_agent/representative_tasks.py',
                 'scripts/check_staged_contract_preflight.py','data/representative-tasks.json']:
        target=a.directory/'integration-sources'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    endpoints=[] if a.oracle_only else model_pool.discover()
    assert a.oracle_only or len(endpoints)==3
    atomic_json(a.directory/'plan.json',{'endpoints':endpoints,'max_model_calls':0 if a.oracle_only else 9,'scope':'Integration only'})
    def smoke(e):
        d=a.directory/f'sdk-{e["port"]}'
        config={'directory':str(d),'endpoint':e,'seed':2026101800,'task_id':'affine_silu','contract':'Integration only',
                'arm':'staged_transfer','policy':'contract','gpu_python':a.gpu_python,'lock_file':str(a.directory/'gpu4.lock')}
        atomic_json(d/'configuration.json',config)
        with (d/'worker.log').open('wb') as log:
            child=subprocess.run([str(STORE/'runtimes/multistage/bin/python'),str(ROOT/'scripts/staged_contract_worker.py'),
                '--configuration',str(d/'configuration.json'),'--smoke'],stdout=log,stderr=subprocess.STDOUT,timeout=600,
                env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT)})
        result={'port':e['port'],'exit_code':child.returncode,'smoke':(d/'smoke.json').exists()}
        if result['smoke']:
            r=json.loads((d/'smoke.json').read_text())
            assert r['event']['tool']=='read' and not r['review']['accept']
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        smokes=list(pool.map(smoke,endpoints))
    with acquire(a.directory/'gpu4.lock') as (lock,waited):
        oracle=evaluate([a.gpu_python,str(ROOT/'scripts/validate_representative_math.py')],4,
                        pass_fds=(lock.fileno(),),require_device_proof=True)
    atomic_json(a.directory/'oracle.json',oracle)
    value={'status':'passed' if all(s['smoke'] and s['exit_code']==0 for s in smokes) and oracle['status']=='oracle_validated' else 'failed',
        'smokes':smokes,'oracle_checks':len(oracle.get('checks',[])),
        'oracle_status':oracle['status'],'scope':'Integration only; no generated kernel or agent success'}
    atomic_json(a.directory/'preflight.json',value)
    print(json.dumps(value))
    if value['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
