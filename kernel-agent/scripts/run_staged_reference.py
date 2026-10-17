"""Frozen paired v2-guard versus capability-aligned staged agent experiments."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from kernel_agent import model_pool  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.protocol import decision  # noqa: E402
from kernel_agent.staged_contract_v4 import Tools as ContractTools  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.run_repair_study import Study, source_files  # noqa: E402

STORE = Path('/data/home/chenyizhou/kernel-agent-storage')
RUNTIME = STORE/'runtimes/multistage/bin/python'
PARENTS = STORE/'experiments/guarded-parents-v3/parents.json'
ARMS = ['guarded','contract']


def make_plan(cohort):
    tasks=['gemm','attention']
    blocks=[{'id':f'generation-{t}-s{s}','task_id':t,'seed':s,'stratum':'generation','model_call_cap':48}
            for t in tasks for s in [2026101843,2026101844]]
    random.Random(2026101800).shuffle(blocks)
    executed = {'scripts/run_staged_reference.py','scripts/staged_reference_worker.py','scripts/staged_contract_worker.py','scripts/run_repair_study.py',
        'scripts/run_campaign.py','scripts/local_proxy.py','scripts/evaluate_candidate.py','scripts/evaluate_serial.py',
        'scripts/evaluate_coexecuted.py','scripts/profile_staged_kernel.py','scripts/evaluate_representative.py',
        'scripts/evaluate_representative_serial.py','scripts/profile_representative.py','scripts/evaluate_representative_verified.py',
        'scripts/evaluate_representative_verified_serial.py','data/tasks.json','pyproject.toml','uv.lock'}
    sources = {p:h for p,h in source_files().items() if p.startswith(('src/','references/')) or p in executed}
    for p in ['data/repair-transfer-tasks.json','data/representative-tasks.json','docs/staged-contract-plan.md',
              'flows/staged_kernel_contract_v4.py','flows/staged_kernel_guarded_v2.py','flows/staged_kernel_representative_reference.py',
              'tests/unit/test_staged_contract.py','tests/integration/test_staged_contract_flow.py',
              'tests/unit/test_staged_contract_v2.py','tests/unit/test_representative_policy.py','tests/integration/test_staged_contract_v2_flow.py','docs/staged-module-correction.md','docs/staged-arguments-correction.md','tests/unit/test_staged_contract_v4.py','tests/integration/test_staged_contract_v4_flow.py','tests/unit/test_structured_recovery.py','tests/unit/test_representative_reference_router.py','tests/integration/test_staged_reference_smoke.py','docs/staged-recovery-transfer-plan.md','docs/staged-reference-correction.md']:
        sources[p] = hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    return {'schema_version':1,'kind':'kernel_reference_staged_'+cohort,'cohort':cohort,'blocks':blocks,
        'arms':ARMS,'tasks':tasks,'scheduled_arms':2*len(blocks),'calls_per_block':48,
        'max_model_calls':48*len(blocks),'context_limit':8192,'max_output_tokens':2048,
        'call_cap_per_arm':24,'candidate_cap_per_arm':6,'gpu_search_cap_per_arm':9,'profile_cap_per_arm':2,
        'source_sha256':sources,'evaluation_gpu':4,'parents':{},
        'runtime':{},'scope':'Frozen within-process paired controls: latest guarded v2 versus conservative complete-reply recovery, evidence-directed tool mask and optional bounded source facts bundle. '
        'Common process-local reference backend routes new task reads before legacy lookup; policy bundle unchanged from recovery-v4. Same call/source/GPU ceilings, unequal actual work; not single-component ablation. '
        'Prospective representative cohort: GEMM/attention, each2NEW generation seeds; independent from argument-core. '
        'Shared GPU; no external blind benchmark, full KDA reproduction or universal speedup claim.'}


class ContractStudy(Study):
    def __init__(self,directory,gpu_python,cohort):
        super().__init__(directory,gpu_python,plan=make_plan(cohort))

    def verify_sources(self):
        for name,sha in self.plan['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, 'Frozen source drift: '+name
        if self.plan['parents']:
            from kernel_agent.campaign import digest
            for p in self.plan['parents']['parents'].values():
                assert digest(json.loads(Path(p['path']).read_text()))==p['parent_digest']

    def run_block(self,block,endpoint):
        self.verify_sources()
        folder = self.directory/'blocks'/block['id']
        folder.mkdir(parents=True,exist_ok=True)
        dataset = 'representative-tasks.json' if block['task_id'] in {'gemm','attention'} else 'repair-transfer-tasks.json'
        spec = json.loads((ROOT/'data'/dataset).read_text())[block['task_id']]
        order = list(ARMS)
        random.Random(block['id']).shuffle(order)
        result = {'block':block,'endpoint':endpoint,'arms':{},'arm_order':order,'started_utc':model_pool.utc()}
        for arm in order:
            self.verify_sources()
            directory = folder/arm
            config = {'directory':str(directory),'endpoint':endpoint,'seed':block['seed'],
                'task_id':block['task_id'],'contract':spec['contract'],'arm':'staged_transfer','policy':arm,
                'flow':'staged_kernel_contract_v4.py' if arm=='contract' else (
                    'staged_kernel_representative_reference.py' if block['task_id'] in {'gemm','attention'} else 'staged_kernel_guarded_v2.py'),
                'gpu_python':self.gpu_python,'lock_file':str(self.directory/'gpu4.lock'),
                'initial_path':self.plan['parents']['parents'][block['task_id']]['path'] if block['stratum']=='repair' else ''}
            atomic_json(directory/'configuration.json',config)
            with (directory/'worker.log').open('wb') as log:
                child = subprocess.run([str(RUNTIME),str(ROOT/'scripts/staged_reference_worker.py'),
                    '--configuration',str(directory/'configuration.json')],stdout=log,stderr=subprocess.STDOUT,
                    timeout=2600,env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'PYTHONUNBUFFERED':'1'})
            path = directory/'result.json'
            value = json.loads(path.read_text()) if path.exists() else {'selection':{'source_sha256':None},
                'failure':{'worker_exit':child.returncode},'stages':[],'model_opportunities':0,'final':None}
            result['arms'][arm]=value
        result['generation_committed_utc']=model_pool.utc()
        atomic_json(folder/'generation.json',result)
        for arm,value in result['arms'].items():
            sha=value['selection']['source_sha256']
            if sha:
                tools=ContractTools(folder/arm,block['task_id'],'staged_transfer',self.gpu_python,self.directory/'gpu4.lock')
                value['final']=tools.evaluate({'path':str(folder/arm/'candidates'/(sha+'.py')),'source_sha256':sha},final=True)
                value['final_decision']=decision(value['final'])
            else:
                value['final_decision']={'correct':False,'promoted':False}
            atomic_json(folder/'final'/(arm+'.json'),value)
        result['finished_utc']=model_pool.utc()
        atomic_json(folder/'result.json',result)
        durations=[]
        for arm in order:
            for p in (folder/arm/'requests').glob('call-*.json'):
                if not p.stem.removeprefix('call-').isdigit():
                    continue
                r=json.loads(p.read_text())
                if r.get('status')=='completed':
                    durations.append(r['elapsed_seconds'])
        return result,sum(durations)/len(durations) if durations else 60


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--cohort',choices=['representative'],required=True)
    p.add_argument('--gpu-python',required=True)
    p.add_argument('--background',action='store_true')
    a=p.parse_args()
    if not a.directory.resolve().is_relative_to(STORE/'experiments'):
        raise ValueError('All new raw data must remain in remote experiment storage')
    if a.background:
        if a.directory.exists():
            raise FileExistsError('Never replay an existing ledger')
        endpoints=model_pool.discover()
        if len(endpoints)!=3:
            raise ValueError('Need all three authorized Qwen processes')
        study=ContractStudy(a.directory,a.gpu_python,a.cohort)
        command=[sys.executable,str(STORE/'bin/run.py'),'run_staged_reference.py','--directory',str(a.directory),
                 '--cohort',a.cohort,'--gpu-python',a.gpu_python]
        with (a.directory/'supervisor.log').open('wb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONUNBUFFERED':'1'})
        receipt={'pid':child.pid,'started_utc':model_pool.utc(),'plan_sha256':study.state['plan_sha256'],
                 'endpoints':endpoints,'model_call_cap':study.plan['max_model_calls']}
        atomic_json(a.directory/'launch.json',receipt)
        print(json.dumps(receipt))
        return
    with acquire(a.directory/'supervisor.lock',timeout=0):
        study=ContractStudy(a.directory,a.gpu_python,a.cohort)
        started=time.time()
        try:
            study.run()
        finally:
            atomic_json(a.directory/'supervisor-exit.json',{'status':study.state['status'],
                'finished_utc':model_pool.utc(),'elapsed_seconds':time.time()-started})


if __name__=='__main__':
    main()
