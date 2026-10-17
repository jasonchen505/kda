"""Compact integration receipts; never count preparatory work as agent outcomes."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent.campaign import atomic_json,digest  # noqa: E402

STORE=Path('/data/home/chenyizhou/kernel-agent-storage/experiments')


def main():
    base=STORE/'contract-preflight-v1'
    counts={}
    for folder in base.glob('sdk-*'):
        receipts=[json.loads(p.read_text()) for p in (folder/'requests').glob('call-*.json')
                  if p.stem.removeprefix('call-').isdigit()]
        counts[folder.name]={'confirmed_calls':sum(r['status']=='completed' for r in receipts),
                            'unknown_calls':sum(r['status'] in {'dispatched','response_unknown'} for r in receipts)}
    passed=json.loads((STORE/'contract-oracle-v3/oracle.json').read_text())
    data={'sdk_services':counts,'oracle_checks':len(passed['checks']),
        'oracle_device_verified':passed['device_binding']['verified'],
        'oracle_status':passed['status'],'preserved_preparation_failures':[
            {'folder':str(base),'reason':'GPU child import path missing scripts package','scored_model_calls':0},
            {'folder':str(STORE/'contract-oracle-v2'),'reason':'GPU-only environment lacks controller Pydantic; task metadata dependency split','scored_model_calls':0}],
        'scope':'9 real SDK schema-smoke calls and12 oracle checks are integration data, excluded from every agent success denominator. Failed math preparations triggered no model retries.'}
    preparations=[]
    for name in ['module-oracle-v1','module-preflight-v1','module-preflight-v2','arguments-preflight-v1']:
        folder=STORE/name
        if not (folder/'preflight.json').exists():
            continue
        result=json.loads((folder/'preflight.json').read_text())
        receipts=[]
        kernels=[]
        for sdk in folder.glob('sdk-*'):
            receipts += [json.loads(p.read_text()) for p in (sdk/'requests').glob('call-*.json')
                         if p.stem.removeprefix('call-').isdigit()]
            if (sdk/'smoke.json').exists():
                smoke=json.loads((sdk/'smoke.json').read_text())
                kernels.append({'service':sdk.name,'status':smoke['event']['result'].get('status'),
                                'source_sha256':smoke['event']['source_sha256']})
        preparations.append({'folder':str(folder),'status':result['status'],
            'confirmed_calls':sum(r['status']=='completed' for r in receipts),
            'unknown_calls':sum(r['status'] in {'dispatched','response_unknown'} for r in receipts),
            'oracle_checks':result['oracle_checks'],'dispatch_status':result.get('dispatch_status'),
            'integration_kernels':kernels})
    data['module_preparations']=preparations
    reference=STORE/'reference-preflight-v1/preflight.json'
    if reference.exists():
        value=json.loads(reference.read_text())
        data['reference_preflight']={'status':value['status'],'services':value['services'],
            'confirmed_calls':sum(x['confirmed_calls'] for x in value['services']),
            'unknown_calls':sum(x['unknown_calls'] for x in value['services']),
            'gpu_calls':0,'scope':value['scope']}

    data['scope']='All preparatory calls/math/dispatch probes/integration kernels remain separate from every scored agent cohort; failures and actual call counts are retained.'
    data['summary_sha256']=digest(data)
    atomic_json(ROOT/'results/staged-contract-environment.json',data)
    print(json.dumps(data))


if __name__=='__main__':
    main()
