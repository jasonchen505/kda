"""Record closed evidence checks and preparation provenance; no model/GPU calls."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from kernel_agent.campaign import atomic_json,digest  # noqa: E402

STORE=Path('/data/home/chenyizhou/kernel-agent-storage')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory=STORE/'experiments/attention-repair-v1'
    summary=load(ROOT/'results/attention-repair-summary.json')
    assert summary['status']=='complete' and len(summary['blocks'])==6
    assert digest({k:v for k,v in summary.items() if k!='summary_sha256'})==summary['summary_sha256']
    command=[sys.executable,str(ROOT/'scripts/export_attention_repair.py'),
             '--directory',str(directory),'--verify-only']
    audit=subprocess.run(command,capture_output=True,text=True,check=True)
    receipt=json.loads(audit.stdout)
    plan=load(directory/'plan.json')
    assert all(sha(ROOT/name)==value for name,value in plan['source_sha256'].items())
    prep=STORE/'experiments/attention-repair-parents-v1'
    manifest=prep/'evidence-manifest.json'
    if not manifest.exists():
        origins=[]
        old=STORE/'experiments/staged-reference-representative-v1'
        old_manifest=load(old/'evidence-manifest.json')
        launch=load(directory/'launch.json')
        for key,meta in plan['parents']['parents'].items():
            parent=load(Path(meta['path']))
            assert digest(parent)==meta['parent_digest']
            assert hashlib.sha256(parent['code'].encode()).hexdigest()==meta['source_sha256']
            assert (prep/(key+'.py')).read_text()==parent['code']
            origin=Path(parent['origin']['path'])
            assert sha(origin)==old_manifest[str(origin.relative_to(old))]
            source=next(r for r in load(origin)['candidates'] if r['id']==parent['origin']['candidate_id'])
            assert source['code']==parent['code']
            verdict=parent['evaluation']
            assert verdict['device_binding']['verified']
            assert verdict['resource_audit']['before']['observed_utc']<launch['started_utc']
            origins.append({'parent':key,'source_sha256':meta['source_sha256'],
                            'parent_digest':meta['parent_digest'],'origin_file_sha256':sha(origin),
                            'status':verdict['status'],'device_verified':True})
        atomic_json(prep/'preparation-audit.json',{'status':'passed','parents':origins,
            'model_calls':0,'preparation_searches':2,
            'scope':'Copied historical sources unchanged; independent observer measurements precede scored launch'})
        atomic_json(manifest,{str(p.relative_to(prep)):sha(p) for p in sorted(prep.rglob('*'))
                             if p.is_file() and p.name!='evidence-manifest.json' and p.suffix!='.lock'})
    prep_files=load(manifest)
    assert all(sha(prep/name)==value for name,value in prep_files.items())
    archive=STORE/'archives/derived/attention-repair-iteration'
    quick=ROOT/'results/attention-repair-quick-checks.json'
    if quick.exists():
        archive.mkdir(parents=True,exist_ok=True)
        if (archive/quick.name).exists():
            raise FileExistsError('Do not replace archived development checks')
        shutil.move(str(quick),archive/quick.name)
    checks={}
    for name in ['attention-repair-checks.json','attention-repair-runtime-checks.json']:
        value=load(ROOT/'results'/name)
        assert value['status']=='passed'
        count=re.search(r'(\d+) passed',value['commands'][0]['stdout'])
        assert count and int(count[1])==294
        checks[name]={'sha256':sha(ROOT/'results'/name),'tests':294,'status':value['status']}
    observer=load(ROOT/'results/attention-observer-checks.json')
    assert observer['status']=='passed'
    rows=[r for b in summary['blocks'] for r in b['arms'].values()]
    result={'status':'passed','closed_utc':datetime.now(timezone.utc).isoformat(),
        'summary_sha256':summary['summary_sha256'],'plan_sha256':summary['plan_sha256'],
        'manifest_sha256':summary['raw_manifest_sha256'],'raw_files':receipt['raw_files'],
        'outcomes':len(rows),'confirmed_calls':sum(r['actual_calls']-r['unknown_calls'] for r in rows),
        'unknown_calls':sum(r['unknown_calls'] for r in rows),'source_files':summary['source_files'],
        'readonly_audit':receipt,'analysis_script_sha256':sha(ROOT/'scripts/export_attention_repair.py'),
        'preparation':{'raw_files':len(prep_files),'manifest_sha256':sha(manifest),
                       'model_calls':0,'gpu_searches':2},'engineering_checks':checks,
        'observer_check_sha256':sha(ROOT/'results/attention-observer-checks.json'),
        'scope':'Closed six-pair raw audit and preparation provenance. No model or GPU requests; CPU checks are not kernel success.'}
    result['audit_sha256']=digest(result)
    atomic_json(ROOT/'results/attention-repair-closeout-audit.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
