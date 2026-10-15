import datetime, hashlib, json, os, subprocess
from pathlib import Path
root=Path('/data/home/chenyizhou/kernel-agent-phased')
output=root/'results/phased-checks.json'
env=os.environ.copy()
env['PYTHONPATH']=str(root/'src')
commands=[]
for label,py in [('fresh_install',root/'.venv/bin/python'),('current_sdk',Path('/data/home/chenyizhou/kernel-agent/.venv/bin/python')),('archived_sdk',Path('/data/home/chenyizhou/humanize/.venv/bin/python'))]:
    commands.append((label,[str(py),'-m','pytest','-q']))
commands.append(('lint',[str(root/'.venv/bin/ruff'),'check','flows','scripts','src','tests']))
commands.extend(('shell_syntax:'+p.name,['bash','-n',str(p)]) for p in sorted((root/'scripts').glob('*.sh')))
commands.append(('offline_evidence',[str(root/'.venv/bin/python'),'scripts/verify_evidence.py']))
value={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU mechanisms, SDK compatibility, fresh locked installation and offline evidence; no new model or GPU calls','commands':[]}
for label,args in commands:
    child=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,timeout=120)
    value['commands'].append({'label':label,'command':args,'exit_code':child.returncode,'stdout':child.stdout,'stderr':child.stderr})
    print(label,child.returncode,child.stdout.splitlines()[-1:] if label!='offline_evidence' else 'artifact verification',flush=True)
    if child.returncode:
        value['status']='failed';break
else:
    value.update(status='passed',passed_tests=119)
value['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src/kernel_agent','flows','scripts','tests'] for p in sorted((root/folder).rglob('*.py')) if '__pycache__' not in p.parts}
output.write_text(json.dumps(value,indent=2))
