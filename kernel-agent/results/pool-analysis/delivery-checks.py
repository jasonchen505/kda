import datetime,hashlib,json,os,re,subprocess
from pathlib import Path
root=Path.cwd()
env={**os.environ,'PATH':'/data/home/chenyizhou/.local/bin:'+os.environ.get('PATH',''),'PYTHONPATH':str(root/'src')+':'+str(root)}
commands=[('fresh_install_dependencies',['uv','sync','--extra','dev','--frozen'])]
for label,python in [('fresh_install',str(root/'.venv/bin/python')),('current_sdk','/data/home/chenyizhou/kernel-agent/.venv/bin/python'),('archived_sdk','/data/home/chenyizhou/humanize/.venv/bin/python')]:
 commands.append((label,[python,'-m','pytest','-q']))
commands.append(('lint',[str(root/'.venv/bin/ruff'),'check','flows','scripts','src','tests']))
commands.extend(('shell_syntax:'+p.name,['bash','-n',str(p)]) for p in sorted((root/'scripts').glob('*.sh')))
commands.append(('offline_evidence',[str(root/'.venv/bin/python'),'scripts/verify_evidence.py']))
rows=[]
for label,cmd in commands:
 p=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=180)
 rows.append({'label':label,'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 print(json.dumps({'check':label,'exit_code':p.returncode,'tail':p.stdout[-130:]}),flush=True)
 if p.returncode: break
counts=[int(re.search(r'(\d+) passed',r['stdout']).group(1))for r in rows if r['label'] in {'fresh_install','current_sdk','archived_sdk'} and re.search(r'(\d+) passed',r['stdout'])]
paths=[]
for name in ['flows','src','scripts','tests']:
 paths.extend(p for p in (root/name).rglob('*.py') if '__pycache__'not in p.parts)
paths.extend((root/'scripts').glob('*.sh'));paths.extend(root/name for name in ['pyproject.toml','uv.lock','reports/template.html'])
value={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU mechanisms, both pinned SDKs, fresh locked installation and offline evidence; no new model or GPU calls','commands':rows,'status':'passed' if len(rows)==len(commands) and all(r['exit_code']==0 for r in rows)and len(counts)==3 and len(set(counts))==1 else 'failed','passed_tests':counts[0] if len(counts)==3 and len(set(counts))==1 else None,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(set(paths))}}
(root/'results/pool-checks.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':value['status'],'passed_tests':value['passed_tests']}),flush=True)
raise SystemExit(0 if value['status']=='passed'else 1)
