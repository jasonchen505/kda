"""Persist CPU regression/lint checks separately from scored GPU/model runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quick',action='store_true')
    p.add_argument('--runtime',action='store_true')
    a=p.parse_args()
    commands=[[sys.executable,'-m','pytest','-q']+(['tests/unit/test_staged_contract_v3.py','tests/integration/test_staged_contract_v3_flow.py'] if a.quick else []),
              [sys.executable,'-m','ruff','check','src','scripts','tests','flows']]
    rows=[]
    for command in commands:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        rows.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    value={'status':'passed' if all(r['exit_code']==0 for r in rows) else 'failed',
           'commands':rows,'scope':'CPU fake-driver/contract mechanisms and source lint; not model/GPU success'}
    name='staged-arguments'+('-runtime' if a.runtime else '')+('-quick' if a.quick else '')+'-checks.json'
    (ROOT/'results'/name).write_text(json.dumps(value,separators=(',',':'))+'\n')
    print(json.dumps(value))
    if value['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
