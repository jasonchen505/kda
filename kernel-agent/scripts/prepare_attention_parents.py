"""Copy actual failed Attention sources unchanged, revalidate common diagnostics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from kernel_agent.attention_repair import ControlTools  # noqa: E402
from kernel_agent.campaign import atomic_json,digest  # noqa: E402

STORE=Path('/data/home/chenyizhou/kernel-agent-storage')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-python',required=True)
    args=parser.parse_args()
    directory=STORE/'experiments/attention-repair-parents-v1'
    directory.mkdir(exist_ok=False)
    parents={}
    for key,seed,status in [('repair_compile',2026101843,'compile_error'),('repair_numeric',2026101844,'numeric_error')]:
        origin=STORE/f'experiments/staged-reference-representative-v1/blocks/generation-attention-s{seed}/contract/tools.json'
        ledger=json.loads(origin.read_text())
        row=ledger['candidates'][0]
        code=row['code']
        source=directory/(key+'.py')
        source.write_text(code)
        assert hashlib.sha256(code.encode()).hexdigest()==row['source_sha256']
        tools=ControlTools(directory/key,'attention','staged_transfer',args.gpu_python,directory/'gpu4.lock')
        verdict=tools.evaluate({'path':str(source),'source_sha256':row['source_sha256']})
        assert verdict['status']==status, verdict
        parent={'task_id':'attention','code':code,'source_sha256':row['source_sha256'],'evaluation':verdict,
                'origin':{'path':str(origin),'candidate_id':row['id'],'source_sha256':row['source_sha256']},
                'scope':'Unchanged selected historical failed source; independently revalidated common observer, no model calls'}
        target=directory/(key+'.json')
        atomic_json(target,parent)
        parents[key]={'path':str(target),'parent_digest':digest(parent),'source_sha256':row['source_sha256']}
    atomic_json(directory/'parents.json',{'parents':parents,'model_calls':0,'preparation_searches':2})
    print(json.dumps({'status':'prepared','parents':parents,'model_calls':0,'preparation_searches':2}))


if __name__=='__main__':
    main()
