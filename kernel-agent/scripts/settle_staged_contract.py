"""Close gracefully interrupted block dispatch without replaying any model/GPU work."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent.campaign import atomic_json,digest  # noqa: E402


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args()
    r=a.directory
    stop=json.loads((r/'stop-request.json').read_text())
    if not (r/'supervisor-exit.json').exists():
        raise SystemExit('Still settling already-dispatched pairs; no files changed.')
    state=json.loads((r/'checkpoint.json').read_text())
    assert all(d['utc']<stop['requested_utc'] for d in state['dispatches'])
    rows=[]
    for row in state['blocks']:
        path=r/'blocks'/row['block']['id']/'result.json'
        item={'block':row['block'],'endpoint':row.get('endpoint')}
        if row['status'] in {'complete','running'}:
            assert path.exists(),'A dispatched pair did not settle: '+str(path)
            value=json.loads(path.read_text())
            assert set(value['arms'])=={'guarded','contract'}
            item.update(status='complete',result_sha256=digest(value))
            if row['status']=='complete':
                assert item['result_sha256']==row['result_sha256']
        else:
            assert row['status']=='pending' and not path.exists()
            assert not list((path.parent).glob('*/requests/call-*.request.json'))
            item['status']='not_dispatched_by_stop'
        rows.append(item)
    value={'status':'stopped_diagnostic','plan_sha256':state['plan_sha256'],
        'original_checkpoint_file_sha256':hashlib.sha256((r/'checkpoint.json').read_bytes()).hexdigest(),
        'stop_requested_utc':stop['requested_utc'],'blocks':rows,
        'scope':'Additive settlement only. Original checkpoint/requests/results unchanged; no model/GPU work replayed. Pending planned outcomes remain unfinished.'}
    value['settlement_sha256']=digest(value)
    target=r/'settlement.json'
    if target.exists():
        assert json.loads(target.read_text())==value
    else:
        atomic_json(target,value)
    print(json.dumps({'status':value['status'],'settled_pairs':sum(b['status']=='complete' for b in rows),
        'unstarted_pairs':sum(b['status']!='complete' for b in rows),'settlement_sha256':value['settlement_sha256']}))


if __name__=='__main__':
    main()
