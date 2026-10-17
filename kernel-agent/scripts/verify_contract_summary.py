"""Stdlib-only published hash/count checks; raw verification runs only on the server."""
import hashlib
import json
import sys
from pathlib import Path


def main(path):
    d=json.loads(Path(path).read_text())
    expected=d.pop('summary_sha256')
    raw=json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest()==expected
    cases=[v for b in d['blocks'] for v in b['arms'].values()]
    arms={'control','repair'} if d['kind']=='kernel_attention_repair_v1' else {'guarded','contract'}
    assert all(set(b['arms'])==arms for b in d['blocks'])
    assert len(cases)<=d['scheduled_outcomes']
    if d['status']=='complete':
        assert len(cases)==d['scheduled_outcomes'] and d['raw_manifest_files']>0
    for r in cases:
        assert 0<=r['actual_calls']<=24 and r['submissions']<=6 and r['new_searches']<=9
        assert r['full_stage_complete']==(len(r['stages'])==3 and all(p['accepted'] for p in r['stages']))
        assert r['full_stage_and_final_correct']==(r['full_stage_complete'] and r['final_decision']['correct'])
    print(json.dumps({'status':'passed','summary_sha256':expected,'scope':'Compact self-hash/counts only; no model or CUDA validation'}))


if __name__=='__main__':
    main(sys.argv[1])
