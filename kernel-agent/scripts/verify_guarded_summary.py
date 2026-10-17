"""Lean self-hash/count validation; no unavailable raw or numerical reexecution."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    v = json.loads(args.summary.read_text())
    expected = v.pop('summary_sha256')
    assert hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == expected
    assert v['scheduled_blocks'] == 6 and v['scheduled_outcomes'] == 12
    assert len(v['blocks']) == v['completed_blocks'] <= 6
    for b in v['blocks']:
        assert set(b['arms']) == {'previous', 'guarded'}
        for r in b['arms'].values():
            assert r['actual_calls'] <= r['opportunities'] <= 24
            assert r['submissions'] <= 6 and r['new_searches'] <= 9
            assert r['unknown_calls'] <= r['actual_calls']
            assert r['final_decision']['promoted'] <= r['final_decision']['correct']
    if v['status'] == 'complete':
        assert len(v['blocks']) == 6
        assert all(sum(b['stratum'] == s for b in v['blocks']) == 3 for s in ['generation', 'repair'])
        assert len(v['raw_manifest_sha256']) == 64
    print(json.dumps({'status': 'passed', 'summary_sha256': expected,
        'scope': 'Published hash/count consistency only; full raw reconstruction is server-only.'}))


if __name__ == '__main__':
    main()
