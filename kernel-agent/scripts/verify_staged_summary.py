"""Lean summary hash/count checks only; no remote raw audit or CUDA execution."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    value = json.loads(args.summary.read_text())
    expected = value.pop('summary_sha256')
    actual = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert actual == expected, 'public summary self-hash changed'
    assert set(value['arms']) == {'fixed', 'staged', 'staged_transfer'}
    assert value['scheduled_blocks'] == 3 and value['scheduled_outcomes'] == 9
    assert len(value['blocks']) == value['completed_blocks'] <= 3
    for arm, total in value['arms'].items():
        rows = [b['arms'][arm] for b in value['blocks']]
        assert total['resolved'] == len(rows)
        assert total['correct'] == sum(r['final_decision']['correct'] for r in rows)
        assert total['promoted'] == sum(r['final_decision']['promoted'] for r in rows)
        assert total['calls'] == sum(r['actual_calls'] for r in rows) <= 72
        assert all(r['allocated_opportunities'] <= 24 and r['submissions'] <= 6 for r in rows)
    if value['status'] == 'complete':
        assert len(value['blocks']) == 3 and all(r['resolved'] == 3 for r in value['arms'].values())
        assert len(value['manifest_sha256']) == 64
    print(json.dumps({'status': 'passed', 'summary_sha256': expected,
        'scope': 'Self-hash and published count consistency only. No unavailable raw reconstruction, model call or GPU check.'}))


if __name__ == '__main__':
    main()
