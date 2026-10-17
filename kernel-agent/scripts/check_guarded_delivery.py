"""Presentation/export lint and lean summary checks; no model or GPU execution."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from kernel_agent.campaign import atomic_json  # noqa: E402


def main():
    commands = [[sys.executable, '-m', 'ruff', 'check', 'src', 'scripts', 'tests', 'flows']]
    for name in ['staged-guarded-summary.json', 'staged-guarded-corrected-summary.json']:
        if (ROOT/'results'/name).exists():
            commands.append([sys.executable, str(ROOT/'scripts/verify_guarded_summary.py'), str(ROOT/'results'/name)])
    rows = []
    for command in commands:
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        rows.append({'command': command, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
    status = 'passed' if all(r['exit_code'] == 0 for r in rows) else 'failed'
    atomic_json(ROOT/'results/staged-guarded-delivery-checks.json', {'status': status, 'commands': rows,
        'scope': 'Source lint and compact hash/count verification; not raw reconstruction or CUDA numerics.'})
    print(json.dumps({'status': status, 'commands': len(rows), 'errors': [r for r in rows if r['exit_code']]}))
    raise SystemExit(0 if status == 'passed' else 1)


if __name__ == '__main__':
    main()
