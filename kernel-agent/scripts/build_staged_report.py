"""Render the new port cohort without modifying any frozen experiment source."""
import contextlib
import io
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path(str(ROOT / 'scripts/build_grounded_report.py'), run_name='__main__')
payload = namespace['payload']
for key, filename in [('staged', 'staged-summary.json'), ('staged_checks', 'staged-checks.json'),
                      ('staged_runtime_checks', 'staged-runtime-checks.json'),
                      ('staged_environment', 'staged-environment.json'),
                      ('staged_initial', 'staged-initial-summary.json'),
                      ('staged_knowledge', 'staged-knowledge-diagnostic.json')]:
    path = ROOT / 'results' / filename
    payload[key] = json.loads(path.read_text()) if path.exists() else None
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
target = ROOT / 'reports/kernel-agent-report.html'
target.write_text((ROOT / 'reports/template.html').read_text().replace('@@DATA@@', encoded))
print(target)
