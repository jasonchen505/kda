"""Add the new paired generation/repair cohort; preserve earlier independent summaries."""
import contextlib
import io
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path(str(ROOT/'scripts/build_staged_report.py'), run_name='__main__')
payload = namespace['payload']
for key, name in [('guarded', 'staged-guarded-summary.json'), ('guarded_initial', 'staged-guarded-summary.json'),
                  ('guarded_checks', 'staged-guarded-checks.json'),
                  ('guarded_runtime_checks', 'staged-guarded-runtime-checks.json')]:
    path = ROOT/'results'/name
    payload[key] = json.loads(path.read_text()) if path.exists() else None
for key, name in [('guarded', 'staged-guarded-corrected-summary.json'), ('guarded_checks', 'staged-guarded-v2-checks.json'),
                  ('guarded_runtime_checks', 'staged-guarded-v2-runtime-checks.json')]:
    path = ROOT/'results'/name
    if path.exists():
        payload[key] = json.loads(path.read_text())
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
target = ROOT/'reports/kernel-agent-report.html'
target.write_text((ROOT/'reports/template.html').read_text().replace('@@DATA@@', encoded))
print(target)
