"""Add the separate budget follow-up to the existing frozen report builder."""
import contextlib
import io
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path(str(ROOT / 'scripts/build_report.py'), run_name='__main__')
payload = namespace['payload']
for key, filename in [('grounded_budget', 'grounded-budget-summary.json'), ('grounded_budget_checks', 'grounded-budget-checks.json'), ('grounded_costs', 'grounded-costs.json'), ('grounded_budget_costs', 'grounded-budget-costs.json')]:
    path = ROOT / 'results' / filename
    payload[key] = json.loads(path.read_text()) if path.exists() else None
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
target = ROOT / 'reports/kernel-agent-report.html'
target.write_text((ROOT / 'reports/template.html').read_text().replace('@@DATA@@', encoded))
print(target)
