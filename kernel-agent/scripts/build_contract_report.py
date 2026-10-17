"""Render new independent cohorts above historical results without pooling outcomes."""
import contextlib
import io
import json
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    previous=runpy.run_path(str(ROOT/'scripts/build_guarded_report.py'),run_name='__main__')
payload=previous['payload']
for key,name in [('contract_core','staged-contract-core-summary.json'),('contract_initial','staged-contract-core-summary.json'),('contract_parameter_initial','staged-module-core-summary.json'),
                 ('contract_representative','staged-contract-representative-summary.json'),
                 ('contract_checks','staged-contract-checks.json'),
                 ('contract_runtime_checks','staged-contract-runtime-checks.json')]:
    path=ROOT/'results'/name
    payload[key]=json.loads(path.read_text()) if path.exists() else None
for key,name in [('contract_core','staged-module-core-summary.json'),('contract_representative','staged-module-representative-summary.json'),('contract_checks','staged-module-checks.json'),('contract_runtime_checks','staged-module-runtime-checks.json')]:
    path=ROOT/'results'/name
    if path.exists():
        payload[key]=json.loads(path.read_text())
for key,name in [('contract_core','staged-arguments-core-summary.json'),('contract_representative','staged-arguments-representative-summary.json'),('contract_checks','staged-arguments-checks.json'),('contract_runtime_checks','staged-arguments-runtime-checks.json')]:
    path=ROOT/'results'/name
    if path.exists():
        payload[key]=json.loads(path.read_text())
for key,name in [('contract_extension_initial','staged-recovery-representative-summary.json'),('contract_checks','staged-recovery-checks.json'),('contract_runtime_checks','staged-recovery-runtime-checks.json'),('contract_environment','staged-contract-environment.json'),('contract_parser_diagnostic','structural-recovery-diagnostic.json')]:
    path=ROOT/'results'/name
    if path.exists():
        payload[key]=json.loads(path.read_text())
for key,name in [('contract_representative','staged-reference-representative-summary.json'),('contract_checks','staged-reference-checks.json'),('contract_runtime_checks','staged-reference-runtime-checks.json')]:
    path=ROOT/'results'/name
    if path.exists():
        payload[key]=json.loads(path.read_text())
for key,name in [('attention_repair','attention-repair-summary.json'),('attention_checks','attention-repair-checks.json'),('attention_runtime_checks','attention-repair-runtime-checks.json'),('attention_closeout','attention-repair-closeout-audit.json')]:
    path=ROOT/'results'/name
    if path.exists():
        payload[key]=json.loads(path.read_text())
encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
target=ROOT/'reports/kernel-agent-report.html'
target.write_text((ROOT/'reports/template.html').read_text().replace('@@DATA@@',encoded))
print(target)
