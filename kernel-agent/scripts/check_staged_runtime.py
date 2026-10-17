"""Check mechanisms under the isolated SDK runtime as well as the controller runtime."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runtime = Path('/data/home/chenyizhou/kernel-agent-storage/runtimes/multistage/bin/python')
result = subprocess.run([str(runtime), str(ROOT / 'scripts/check_project.py'), '--output',
                        str(ROOT / 'results/staged-runtime-checks.json')],
                       env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT),
                            'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'HUMANIZE_PRICES': 'off'})
raise SystemExit(result.returncode)
