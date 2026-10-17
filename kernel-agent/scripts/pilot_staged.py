"""Unscored one-arm port integration pilot; do not pool it with the prospective study."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import model_pool  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from scripts.run_staged_study import RUNTIME, STORE  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gpu-python', required=True)
    args = parser.parse_args()
    if not args.directory.resolve().is_relative_to(STORE / 'experiments'):
        raise ValueError('pilot raw must remain remote')
    args.directory.mkdir(parents=True, exist_ok=False)
    endpoints = [e for e in model_pool.discover() if not any(model_pool.validate(e).values())]
    endpoint = model_pool.choose(endpoints, set(), {})
    if endpoint is None:
        raise ValueError('no idle authorized service for the integration pilot')
    spec = json.loads((ROOT / 'data/repair-transfer-tasks.json').read_text())['affine_silu']
    config = {'directory': str(args.directory), 'endpoint': endpoint, 'seed': 2026101799,
              'task_id': 'affine_silu', 'contract': spec['contract'], 'arm': 'staged_transfer',
              'gpu_python': args.gpu_python, 'lock_file': str(args.directory / 'gpu4.lock')}
    atomic_json(args.directory / 'configuration.json', config)
    with (args.directory / 'worker.log').open('wb') as log:
        child = subprocess.run([str(RUNTIME), str(ROOT / 'scripts/staged_worker.py'),
            '--configuration', str(args.directory / 'configuration.json')], stdout=log, stderr=subprocess.STDOUT,
            timeout=2600, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
    print(json.dumps({'exit_code': child.returncode, 'directory': str(args.directory),
                      'scope': 'Unscored integration pilot, no final workloads'}))


if __name__ == '__main__':
    main()
