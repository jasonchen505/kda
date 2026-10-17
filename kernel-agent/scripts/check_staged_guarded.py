"""Separate real SDK/service and profiler smoke; fixture is never an agent success."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import model_pool, staged  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.staged_tools import Tools  # noqa: E402
from scripts.run_staged_study import RUNTIME, STORE  # noqa: E402

FIXTURE = '''import torch
import triton
import triton.language as tl
@triton.jit
def probe(X, Y, Z, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    y = tl.load(Y + i, i < N, other=0).to(tl.float32)
    tl.store(Z + i, x + y, i < N)
def run(x, y):
    z = torch.empty_like(x)
    probe[(triton.cdiv(x.numel(), 256),)](x, y, z, x.numel(), 256)
    return z
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gpu-python', required=True)
    args = parser.parse_args()
    if not args.directory.resolve().is_relative_to(STORE / 'experiments'):
        raise ValueError('preflight raw data must remain remote')
    args.directory.mkdir(parents=True, exist_ok=False)
    results = []
    endpoints = model_pool.discover()
    if len(endpoints) != 3:
        raise ValueError('three local services required')
    for index, endpoint in enumerate(endpoints):
        folder = args.directory / f'sdk-{index}'
        config = {'directory': str(folder), 'endpoint': endpoint, 'seed': 2026101790+index,
                  'task_id': 'vector_add', 'contract': 'integration smoke, not a scored task',
                  'arm': 'staged_transfer', 'policy': 'guarded', 'gpu_python': args.gpu_python,
                  'lock_file': str(args.directory / 'gpu4.lock')}
        atomic_json(folder / 'configuration.json', config)
        with (folder / 'worker.log').open('wb') as log:
            result = subprocess.run([str(RUNTIME), str(ROOT / 'scripts/staged_guarded_worker.py'),
                '--configuration', str(folder / 'configuration.json'), '--smoke'],
                stdout=log, stderr=subprocess.STDOUT, timeout=400,
                env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
        smoke_path = folder / 'smoke.json'
        smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        passed = (result.returncode == 0 and [p['tool'] for p in smoke.get('plan', {}).get('steps', [])] == ['read']
                  and smoke.get('action', {}).get('tool') == 'read'
                  and smoke.get('action', {}).get('argument') == 'guide'
                  and not smoke.get('review', {}).get('accept', True) and smoke.get('review', {}).get('claims') == []
                  and 'text' in smoke.get('event', {}).get('result', {}))
        results.append({'endpoint': endpoint, 'passed': passed, 'exit_code': result.returncode})
        if not passed:
            atomic_json(args.directory / 'preflight.json', {'sdk': results, 'status': 'sdk_failed'})
            raise ValueError('real SDK integration failed; inspect remote worker.log')
    tools = Tools(args.directory / 'profiler', 'vector_add', 'staged', args.gpu_python, args.directory / 'gpu4.lock')
    event = tools.execute({'tool': 'submit', 'argument': '', 'edits': [], 'code_lines': FIXTURE.splitlines(),
                           'reason': 'Handwritten profiler capability fixture, excluded from agent scores'}, 'preflight')
    evaluated = tools.execute({'tool': 'evaluate', 'argument': event['source_sha256']}, 'preflight')
    profile = tools.execute({'tool': 'profile', 'argument': event['source_sha256']}, 'preflight')
    value = {'status': 'passed' if evaluated['result'].get('status') == 'correct'
             and profile['result'].get('status') in {'profiled', 'profiled_activity'} else 'profiler_failed',
             'sdk': results, 'profile': profile, 'fixture_sha256': staged.text_hash(FIXTURE),
             'scope': 'Three real typed plan/action/review calls per service plus one handwritten GPU/profiler fixture. '
                      'Not generated candidates, not part of efficacy study.'}
    atomic_json(args.directory / 'preflight.json', value)
    print(json.dumps({'status': value['status'], 'sdk_services': len(results),
                      'profile_status': profile['result'].get('status')}))
    if value['status'] != 'passed':
        raise ValueError('profile preflight failed; preserve exact record')


if __name__ == '__main__':
    main()
