"""Prospective paired migration study: fixed controller, staged agent and transferred repair."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import model_pool, staged  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.protocol import decision  # noqa: E402
from kernel_agent.resource_lock import acquire  # noqa: E402
from kernel_agent.staged_tools import Tools  # noqa: E402
from scripts.run_repair_study import Study, source_files  # noqa: E402

STORE = Path('/data/home/chenyizhou/kernel-agent-storage')
RUNTIME = STORE / 'runtimes/multistage/bin/python'
TASKS = ['affine_silu', 'scale_transpose', 'l2_normalize']
SEEDS = [2026101702]


def make_plan():
    blocks = [{'id': f'{t}-s{s}', 'task_id': t, 'seed': s, 'model_call_cap': 72} for t in TASKS for s in SEEDS]
    random.Random(20261017).shuffle(blocks)
    sources = source_files()
    executed = {'data/tasks.json', 'pyproject.toml', 'uv.lock',
                'scripts/run_staged_corrected.py', 'scripts/staged_corrected_worker.py',
                'scripts/run_repair_study.py', 'scripts/run_campaign.py', 'scripts/local_proxy.py',
                'scripts/evaluate_serial.py', 'scripts/evaluate_coexecuted.py', 'scripts/evaluate_candidate.py',
                'scripts/profile_staged_kernel.py', 'scripts/summarize_staged.py'}
    sources = {name: sha for name, sha in sources.items()
               if name.startswith(('src/', 'references/')) or name in executed}
    for name in ['flows/staged_kernel_corrected.py', 'docs/staged-confirmation-plan.md', 'docs/staged-study-plan.md', 'data/repair-transfer-tasks.json',
                 'tests/unit/test_staged.py', 'tests/integration/test_staged_gateway.py',
                 'tests/integration/test_staged_flow.py', 'tests/unit/test_staged_phase_interface.py',
                 'tests/integration/test_staged_corrected_flow.py']:
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    requirements = RUNTIME.parent.parent / 'requirements-installed.txt'
    return {'schema_version': 1, 'kind': 'rtx4090_staged_workflow_transfer_corrected', 'blocks': blocks,
            'arms': list(staged.ARMS), 'scheduled_arms': 9, 'calls_per_block': 72,
            'max_model_calls': 216, 'max_output_tokens': 2048, 'context_limit': 8192,
            'candidate_cap_per_arm': 6, 'call_cap_per_arm': 24, 'evaluation_gpu': 4,
            'source_sha256': sources, 'seeds': SEEDS, 'tasks': TASKS,
            'runtime': {}, 'staged_runtime': str(RUNTIME),
            'staged_requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest(),
            'scope': 'First-port feasibility cohort: three paired development-family blocks/one fresh seed. '
            'Same call/token/candidate/tool ceilings, actual spending differs. Full-workflow bundle, '
            'not an isolated planning effect or original B200 contest reproduction. '
            'All roles same Qwen process/fresh contexts; no cross-root weight equivalence or blind generalization.'}


class StagedStudy(Study):
    def __init__(self, directory, gpu_python):
        super().__init__(directory, gpu_python, plan=make_plan())

    def verify_sources(self):
        for name, expected in self.plan['source_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError('frozen study source drift: ' + name)

    def run_block(self, block, endpoint):
        self.verify_sources()
        folder = self.directory / 'blocks' / block['id']
        folder.mkdir(parents=True, exist_ok=True)
        spec = json.loads((ROOT / 'data/repair-transfer-tasks.json').read_text())[block['task_id']]
        arms = list(staged.ARMS)
        random.Random(block['seed']).shuffle(arms)
        result = {'block': block, 'endpoint': endpoint, 'arms': {}, 'arm_order': arms,
                  'started_utc': model_pool.utc()}
        for arm in arms:
            self.verify_sources()
            directory = folder / arm
            configuration = {'directory': str(directory), 'endpoint': endpoint,
                'seed': block['seed'], 'task_id': block['task_id'], 'contract': spec['contract'],
                'arm': arm, 'gpu_python': self.gpu_python, 'lock_file': str(self.directory / 'gpu4.lock')}
            atomic_json(directory / 'configuration.json', configuration)
            env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT), 'PYTHONUNBUFFERED': '1'}
            with (directory / 'worker.log').open('wb') as log:
                child = subprocess.run([str(RUNTIME), str(ROOT / 'scripts/staged_corrected_worker.py'),
                                        '--configuration', str(directory / 'configuration.json')],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=2600, env=env)
            path = directory / 'result.json'
            if path.exists():
                value = json.loads(path.read_text())
            else:
                value = {'task_id': block['task_id'], 'arm': arm, 'selection': {'source_sha256': None},
                         'model_opportunities': 0, 'stages': [], 'failure': {'worker_exit': child.returncode},
                         'final': None}
            result['arms'][arm] = value
        # All arms commit search selections before the supervisor accesses any final workload.
        result['generation_committed_utc'] = model_pool.utc()
        atomic_json(folder / 'generation.json', result)
        for arm, value in result['arms'].items():
            sha = value['selection']['source_sha256']
            if sha:
                tools = Tools(folder / arm, block['task_id'], arm, self.gpu_python, self.directory / 'gpu4.lock')
                row = {'path': str(folder / arm / 'candidates' / (sha + '.py')), 'source_sha256': sha}
                value['final'] = tools.evaluate(row, final=True)
                value['final_decision'] = decision(value['final'])
            else:
                value['final_decision'] = {'correct': False, 'promoted': False}
            atomic_json(folder / 'final' / (arm + '.json'), value)
        result['finished_utc'] = model_pool.utc()
        atomic_json(folder / 'result.json', result)
        self.verify_sources()
        durations = []
        for arm in arms:
            for path in (folder / arm / 'requests').glob('call-*.json'):
                row = json.loads(path.read_text())
                if row.get('status') == 'completed':
                    durations.append(row['elapsed_seconds'])
        return result, sum(durations) / len(durations) if durations else 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gpu-python', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if not args.directory.resolve().is_relative_to(STORE / 'experiments'):
        raise ValueError('raw staged experiments must remain in remote storage')
    if args.background:
        if args.directory.exists():
            raise FileExistsError('background launch needs a new ledger; never replay a prior run')
        endpoints = model_pool.discover()
        if len(endpoints) != 3:
            raise ValueError('discover all three authorized services before launching')
        study = StagedStudy(args.directory, args.gpu_python)
        command = [sys.executable, str(STORE / 'bin/run.py'), 'run_staged_corrected.py',
                   '--directory', str(args.directory), '--gpu-python', args.gpu_python]
        with (args.directory / 'supervisor.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        receipt = {'pid': child.pid, 'started_utc': model_pool.utc(), 'plan_sha256': study.state['plan_sha256'],
                   'model_call_cap': 216, 'endpoints': endpoints, 'source_root': str(ROOT)}
        atomic_json(args.directory / 'launch.json', receipt)
        print(json.dumps(receipt))
        return
    with acquire(args.directory / 'supervisor.lock', timeout=0):
        study = StagedStudy(args.directory, args.gpu_python)
        started = time.time()
        if args.prepare:
            print(json.dumps({'status': 'prepared', 'plan_sha256': study.state['plan_sha256']}))
            return
        try:
            study.run()
        finally:
            atomic_json(args.directory / 'supervisor-exit.json', {'status': study.state['status'],
                         'finished_utc': model_pool.utc(), 'elapsed_seconds': time.time()-started})


if __name__ == '__main__':
    main()
