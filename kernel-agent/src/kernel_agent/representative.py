"""Additive GEMM/attention oracle adapters; historical evaluators remain immutable."""
import json
import os
import subprocess
from pathlib import Path

from kernel_agent import staged_contract
from kernel_agent.representative_tasks import specs
from kernel_agent.staged_guard_v2 import Tools as GuardTools

ROOT = Path(__file__).resolve().parents[2]


class RepresentativeMixin:
    def evaluate(self, row, final=False):
        if self.task_id not in specs():
            return super().evaluate(row, final)
        if not final:
            if self.gpu_calls >= 9:
                raise ValueError('search GPU budget exhausted')
            self.gpu_calls += 1
            self.save()
        command = [self.gpu_python, str(ROOT/'scripts/evaluate_representative_serial.py'),
            '--lock-file', str(self.lock_file), self.task_id, row['path']]
        if final:
            command.append('--final')
        child = subprocess.run(command, capture_output=True, text=True, timeout=400,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT)})
        value = json.loads(child.stdout) if child.returncode == 0 else {
            'status': 'evaluator_error', 'message': (child.stdout+child.stderr)[-1800:]}
        if value.get('status') == 'correct' and value.get('source_sha256') != row['source_sha256']:
            raise ValueError('stale representative source identity')
        return value

    def profile(self, row):
        if self.task_id not in specs():
            return super().profile(row)
        from kernel_agent.resource_lock import acquire
        from scripts.evaluate_coexecuted import evaluate
        if self.profiles >= 2 or row['evaluation'].get('status') != 'correct':
            raise ValueError('profile needs a correct source and remaining profile budget')
        self.profiles += 1
        self.save()
        folder = self.directory/f'profile-{self.profiles}'
        folder.mkdir(exist_ok=False)
        command = [self.gpu_python, str(ROOT/'scripts/profile_representative.py'), self.task_id,
            row['path'], '--trace', str(folder/'cuda-activity.json')]
        with acquire(self.lock_file) as (lock, waited):
            result = evaluate(command, 4, pass_fds=(lock.fileno(),), require_device_proof=True)
        result.update(queue_wait_seconds=waited, source_sha256=row['source_sha256'],
                      hardware_counters_available=False)
        from kernel_agent.campaign import atomic_json
        atomic_json(folder/'profile.json', result)
        return result

    def execute(self, action, stage):
        value = staged_contract.unwrap(action)
        value = value.model_dump(mode='json') if hasattr(value, 'model_dump') else value
        event = super().execute(action, stage)
        if self.task_id in specs() and value['tool'] == 'read' and value['argument'] in {'api', 'example'}:
            if event['result'].get('status') not in {'repeat_refused', 'tool_error'}:
                event['result'] = {'text': specs()[self.task_id]['api_notes'],
                    'scope': 'Task API notes only; no complete generated candidate supplied'}
                self.save()
        return event


class ContractTools(RepresentativeMixin, staged_contract.Tools):
    pass


class ExtendedGuardTools(RepresentativeMixin, GuardTools):
    pass
