"""Bounded real tools in an isolated remote workspace; never expose final workloads."""
import csv
import io
import json
import os
import subprocess
import time
from pathlib import Path

from kernel_agent import line_protocol, span_repair, staged
from kernel_agent.campaign import atomic_json
from kernel_agent.knowledge import retrieve
from kernel_agent.protocol import admit
from kernel_agent.references import retrieve_example
from kernel_agent.resource_lock import acquire

ROOT = Path(__file__).resolve().parents[2]
STORE = Path('/data/home/chenyizhou/kernel-agent-storage')
NCU = Path('/opt/nvidia/nsight-compute/2024.1.1/ncu')
METRICS = ['gpu__time_duration.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
           'dram__throughput.avg.pct_of_peak_sustained_elapsed', 'launch__registers_per_thread',
           'sm__warps_active.avg.pct_of_peak_sustained_active']
GUIDE = ('Profile the latest correct candidate, then compare measured duration, SM/DRAM throughput, '
         'registers and active warps. High/low counters alone are not a proof of bottleneck. '
         'Use per-shape search times for configuration decisions. NCU replay is intrusive and shared-GPU '
         'timing is conditional; never substitute a failed profile with guessed counters. '
         'Use portable Ada features, not Blackwell TMEM/tcgen05. '
         'Triton constexpr arguments take tl types, not torch dtype objects. '
         'Stage3 may keep the generic implementation if measured specialization has no benefit; explain why.')


def compact(result, probes=True):
    keys = ('status', 'message', 'shape', 'dtype', 'checks_passed', 'timings', 'semantic_probes',
            'counterexample', 'source_sha256')
    value = {k: result[k] for k in keys if k in result}
    if not probes:
        value.pop('counterexample', None)
        value.pop('semantic_probes', None)
    if 'message' in value:
        value['message'] = str(value['message'])[-1200:]
    return value


def parse_metrics(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '"Metric Name"' in line), None)
    if start is None:
        return []
    return [{k: row.get(k) for k in ['Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value']}
            for row in csv.DictReader(io.StringIO('\n'.join(lines[start:])))
            if row.get('Metric Name') in METRICS]


class Tools:
    def __init__(self, directory, task_id, arm, gpu_python, lock_file):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.task_id, self.arm, self.gpu_python, self.lock_file = task_id, arm, gpu_python, Path(lock_file)
        self.rows, self.proposals, self.events = [], [], []
        self.submissions, self.gpu_calls, self.profiles = 0, 0, 0

    def save(self):
        atomic_json(self.directory / 'tools.json', {'candidates': self.rows, 'proposals': self.proposals, 'events': self.events,
                    'submissions': self.submissions, 'gpu_calls': self.gpu_calls, 'profiles': self.profiles})

    def evaluate(self, row, final=False):
        if not final and self.gpu_calls >= 9:
            raise ValueError('search GPU tool budget exhausted')
        if not final:
            self.gpu_calls += 1
            self.save()  # allocate before starting the subprocess; final must not overwrite the search ledger
        command = [self.gpu_python, str(ROOT / 'scripts/evaluate_serial.py'), '--gpu-index', '4',
                   '--lock-file', str(self.lock_file), '--require-device-proof', self.task_id, row['path']]
        command += ['--final'] if final else ['--diagnostics', '--semantic-probes']
        child = subprocess.run(command, capture_output=True, text=True, timeout=400,
                               env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
        result = json.loads(child.stdout) if child.returncode == 0 else {
            'status': 'evaluator_error', 'message': (child.stdout + child.stderr)[-1800:]}
        if result.get('status') == 'correct' and result.get('source_sha256') != row['source_sha256']:
            raise ValueError('stale evaluator source identity')
        return result

    def profile(self, row):
        if self.profiles >= 2:
            raise ValueError('two-profile budget exhausted')
        if row['evaluation'].get('status') != 'correct':
            raise ValueError('profile requires independent search correctness first')
        self.profiles += 1
        self.save()
        directory = self.directory / f'profile-{self.profiles}'
        directory.mkdir(exist_ok=False)
        shape = row['evaluation']['timings'][1]['shape']
        uuid = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
        ncu_command = [str(NCU), '--profile-from-start', 'off', '--target-processes', 'all',
                   '--launch-count', '1', '--metrics', ','.join(METRICS), '--csv', '--page', 'raw',
                   '--export', str(directory / 'candidate'), self.gpu_python,
                   str(ROOT / 'scripts/profile_staged_kernel.py'), self.task_id, row['path'],
                   '--shape', json.dumps(shape)]
        started = time.monotonic()
        with acquire(self.lock_file) as (lock, waited):
            before = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=uuid,utilization.gpu,memory.used',
                                              '--format=csv,noheader'], text=True).strip()
            child = subprocess.run(ncu_command, capture_output=True, text=True, timeout=150, pass_fds=(lock.fileno(),),
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': uuid,
                     'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
        (directory / 'stdout.txt').write_text(child.stdout)
        (directory / 'stderr.txt').write_text(child.stderr)
        proof = None
        for line in child.stdout.splitlines():
            try:
                candidate = json.loads(line)
                if 'device_audit' in candidate:
                    proof = candidate['device_audit']
            except (ValueError, TypeError):
                continue
        verified = bool(proof and proof['device_count'] == 1 and proof['device_index'] == 0
                        and str(proof['gpu_uuid']).lower().removeprefix('gpu-') == uuid.lower().removeprefix('gpu-')
                        and proof['visible_devices'] == uuid)
        metrics = parse_metrics(child.stdout)
        result = {'status': 'profiled' if child.returncode == 0 and verified and len(metrics) == len(METRICS)
                  else 'profile_unavailable', 'exit_code': child.returncode, 'metrics': metrics,
                  'device_verified': verified, 'device_audit': proof, 'gpu_uuid': uuid,
                  'before': before, 'source_sha256': row['source_sha256'], 'shape': shape,
                  'queue_wait_seconds': waited, 'elapsed_seconds': time.monotonic() - started,
                  'message': (child.stderr + child.stdout)[-800:] if child.returncode else '',
                  'scope': 'NCU2024.1.1 candidate-only range; shared GPU counters, not isolated benchmark'}
        if 'ERR_NVGPUCTRPERM' in child.stdout + child.stderr:
            command = [self.gpu_python, str(ROOT / 'scripts/profile_staged_kernel.py'), self.task_id,
                       row['path'], '--shape', json.dumps(shape),
                       '--activity-output', str(directory / 'cuda-activity.json')]
            with acquire(self.lock_file) as (lock, _):
                trace = subprocess.run(command, capture_output=True, text=True, timeout=150,
                    pass_fds=(lock.fileno(),), env={**os.environ, 'CUDA_VISIBLE_DEVICES': uuid,
                        'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
            (directory / 'activity-stdout.txt').write_text(trace.stdout)
            (directory / 'activity-stderr.txt').write_text(trace.stderr)
            activity = json.loads(trace.stdout) if trace.returncode == 0 else {}
            actual = activity.get('device_audit', {})
            trace_verified = (actual.get('visible_devices') == uuid and actual.get('device_count') == 1
                and str(actual.get('gpu_uuid')).lower().removeprefix('gpu-') == uuid.lower().removeprefix('gpu-'))
            result['ncu_failure'] = result['message']
            result.update(status='profiled_activity' if trace_verified and any('CUDA' in r['device_type']
                          for r in activity.get('activity', [])) else 'profile_unavailable',
                          activity=activity.get('activity', []), hardware_counters_available=False,
                          activity_device_verified=trace_verified,
                          backend='torch.profiler CUDA activity fallback after NCU permission refusal',
                          scope='Actual CUDA kernel/launch activity; no hardware-counter bottleneck attribution',
                          elapsed_seconds=time.monotonic()-started)
        atomic_json(directory / 'profile.json', result)
        return result

    def execute(self, action, stage):
        action = action.model_dump(mode='json') if isinstance(action, staged.Action) else action
        name, argument = action['tool'], action.get('argument', '')
        event = {'id': f'e{len(self.events)+1}', 'tool': name, 'stage': stage,
                 'source_sha256': self.proposals[-1]['source_sha256'] if self.proposals else None,
                 'reason': action.get('reason', ''), 'started': time.time()}
        try:
            if name == 'read':
                if argument == 'api':
                    result = {'text': retrieve({'status': 'task_contract'}, self.task_id)['context']}
                elif argument == 'example':
                    result = retrieve_example(self.task_id)
                elif argument == 'guide':
                    result = {'text': GUIDE}
                elif argument == 'source' or argument.startswith('source:'):
                    source = self.proposals[-1] if argument == 'source' else next(
                        r for r in self.proposals if r['source_sha256'] == argument.split(':', 1)[1])
                    result = {'sha256': source['source_sha256'], 'numbered_source': '\n'.join(
                        f'{i}: {line}' for i, line in enumerate(source['code'].splitlines(), 1))}
                elif argument == 'evidence':
                    result = {'evaluations': [compact(r['evaluation']) for r in self.rows],
                              'profile_events': [e for e in self.events if e['tool'] == 'profile']}
                else:
                    raise ValueError('read only api/example/guide/source/evidence')
            elif name == 'research':
                if not argument or len(argument) > 90:
                    raise ValueError('research query must be 1..90 characters')
                base = STORE / 'upstreams/kda/skills/KernelWiki'
                child = subprocess.run(['rg', '-l', '-i', '-F', '--', argument, str(base / 'wiki')],
                                       capture_output=True, text=True, timeout=10)
                pages = []
                for filename in child.stdout.splitlines()[:2]:
                    path = Path(filename)
                    if not path.resolve().is_relative_to(base.resolve()):
                        raise ValueError('knowledge source escaped pinned repository')
                    text = path.read_text()
                    pages.append({'path': str(path.relative_to(base)), 'sha256': staged.text_hash(text),
                                  'excerpt': text[:1800], 'truncated': len(text) > 1800})
                result = {'query': argument, 'pages': pages,
                          'warning': 'Pinned Hopper/Blackwell KB; architecture-specific techniques may not apply to Ada'}
            elif name in {'submit', 'patch'}:
                if self.submissions >= staged.CANDIDATE_CAP:
                    raise ValueError('candidate budget exhausted')
                self.submissions += 1
                self.save()
                if name == 'patch':
                    if self.arm != 'staged_transfer' or not self.proposals:
                        raise ValueError('indexed patch only in transfer arm with a parent')
                    proposed = span_repair.apply(json.dumps({'draft': action['reason'] or 'model patch',
                        'plan': 'apply exact requested edit', 'base_sha256': argument,
                        'edits': action['edits']}), self.proposals[-1]['code'])
                else:
                    proposed = line_protocol.proposal(json.dumps({'draft': action['reason'] or 'model source',
                        'plan': 'independent tool evaluation', 'code_lines': action['code_lines']}))
                # Preserve exact model source even when admission fails; a fresh session must see its own failure.
                sha = staged.text_hash(proposed['code'])
                artifact = self.directory / 'proposals' / (sha + '.py.txt')
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_text(proposed['code'])
                pending = {'id': f'c{self.submissions}', 'code': proposed['code'], 'source_sha256': sha,
                           'parent_sha256': event['source_sha256'], 'stage': stage,
                           'proposal_path': str(artifact), 'evaluation': {'status': 'not_admitted'}}
                self.proposals.append(pending)
                event['source_sha256'] = sha
                self.save()
                admit(proposed['code'])
                path = self.directory / 'candidates' / (sha + '.py')
                path.parent.mkdir(exist_ok=True)
                path.write_text(proposed['code'])
                row = pending
                row.update(path=str(path), evaluation={'status': 'not_evaluated'})
                self.rows.append(row)
                result = {'status': 'admitted', 'source_sha256': sha, 'candidate_id': row['id']}
                event['source_sha256'] = sha
            elif name in {'evaluate', 'profile'}:
                row = next(r for r in reversed(self.rows) if r['source_sha256'] == argument)
                if name == 'evaluate':
                    result = self.evaluate(row)
                    row['evaluation'] = result
                else:
                    result = self.profile(row)
                event['source_sha256'] = row['source_sha256']
            elif name == 'finish':
                result = {'status': 'review_requested'}
            else:
                raise ValueError('unknown tool')
        except (ValueError, SyntaxError, IndexError, StopIteration, OSError, subprocess.SubprocessError) as error:
            result = {'status': 'tool_error', 'message': str(error)[-1200:]}
            if name == 'submit' and self.proposals and self.proposals[-1]['evaluation']['status'] == 'not_admitted':
                self.proposals[-1]['evaluation'] = {'status': 'protocol_error', 'message': str(error)[-1200:]}
        event.update(result=result, finished=time.time())
        self.events.append(event)
        self.save()
        return event
