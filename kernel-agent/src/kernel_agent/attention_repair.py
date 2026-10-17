"""V4 paired control and diagnostic-grounded repair; identical evaluator/budgets."""
import json
import os
import subprocess

from kernel_agent import staged_contract_v4 as previous
from kernel_agent.repair_progress import alpha_identity, compiler_observation
from kernel_agent.representative import ROOT


class ObservedEvaluator:
    def evaluate(self, row, final=False):
        if not final:
            if self.gpu_calls >= 9:
                raise ValueError('search GPU budget exhausted')
            self.gpu_calls += 1
            self.save()
        command = [self.gpu_python, str(ROOT/'scripts/evaluate_attention_observed_serial.py'),
                   '--lock-file', str(self.lock_file), self.task_id, row['path']]
        if final:
            command.append('--final')
        child = subprocess.run(command, capture_output=True, text=True, timeout=400,
                               env={**os.environ, 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT)})
        result = json.loads(child.stdout) if child.returncode == 0 else {
            'status': 'evaluator_error', 'message': (child.stdout+child.stderr)[-1800:]}
        if result.get('status') == 'correct':
            assert result['source_sha256'] == row['source_sha256']
            assert sum(result['candidate_jit_dispatches'].values()) > 0
        return result


class ControlTools(ObservedEvaluator, previous.Tools):
    def repair_feedback(self):
        return None


class RepairTools(ControlTools):
    def evaluate(self, row, final=False):
        identity = alpha_identity(row['code'])
        # A changed diagnostic expression alone is not required: a type annotation,
        # helper, launch binding or any other executable change may fix the issue.
        # Refuse only whole-module rename equivalence. No kernel source is edited.
        if not final and identity:
            prior = next((r for r in reversed(self.rows) if r is not row
                          and r['evaluation'].get('status') in {'correct', 'compile_error', 'runtime_error', 'numeric_error'}
                          and alpha_identity(r['code']) == identity), None)
            if prior:
                return {'status': 'repair_no_progress', 'prior_source_sha256': prior['source_sha256'],
                        'prior_status': prior['evaluation']['status'], 'alpha_identity': identity,
                        'message': 'Whole module differs only in bound identifiers/formatting. Previous result remains relevant; '
                                   'this is not a repair or a new implementation. No GPU evaluation ran. '
                                   'Change the failing operation, its types/dependencies or launch bindings; explanation alone is insufficient.'}
        result = super().evaluate(row, final)
        if not final:
            anchor = compiler_observation(row['code'], result)
            if anchor:
                result['compiler_observation'] = anchor
        return result

    def repair_feedback(self):
        failures = [e for e in self.events if e['tool'] == 'evaluate' and
                    e['result'].get('status') in {'compile_error', 'runtime_error', 'numeric_error'}]
        if not failures:
            return None
        event = failures[-1]
        value = event['result']
        source = next(r['code'] for r in self.proposals if r['source_sha256'] == event['source_sha256'])
        anchor = value.get('compiler_observation') or compiler_observation(source, value)
        repeats = sum(e['result'].get('status') == value['status'] and
                      str(e['result'].get('message', '')).splitlines()[-1:] == str(value.get('message', '')).splitlines()[-1:]
                      for e in failures)
        return {'failed_event': event['id'], 'failed_source_sha256': event['source_sha256'],
                'status': value['status'], 'same_diagnostic_occurrences': repeats,
                'compiler_observation': anchor,
                'numeric_witness': value.get('numeric_witness'),
                'semantic_relations': value.get('semantic_relations'),
                'last_refusal': self.events[-1]['result'] if self.events[-1]['result'].get('status') in
                                {'repair_no_progress', 'semantic_repeat'} else None,
                'repair_contract': 'Use the actual failed source and counterexample. State in reason which operation/type/dependency/launch binding changed '
                                   'and which observed failure it targets, then submit the complete changed module. Renaming or repeating an explanation is not progress. '
                                   'Compiler location and numeric relations are observations, not proven root causes. '
                                   'If the same failure recurs, reconsider the implementation rather than repeat the same local edit. '
                                   'Final workloads are hidden. No extra tool or model budget.'}
