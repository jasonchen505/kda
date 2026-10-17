"""Actual SDK fake-driver wiring; fixtures are not generated-kernel or GPU successes."""
import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent import staged_guard as g
from kernel_agent.staged import text_hash
from kernel_agent.staged_tools import Tools

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


def action(tool, code='', argument=''):
    return json.dumps({'tool': tool, 'code_lines': code.splitlines(), 'argument': argument, 'edits': [], 'reason': 'synthetic fixture'})


def planned(names):
    return g.Plan(objective='fixture', steps=[g.Step(tool=n, argument='', purpose='fixture') for n in names], fallback='same budget').model_dump_json()


def measured(self, row, final=False):
    assert not final
    return {'status': 'correct', 'source_sha256': row['source_sha256'], 'checks_passed': 28,
            'timings': [{'shape': [n], 'candidate_us': 1., 'compiled_us': 1.2} for n in [1, 2, 3]],
            'device_binding': {'verified': True}}


def test_guarded_complete_stages_use_real_sdk_but_only_synthetic_measurements(monkeypatch, tmp_path):
    monkeypatch.setattr(Tools, 'evaluate', measured)
    monkeypatch.setattr(Tools, 'profile', lambda self, row: {'status': 'profiled_activity',
                        'source_sha256': row['source_sha256'], 'hardware_counters_available': False})
    variant = SOURCE.replace('pass', 'a = 1')
    writer = fakes.FakeAgentDriver(reply=[action('submit', SOURCE), action('evaluate', argument='latest'), action('finish'),
        action('profile', argument='latest_correct'), action('submit', variant), action('evaluate', argument='latest'), action('finish'), action('finish')])
    planner = fakes.FakeAgentDriver(reply=[json.dumps({'text': 'draft'}), planned(['submit', 'evaluate', 'finish']),
        json.dumps({'text': 'draft'}), planned(['profile', 'submit', 'evaluate', 'finish']),
        json.dumps({'text': 'draft'}), planned(['finish'])])
    reviewer = fakes.FakeAgentDriver(reply=[json.dumps({'accept': True, 'evidence_ids': ids, 'missing': '', 'next_step': '',
        'claims': [{'kind': kind, 'evidence_ids': refs, 'hypothesis': ''}]})
        for ids, kind, refs in [(['e2'], 'search_correct', ['e2']),
                               (['e2', 'e4', 'e6'], 'activity_observed', ['e4']),
                               (['e2', 'e6'], 'search_compared', ['e2', 'e6'])]])
    asyncio.run(fakes.run_fake(str(ROOT/'flows/staged_kernel_guarded.py'), 'fixture',
        agents={'planner': planner, 'writer': writer, 'reviewer': reviewer}, params={
        'directory': str(tmp_path), 'task_id': 'affine_silu', 'contract': 'fixture', 'arm': 'staged_transfer',
        'gpu_python': 'unused', 'lock_file': str(tmp_path/'lock'), 'policy': 'guarded'}))
    result = json.loads((tmp_path/'result.json').read_text())
    assert [p['accepted'] for p in result['stages']] == [True, True, True]
    assert result['model_opportunities'] == 17 and result['final'] is None
    assert result['selection']['source_sha256'] == text_hash(SOURCE)


def test_plan_revision_spends_existing_writer_action_not_extra_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(Tools, 'evaluate', measured)
    planner = fakes.FakeAgentDriver(reply=[json.dumps({'text': 'draft'}), planned(['read']), planned(['submit', 'evaluate', 'finish'])])
    writer = fakes.FakeAgentDriver(reply=[action('submit', SOURCE), action('evaluate', argument='latest'), action('finish')])
    reviewer = fakes.FakeAgentDriver(reply=[json.dumps({'accept': False, 'evidence_ids': ['e2'], 'missing': 'synthetic stop',
                                                      'next_step': '', 'claims': []})]*2)
    asyncio.run(fakes.run_fake(str(ROOT/'flows/staged_kernel_guarded.py'), 'fixture',
        agents={'planner': planner, 'writer': writer, 'reviewer': reviewer}, params={
        'directory': str(tmp_path), 'task_id': 'affine_silu', 'contract': 'fixture', 'arm': 'staged_transfer',
        'gpu_python': 'unused', 'lock_file': str(tmp_path/'lock'), 'policy': 'guarded'}))
    calls = json.loads((tmp_path/'calls.json').read_text())
    assert (tmp_path/'docs/research/plan-revision.json').exists()
    assert [r['role'] for r in calls[:6]] == ['planner']*3+['writer']*3
    assert json.loads((tmp_path/'docs/research/plan-revision.json').read_text())['errors'] == []
