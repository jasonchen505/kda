"""Full phase wiring with synthetic model/GPU data, explicitly excluded from agent scores."""
import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.staged import text_hash
from kernel_agent.staged_tools import Tools

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


def test_three_phases_commit_selection_and_never_run_final(monkeypatch, tmp_path):
    sources = [SOURCE, SOURCE + '# synthetic second source\n']
    hashes = [text_hash(s) for s in sources]

    def measured(self, row, final=False):
        assert not final, 'Flow must defer final data to the supervisor'
        return {'status': 'correct', 'source_sha256': row['source_sha256'], 'checks_passed': 28,
                'timings': [{'shape': [n], 'candidate_us': 1., 'compiled_us': 1.2 if row['source_sha256'] == hashes[1] else .8}
                            for n in [1, 2, 3]], 'device_binding': {'verified': True}}

    monkeypatch.setattr(Tools, 'evaluate', measured)
    monkeypatch.setattr(Tools, 'profile', lambda self, row: {'status': 'profiled_activity',
                        'source_sha256': row['source_sha256'], 'hardware_counters_available': False,
                        'activity': [{'name': 'synthetic', 'device_type': 'CUDA', 'duration_us': 1.}]})

    def action(tool, argument='', code=None):
        return json.dumps({'tool': tool, 'argument': argument, 'code_lines': code.splitlines() if code else [],
                           'edits': [], 'reason': 'Synthetic wiring fixture, not a real generated kernel'})

    writer = fakes.FakeAgentDriver(reply=[action('submit', code=sources[0]), action('evaluate', hashes[0]),
        action('finish'), action('profile', hashes[0]), action('finish'), action('submit', code=sources[1]),
        action('evaluate', hashes[1]), action('finish')])
    planner = fakes.FakeAgentDriver(reply=[json.dumps({'text': 'Synthetic draft/plan'})] * 6)
    reviewer = fakes.FakeAgentDriver(reply=[json.dumps({'accept': True, 'evidence_ids': ids,
        'missing': '', 'next_step': 'finish'}) for ids in [['e2'], ['e2', 'e4'], ['e2', 'e7']]])
    asyncio.run(fakes.run_fake(str(ROOT / 'flows/staged_kernel_corrected.py'), 'fixture',
        agents={'planner': planner, 'writer': writer, 'reviewer': reviewer}, params={
            'directory': str(tmp_path), 'task_id': 'affine_silu', 'contract': 'synthetic fixture',
            'arm': 'staged_transfer', 'gpu_python': 'unused', 'lock_file': str(tmp_path / 'lock')}))
    result = json.loads((tmp_path / 'result.json').read_text())
    assert [p['accepted'] for p in result['stages']] == [True, True, True]
    assert result['selection']['source_sha256'] == hashes[1]
    assert result['final'] is None and result['model_opportunities'] == 17
