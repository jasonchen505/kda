"""Exercise the real Humanize flow with fake models; never count it as an agent success."""
import asyncio
import json
from pathlib import Path

import pytest
from hmz.sdk import fakes

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('claimed_accept', [False, True])
def test_failed_or_forged_review_blocks_later_phases(tmp_path, claimed_accept):
    planner = fakes.FakeAgentDriver(reply=[json.dumps({'text': 'draft fixture'}),
                                         json.dumps({'text': 'plan fixture'})])
    writer = fakes.FakeAgentDriver(reply=[json.dumps({'tool': 'finish', 'argument': '', 'code_lines': [],
                                                    'edits': [], 'reason': 'fake untested completion'})] * 2)
    reviewer = fakes.FakeAgentDriver(reply=[json.dumps({'accept': claimed_accept,
        'evidence_ids': ['invented'], 'missing': 'No actual GPU evidence', 'next_step': 'implement'})] * 2)
    asyncio.run(fakes.run_fake(str(ROOT / 'flows/staged_kernel.py'), 'fixture contract',
        agents={'planner': planner, 'writer': writer, 'reviewer': reviewer}, params={
            'directory': str(tmp_path), 'task_id': 'affine_silu', 'contract': 'fixture contract',
            'arm': 'staged_transfer', 'gpu_python': 'unused', 'lock_file': str(tmp_path / 'lock')}))
    value = json.loads((tmp_path / 'result.json').read_text())
    assert value['failure']['type'] == 'StageNotAccepted'
    assert len(value['stages']) == 1 and not value['stages'][0]['accepted']
    assert len(planner.prompts) == 2 and len(writer.prompts) == 2 and len(reviewer.prompts) == 2
    assert value['selection']['source_sha256'] is None and value['final'] is None
    assert not (tmp_path / 'docs/profile_refine').exists()
