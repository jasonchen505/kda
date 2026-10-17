"""Known response errors spend an existing action; uncertain calls are not replayed."""
import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.campaign import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]


def test_received_bad_writer_reply_does_not_cancel_remaining_actions(tmp_path):
    response = {'choices': [{'message': {'content': '{'}}]}
    atomic_json(tmp_path / 'requests/call-03.response.json', response)
    atomic_json(tmp_path / 'requests/call-03.json', {'status': 'completed', 'response_sha256': digest(response)})
    planner = fakes.FakeAgentDriver(reply=[json.dumps({'text': 'draft fixture'}), json.dumps({'text': 'plan fixture'})])
    finish = json.dumps({'tool': 'finish', 'argument': '', 'code_lines': [], 'edits': [], 'reason': 'fake'})
    writer = fakes.FakeAgentDriver(reply=['{', finish, finish])
    reviewer = fakes.FakeAgentDriver(reply=[json.dumps({'accept': False, 'evidence_ids': [],
        'missing': 'No kernel', 'next_step': 'implement'})] * 2)
    asyncio.run(fakes.run_fake(str(ROOT / 'flows/staged_kernel_corrected.py'), 'fixture',
        agents={'planner': planner, 'writer': writer, 'reviewer': reviewer}, params={
            'directory': str(tmp_path), 'task_id': 'affine_silu', 'contract': 'fixture',
            'arm': 'staged_transfer', 'gpu_python': 'unused', 'lock_file': str(tmp_path / 'lock')}))
    calls = json.loads((tmp_path / 'calls.json').read_text())
    assert len(writer.prompts) == 3 and len(calls) == 7
    assert calls[2]['status'] == 'received_reply_error'
    tools = json.loads((tmp_path / 'tools.json').read_text())
    assert tools['events'][0]['tool'] == 'decode_reply' and tools['submissions'] == 0
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['failure']['type'] == 'StageNotAccepted' and len(result['stages']) == 1
