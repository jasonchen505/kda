"""Tool observability at the corrected phase boundary; no model/GPU results."""
from kernel_agent.campaign import atomic_json, digest
from kernel_agent.staged import Document
from kernel_agent.staged_phase_interface import Tools, review_events, recoverable_writer_reply


def test_reviewer_sees_failed_reads_and_empty_searches():
    events = [
        {'id': 'e1', 'tool': 'read', 'stage': 'research', 'source_sha256': None,
         'result': {'status': 'tool_error', 'message': 'No generated source exists'}},
        {'id': 'e2', 'tool': 'research', 'stage': 'research', 'source_sha256': None,
         'result': {'status': 'no_matches', 'query': 'fixture', 'pages': []}},
    ]
    packet = review_events(events)
    assert [r['id'] for r in packet] == ['e1', 'e2']
    assert packet[0]['result']['message'] == 'No generated source exists'
    assert packet[1]['result']['status'] == 'no_matches'


def test_empty_workspace_error_is_actionable_without_inventing_code(tmp_path):
    tools = Tools(tmp_path, 'affine_silu', 'staged_transfer', 'unused', tmp_path / 'lock')
    event = tools.execute({'tool': 'read', 'argument': 'source'}, 'research')
    assert event['result']['status'] == 'tool_error'
    assert 'submit' in event['result']['message'] and 'read api/example' in event['result']['message']
    assert tools.rows == [] and tools.proposals == [] and tools.gpu_calls == 0


def test_recovery_requires_received_invalid_hash_bound_response(tmp_path):
    response = {'choices': [{'message': {'content': '{'}}]}
    atomic_json(tmp_path / 'requests/call-03.response.json', response)
    path = tmp_path / 'requests/call-03.json'
    for state in ['response_unknown', 'dispatched', 'not_dispatched']:
        atomic_json(path, {'status': state, 'response_sha256': digest(response)})
        assert not recoverable_writer_reply(tmp_path, 'call-03', Document)
    atomic_json(path, {'status': 'completed', 'response_sha256': digest(response)})
    assert recoverable_writer_reply(tmp_path, 'call-03', Document)
    atomic_json(path, {'status': 'completed', 'response_sha256': 'forged'})
    assert not recoverable_writer_reply(tmp_path, 'call-03', Document)
