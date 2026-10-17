"""Integrity tests for staged acceptance and tool evidence, not agent successes."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kernel_agent import staged
from kernel_agent.staged_tools import Tools, parse_metrics


def review(ids):
    return staged.Review(accept=True, evidence_ids=ids, missing='', next_step='')


def evidence():
    rows = [{'id': 'c1', 'source_sha256': 'aaa', 'evaluation': {'status': 'correct'}}]
    events = [{'id': 'e1', 'tool': 'evaluate', 'source_sha256': 'aaa',
               'result': {'status': 'correct', 'timings': [{}, {}, {}]}}]
    return rows, events


def test_reviewer_cannot_forge_measurement_ids():
    rows, events = evidence()
    assert not staged.stage_gate('research', rows, events, review(['imagined']))[0]
    assert staged.stage_gate('research', rows, events, review(['e1']))[0]


@pytest.mark.parametrize('status,sha', [('profile_unavailable', 'aaa'), ('profiled', 'different')])
def test_profile_gate_rejects_failure_and_stale_source(status, sha):
    rows, events = evidence()
    events.append({'id': 'p1', 'tool': 'profile', 'source_sha256': sha, 'result': {'status': status}})
    assert not staged.stage_gate('profile_refine', rows, events, review(['e1', 'p1']))[0]


def test_profile_gate_needs_actual_correct_source_and_counters():
    rows, events = evidence()
    events.append({'id': 'p1', 'tool': 'profile', 'source_sha256': 'aaa', 'result': {'status': 'profiled'}})
    assert staged.stage_gate('profile_refine', rows, events, review(['e1', 'p1']))[0]
    rows[0]['evaluation']['status'] = 'numeric_error'
    assert not staged.stage_gate('profile_refine', rows, events, review(['e1', 'p1']))[0]


def test_shape_gate_requires_measured_distribution():
    rows, events = evidence()
    assert not staged.stage_gate('shape_tune', rows, events, review(['e1']))[0]
    rows.append({'id': 'c2', 'source_sha256': 'bbb', 'evaluation': {'status': 'correct'}})
    events.append({'id': 'e2', 'tool': 'evaluate', 'source_sha256': 'bbb',
                   'result': {'status': 'correct', 'timings': [{}, {}, {}]}})
    assert staged.stage_gate('shape_tune', rows, events, review(['e1', 'e2']))[0]
    events[0]['result']['timings'] = [{}, {}]
    assert not staged.stage_gate('shape_tune', rows, events, review(['e1', 'e2']))[0]


def test_final_workloads_cannot_be_requested_as_a_tool(tmp_path):
    with pytest.raises(ValidationError):
        staged.Action(tool='final', argument='', code_lines=[], edits=[], reason='')
    tools = Tools(tmp_path, 'affine_silu', 'staged', 'unused', tmp_path / 'lock')
    result = tools.execute({'tool': 'read', 'argument': '../final.json'}, 'research')
    assert result['result']['status'] == 'tool_error'


def test_failed_patch_retains_source_and_spends_submission(tmp_path):
    tools = Tools(tmp_path, 'affine_silu', 'staged_transfer', 'unused', tmp_path / 'lock')
    source = 'import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'
    parent = {'id': 'c1', 'code': source, 'source_sha256': staged.text_hash(source),
              'evaluation': {'status': 'numeric_error'}}
    tools.rows.append(parent)
    tools.proposals.append(parent)
    result = tools.execute({'tool': 'patch', 'argument': 'forged', 'reason': 'test', 'edits': [
        {'start_line': 4, 'end_line': 4, 'replacement_lines': ['    return']}], 'code_lines': []}, 'research')
    assert result['result']['status'] == 'tool_error'
    assert tools.rows == [parent] and tools.rows[0]['code'] == source
    assert tools.submissions == 1
    ledger = json.loads((tmp_path / 'tools.json').read_text())
    assert ledger['submissions'] == 1


def test_rejected_source_is_visible_for_fresh_session_repair(tmp_path):
    tools = Tools(tmp_path, 'affine_silu', 'staged_transfer', 'unused', tmp_path / 'lock')
    code = 'import triton\n@triton.jit\n_def kernel():\n    pass\ndef run(x):\n    return x\n'
    result = tools.execute({'tool': 'submit', 'argument': '', 'reason': 'fixture',
                            'code_lines': code.splitlines(), 'edits': []}, 'research')
    assert result['result']['status'] == 'tool_error' and tools.rows == []
    assert tools.proposals[-1]['code'] == code
    assert Path(tools.proposals[-1]['proposal_path']).read_text() == code
    assert code in staged.source_context(tools.proposals, False)[0][1]


def test_profile_csv_retains_units_and_ignores_other_lines():
    data = '\n'.join(['==PROF== Connected', '"Kernel Name","Metric Name","Metric Unit","Metric Value"',
                      '"probe","gpu__time_duration.sum","nsecond","1234"',
                      '{"device_audit":{}}'])
    assert parse_metrics(data) == [{'Kernel Name': 'probe', 'Metric Name': 'gpu__time_duration.sum',
                                    'Metric Unit': 'nsecond', 'Metric Value': '1234'}]
