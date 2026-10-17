"""Guards prevent real side effects or unsupported acceptance; no model efficacy claims."""
import json

import pytest
from pydantic import ValidationError

from kernel_agent import staged_guard as g
from kernel_agent.staged import Review, text_hash

SOURCE = 'import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


def plan(names):
    return g.Plan(objective='fixture', steps=[g.Step(tool=n, argument='', purpose='fixture') for n in names], fallback='same budget')


def tools(tmp_path):
    return g.Tools(tmp_path, 'affine_silu', 'staged_transfer', 'unused', tmp_path / 'lock')


def test_plan_rejects_overspending_and_missing_source_dependencies():
    assert g.plan_issues(plan(['read', 'profile', 'submit', 'evaluate', 'finish']), 4, 'profile_refine', True)
    assert g.plan_issues(plan(['evaluate', 'submit']), 6, 'research')
    assert not g.plan_issues(plan(['submit', 'evaluate', 'finish']), 6, 'research')
    with pytest.raises(ValidationError):
        plan(['benchmark'])


def test_reserved_evaluation_actions_remove_reference_detours(tmp_path):
    t = tools(tmp_path)
    assert 'read' in t.allowed('research', 6)
    assert set(t.allowed('research', 2)) == {'submit'}
    t.execute({'tool': 'submit', 'code_lines': SOURCE.splitlines(), 'reason': 'synthetic fixture'}, 'research')
    assert t.allowed('research', 1) == ['evaluate']
    assert g.action_schema(['evaluate']).model_json_schema()['properties']['tool']['const'] == 'evaluate'


def test_aliases_and_duplicate_profile_never_run_second_measurement(monkeypatch, tmp_path):
    t = tools(tmp_path)
    t.execute({'tool': 'submit', 'code_lines': SOURCE.splitlines(), 'reason': 'synthetic fixture'}, 'research')
    t.rows[-1]['evaluation'] = {'status': 'correct'}
    called = []
    monkeypatch.setattr(t, 'profile', lambda row: called.append(row['source_sha256']) or {'status': 'profiled_activity'})
    t.execute({'tool': 'profile', 'argument': 'latest source SHA'}, 'profile_refine')
    event = t.execute({'tool': 'profile', 'argument': 'latest_correct'}, 'profile_refine')
    assert called == [text_hash(SOURCE)] and event['result']['status'] == 'repeat_refused'


def test_missing_source_is_actionable_and_old_reference_is_not_reexecuted(tmp_path, monkeypatch):
    t = tools(tmp_path)
    assert 'Unknown source' in t.execute({'tool': 'evaluate', 'argument': 'invented'}, 'research')['result']['message']
    assert 'api/example/guide' in t.execute({'tool': 'read', 'argument': 'source'}, 'research')['result']['message']
    from kernel_agent import staged_tools
    called = []
    monkeypatch.setattr(staged_tools, 'retrieve_example', lambda task: called.append(task) or {'text': 'fixture'})
    t.execute({'tool': 'read', 'argument': 'example'}, 'research')
    t.execute({'tool': 'read', 'argument': 'example'}, 'research')
    assert called == ['affine_silu']


def test_claims_reject_forgery_and_keep_hypotheses_unverified(tmp_path):
    t = tools(tmp_path)
    errors, _ = g.validate_claims([g.Claim(kind='search_correct', evidence_ids=['fake'], hypothesis='')], t)
    assert errors
    errors, facts = g.validate_claims([g.Claim(kind='hypothesis', evidence_ids=[], hypothesis='launch overhead may matter')], t)
    assert not errors and facts[0]['verified'] is False
    with pytest.raises(ValidationError):
        g.Claim(kind='launch_overhead_proven', evidence_ids=[], hypothesis='')


def test_comment_only_variants_and_different_shapes_cannot_pass_comparison(tmp_path):
    t = tools(tmp_path)
    for i, code in enumerate([SOURCE, SOURCE+'# comment\n']):
        sha = text_hash(code)
        t.rows.append({'source_sha256': sha, 'code': code})
        t.events.append({'id': f'e{i}', 'tool': 'evaluate', 'source_sha256': sha,
                         'result': {'status': 'correct', 'timings': [{'shape': [n]} for n in [1, 2, 3]]}})
    claim = g.Claim(kind='search_compared', evidence_ids=['e0', 'e1'], hypothesis='')
    assert g.validate_claims([claim], t)[0]
    t.rows[-1]['code'] = SOURCE.replace('pass', 'a = 1')
    assert not g.validate_claims([claim], t)[0]
    t.events[-1]['result']['timings'] = [{'shape': [n]} for n in [4, 5, 6]]
    assert g.validate_claims([claim], t)[0]


def test_shared_failed_parent_is_identical_and_spends_no_model_submission(tmp_path):
    path = tmp_path/'parent.json'
    path.write_text(json.dumps({'code': SOURCE, 'source_sha256': text_hash(SOURCE), 'evaluation': {'status': 'numeric_error'}}))
    t = tools(tmp_path/'arm')
    g.import_parent(t, path)
    assert t.proposals[0]['code'] == SOURCE and t.submissions == 0 and t.gpu_calls == 0
    assert t.events[0]['shared_parent'] and t.rows[0]['imported_parent']


def test_plain_reviewer_cannot_bless_unavailable_profile(tmp_path):
    t = tools(tmp_path)
    review = g.Review(**Review(accept=True, evidence_ids=['fake'], missing='', next_step='').model_dump(), claims=[])
    assert not g.gate('profile_refine', t, review)[0]
