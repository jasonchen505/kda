"""Counterexamples to parameter confusion and accepting a refused measurement."""
import pytest
from pydantic import ValidationError

from kernel_agent import staged_guard as old, staged_guard_v2 as new


def test_source_only_actions_cannot_emit_reference_argument():
    schema = new.action_schema(['submit', 'patch'])
    values = {'tool': 'patch', 'argument': 'guide', 'code_lines': [], 'edits': [], 'reason': 'fixture'}
    with pytest.raises(ValidationError):
        schema.model_validate(values)
    values['argument'] = 'latest'
    assert schema.model_validate(values).argument == 'latest'


def test_plan_revision_schema_has_actual_remaining_limit():
    step = {'tool': 'read', 'argument': 'api', 'purpose': 'fixture'}
    with pytest.raises(ValidationError):
        new.plan_schema(5).model_validate({'objective': 'fixture', 'steps': [step]*6, 'fallback': 'same budget'})


def test_refused_evaluate_cannot_satisfy_actual_variant_requirement(tmp_path):
    t = new.Tools(tmp_path, 'affine_silu', 'staged_transfer', 'unused', tmp_path/'lock')
    t.proposals = [{'source_sha256': 'base', 'code': 'x=1'}, {'source_sha256': 'variant', 'code': 'x=2'}]
    t.phase_entry, t.phase_start = 'base', 0
    event = {'tool': 'evaluate', 'source_sha256': 'variant', 'result': {'status': 'repeat_refused'}}
    t.events = [event]
    assert old.Tools.variant_tested(t) and not t.variant_tested()
    event['result'] = {'status': 'compile_error'}
    event['resolved_argument'] = 'variant'
    assert t.variant_tested()


def test_read_plan_contract_matches_executor_capabilities():
    p = old.Plan(objective='fixture', steps=[old.Step(tool='read', argument='source', purpose='fixture'),
        old.Step(tool='submit', argument='', purpose='fixture'), old.Step(tool='evaluate', argument='latest', purpose='fixture')], fallback='stop')
    assert new.plan_issues(p, 6, 'research', True)
