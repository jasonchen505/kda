"""Real observed missing-tag/suffix failure; ambiguity and truncation stay rejected."""
import json

from kernel_agent import staged_contract_v3 as c
from kernel_agent.structured_recovery import recover


def test_unique_missing_tag_recovers_exact_model_source_and_logs_ignored_tail():
    schema=c.action_schema(['submit_and_evaluate','read','research'])
    source=['import triton','def run(x):','    return x']
    raw={'argument':'','code_lines':source,'reason':'model source'}
    content=json.dumps(raw)+'"}]"}] prompt noise'
    result=recover(content,schema,['submit_and_evaluate','read','research'])
    assert result[0].tool=='submit_and_evaluate' and result[0].code_lines==source
    assert result[1]['inferred_tool']=='submit_and_evaluate' and result[1]['tail_chars']>0


def test_missing_tag_is_not_guessed_when_two_tools_match():
    schema=c.action_schema(['compare','finish'])
    raw={'argument':'','code_lines':[],'reason':'ambiguous'}
    assert recover(json.dumps(raw),schema,['compare','finish']) is None


def test_truncation_missing_fields_and_wrong_arguments_are_not_repaired():
    schema=c.action_schema(['submit_and_evaluate'])
    raw={'argument':'latest_correct','code_lines':['import triton'],'reason':'wrong argument'}
    assert recover(json.dumps(raw),schema,['submit_and_evaluate']) is None
    assert recover('{"argument":"","code_lines":["incomplete',schema,['submit_and_evaluate']) is None
    assert recover('{"argument":"","code_lines":["import triton"]}',schema,['submit_and_evaluate']) is None
