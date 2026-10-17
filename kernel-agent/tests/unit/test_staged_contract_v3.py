"""Observed repeated source arguments must be constrained and precisely diagnosed."""
import json

import pytest
from pydantic import ValidationError

from kernel_agent import staged_contract_v3 as c


def test_source_only_schema_cannot_accept_the_repeated_latest_correct_argument():
    schema=c.action_schema(['submit_and_evaluate'])
    assert schema.model_json_schema()['properties']['argument']['const']==''
    bad={'tool':'submit_and_evaluate','argument':'latest_correct','code_lines':['import triton'],'reason':'fixture'}
    with pytest.raises(ValidationError):
        schema.model_validate(bad)
    diagnostic=c.diagnose_reply(json.dumps(bad),schema)
    assert diagnostic['received_argument']=='latest_correct'
    assert diagnostic['errors'][0]['location']==['argument']
    assert '"argument": ""' in c.wire_format('writer') or 'argument=""' in c.wire_format('writer')


def test_mixed_tool_schema_still_enforces_cross_field_preconditions():
    schema=c.action_schema(['submit_and_evaluate','profile'])
    bad={'tool':'submit_and_evaluate','argument':'latest_correct','code_lines':['import triton'],'reason':'fixture'}
    diagnostic=c.diagnose_reply(json.dumps(bad),schema)
    assert 'Invalid argument for submit_and_evaluate' in diagnostic['errors'][0]['message']


def test_failed_source_feedback_retains_real_finite_witness_without_hidden_data():
    witness={'phase':'search_only','inputs':[],'expected':{'values':[1.]},'actual':{'values':[2.]}}
    value={'status':'numeric_error','counterexample':witness,'message':'wrong'}
    assert c.compact_verdict(value)['finite_search_witness']==witness
    assert 'finite_search_witness' not in c.compact_verdict({'status':'correct'})
