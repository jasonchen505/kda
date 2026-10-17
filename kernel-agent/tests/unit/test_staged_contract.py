"""Regression counterexamples for observed phase2 failures; no model/GPU scores."""
import pytest
from pydantic import ValidationError

from kernel_agent import staged_contract as c
from kernel_agent.staged_tools import Tools as BaseTools

SOURCE='import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


def measured(self,row,final=False):
    self.gpu_calls+=1
    return {'status':'correct','source_sha256':row['source_sha256'],'checks_passed':28,
        'timings':[{'shape':[n],'candidate_us':1.,'compiled_us':1.2} for n in [1,2,3]]}


def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(BaseTools,'evaluate',measured)
    monkeypatch.setattr(BaseTools,'profile',lambda self,row:{'status':'profiled_activity',
        'source_sha256':row['source_sha256'],'activity':[{'device_type':'CUDA','duration_us':2.}],
        'hardware_counters_available':False})
    return c.Tools(tmp_path,'affine_silu','staged_transfer','unused',tmp_path/'lock')


def _test_source(tools,source,stage):
    return tools.execute({'tool':'test','code_lines':source.splitlines(),'argument':'','reason':'fixture'},stage)


def test_atomic_test_really_evaluates_once_and_comment_repeat_does_not(monkeypatch,tmp_path):
    tools=setup(monkeypatch,tmp_path)
    event=_test_source(tools,SOURCE,'research')
    assert event['result']['status']=='correct' and tools.gpu_calls==1
    _test_source(tools,SOURCE+'# changed comment\n','profile_refine')
    assert tools.gpu_calls==1 and tools.events[-1]['result']['status']=='semantic_repeat'


def test_phase2_accepts_real_negative_result_without_counter_or_speedup(monkeypatch,tmp_path):
    tools=setup(monkeypatch,tmp_path)
    _test_source(tools,SOURCE,'research')
    tools.start_phase()
    tools.execute({'tool':'profile','argument':'latest_correct','code_lines':[]},'profile_refine')
    monkeypatch.setattr(BaseTools,'evaluate',lambda self,row,final=False:{'status':'compile_error','message':'real compiler refusal'})
    _test_source(tools,SOURCE.replace('pass','a = 1'),'profile_refine')
    ready=tools.readiness('profile_refine')
    assert ready['ready']
    review=c.Review(accept=True,evidence_ids=ready['required_evidence_ids'],selected_candidate='c1',
        decision='tested_negative',limitation='activity_only_no_causal_attribution',hypotheses=[],reason='retain correct baseline')
    assert c.gate('profile_refine',tools,review)[0]
    assert not tools.readiness('shape_tune')['ready']


def test_current_baseline_and_actual_variant_are_required(monkeypatch,tmp_path):
    tools=setup(monkeypatch,tmp_path)
    _test_source(tools,SOURCE,'research')
    tools.start_phase()
    tools.emit('profile','profile_refine',{'status':'profiled_activity'},sha='forged')
    _test_source(tools,SOURCE+'# same AST\n','profile_refine')
    assert len(tools.readiness('profile_refine')['missing'])==2


def test_shape_comparison_requires_two_correct_distinct_ast_and_real_compare(monkeypatch,tmp_path):
    tools=setup(monkeypatch,tmp_path)
    _test_source(tools,SOURCE,'research')
    _test_source(tools,SOURCE.replace('pass','a = 1'),'profile_refine')
    tools.start_phase()
    assert tools.comparison()['selected_candidate']=='c1'
    assert not tools.readiness('shape_tune')['ready']
    tools.execute({'tool':'compare','argument':'','code_lines':[]},'shape_tune')
    assert tools.readiness('shape_tune')['ready']
    tools.rows[-1]['evaluation']['timings'][-1]['shape']=[4]
    assert tools.comparison() is None


def test_tool_specific_schema_rejects_patch_guide_and_code_in_profile():
    schema=c.action_schema(['profile','test'])
    with pytest.raises(ValidationError):
        schema.model_validate({'action':{'tool':'profile','argument':'guide','code_lines':[],'reason':'wrong'}})
    with pytest.raises(ValidationError):
        schema.model_validate({'action':{'tool':'profile','argument':'latest_correct','code_lines':['bad'],'reason':'wrong'}})


def test_single_citation_list_and_winner_validation(monkeypatch,tmp_path):
    tools=setup(monkeypatch,tmp_path)
    _test_source(tools,SOURCE,'research')
    review=c.Review(accept=True,evidence_ids=['e999'],selected_candidate='c2',decision='keep_generic',
        limitation='activity_only_no_causal_attribution',hypotheses=['unverified'],reason='fixture')
    assert len(c.gate('research',tools,review)[1])>=3
