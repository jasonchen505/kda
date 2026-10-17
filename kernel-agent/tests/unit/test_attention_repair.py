"""Real feedback/refusal state transitions with synthetic independent outcomes."""
import json

from kernel_agent.attention_repair import ControlTools, ObservedEvaluator, RepairTools
from kernel_agent.staged_guard import import_parent

SOURCE = 'import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(X, D):\n    scale = D ** 0.5\ndef run(x):\n    kernel[(1,)](x, 24)\n    return x\n'


def action(code):
    return {'tool':'submit_and_evaluate','argument':'','code_lines':code.splitlines(),'reason':'synthetic repair'}


def test_rename_refusal_spends_source_but_not_gpu_and_annotation_fix_is_allowed(monkeypatch,tmp_path):
    def evaluate(self,row,final=False):
        self.gpu_calls += 1
        return {'status':'compile_error','message':"    scale = D ** 0.5\n              ^\nAttributeError('pow')"}
    monkeypatch.setattr(ObservedEvaluator,'evaluate',evaluate)
    t=RepairTools(tmp_path,'attention','staged_transfer','unused',tmp_path/'lock')
    t.start_phase()
    t.execute(action(SOURCE),'research')
    first=t.rows[0]['source_sha256']
    t.execute(action(SOURCE.replace('scale','factor')),'research')
    assert t.submissions==2 and t.gpu_calls==1
    assert t.events[-1]['result']['status']=='repair_no_progress'
    assert t.events[-1]['result']['prior_source_sha256']==first
    assert t.repair_feedback()['failed_source_sha256']==first
    assert not t.readiness('research')['ready']
    t.execute(action(SOURCE.replace('D):','D: tl.constexpr):')),'research')
    assert t.gpu_calls==2 and t.events[-1]['result']['status']=='compile_error'
    assert json.loads((tmp_path/'tools.json').read_text())['submissions']==3


def test_numeric_feedback_is_source_bound_and_control_hides_it(monkeypatch,tmp_path):
    witness={'coordinates':[{'index':[0,1,5,21],'actual':1.,'expected':2.}], 'mismatches':313}
    monkeypatch.setattr(ObservedEvaluator,'evaluate',lambda self,row,final=False:{
        'status':'numeric_error','message':'actual failed search','numeric_witness':witness,
        'semantic_relations':[{'kind':'zero_q_uniform','passed':False}]})
    for cls in [ControlTools,RepairTools]:
        t=cls(tmp_path/cls.__name__,'attention','staged_transfer','unused',tmp_path/'lock')
        t.start_phase()
        t.execute(action(SOURCE),'research')
        if cls is ControlTools:
            assert t.repair_feedback() is None
        else:
            feedback=t.repair_feedback()
            assert feedback['numeric_witness']==witness and feedback['failed_event']=='e2'
            assert feedback['failed_source_sha256']==t.rows[0]['source_sha256']


def test_imported_failed_parent_does_not_need_gpu_to_reject_a_rename(monkeypatch,tmp_path):
    from kernel_agent.staged import text_hash
    parent=tmp_path/'parent.json'
    parent.write_text(json.dumps({'code':SOURCE,'source_sha256':text_hash(SOURCE),
        'evaluation':{'status':'compile_error','message':'actual parent failure'}}))
    monkeypatch.setattr(ObservedEvaluator,'evaluate',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('unexpected GPU call')))
    t=RepairTools(tmp_path/'arm','attention','staged_transfer','unused',tmp_path/'lock')
    import_parent(t,parent)
    t.start_phase()
    t.execute(action(SOURCE.replace('kernel','renamed')),'research')
    assert t.gpu_calls==0 and t.submissions==1 and t.events[-1]['result']['status']=='repair_no_progress'
