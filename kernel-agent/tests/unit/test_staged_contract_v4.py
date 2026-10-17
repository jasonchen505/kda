"""Required real-evidence actions and exact-source protocol recovery, CPU only."""
from kernel_agent import staged_contract_v4 as c
from kernel_agent.staged_tools import Tools as BaseTools

SOURCE='import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


def test_next_actions_follow_actual_phase_evidence(monkeypatch,tmp_path):
    monkeypatch.setattr(BaseTools,'evaluate',lambda self,row,final=False:{
        'status':'correct','source_sha256':row['source_sha256'],'checks_passed':28,
        'timings':[{'shape':[n],'candidate_us':1.,'compiled_us':1.2} for n in [1,2,3]]})
    monkeypatch.setattr(BaseTools,'profile',lambda self,row:{'status':'profiled_activity',
        'source_sha256':row['source_sha256'],'activity':[]})
    t=c.Tools(tmp_path,'l2_normalize','staged_transfer','unused',tmp_path/'lock')
    assert t.allowed('research',4)==['submit_and_evaluate']
    t.execute({'tool':'submit_and_evaluate','argument':'','code_lines':SOURCE.splitlines(),'reason':'fixture'},'research')
    t.start_phase()
    assert t.allowed('profile_refine',4)==['profile']
    t.execute({'tool':'profile','argument':'latest_correct','code_lines':[],'reason':'fixture'},'profile_refine')
    assert t.allowed('profile_refine',3)==['submit_and_evaluate']
    t.execute({'tool':'submit_and_evaluate','argument':'','code_lines':SOURCE.replace('pass','a = 1').splitlines(),'reason':'fixture'},'profile_refine')
    t.start_phase()
    assert t.allowed('shape_tune',4)==['compare']
    assert t.source_facts()['source_sha256']==t.proposals[-1]['source_sha256']
    assert not t.readiness('shape_tune')['ready']
    t.execute({'tool':'compare','argument':'','code_lines':[],'reason':'fixture'},'shape_tune')
    assert t.readiness('shape_tune')['ready']


def test_repeated_source_remains_rejected_and_actionable(monkeypatch,tmp_path):
    monkeypatch.setattr(BaseTools,'evaluate',lambda self,row,final=False:{'status':'correct','source_sha256':row['source_sha256'],'timings':[]})
    t=c.Tools(tmp_path,'l2_normalize','staged_transfer','unused',tmp_path/'lock')
    action={'tool':'submit_and_evaluate','argument':'','code_lines':SOURCE.splitlines(),'reason':'fixture'}
    t.execute(action,'research')
    t.start_phase()
    t.execute(action,'profile_refine')
    assert t.events[-1]['result']['status']=='semantic_repeat'
    assert 'identical' in t.packet('profile_refine')['repeat_feedback']
    assert not t.readiness('profile_refine')['ready']
