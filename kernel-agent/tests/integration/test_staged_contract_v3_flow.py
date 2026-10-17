"""Full real-SDK wiring with fake measurements; never count as model/GPU success."""
import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.staged_tools import Tools

ROOT=Path(__file__).resolve().parents[2]
SOURCE='import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


def test_three_phases_atomic_experiments_and_single_list_reviews(monkeypatch,tmp_path):
    def measured(self,row,final=False):
        return {'status':'correct','source_sha256':row['source_sha256'],'checks_passed':28,
            'timings':[{'shape':[n],'candidate_us':1.,'compiled_us':1.2} for n in [1,2,3]]}
    monkeypatch.setattr(Tools,'evaluate',measured)
    monkeypatch.setattr(Tools,'profile',lambda self,row:{'status':'profiled_activity',
        'source_sha256':row['source_sha256'],'hardware_counters_available':False,'activity':[]})
    def action(tool,code=''):
        return json.dumps({'tool':tool,'argument':'latest_correct' if tool=='profile' else '',
            'code_lines':code.splitlines(),'reason':'synthetic'})
    writer=fakes.FakeAgentDriver(reply=[action('submit_and_evaluate',SOURCE),action('profile'),
        action('submit_and_evaluate',SOURCE.replace('pass','a = 1')),action('compare')])
    plans=[]
    for measure,names in [('search_only',['submit_and_evaluate']),('activity_and_search',['profile','submit_and_evaluate']),('common_shape_search',['compare'])]:
        plans.extend([json.dumps({'text':'synthetic draft'}),json.dumps({'hypothesis':'unverified fixture',
            'measurement':measure,'steps':[{'tool':n,'purpose':'fixture'} for n in names],'negative_result':'retain baseline'})])
    planner=fakes.FakeAgentDriver(reply=plans)
    reviewer=fakes.FakeAgentDriver(reply=[json.dumps({'accept':True,'evidence_ids':ids,
        'selected_candidate':'c1','decision':'keep_generic','limitation':'activity_only_no_causal_attribution',
        'hypotheses':[],'reason':'synthetic evidence only'}) for ids in [['e2'],['e2','e3','e5'],['e2','e5','e6']]])
    asyncio.run(fakes.run_fake(str(ROOT/'flows/staged_kernel_contract_v3.py'),'fixture',
        agents={'planner':planner,'writer':writer,'reviewer':reviewer},params={
            'directory':str(tmp_path),'task_id':'affine_silu','contract':'fixture','arm':'staged_transfer',
            'gpu_python':'unused','lock_file':str(tmp_path/'lock')}))
    result=json.loads((tmp_path/'result.json').read_text())
    assert [s['accepted'] for s in result['stages']]==[True,True,True]
    assert result['model_opportunities']==13 and result['final'] is None
    events=json.loads((tmp_path/'tools.json').read_text())['events']
    assert [e['tool'] for e in events]==['submit','evaluate','profile','submit','evaluate','compare']
