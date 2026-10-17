"""Full real-SDK wiring with fake measurements; never count as model/GPU success."""
import asyncio
import json
from pathlib import Path

import pytest
from hmz.sdk import fakes

from kernel_agent.campaign import digest

from kernel_agent.attention_repair import ObservedEvaluator as Tools
from kernel_agent.staged_tools import Tools as BaseTools

ROOT=Path(__file__).resolve().parents[2]
SOURCE='import triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\n'


@pytest.mark.parametrize('recover_first',[False,True])
@pytest.mark.parametrize('policy',['control','repair'])
def test_three_phases_atomic_experiments_and_single_list_reviews(monkeypatch,tmp_path,recover_first,policy):
    def measured(self,row,final=False):
        return {'status':'correct','source_sha256':row['source_sha256'],'checks_passed':28,
            'timings':[{'shape':[n],'candidate_us':1.,'compiled_us':1.2} for n in [1,2,3]]}
    monkeypatch.setattr(Tools,'evaluate',measured)
    monkeypatch.setattr(BaseTools,'profile',lambda self,row:{'status':'profiled_activity',
        'source_sha256':row['source_sha256'],'hardware_counters_available':False,'activity':[]})
    def action(tool,code=''):
        return json.dumps({'tool':tool,'argument':'latest_correct' if tool=='profile' else '',
            'code_lines':code.splitlines(),'reason':'synthetic'})
    first=action('submit_and_evaluate',SOURCE)
    if recover_first:
        value=json.loads(first)
        del value['tool']
        first=json.dumps(value)+' "}]"}] repeated suffix'
        folder=tmp_path/'requests'
        folder.mkdir()
        response={'choices':[{'message':{'content':first}}]}
        (folder/'call-03.response.json').write_text(json.dumps(response))
        (folder/'call-03.json').write_text(json.dumps({'status':'completed','response_sha256':digest(response)}))
    writer=fakes.FakeAgentDriver(reply=[first,action('profile'),
        action('submit_and_evaluate',SOURCE.replace('pass','a = 1')),action('compare')])
    plans=[]
    for measure,names in [('search_only',['submit_and_evaluate']),('activity_and_search',['profile','submit_and_evaluate']),('common_shape_search',['compare'])]:
        plans.extend([json.dumps({'text':'synthetic draft'}),json.dumps({'hypothesis':'unverified fixture',
            'measurement':measure,'steps':[{'tool':n,'purpose':'fixture'} for n in names],'negative_result':'retain baseline'})])
    planner=fakes.FakeAgentDriver(reply=plans)
    reviewer=fakes.FakeAgentDriver(reply=[json.dumps({'accept':True,'evidence_ids':ids,
        'selected_candidate':'c1','decision':'keep_generic','limitation':'activity_only_no_causal_attribution',
        'hypotheses':[],'reason':'synthetic evidence only'}) for ids in [['e2'],['e2','e3','e5'],['e2','e5','e6']]])
    asyncio.run(fakes.run_fake(str(ROOT/'flows/staged_kernel_attention_repair.py'),'fixture',
        agents={'planner':planner,'writer':writer,'reviewer':reviewer},params={
            'directory':str(tmp_path),'task_id':'affine_silu','contract':'fixture','arm':'staged_transfer',
            'gpu_python':'unused','lock_file':str(tmp_path/'lock'),'policy':policy}))
    result=json.loads((tmp_path/'result.json').read_text())
    assert [s['accepted'] for s in result['stages']]==[True,True,True]
    assert result['model_opportunities']==13 and result['final'] is None
    events=json.loads((tmp_path/'tools.json').read_text())['events']
    assert [e['tool'] for e in events]==['submit','evaluate','profile','submit','evaluate','compare']
    calls=json.loads((tmp_path/'calls.json').read_text())
    assert calls[2]['status']==('received_prefix_recovered' if recover_first else 'sdk_completed')
    if recover_first:
        assert calls[2]['recovery']['inferred_tool']=='submit_and_evaluate'
        source=json.loads((tmp_path/'tools.json').read_text())['proposals'][0]['code']
        assert source==SOURCE
