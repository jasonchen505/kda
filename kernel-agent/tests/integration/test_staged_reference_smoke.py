"""Exercise the actual smoke flow and reference backend; synthetic roles only."""
import asyncio
import json
from pathlib import Path

from hmz.sdk import fakes

from kernel_agent.representative_reference_router import routed_references

ROOT=Path(__file__).resolve().parents[2]


def test_worker_reference_smoke_flow_uses_real_example_route(tmp_path):
    replies={'planner':{'text':'inspect example route'},'writer':{'tool':'read','argument':'example',
        'code_lines':[],'edits':[],'reason':'integration'},'reviewer':{'text':'reference route only'}}
    with routed_references():
        asyncio.run(fakes.run_fake(str(ROOT/'flows/staged_reference_smoke.py'),'integration',
            agents={role:fakes.FakeAgentDriver(reply=json.dumps(value)) for role,value in replies.items()},
            params={'directory':str(tmp_path),'task_id':'attention','contract':'integration','arm':'staged_transfer',
                    'gpu_python':'unused','lock_file':str(tmp_path/'lock')}))
    result=json.loads((tmp_path/'smoke.json').read_text())
    assert result['status']=='reference_route_verified'
    assert result['event']['tool']=='read' and result['event']['resolved_argument']=='example'
    assert len(json.loads((tmp_path/'calls.json').read_text()))==3
