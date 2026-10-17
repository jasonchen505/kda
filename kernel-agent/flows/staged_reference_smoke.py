"""Real read-example route smoke; no kernel or full-stage success is scored."""
import json
from pathlib import Path

from hmz.flows import Agent,AgentCollection,EnvCollection,FlowParams,flow

from kernel_agent import staged,staged_verified_guard
from kernel_agent.campaign import atomic_json
from kernel_agent.representative_tasks import specs
from kernel_agent.staged_gateway import MARKER


class Agents(AgentCollection):
    planner: Agent
    writer: Agent
    reviewer: Agent


class Envs(EnvCollection):
    pass


class Params(FlowParams):
    directory:str
    task_id:str
    contract:str
    arm:str
    gpu_python:str
    lock_file:str
    smoke:bool=False
    initial_path:str=''
    policy:str='guarded'


@flow(agents=Agents,envs=Envs,params=Params)
async def reference_smoke(task,*,agents,envs,params,ctx):
    d=Path(params.directory)
    calls=[]
    async def ask(role,schema,instruction):
        t={'id':f'call-{len(calls)+1:02d}','allocated':len(calls)+1,'role':role,
            'stage':'smoke','schema':schema.model_json_schema(),'status':'allocated_before_dispatch'}
        calls.append(t)
        atomic_json(d/'calls.json',calls)
        atomic_json(d/'control.json',t)
        session=await agents[role].spawn()
        reply=await agents[role].run(MARKER+json.dumps({'ticket':t['id'],
            'sections':[('smoke',instruction,True)]}),session=session,output_schema=schema)
        t.update(status='sdk_completed',reply=reply.model_dump(mode='json'))
        atomic_json(d/'calls.json',calls)
        return reply
    await ask('planner',staged.Document,'Integration only. Return exactly {"text":"inspect example route"}. No extra fields.')
    action=await ask('writer',staged_verified_guard.action_schema(['read']),
        'Integration only. Return exactly {"tool":"read","argument":"example","code_lines":[],"edits":[],"reason":"integration"}. No kernel or extra fields.')
    assert action.tool=='read' and action.argument=='example'
    tools=staged_verified_guard.Tools(d,params.task_id,params.arm,params.gpu_python,Path(params.lock_file))
    event=tools.execute(action,'smoke')
    assert event['result']['text']==specs()[params.task_id]['api_notes']
    await ask('reviewer',staged.Document,'Integration only: an example read succeeded; no kernel or study was completed. Return exactly {"text":"reference route only"}.')
    atomic_json(d/'smoke.json',{'status':'reference_route_verified','event':event,
        'scope':'Actual SDK/worker/read-example integration only;0GPU calls,no scored kernel or full-stage outcome'})
    return 'reference integration only'
