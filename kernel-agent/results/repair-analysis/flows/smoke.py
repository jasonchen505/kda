"""Exercise the real Humanize coding-agent driver before kernel experiments."""
from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, LocalEnv, flow
from kernel_agent import humanize_api


class Agents(AgentCollection):
    builder: Agent


class Envs(EnvCollection):
    workspace: LocalEnv


@flow(agents=Agents, envs=Envs, params=FlowParams)
async def smoke(task, *, agents, envs, params, ctx):
    session = await humanize_api.spawn(agents["builder"], envs["workspace"])
    return await humanize_api.run(agents["builder"], task, session=session, workspace=envs["workspace"])
