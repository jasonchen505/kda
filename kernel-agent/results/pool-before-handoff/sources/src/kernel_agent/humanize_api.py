"""Bind the workspace through either of the two verified Humanize public APIs."""
import inspect

from hmz.flows import Agent

# 77e3d065 binds a session at spawn; d7f732e9 binds each turn at run.
# Inspect the public contract, not an internal driver or a version alias.
SESSION_ENV = "env" in inspect.signature(Agent.spawn).parameters


async def spawn(agent, workspace):
    return await agent.spawn(env=workspace) if SESSION_ENV else await agent.spawn()


async def run(agent, prompt, *, session, workspace):
    if SESSION_ENV:
        return await agent.run(prompt, session=session)
    return await agent.run(prompt, session=session, env=workspace)
