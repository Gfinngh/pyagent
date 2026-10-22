"""23 — Deadlines and cancellation.

Two distinct mechanisms:

  run_timeout_ms   the whole agent run has a wall-clock budget → AgentTimeoutError
  asyncio task     outer cancellation → asyncio.CancelledError

Neither is a tool-level timeout (that is examples/11).

    pixi run python examples/23_cancellation.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from examples._common import section
from examples._fakes import ScriptedClient, make_tool_call, text_response, tool_call_response
from pyagent.agent import Agent, AgentTimeoutError
from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool


class Empty(BaseModel):
    """No arguments."""


async def hang(args: Empty, context: ExecutionContext | None = None) -> str:
    _ = args, context
    await asyncio.sleep(30)
    return "never"


hang_tool = FunctionTool(
    name="hang",
    description="Sleeps for 30 seconds.",
    schema=Empty,
    func=hang,
)


def one_call_client() -> ScriptedClient:
    return ScriptedClient(
        [
            tool_call_response([make_tool_call("c1", "hang")]),
            text_response("unreachable"),
        ]
    )


async def main() -> None:
    section("1. run_timeout_ms — AgentTimeoutError")
    agent = Agent(
        client=one_call_client(),
        tools=[hang_tool],
        run_timeout_ms=100,
        max_steps=2,
    )
    try:
        await agent.run("go")
        print("  unexpected: run completed")
    except AgentTimeoutError as exc:
        print(f"  AgentTimeoutError: {exc}")
        print(f"  metadata: {exc.metadata}")

    section("2. outer task cancellation — CancelledError")
    agent2 = Agent(client=one_call_client(), tools=[hang_tool], max_steps=2)
    task = asyncio.create_task(agent2.run("go"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
        print("  unexpected: task completed")
    except asyncio.CancelledError:
        print("  CancelledError propagated to the caller")

    section("notes")
    print("A deadline is an error result; cancellation is a control-flow signal.")
    print("Never convert CancelledError into AgentTimeoutError — the CLI relies")
    print("on the distinction for its exit code.")


if __name__ == "__main__":
    asyncio.run(main())
