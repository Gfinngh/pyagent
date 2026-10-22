"""11 — Per-tool timeouts.

A tool may carry `timeout_ms`. `Agent.act` races the call against the limit; on
expiry the tool is cancelled and the model receives an error `ToolResult`. The
run continues — the model can react and move on.

    pixi run python examples/11_tool_timeouts.py
"""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from examples._common import collect_tool_results, print_final, print_trace, section
from examples._fakes import ScriptedClient, make_tool_call, text_response, tool_call_response
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool


class Empty(BaseModel):
    """No arguments."""


async def slow(args: Empty, context: ExecutionContext | None = None) -> str:
    _ = args, context
    await asyncio.sleep(0.5)
    return "slow finished"


slow_tool = FunctionTool(
    name="slow",
    description="Sleeps 500 ms; times out at 50 ms.",
    schema=Empty,
    func=slow,
    timeout_ms=50,
)


async def main() -> None:
    agent = Agent(
        client=ScriptedClient(
            [
                tool_call_response([make_tool_call("c1", "slow")]),
                text_response("the tool timed out, moving on"),
            ]
        ),
        tools=[slow_tool],
        max_steps=3,
    )

    section("run")
    started = time.monotonic()
    result = await agent.run("go")
    elapsed = time.monotonic() - started

    print_trace(result)

    section("outcome")
    for r in collect_tool_results(result):
        print(f"  status:  {r.status}")
        print(f"  content: {r.content}")

    print(f"  wall time: {elapsed * 1000:.0f} ms (tool limit was 50 ms)")

    section("final answer")
    print_final(result)

    section("notes")
    print("A run-level deadline (run_timeout_ms) raises AgentTimeoutError instead")
    print("of producing a tool result. See examples/23_cancellation.py.")


if __name__ == "__main__":
    asyncio.run(main())
