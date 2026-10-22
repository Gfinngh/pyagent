"""12 — Concurrency in the tool batch.

When every approved call in a batch is concurrency-safe (the default for
`read`-tier tools), the agent runs them in parallel with `asyncio.gather`. One
non-safe tool forces the whole batch to run sequentially. Result order always
matches emission order.

This example times both shapes so the difference is visible.

    pixi run python examples/12_parallel_tool_calls.py
"""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from examples._common import section
from examples._fakes import ScriptedClient, make_tool_call, text_response, tool_call_response
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool


class Empty(BaseModel):
    """No arguments."""


def make_sleeper(name: str, permission: str) -> FunctionTool:
    async def impl(args: Empty, context: ExecutionContext | None = None) -> str:
        _ = args, context
        await asyncio.sleep(0.2)
        return f"{name} done"

    return FunctionTool(
        name=name,
        description=f"Sleeps 200 ms ({permission})",
        schema=Empty,
        func=impl,
        permission=permission,  # type: ignore[arg-type]
    )


async def time_run(label: str, permission: str) -> float:
    tools = [make_sleeper(n, permission) for n in ("a", "b", "c")]
    client = ScriptedClient(
        [
            tool_call_response(
                [
                    make_tool_call("c1", "a"),
                    make_tool_call("c2", "b"),
                    make_tool_call("c3", "c"),
                ]
            ),
            text_response("done"),
        ]
    )
    agent = Agent(
        client=client,
        tools=tools,
        confirm_tool_call=lambda tc: True,
        max_steps=3,
    )
    started = time.monotonic()
    await agent.run("go")
    elapsed = time.monotonic() - started

    section(label)
    print(f"  wall time: {elapsed * 1000:.0f} ms")
    return elapsed


async def main() -> None:
    print("Three tools, each sleeping 200 ms.")

    parallel = await time_run("all read → concurrency-safe → parallel", "read")
    sequential = await time_run("all mutate → not safe → sequential", "mutate")

    section("interpretation")
    print(f"  parallel:   ~{parallel * 1000:.0f} ms (expect ≈ 200 ms)")
    print(f"  sequential: ~{sequential * 1000:.0f} ms (expect ≈ 600 ms)")
    print("\nOpt a mutate tool into parallel with concurrency_safe=True — only when")
    print("it is genuinely safe to run concurrently with itself.")


if __name__ == "__main__":
    asyncio.run(main())
