"""03 — Multiple tools.

A mix of shipped tools and a custom one. The model decides which to call and in
what order. Shows:

  - the shipped `add`, `calculator`, and `greet` tools
  - a custom `get_time` tool built from `FunctionTool` + a pydantic model
  - the agent chaining tool calls in one run

    pixi run python examples/03_multi_tool.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import BaseModel

from examples._common import build_llm_client, print_final, print_trace, section
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool
from pyagent.tools.add import add_tool
from pyagent.tools.calculator import calculator_tool
from pyagent.tools.greet import greet_tool


class TimeInput(BaseModel):
    """No arguments; the model still needs a schema."""


def get_time(
    args: TimeInput,
    context: ExecutionContext | None = None,
) -> str:
    """Return the current UTC time."""
    _ = args, context
    return datetime.now(UTC).isoformat()


time_tool = FunctionTool(
    name="get_time",
    description="Return the current UTC time as an ISO 8601 string.",
    schema=TimeInput,
    func=get_time,
)


async def main() -> None:
    agent = Agent(
        client=build_llm_client(),
        tools=[add_tool, calculator_tool, greet_tool, time_tool],
        max_steps=6,
    )

    prompt = "Greet Alice, tell me what time it is in UTC, and compute 10 * 4."
    section("run")
    result = await agent.run(prompt)
    print_trace(result)

    section("final answer")
    print_final(result)


if __name__ == "__main__":
    asyncio.run(main())
