"""02 — Tool calling.

One tool. Watch the model invoke `calculator` and use the result. Demonstrates
the full round trip: model emits a call → agent runs the tool → result goes back
to the model → model produces a final answer.

    pixi run python examples/02_tool_calling.py
"""

from __future__ import annotations

import asyncio

from examples._common import build_llm_client, print_final, print_trace, section
from pyagent.agent import Agent
from pyagent.tools.calculator import calculator_tool


async def main() -> None:
    agent = Agent(
        client=build_llm_client(),
        tools=[calculator_tool],
        max_steps=4,
    )

    section("run")
    result = await agent.run("Use the calculator tool to compute 2 + 3, then tell me the answer.")
    print_trace(result)

    section("final answer")
    print_final(result)


if __name__ == "__main__":
    asyncio.run(main())
