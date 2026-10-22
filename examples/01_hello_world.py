"""01 — Hello world.

The smallest possible agent: one user message, one LLM call, one text
response. No tools, no structured output, no persistence.

    pixi run python examples/01_hello_world.py
"""

from __future__ import annotations

import asyncio

from examples._common import build_llm_client, print_final, print_trace, section
from pyagent.agent import Agent


async def main() -> None:
    agent = Agent(client=build_llm_client())

    section("run")
    result = await agent.run('Say "hello" and nothing else.')
    print_trace(result)

    section("final answer")
    print_final(result)


if __name__ == "__main__":
    asyncio.run(main())
