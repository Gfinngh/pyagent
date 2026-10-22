"""07 — MCP tools.

Load the MCP server's tool catalogue and call tools directly, with no LLM in the
loop. Every tool can also be handed to `Agent(tools=...)` — that is what the
other examples do.

    pixi run python examples/07_mcp_tools.py
"""

from __future__ import annotations

import asyncio

from examples._common import extract_text, mcp_with_workspace, section
from pyagent.client import ToolOverrides

WORKSPACE = ".examples/workspace"


async def main() -> None:
    overrides = {
        "write_file": ToolOverrides(permission="mutate", timeout_ms=10_000),
        "execute_python": ToolOverrides(permission="dangerous", timeout_ms=15_000),
    }

    async with mcp_with_workspace(WORKSPACE, overrides=overrides) as tools:
        by_name = {t.name: t for t in tools}

        section(f"{len(tools)} tools available")
        for tool in tools:
            first_line = (tool.description or "").splitlines()[0]
            print(f"  {tool.name:<16} [{tool.permission:<9}] {first_line}")

        section('calculator(operator="add", first_number=2, second_number=3)')
        calc = by_name["calculator"]
        raw = await calc.call({"operator": "add", "first_number": 2, "second_number": 3})
        print(f"  → {extract_text(raw)}")

        section('greet(name="Alice")')
        raw = await by_name["greet"].call({"name": "Alice"})
        print(f"  → {extract_text(raw)}")

        section("notes")
        print("Pass these tools to Agent(tools=...) to let the model use them.")


if __name__ == "__main__":
    asyncio.run(main())
