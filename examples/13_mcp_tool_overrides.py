"""13 — MCP tool overrides.

Discovered MCP tools default to `permission="read"` and `timeout_ms=0`: the
server does not advertise a permission, so the client decides. `load_mcp_tools`
accepts `ToolOverrides` keyed by tool name — that is the Python mechanism that
keeps `write_file` from being treated as a read tool.

`main.py`'s `TOOL_OVERRIDES` is the canonical configuration:

    search_web      read,      15_000 ms
    write_file      mutate,    10_000 ms
    execute_python  dangerous, 15_000 ms

    pixi run python examples/13_mcp_tool_overrides.py
"""

from __future__ import annotations

import asyncio
import sys

from examples._common import section
from pyagent.client import (
    McpConnection,
    McpServerParams,
    ToolOverrides,
    load_mcp_tools,
)


async def dump(label: str, overrides: dict[str, ToolOverrides] | None) -> None:
    section(label)
    params = McpServerParams(command=sys.executable, args=["-m", "pyagent.server"])
    async with McpConnection(params) as conn:
        tools = await load_mcp_tools(conn, overrides=overrides)
        for tool in tools:
            print(f"  {tool.name:<16} permission={tool.permission:<9} timeout_ms={tool.timeout_ms}")


async def main() -> None:
    await dump("no overrides — every tool defaults to read", None)

    await dump(
        "overrides applied by tool name",
        {
            "search_web": ToolOverrides(permission="read", timeout_ms=15_000),
            "write_file": ToolOverrides(permission="mutate", timeout_ms=10_000),
            "execute_python": ToolOverrides(permission="dangerous", timeout_ms=15_000),
        },
    )

    section("why this matters")
    print("Without an override, write_file would be a `read` tool: no confirmation")
    print("callback, no --approve gate, no audit trail. The override is the lock.")


if __name__ == "__main__":
    asyncio.run(main())
