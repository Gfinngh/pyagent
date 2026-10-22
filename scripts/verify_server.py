"""Phase 7: verify MCP server - spawn over stdio, list tools, call them.

MCP Python SDK v2 renamed protocol fields from camelCase to snake_case
(inputSchema -> input_schema, isError -> is_error). Wire format is unchanged.

Tool errors: the SDK masks plain exceptions. To surface the actual message
to the client, the server raises ``ToolError``.

Tools registered: calculator, greet, search_web, write_file, list_files,
read_file. Presence is asserted for all six; call paths for the network and
filesystem tools are covered in dedicated verify scripts.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def text_of(result: Any) -> str:
    raw_content = getattr(result, "content", None)
    if not isinstance(raw_content, list):
        return ""
    blocks: list[Any] = cast(list[Any], raw_content)
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pyagent.server"],
    )

    print("\nTest 1: spawn server, initialize session")
    async with (
        stdio_client(params, errlog=sys.stderr) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        check(True, "session initialized")

        print("\nTest 2: list_tools returns all six tools")
        tools_result = await session.list_tools()
        tool_names = sorted(t.name for t in tools_result.tools)
        for expected in (
            "calculator",
            "greet",
            "search_web",
            "write_file",
            "list_files",
            "read_file",
        ):
            check(expected in tool_names, f"{expected} present (got {tool_names})")

        print("\nTest 3: calculator schema shape")
        calc_matches = [t for t in tools_result.tools if t.name == "calculator"]
        if not calc_matches:
            check(False, "calculator tool not found")
        else:
            calc = calc_matches[0]
            schema = calc.input_schema
            check(
                schema.get("type") == "object",
                f"input_schema.type=object (got {schema.get('type')!r})",
            )
            props = schema.get("properties", {})
            check(isinstance(props, dict), "properties is a dict")
            if isinstance(props, dict):
                props_typed = cast(dict[str, object], props)
                check("operator" in props_typed, "properties has operator")
                check("first_number" in props_typed, "properties has first_number")
                check("second_number" in props_typed, "properties has second_number")

        print("\nTest 4: call greet")
        greet_result = await session.call_tool("greet", {"name": "World"})
        check(not greet_result.is_error, "greet did not error")
        greet_text = text_of(greet_result)
        check(
            "Hello, World." in greet_text,
            f"greet -> 'Hello, World.' (got {greet_text!r})",
        )

        print("\nTest 5: call calculator (add)")
        add_result = await session.call_tool(
            "calculator",
            {"operator": "add", "first_number": 2, "second_number": 2},
        )
        check(not add_result.is_error, "add did not error")
        add_text = text_of(add_result)
        check("4" in add_text, f"2+2 result contains '4' (got {add_text!r})")

        print("\nTest 6: call calculator (multiply, larger)")
        mul_result = await session.call_tool(
            "calculator",
            {"operator": "multiply", "first_number": 12345, "second_number": 67890},
        )
        check(not mul_result.is_error, "multiply did not error")
        mul_text = text_of(mul_result)
        check(
            "838102050" in mul_text.replace(",", ""),
            f"product is 838102050 (got {mul_text!r})",
        )

        print("\nTest 7: divide by zero surfaces as a tool error with message")
        try:
            div_result = await session.call_tool(
                "calculator",
                {"operator": "divide", "first_number": 1, "second_number": 0},
            )
        except Exception as e:
            check(
                "divide" in str(e).lower() or "zero" in str(e).lower(),
                f"raised tool error mentions divide/zero (got {e!r})",
            )
        else:
            check(bool(div_result.is_error), "divide-by-zero is_error=True")
            div_text = text_of(div_result)
            check(
                "divide" in div_text.lower() or "zero" in div_text.lower(),
                f"error text mentions divide/zero (got {div_text!r})",
            )


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 7 (server) verified.")


if __name__ == "__main__":
    run()
