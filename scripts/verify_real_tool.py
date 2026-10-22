"""Verify overrides flow to real MCP tools, and search_web behaves on network.

Four scenarios:

  1. ``load_mcp_tools(conn)`` without overrides - every RemoteTool is
     ``permission="read"``, ``timeout_ms=0`` (unchanged defaults)
  2. ``load_mcp_tools(conn, overrides)`` - search_web gets the configured
     ``permission`` and ``timeout_ms``; other tools keep defaults
  3. Missing ``TAVILY_API_KEY`` produces a clean error ``ToolResult``
  4. Real search works when ``TAVILY_API_KEY`` is set

Tests 1-2 are deterministic. Test 3 is deterministic when the key is unset.
Test 4 requires network egress and a live Tavily key.
"""

from __future__ import annotations

import asyncio
import os
import sys

from pyagent.agent import Agent
from pyagent.client import (
    McpConnection,
    McpServerParams,
    ToolOverrides,
    load_mcp_tools,
)
from pyagent.context import ExecutionContext
from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import ToolCall
from pyagent.tools import MultipartContent, RemoteTool, TextPart

_FAILURES: list[str] = []
_SKIPPED: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def skip(msg: str) -> None:
    _SKIPPED.append(msg)
    print(f"  \u26a0 SKIP: {msg}")


def server_params() -> McpServerParams:
    return McpServerParams(command=sys.executable, args=["-m", "pyagent.server"])


def find_tool(tools: list[RemoteTool], name: str) -> RemoteTool | None:
    for t in tools:
        if t.name == name:
            return t
    return None


class _NullClient:
    """Fake LLM client. Its ``generate`` is never expected to be called here."""

    async def generate(self, request: LlmRequest) -> LlmResponse:
        _ = request
        raise RuntimeError("_NullClient.generate should not be called in this test")


async def _main() -> None:
    print("\nSetup: connect and discover tools")
    async with McpConnection(server_params()) as conn:
        discovered = await conn.list_tools()
        names = sorted(t.name for t in discovered)
        print(f"  \u2139 discovered: {names}")
        check("calculator" in names, "calculator present")
        check("greet" in names, "greet present")
        check("search_web" in names, f"search_web present (got {names})")

        # --------------------------------------------------------------
        print("\nTest 1: default overrides - read permission, no timeout")
        defaults = await load_mcp_tools(conn)
        for tool in defaults:
            check(
                tool.permission == "read",
                f"{tool.name}: permission='read' (got {tool.permission})",
            )
            check(
                tool.timeout_ms == 0,
                f"{tool.name}: timeout_ms=0 (got {tool.timeout_ms})",
            )

        # --------------------------------------------------------------
        print("\nTest 2: explicit overrides reach the target tool only")
        overrides = {
            "search_web": ToolOverrides(permission="read", timeout_ms=15_000),
        }
        configured = await load_mcp_tools(conn, overrides=overrides)
        sw = find_tool(configured, "search_web")
        calc = find_tool(configured, "calculator")
        check(sw is not None, "search_web wrapped")
        check(calc is not None, "calculator wrapped")
        if sw is None or calc is None:
            return
        check(
            sw.timeout_ms == 15_000,
            f"search_web.timeout_ms=15000 (got {sw.timeout_ms})",
        )
        check(
            sw.permission == "read",
            f"search_web.permission='read' (got {sw.permission})",
        )
        check(
            calc.timeout_ms == 0,
            f"calculator.timeout_ms=0 (untouched) (got {calc.timeout_ms})",
        )
        check(
            calc.permission == "read",
            f"calculator.permission='read' (untouched) (got {calc.permission})",
        )

        # --------------------------------------------------------------
        print("\nTest 3: missing TAVILY_API_KEY -> error ToolResult, no crash")
        key_present = bool(os.environ.get("TAVILY_API_KEY"))
        if key_present:
            skip("TAVILY_API_KEY is set; missing-key path not exercised")
        else:
            agent = Agent(client=_NullClient(), tools=[sw])
            tc = ToolCall.model_validate(
                {
                    "tool_call_id": "tc-search",
                    "name": "search_web",
                    "arguments": {"query": "python", "max_results": 1},
                }
            )
            results = await agent.act(ExecutionContext(), [tc])
            check(len(results) == 1, f"one ToolResult (got {len(results)})")
            if results:
                r = results[0]
                check(r.status == "error", f"status=error (got {r.status})")
                msg = str(r.content[0]).lower()
                check(
                    "tavily_api_key" in msg,
                    f"message names the env var (got {r.content[0]!r})",
                )
                check(
                    "not set" in msg,
                    f"message says 'not set' (got {r.content[0]!r})",
                )

        # --------------------------------------------------------------
        print("\nTest 4: real search returns text (requires TAVILY_API_KEY)")
        if not key_present:
            skip("TAVILY_API_KEY not set; skipping live search")
        else:
            try:
                output = await sw.call({"query": "what is the capital of France", "max_results": 1})
            except Exception as exc:
                check(False, f"search_web.call raised: {exc!r}")
            else:
                check(
                    isinstance(output, MultipartContent),
                    f"returns MultipartContent (got {type(output).__name__})",
                )
                if isinstance(output, MultipartContent):
                    text = "".join(p.text for p in output.parts if isinstance(p, TextPart))
                    check(len(text) > 0, f"text non-empty (got {len(text)} chars)")
                    check(
                        "paris" in text.lower() or "france" in text.lower(),
                        f"result mentions the query subject (got {text[:120]!r}…)",
                    )


async def main() -> None:
    await _main()


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        if _SKIPPED:
            print(f"\n\u26a0 {len(_SKIPPED)} skipped:")
            for s in _SKIPPED:
                print(f"  - {s}")
        sys.exit(1)
    if _SKIPPED:
        print(f"\n\u2705 Real-tool verification passed with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Real-tool verification passed.")


if __name__ == "__main__":
    run()
