"""Phase 19: tool permission tiers.

Eleven scenarios, all deterministic (fake tool, fake client, asyncio only):

  1. read tool runs without a callback
  2. mutate tool with an async callback returning True runs
  3. mutate tool with a sync callback returning True runs (sync supported)
  4. mutate tool with a callback returning False is denied
  5. mutate tool without a callback is denied
  6. dangerous tool + callback True + no allow_dangerous is denied
     (callback is NOT consulted - two-lock model)
  7. dangerous tool + callback True + allow_dangerous=True runs
  8. callback raising is recorded as an error ToolResult, not raised
  9. denied tool is never invoked (call_count stays at zero)
 10. invalid permission at construction raises ValueError
 11. mixed batch: read allowed, mutate denied, both results recorded
"""

from __future__ import annotations

import asyncio
import sys
from typing import cast

from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import ContentItem, Message, ToolCall, ToolResult
from pyagent.tools import BaseTool

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class CountedTool(BaseTool):
    """Fake tool that records invocations and returns a fixed value."""

    value: object
    call_count: int

    def __init__(
        self,
        *,
        name: str = "t",
        permission: str = "read",
        value: object = "done",
    ) -> None:
        super().__init__(name=name, description="fake", permission=permission)  # type: ignore[arg-type]
        self.value = value
        self.call_count = 0

    async def call(
        self,
        args: object | None = None,
        context: ExecutionContext | None = None,
    ) -> object:
        _ = (args, context)
        self.call_count += 1
        return self.value


class ScriptedClient:
    """Fake LLM client: pops one behaviour per generate call.

    Behaviours are a ``ToolCall`` (single call), a ``list[ToolCall]`` (parallel
    batch), or a plain ``str`` (final assistant answer).
    """

    def __init__(
        self,
        behaviours: list[ToolCall | list[ToolCall] | str],
    ) -> None:
        self.behaviours = list(behaviours)
        self.calls = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        _ = request
        self.calls += 1
        if not self.behaviours:
            raise AssertionError(f"unexpected extra generate call #{self.calls}")
        nxt = self.behaviours.pop(0)
        if isinstance(nxt, str):
            return LlmResponse(content=[Message(role="assistant", content=nxt)])
        if isinstance(nxt, ToolCall):
            return LlmResponse(content=[nxt])
        return LlmResponse(content=cast(list[ContentItem], list(nxt)))


def tool_call(name: str = "t", tool_call_id: str = "tc1") -> ToolCall:
    return ToolCall.model_validate({"tool_call_id": tool_call_id, "name": name, "arguments": {}})


def all_tool_results(ctx: ExecutionContext) -> list[ToolResult]:
    return [item for ev in ctx.events for item in ev.content if isinstance(item, ToolResult)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_1_read_without_callback() -> None:
    print("\nTest 1: read tool runs without a callback")
    tool = CountedTool(name="calc", permission="read", value=42)
    client = ScriptedClient([tool_call("calc"), "answer"])
    agent = Agent(client=client, tools=[tool], max_steps=4)
    ctx = ExecutionContext()
    result = await agent.run("go", ctx)
    check(result.output == "answer", f"final output (got {result.output!r})")
    check(tool.call_count == 1, f"tool ran once (got {tool.call_count})")
    results = all_tool_results(ctx)
    check(len(results) == 1, f"one ToolResult (got {len(results)})")
    if results:
        check(results[0].status == "success", f"status=success (got {results[0].status})")


async def test_2_mutate_async_callback_true() -> None:
    print("\nTest 2: mutate tool with async callback True runs")
    tool = CountedTool(name="write", permission="mutate", value="wrote")
    calls: list[ToolCall] = []

    async def approve(tc: ToolCall) -> bool:
        calls.append(tc)
        return True

    client = ScriptedClient([tool_call("write"), "done"])
    agent = Agent(client=client, tools=[tool], confirm_tool_call=approve, max_steps=4)
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(tool.call_count == 1, f"tool ran once (got {tool.call_count})")
    check(len(calls) == 1, f"callback consulted once (got {len(calls)})")
    if calls:
        check(calls[0].name == "write", f"callback saw correct tool (got {calls[0].name})")


async def test_3_mutate_sync_callback_true() -> None:
    print("\nTest 3: mutate tool with sync callback True runs")
    tool = CountedTool(name="write", permission="mutate", value="wrote")
    client = ScriptedClient([tool_call("write"), "done"])
    agent = Agent(
        client=client,
        tools=[tool],
        confirm_tool_call=lambda tc: True,
        max_steps=4,
    )
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(tool.call_count == 1, f"sync callback approved, tool ran (got {tool.call_count})")
    results = all_tool_results(ctx)
    check(len(results) == 1 and results[0].status == "success", "success result recorded")


async def test_4_mutate_callback_false() -> None:
    print("\nTest 4: mutate tool with callback False is denied")
    tool = CountedTool(name="write", permission="mutate")
    client = ScriptedClient([tool_call("write"), "gave up"])
    agent = Agent(
        client=client,
        tools=[tool],
        confirm_tool_call=lambda tc: False,
        max_steps=4,
    )
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(tool.call_count == 0, f"tool never invoked (got {tool.call_count})")
    results = all_tool_results(ctx)
    check(len(results) == 1, f"one ToolResult (got {len(results)})")
    if results:
        check(results[0].status == "error", f"status=error (got {results[0].status})")
        msg = str(results[0].content[0]).lower()
        check("denied" in msg, f"message mentions 'denied' (got {results[0].content[0]!r})")
        check("write" in msg, f"message names tool (got {results[0].content[0]!r})")


async def test_5_mutate_no_callback() -> None:
    print("\nTest 5: mutate tool without a callback is denied")
    tool = CountedTool(name="write", permission="mutate")
    client = ScriptedClient([tool_call("write"), "gave up"])
    agent = Agent(client=client, tools=[tool], max_steps=4)
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(tool.call_count == 0, f"tool never invoked (got {tool.call_count})")
    results = all_tool_results(ctx)
    if results:
        msg = str(results[0].content[0]).lower()
        check(
            "confirm_tool_call" in msg,
            f"message mentions the callback (got {results[0].content[0]!r})",
        )


async def test_6_dangerous_no_allow() -> None:
    print("\nTest 6: dangerous + callback True + no allow_dangerous is denied")
    tool = CountedTool(name="drop_db", permission="dangerous")
    calls: list[ToolCall] = []

    async def approve(tc: ToolCall) -> bool:
        calls.append(tc)
        return True

    client = ScriptedClient([tool_call("drop_db"), "gave up"])
    agent = Agent(
        client=client,
        tools=[tool],
        confirm_tool_call=approve,
        allow_dangerous=False,
        max_steps=4,
    )
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(tool.call_count == 0, f"tool never invoked (got {tool.call_count})")
    check(len(calls) == 0, f"callback NOT consulted (got {len(calls)})")
    results = all_tool_results(ctx)
    if results:
        msg = str(results[0].content[0]).lower()
        check("dangerous" in msg, f"message mentions 'dangerous' (got {results[0].content[0]!r})")
        check("allow_dangerous" in msg, f"message names the flag (got {results[0].content[0]!r})")


async def test_7_dangerous_with_allow() -> None:
    print("\nTest 7: dangerous + callback True + allow_dangerous=True runs")
    tool = CountedTool(name="drop_db", permission="dangerous", value="dropped")
    client = ScriptedClient([tool_call("drop_db"), "done"])
    agent = Agent(
        client=client,
        tools=[tool],
        confirm_tool_call=lambda tc: True,
        allow_dangerous=True,
        max_steps=4,
    )
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(tool.call_count == 1, f"tool ran (got {tool.call_count})")
    results = all_tool_results(ctx)
    check(len(results) == 1 and results[0].status == "success", "success result")


async def test_8_callback_raises() -> None:
    print("\nTest 8: callback raising becomes an error ToolResult, not raised")
    tool = CountedTool(name="write", permission="mutate")

    async def boom(tc: ToolCall) -> bool:
        raise RuntimeError("prompt cancelled")

    client = ScriptedClient([tool_call("write"), "gave up"])
    agent = Agent(
        client=client,
        tools=[tool],
        confirm_tool_call=boom,
        max_steps=4,
    )
    ctx = ExecutionContext()
    raised = False
    try:
        await agent.run("go", ctx)
    except Exception:
        raised = True
    check(not raised, "run did not raise")
    check(tool.call_count == 0, f"tool never invoked (got {tool.call_count})")
    results = all_tool_results(ctx)
    if results:
        msg = str(results[0].content[0])
        check(
            "permission check" in msg.lower() or "permission" in msg.lower(),
            f"message mentions 'permission' (got {msg!r})",
        )
        check("prompt cancelled" in msg, f"message carries the exception (got {msg!r})")


async def test_9_denied_never_runs() -> None:
    print("\nTest 9: denied tool never runs (call_count == 0)")
    tool = CountedTool(name="write", permission="mutate")
    client = ScriptedClient([tool_call("write"), "ok"])
    agent = Agent(
        client=client,
        tools=[tool],
        confirm_tool_call=lambda tc: False,
        max_steps=4,
    )
    await agent.run("go", ExecutionContext())
    check(tool.call_count == 0, f"tool.call_count == 0 (got {tool.call_count})")


async def test_10_invalid_permission() -> None:
    print("\nTest 10: invalid permission at construction raises ValueError")
    raised: ValueError | None = None
    try:
        CountedTool(name="bad", permission="admin")
    except ValueError as exc:
        raised = exc
    check(raised is not None, "ValueError raised")
    if raised is not None:
        check("permission" in str(raised).lower(), f"message mentions permission (got {raised})")


async def test_11_mixed_batch() -> None:
    print("\nTest 11: mixed batch - read allowed, mutate denied in same step")
    read_tool = CountedTool(name="reader", permission="read", value="read-result")
    write_tool = CountedTool(name="writer", permission="mutate")
    batch = [tool_call("reader", "tc-read"), tool_call("writer", "tc-write")]
    client = ScriptedClient([batch, "done"])
    agent = Agent(
        client=client,
        tools=[read_tool, write_tool],
        confirm_tool_call=lambda tc: False,
        max_steps=4,
    )
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    check(read_tool.call_count == 1, f"read tool ran (got {read_tool.call_count})")
    check(write_tool.call_count == 0, f"write tool denied (got {write_tool.call_count})")
    results = all_tool_results(ctx)
    check(len(results) == 2, f"two ToolResults (got {len(results)})")
    by_name = {r.name: r for r in results}
    check("reader" in by_name, "reader result present")
    check("writer" in by_name, "writer result present")
    if "reader" in by_name:
        check(by_name["reader"].status == "success", "reader succeeded")
    if "writer" in by_name:
        check(by_name["writer"].status == "error", "writer denied")


async def main() -> None:
    await test_1_read_without_callback()
    await test_2_mutate_async_callback_true()
    await test_3_mutate_sync_callback_true()
    await test_4_mutate_callback_false()
    await test_5_mutate_no_callback()
    await test_6_dangerous_no_allow()
    await test_7_dangerous_with_allow()
    await test_8_callback_raises()
    await test_9_denied_never_runs()
    await test_10_invalid_permission()
    await test_11_mixed_batch()


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 19 (permissions) verified.")


if __name__ == "__main__":
    run()
