"""Phase 6: verify ReAct agent loop (free-form output, no structured)."""

from __future__ import annotations

import asyncio
import os
import sys

from pyagent.agent import Agent, LlmCallError
from pyagent.context import ExecutionContext
from pyagent.llm import LlmClient
from pyagent.models import resolve_model
from pyagent.schemas import Message, ToolCall, ToolResult
from pyagent.tools.calculator import calculator_tool
from scripts._verify_common import is_transient_llm_response

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


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    print("\nTest 1: simple ReAct loop - 2 + 2 with calculator")
    client = LlmClient(model=resolve_model("gemini/gemini-3.1-flash-lite"))
    agent = Agent(
        client=client,
        tools=[calculator_tool],
        instructions="You are a helpful assistant. Use tools when appropriate.",
        max_steps=4,
    )

    try:
        result = await agent.run("What is 2 + 2? Use the calculator.")
    except LlmCallError as e:
        if e.metadata.get("quota_exhausted") is True:
            skip(f"quota exhausted: {e}")
            return
        if is_transient_llm_response(str(e), e.metadata):
            skip(f"transient LLM failure: {e}")
            return
        check(False, f"LLM failed: {e}")
        print(f"    metadata: {e.metadata}")
        return

    check(result.output is not None, f"output is not None (got {result.output!r})")
    check(
        isinstance(result.output, str),
        f"output is str (got {type(result.output).__name__})",
    )
    if isinstance(result.output, str):
        check("4" in result.output, f"output mentions 4 (got {result.output!r})")

    print("\nTest 2: trace shape")
    events = result.context.events
    check(len(events) >= 3, f"at least 3 events (got {len(events)})")
    check(events[0].author == "user", "event[0] is user")
    check(
        result.context.current_step >= 1,
        f"current_step >= 1 (got {result.context.current_step})",
    )

    tool_call_count = sum(1 for ev in events for item in ev.content if isinstance(item, ToolCall))
    tool_result_count = sum(
        1 for ev in events for item in ev.content if isinstance(item, ToolResult)
    )
    check(tool_call_count >= 1, f"at least 1 tool call (got {tool_call_count})")
    check(tool_result_count >= 1, f"at least 1 tool result (got {tool_result_count})")

    print("\nTest 3: final event has no tool calls/results")
    last_event = events[-1]
    last_has_calls = any(isinstance(c, ToolCall) for c in last_event.content)
    last_has_results = any(isinstance(c, ToolResult) for c in last_event.content)
    check(not last_has_calls, "final event has no ToolCall")
    check(not last_has_results, "final event has no ToolResult")
    check(
        any(isinstance(m, Message) and m.role == "assistant" for m in last_event.content),
        "final event has assistant Message",
    )

    print("\nTest 4: unknown tool produces error ToolResult")
    ctx = ExecutionContext()
    bad_call = ToolCall.model_validate(
        {
            "tool_call_id": "x1",
            "name": "nonexistent",
            "arguments": {},
        }
    )
    results = await agent.act(ctx, [bad_call])
    check(len(results) == 1, "one result returned")
    check(results[0].status == "error", f"status=error (got {results[0].status})")
    check(
        "not found" in str(results[0].content[0]).lower(),
        "message mentions 'not found'",
    )

    print("\n\u2705 Phase 6 (agent) verified.")
    print(f"  output: {result.output!r}")
    print(f"  events: {len(events)}, steps: {result.context.current_step}")


def _finalize(*, completed: bool) -> None:
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        if _SKIPPED:
            print(f"\n\u26a0 {len(_SKIPPED)} skipped:")
            for s in _SKIPPED:
                print(f"  - {s}")
        sys.exit(1)
    if not completed:
        print("\n❌ verify script crashed before completing")
        return

    if _SKIPPED:
        print(f"\n\u2705 Phase 6 (agent) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    # Success path already printed "\u2705 Phase 6 (agent) verified." in main().


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
