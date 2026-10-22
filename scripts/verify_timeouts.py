"""Phase 18: tool timeouts + agent timeout.

Six scenarios, all deterministic (fake tool, fake client, asyncio only):

  1. Slow tool within its limit -> succeeds, no deadline hit
  2. Slow tool over its limit -> error ToolResult naming the tool + limit;
     the run continues and the agent still produces a final answer
  3. Agent deadline shorter than the tool delay -> AgentTimeoutError with
     partial-context metadata
  4. No timeouts configured -> unchanged behaviour (backward compat)
  5. Outer task cancellation -> CancelledError propagates, is not swallowed
     by ``act`` and not converted to ``AgentTimeoutError``
  6. Timeout message shape -> tool name and millisecond limit both present
"""

from __future__ import annotations

import asyncio
import sys

from pyagent.agent import Agent, AgentTimeoutError
from pyagent.context import ExecutionContext
from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import Message, ToolCall, ToolResult
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


class SlowTool(BaseTool):
    """Fake tool that sleeps for ``delay_s`` then returns a fixed value."""

    delay_s: float
    return_value: object

    def __init__(
        self,
        name: str = "slow",
        *,
        delay_s: float,
        timeout_ms: int = 0,
        return_value: object = "done",
    ) -> None:
        super().__init__(name=name, description="slow fake tool", timeout_ms=timeout_ms)
        self.delay_s = delay_s
        self.return_value = return_value

    async def call(
        self,
        args: object | None = None,
        context: ExecutionContext | None = None,
    ) -> object:
        _ = (args, context)
        await asyncio.sleep(self.delay_s)
        return self.return_value


class ScriptedClient:
    """Fake LLM client: pops one behaviour per call.

    A ``ToolCall`` item produces a LlmResponse with that call; a plain string
    produces an assistant message (which terminates the loop in free-form
    mode).
    """

    def __init__(self, behaviours: list[ToolCall | str]) -> None:
        self.behaviours = list(behaviours)
        self.calls = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        _ = request
        self.calls += 1
        if not self.behaviours:
            raise AssertionError(f"unexpected extra generate call #{self.calls}")
        nxt = self.behaviours.pop(0)
        if isinstance(nxt, ToolCall):
            return LlmResponse(content=[nxt])
        return LlmResponse(content=[Message(role="assistant", content=nxt)])


def tool_call(name: str = "slow", tool_call_id: str = "tc1") -> ToolCall:
    return ToolCall.model_validate({"tool_call_id": tool_call_id, "name": name, "arguments": {}})


def error_result_from(context: ExecutionContext, name: str = "slow") -> ToolResult | None:
    for event in context.events:
        for item in event.content:
            if isinstance(item, ToolResult) and item.name == name and item.status == "error":
                return item
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_1_slow_tool_within_limit() -> None:
    print("\nTest 1: slow tool within its limit succeeds")
    tool = SlowTool(delay_s=0.05, timeout_ms=500)
    client = ScriptedClient([tool_call(), "all done"])
    agent = Agent(client=client, tools=[tool], max_steps=4)
    ctx = ExecutionContext()
    result = await agent.run("go", ctx)
    check(result.output == "all done", f"final output (got {result.output!r})")
    results = [item for ev in ctx.events for item in ev.content if isinstance(item, ToolResult)]
    check(len(results) == 1, f"one ToolResult (got {len(results)})")
    if results:
        check(results[0].status == "success", f"status=success (got {results[0].status})")
        check(results[0].content == ["done"], f"content=['done'] (got {results[0].content})")


async def test_2_slow_tool_over_limit() -> None:
    print("\nTest 2: slow tool over its limit produces a structured error")
    tool = SlowTool(delay_s=0.50, timeout_ms=50)
    client = ScriptedClient([tool_call(), "recovered"])
    agent = Agent(client=client, tools=[tool], max_steps=4)
    ctx = ExecutionContext()
    result = await agent.run("go", ctx)
    check(
        result.output == "recovered",
        f"run continued to a final answer (got {result.output!r})",
    )
    err = error_result_from(ctx)
    check(err is not None, "an error ToolResult was recorded")
    if err is not None:
        msg = str(err.content[0]).lower()
        check("timed out" in msg, f"message mentions 'timed out' (got {err.content[0]!r})")
        check("slow" in msg, f"message names the tool (got {err.content[0]!r})")
        check("50" in str(err.content[0]), f"message names the limit (got {err.content[0]!r})")


async def test_3_agent_deadline() -> None:
    print("\nTest 3: agent deadline shorter than tool delay raises AgentTimeoutError")
    tool = SlowTool(delay_s=0.50, timeout_ms=0)  # tool has no own limit
    client = ScriptedClient([tool_call()])  # never reaches the second turn
    agent = Agent(client=client, tools=[tool], max_steps=4, run_timeout_ms=50)
    ctx = ExecutionContext()
    caught: AgentTimeoutError | None = None
    try:
        await agent.run("go", ctx)
    except AgentTimeoutError as e:
        caught = e
    check(caught is not None, "AgentTimeoutError raised")
    if caught is not None:
        meta = caught.metadata
        check(
            meta.get("timeout_ms") == 50, f"metadata.timeout_ms=50 (got {meta.get('timeout_ms')})"
        )
        check(
            meta.get("events_count") is not None,
            f"metadata.events_count present (got {meta.get('events_count')})",
        )
        check(
            meta.get("current_step") == 0,
            f"timeout during step 1: current_step=0 (got {meta.get('current_step')})",
        )


async def test_4_no_timeouts() -> None:
    print("\nTest 4: no timeouts configured -> unchanged behaviour")
    tool = SlowTool(delay_s=0.01, timeout_ms=0)
    client = ScriptedClient([tool_call(), "ok"])
    agent = Agent(client=client, tools=[tool], max_steps=4, run_timeout_ms=0)
    ctx = ExecutionContext()
    result = await agent.run("go", ctx)
    check(result.output == "ok", f"output='ok' (got {result.output!r})")
    check(tool.timeout_ms == 0, f"tool.timeout_ms=0 (got {tool.timeout_ms})")


async def test_5_outer_cancellation() -> None:
    print("\nTest 5: outer cancellation propagates as CancelledError")
    tool = SlowTool(delay_s=1.0, timeout_ms=0)  # runs forever within test window
    client = ScriptedClient([tool_call()])
    agent = Agent(client=client, tools=[tool], max_steps=4, run_timeout_ms=0)

    task = asyncio.create_task(agent.run("go"))
    await asyncio.sleep(0.05)  # let the tool start
    task.cancel()
    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True
    except AgentTimeoutError:
        check(False, "outer cancel was misreported as AgentTimeoutError")
        return
    check(cancelled, "task was cancelled (CancelledError propagated)")
    check(task.cancelled(), "task.cancelled() reports True")


async def test_6_message_shape() -> None:
    print("\nTest 6: timeout message names the tool and the limit")
    tool = SlowTool(delay_s=0.50, timeout_ms=75, name="query_db")
    client = ScriptedClient([tool_call(name="query_db"), "done"])
    agent = Agent(client=client, tools=[tool], max_steps=4)
    ctx = ExecutionContext()
    await agent.run("go", ctx)
    err = error_result_from(ctx, name="query_db")
    check(err is not None, "error result recorded for query_db")
    if err is not None:
        text = str(err.content[0])
        check("query_db" in text, f"tool name in message (got {text!r})")
        check("75" in text, f"limit in message (got {text!r})")
        check("timed out" in text.lower(), f"'timed out' in message (got {text!r})")


async def main() -> None:
    await test_1_slow_tool_within_limit()
    await test_2_slow_tool_over_limit()
    await test_3_agent_deadline()
    await test_4_no_timeouts()
    await test_5_outer_cancellation()
    await test_6_message_shape()


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 18 (tool + agent timeouts) verified.")


if __name__ == "__main__":
    run()
