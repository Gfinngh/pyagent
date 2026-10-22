"""Verify parallel tool execution.

Eight scenarios, all deterministic (fake tools, fake client, asyncio only):

  1. Three parallel reads run concurrently (wall clock about 1x delay, not 3x)
  2. Result order matches emission order despite parallel execution
  3. Mixed batch (read + mutate) forces ALL calls sequential
  4. All-mutations batch runs sequentially (wall clock about 3x delay)
  5. Exception isolation: one raising tool does not cancel siblings
  6. Timeout isolation: one slow tool's timeout does not affect siblings
  7. Outer cancellation propagates as CancelledError
  8. Permission denial in a mixed batch preserves order (read ok, mutate
     denied, read ok)
"""

from __future__ import annotations

import asyncio
import sys
import time

from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import ToolCall
from pyagent.tools import BaseTool, Permission

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


class FakeTool(BaseTool):
    """Tool that sleeps, optionally raises, and records timing."""

    delay_s: float
    value: object
    raise_exc: BaseException | None
    call_times: list[tuple[float, float]]

    def __init__(
        self,
        name: str = "t",
        *,
        delay_s: float = 0.0,
        permission: Permission = "read",
        value: object = "ok",
        timeout_ms: int = 0,
        concurrency_safe: bool | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description="fake",
            permission=permission,
            concurrency_safe=concurrency_safe,
            timeout_ms=timeout_ms,
        )
        self.delay_s = delay_s
        self.value = value
        self.raise_exc = raise_exc
        self.call_times = []

    async def call(
        self,
        args: object | None = None,
        context: ExecutionContext | None = None,
    ) -> object:
        _ = (args, context)
        start = time.monotonic()
        try:
            if self.delay_s > 0:
                await asyncio.sleep(self.delay_s)
            if self.raise_exc is not None:
                raise self.raise_exc
            return self.value
        finally:
            self.call_times.append((start, time.monotonic()))


class _NullClient:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        _ = request
        raise RuntimeError("_NullClient.generate should not be called")


def tool_call(name: str, tc_id: str, args: dict[str, object] | None = None) -> ToolCall:
    return ToolCall.model_validate(
        {
            "tool_call_id": tc_id,
            "name": name,
            "arguments": args or {},
        }
    )


def _max_overlap(times: list[tuple[float, float]]) -> int:
    """Maximum number of intervals overlapping at any instant."""
    events: list[tuple[float, int]] = []
    for start, end in times:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    current = 0
    best = 0
    for _, delta in events:
        current += delta
        if current > best:
            best = current
    return best


async def test_1_parallel_reads_concurrent() -> None:
    print("\nTest 1: three parallel reads run concurrently")
    tool = FakeTool("r", delay_s=0.2)
    agent = Agent(client=_NullClient(), tools=[tool])
    calls = [tool_call("r", f"tc{i}", {"i": i}) for i in range(3)]

    start = time.monotonic()
    results = await agent.act(ExecutionContext(), calls)
    elapsed = time.monotonic() - start

    print(f"  \u2139 elapsed: {elapsed:.2f}s (sequential would be ~0.6s)")
    check(len(results) == 3, f"three results (got {len(results)})")
    check(elapsed < 0.4, f"elapsed < 0.4s (got {elapsed:.2f}s)")
    check(
        _max_overlap(tool.call_times) == 3,
        f"all 3 calls overlapped (got {_max_overlap(tool.call_times)})",
    )


async def test_2_order_preserved() -> None:
    print("\nTest 2: result order matches emission order")
    tool = FakeTool("r", delay_s=0.05)
    agent = Agent(client=_NullClient(), tools=[tool])
    calls = [tool_call("r", f"tc{i}", {"i": i}) for i in range(4)]

    results = await agent.act(ExecutionContext(), calls)
    check(len(results) == 4, "four results")
    for i, (r, c) in enumerate(zip(results, calls, strict=False)):
        check(
            r.tool_call_id == c.tool_call_id,
            f"results[{i}].tool_call_id == calls[{i}].tool_call_id "
            f"(got {r.tool_call_id!r}, expected {c.tool_call_id!r})",
        )


async def test_3_mixed_forces_sequential() -> None:
    print("\nTest 3: read + mutate in one batch forces all sequential")
    read_tool = FakeTool("r", delay_s=0.1)
    write_tool = FakeTool("w", delay_s=0.1, permission="mutate")
    agent = Agent(
        client=_NullClient(),
        tools=[read_tool, write_tool],
        confirm_tool_call=lambda tc: True,
    )
    calls = [tool_call("r", "tc-r"), tool_call("w", "tc-w")]

    start = time.monotonic()
    results = await agent.act(ExecutionContext(), calls)
    elapsed = time.monotonic() - start

    print(f"  \u2139 elapsed: {elapsed:.2f}s (sequential ~0.2s, parallel ~0.1s)")
    check(len(results) == 2, "two results")
    check(elapsed >= 0.18, f"elapsed >= 0.18s (got {elapsed:.2f}s)")
    check(
        _max_overlap(read_tool.call_times + write_tool.call_times) == 1,
        "no overlap (fully sequential)",
    )
    check(results[0].status == "success", "read succeeded")
    check(results[1].status == "success", "write succeeded")


async def test_4_all_mutations_sequential() -> None:
    print("\nTest 4: three mutations run sequentially")
    tool = FakeTool("w", delay_s=0.1, permission="mutate")
    agent = Agent(
        client=_NullClient(),
        tools=[tool],
        confirm_tool_call=lambda tc: True,
    )
    calls = [tool_call("w", f"tc{i}") for i in range(3)]

    start = time.monotonic()
    results = await agent.act(ExecutionContext(), calls)
    elapsed = time.monotonic() - start

    print(f"  \u2139 elapsed: {elapsed:.2f}s (sequential ~0.3s)")
    check(len(results) == 3, "three results")
    check(elapsed >= 0.28, f"elapsed >= 0.28s (got {elapsed:.2f}s)")
    check(
        _max_overlap(tool.call_times) == 1,
        f"no overlap (got {_max_overlap(tool.call_times)})",
    )


async def test_5_exception_isolation() -> None:
    print("\nTest 5: one raising tool does not cancel siblings")
    ok1 = FakeTool("ok1", delay_s=0.05)
    raiser = FakeTool("boom", delay_s=0.05, raise_exc=RuntimeError("kaboom"))
    ok2 = FakeTool("ok2", delay_s=0.05)
    agent = Agent(client=_NullClient(), tools=[ok1, raiser, ok2])
    calls = [
        tool_call("ok1", "tc1"),
        tool_call("boom", "tc2"),
        tool_call("ok2", "tc3"),
    ]

    results = await agent.act(ExecutionContext(), calls)
    check(len(results) == 3, f"three results (got {len(results)})")
    by_id = {r.tool_call_id: r for r in results}
    check(by_id["tc1"].status == "success", "ok1 succeeded")
    check(by_id["tc2"].status == "error", "boom errored")
    check("kaboom" in str(by_id["tc2"].content[0]), "error message preserved")
    check(by_id["tc3"].status == "success", "ok2 succeeded")
    check(bool(ok1.call_times) and bool(ok2.call_times), "siblings actually ran")


async def test_6_timeout_isolation() -> None:
    print("\nTest 6: one slow tool times out, siblings unaffected")
    slow = FakeTool("slow", delay_s=0.5, timeout_ms=100)
    fast1 = FakeTool("f1", delay_s=0.05)
    fast2 = FakeTool("f2", delay_s=0.05)
    agent = Agent(client=_NullClient(), tools=[slow, fast1, fast2])
    calls = [
        tool_call("slow", "tc-slow"),
        tool_call("f1", "tc-f1"),
        tool_call("f2", "tc-f2"),
    ]

    start = time.monotonic()
    results = await agent.act(ExecutionContext(), calls)
    elapsed = time.monotonic() - start

    print(f"  \u2139 elapsed: {elapsed:.2f}s (should be ~0.1s, not 0.6s)")
    check(len(results) == 3, "three results")
    by_id = {r.tool_call_id: r for r in results}
    check(by_id["tc-slow"].status == "error", "slow errored")
    check(
        "timed out" in str(by_id["tc-slow"].content[0]).lower(),
        "message names timeout",
    )
    check(by_id["tc-f1"].status == "success", "fast1 succeeded")
    check(by_id["tc-f2"].status == "success", "fast2 succeeded")
    check(elapsed < 0.3, f"elapsed < 0.3s (got {elapsed:.2f}s)")


async def test_7_cancellation_propagates() -> None:
    print("\nTest 7: outer cancellation propagates as CancelledError")
    tool = FakeTool("slow", delay_s=1.0)
    agent = Agent(client=_NullClient(), tools=[tool])
    calls = [tool_call("slow", "tc1"), tool_call("slow", "tc2")]

    task = asyncio.create_task(agent.act(ExecutionContext(), calls))
    await asyncio.sleep(0.05)
    task.cancel()

    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True
    except Exception as exc:
        check(False, f"unexpected exception type: {type(exc).__name__}: {exc}")
        return

    check(cancelled, "CancelledError propagated")
    check(task.cancelled(), "task.cancelled() is True")


async def test_8_denial_preserves_order() -> None:
    print("\nTest 8: permission denial preserves order in a mixed batch")
    read_tool = FakeTool("r", delay_s=0.0)
    write_tool = FakeTool("w", delay_s=0.0, permission="mutate")
    agent = Agent(
        client=_NullClient(),
        tools=[read_tool, write_tool],
        confirm_tool_call=lambda tc: False,  # deny all mutations
    )
    calls = [
        tool_call("r", "tc-r1"),
        tool_call("w", "tc-w"),
        tool_call("r", "tc-r2"),
    ]

    results = await agent.act(ExecutionContext(), calls)
    check(len(results) == 3, "three results")
    check(
        [r.tool_call_id for r in results] == ["tc-r1", "tc-w", "tc-r2"],
        f"order preserved (got {[r.tool_call_id for r in results]})",
    )
    check(results[0].status == "success", "r1 succeeded")
    check(results[1].status == "error", "w denied")
    check(
        "denied" in str(results[1].content[0]).lower(),
        "denial message present",
    )
    check(results[2].status == "success", "r2 succeeded")


async def main() -> None:
    await test_1_parallel_reads_concurrent()
    await test_2_order_preserved()
    await test_3_mixed_forces_sequential()
    await test_4_all_mutations_sequential()
    await test_5_exception_isolation()
    await test_6_timeout_isolation()
    await test_7_cancellation_propagates()
    await test_8_denial_preserves_order()


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Parallel tool execution verified.")


if __name__ == "__main__":
    run()
