"""Phase 14b: Agent.resume() across the three interruption states.

Builds a real log with one agent run, then truncates it to simulate:

  State C - cut after the tool-result event. Resume calls the LLM once and
            produces the final answer.
  State B - cut after the response-with-tool-call event. Resume must execute
            the pending calculator call first (Gemini rejects a request with a
            functionCall that has no functionResponse), then call the LLM.
  State A - full log. Resume is a no-op: no new events, no new steps, output
            unchanged.

Skip conditions (not workflow bugs):
  - ``quota_exhausted``: Gemini daily quota wall
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pyagent.agent import Agent, LlmCallError
from pyagent.context import ExecutionContext
from pyagent.event_log import JsonlEventLog
from pyagent.llm import LlmClient
from pyagent.models import resolve_model
from pyagent.schemas import ToolCall, ToolResult
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


def _skip_if_llm_error(exc: LlmCallError) -> Literal[True]:
    """Record skip if quota/transient, else record failure. Always returns True.

    Returning ``Literal[True]`` lets pyright narrow ``if _skip_if_llm_error(e):``
    as an always-taken branch - the caller's ``return`` is provably executed, so
    variables assigned in the ``try`` block are provably bound after the
    ``except`` block.
    """
    if exc.metadata.get("quota_exhausted") is True:
        skip(f"quota exhausted: {exc}")
        return True
    if is_transient_llm_response(str(exc), exc.metadata):
        skip(f"transient LLM failure: {exc}")
        return True
    check(False, f"LLM failed: {exc}")
    return True


def mentions_four(output: str | dict[str, object] | None) -> bool:
    if isinstance(output, str):
        return "4" in output
    if isinstance(output, dict):
        return any("4" in str(v) for v in output.values())
    return False


async def build_log(path: Path, client: LlmClient) -> list[str] | None:
    """Run an agent once, write the log, return its lines. None on failure."""
    agent = Agent(
        client=client,
        tools=[calculator_tool],
        instructions="You are a helpful assistant. Use tools when appropriate.",
        max_steps=4,
    )
    try:
        result = await agent.run("Use the calculator to compute 2 + 2. Do not compute it yourself.")
    except LlmCallError as e:
        _skip_if_llm_error(e)
        return None

    if result.output is None:
        skip("setup run produced no output")
        return None

    log = JsonlEventLog(path)
    for event in result.context.events:
        log.append(event)
    log.close()

    lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
    check(len(lines) >= 4, f"setup log has >=4 lines (got {len(lines)})")
    return lines


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    model = resolve_model("gemini/gemini-3.1-flash-lite")
    client = LlmClient(model=model, max_retries=0, timeout_ms=30_000)
    agent = Agent(
        client=client,
        tools=[calculator_tool],
        instructions="You are a helpful assistant. Use tools when appropriate.",
        max_steps=4,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"

        print("\nSetup: generate a real 4-event log")
        lines = await build_log(source, client)
        if lines is None:
            return

        # ---------------------------------------------------------------
        print("\nTest 1: State C - interrupted after tool_result")
        state_c = root / "state_c.jsonl"
        state_c.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
        ctx_c = ExecutionContext.from_log(state_c)
        check(ctx_c.final_result is None, "from_log reports unfinished")
        check(ctx_c.current_step == 1, f"current_step=1 (got {ctx_c.current_step})")

        try:
            result_c = await agent.resume(ctx_c)
        except LlmCallError as e:
            _skip_if_llm_error(e)
            return

        check(result_c.output is not None, f"output set (got {result_c.output!r})")
        check(
            mentions_four(result_c.output),
            f"output mentions 4 (got {result_c.output!r})",
        )
        check(
            len(ctx_c.events) == 4,
            f"events grew 3 -> 4 (got {len(ctx_c.events)})",
        )
        check(
            ctx_c.current_step == 2,
            f"current_step 1 -> 2 (got {ctx_c.current_step})",
        )

        # ---------------------------------------------------------------
        print("\nTest 2: State B - interrupted mid-tool-call")
        state_b = root / "state_b.jsonl"
        state_b.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
        ctx_b = ExecutionContext.from_log(state_b)
        check(ctx_b.final_result is None, "from_log reports unfinished")
        check(ctx_b.current_step == 1, f"current_step=1 (got {ctx_b.current_step})")
        pending_before = any(
            isinstance(item, ToolCall) for ev in ctx_b.events for item in ev.content
        )
        check(pending_before, "dangling ToolCall present before resume")

        try:
            result_b = await agent.resume(ctx_b)
        except LlmCallError as e:
            _skip_if_llm_error(e)
            return

        recovery_results = [
            item for ev in ctx_b.events for item in ev.content if isinstance(item, ToolResult)
        ]
        check(
            len(recovery_results) >= 1,
            f"recovery appended a ToolResult (got {len(recovery_results)})",
        )
        if recovery_results:
            check(
                recovery_results[0].status == "success",
                f"recovery result status=success (got {recovery_results[0].status})",
            )
            check(
                recovery_results[0].content == [4.0],
                f"recovery result is 4.0 (got {recovery_results[0].content})",
            )
        check(
            mentions_four(result_b.output),
            f"output mentions 4 (got {result_b.output!r})",
        )
        check(
            len(ctx_b.events) == 4,
            f"events grew 2 -> 4 (got {len(ctx_b.events)})",
        )

        # ---------------------------------------------------------------
        print("\nTest 3: State A - completed log, no-op resume")
        ctx_a = ExecutionContext.from_log(source)
        before_events = len(ctx_a.events)
        before_step = ctx_a.current_step
        before_result = ctx_a.final_result
        check(before_result is not None, "from_log reports completed")

        try:
            result_a = await agent.resume(ctx_a)
        except LlmCallError as e:
            _skip_if_llm_error(e)
            return

        check(
            len(ctx_a.events) == before_events,
            f"events unchanged {before_events} (got {len(ctx_a.events)})",
        )
        check(
            ctx_a.current_step == before_step,
            f"current_step unchanged {before_step} (got {ctx_a.current_step})",
        )
        check(
            ctx_a.final_result == before_result,
            f"final_result unchanged (got {ctx_a.final_result!r})",
        )
        check(
            result_a.output == before_result,
            f"returned output unchanged (got {result_a.output!r})",
        )

        # ---------------------------------------------------------------
        print("\nTest 4: resume(ctx, user_input) continues a completed run")
        ctx_d = ExecutionContext.from_log(source)
        check(ctx_d.final_result is not None, "start state is completed")

        try:
            result_d = await agent.resume(ctx_d, "Now what is that result times 10?")
        except LlmCallError as e:
            _skip_if_llm_error(e)
            return

        check(result_d.output is not None, f"output set (got {result_d.output!r})")
        if isinstance(result_d.output, str):
            digits = result_d.output.replace(",", "").replace(" ", "")
            check(
                "40" in digits,
                f"answer mentions 40 (got {result_d.output!r})",
            )


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
        print(f"\n\u2705 Phase 14 (Agent.resume) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 14 (Agent.resume) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
