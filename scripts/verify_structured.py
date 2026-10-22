"""Phase 6: verify structured output via injected final_answer tool."""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic import BaseModel, Field

from pyagent.agent import Agent, LlmCallError
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


class MathAnswer(BaseModel):
    answer: int = Field(description="The numeric result.")
    explanation: str = Field(description="Brief reasoning.")


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    print("\nTest 1: output_type injects final_answer tool")
    client = LlmClient(model=resolve_model("gemini/gemini-3-flash-preview"))
    agent = Agent(
        client=client,
        tools=[calculator_tool],
        instructions="Solve math problems. Use the calculator if needed.",
        output_type=MathAnswer,
        max_steps=4,
    )

    tool_names = {t.name for t in agent.tools}
    check("calculator" in tool_names, "calculator is in tools")
    check("final_answer" in tool_names, "final_answer is in tools")
    check(
        agent.output_tool_name == "final_answer",
        f"output_tool_name set (got {agent.output_tool_name})",
    )

    print("\nTest 2: structured run - 12 * 7")
    try:
        result = await agent.run("What is 12 * 7?")
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
        isinstance(result.output, dict),
        f"output is dict (got {type(result.output).__name__})",
    )
    if isinstance(result.output, dict):
        check(
            "answer" in result.output,
            f"has 'answer' key (got {list(result.output)})",
        )
        check("explanation" in result.output, "has 'explanation' key")
        check(
            result.output.get("answer") == 84,
            f"answer=84 (got {result.output.get('answer')})",
        )
        check(
            "output" not in result.output,
            "output is NOT the {output: ...} wrapper",
        )

    print("\nTest 3: trace includes final_answer tool call + result")
    events = result.context.events
    final_answer_calls = [
        item
        for ev in events
        for item in ev.content
        if isinstance(item, ToolCall) and item.name == "final_answer"
    ]
    final_answer_results = [
        item
        for ev in events
        for item in ev.content
        if isinstance(item, ToolResult) and item.name == "final_answer"
    ]
    check(
        len(final_answer_calls) >= 1,
        f"at least 1 final_answer call (got {len(final_answer_calls)})",
    )
    check(
        len(final_answer_results) >= 1,
        f"at least 1 final_answer result (got {len(final_answer_results)})",
    )

    print(f"\n  \u2139 output: {result.output!r}")


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
        print(f"\n\u2705 Phase 6 (structured) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 6 (structured) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
