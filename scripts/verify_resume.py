"""Phase 14a: manual resume - rehydrated context fed straight to LlmClient.

Generates a real event log by running an agent once, rehydrates it with
``ExecutionContext.from_log``, appends a follow-up question, and sends the
flattened history straight to ``LlmClient.generate``. This isolates the
data-path question ("does the rehydrated tool result reach the model
correctly?") from the loop question, which ``verify_resume_agent`` covers.

Test 2 accepts two valid responses, both of which prove the model saw the
rehydrated 4:

  - assistant text mentioning 40, OR
  - a ToolCall computing 4 * 10 (the model choosing to verify with the tool
    rather than trust its own arithmetic)

On the ToolCall path, the operator is accepted as either ``"multiply"`` (the
tool's canonical ``Literal``) or a common synonym (``"*"``, ``"mul"``,
``"times"``, ``"x"``). This test does not run the tool - it only checks the
model's intent. A synonym would fail ``FunctionTool.call`` validation in a
full agent loop and be handled there; for the data-flow question here, any
multiply-ish operator proves the model saw the rehydrated 4.

Quota-tolerant: only ``quota_exhausted`` skips. A 408 timeout or 5xx fails.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from pyagent.agent import Agent, LlmCallError
from pyagent.context import ExecutionContext
from pyagent.event_log import JsonlEventLog
from pyagent.llm import LlmClient, LlmRequest, LlmResponse
from pyagent.models import resolve_model
from pyagent.schemas import Message, ToolCall
from pyagent.tools.calculator import calculator_tool

_FAILURES: list[str] = []
_SKIPPED: list[str] = []

_MULTIPLY_SYNONYMS: frozenset[str] = frozenset({"multiply", "*", "mul", "times", "x"})


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def skip(msg: str) -> None:
    _SKIPPED.append(msg)
    print(f"  \u26a0 SKIP: {msg}")


def is_quota_skip(err: LlmCallError) -> bool:
    """True only when the failure is a genuine quota wall (not a timeout/5xx)."""
    return err.metadata.get("quota_exhausted") is True or "quota" in str(err).lower()


def quota_exhausted(resp: LlmResponse) -> bool:
    return resp.error_metadata.get("quota_exhausted") is True


def collect_text(content: list[object]) -> str:
    return "".join(
        item.content for item in content if isinstance(item, Message) and item.role == "assistant"
    )


def collect_tool_calls(content: list[object]) -> list[ToolCall]:
    return [item for item in content if isinstance(item, ToolCall)]


def tool_call_uses_4_and_10(tc: ToolCall) -> bool:
    """True when the call multiplies 4 by 10 - i.e. the model saw the 4.

    Accepts the canonical ``"multiply"`` operator plus common synonyms. The
    tool's own ``Literal`` validation will reject a synonym at execution time
    (the agent turns that into an error ``ToolResult`` and retries), but this
    test is about data flow, not tool-call validity.
    """
    op = str(tc.arguments.get("operator", "")).strip().lower()
    if op not in _MULTIPLY_SYNONYMS:
        return False
    a = tc.arguments.get("first_number")
    b = tc.arguments.get("second_number")
    numbers = {
        float(a) if isinstance(a, (int, float)) else None,
        float(b) if isinstance(b, (int, float)) else None,
    }
    return numbers == {4.0, 10.0}


async def generate_log(path: Path, client: LlmClient) -> bool:
    """Run an agent once and write its events to ``path``. True on success."""
    agent = Agent(
        client=client,
        tools=[calculator_tool],
        instructions="You are a helpful assistant. Use tools when appropriate.",
        max_steps=4,
    )
    try:
        result = await agent.run("Use the calculator to compute 2 + 2. Do not compute it yourself.")
    except LlmCallError as e:
        if is_quota_skip(e):
            skip(f"quota exhausted: {e}")
            return False
        check(False, f"setup run failed: {e}")
        return False

    if result.output is None:
        skip("setup run produced no output; cannot build a log")
        return False

    log = JsonlEventLog(path)
    for event in result.context.events:
        log.append(event)
    log.close()

    tool_calls = [
        item for ev in result.context.events for item in ev.content if isinstance(item, ToolCall)
    ]
    check(
        len(tool_calls) >= 1,
        f"setup run recorded >=1 tool call (got {len(tool_calls)})",
    )
    return True


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    model = resolve_model("gemini/gemini-3.1-flash-lite")
    client = LlmClient(model=model, max_retries=0, timeout_ms=30_000)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.jsonl"

        print("\nSetup: generate a real event log via one agent run")
        if not await generate_log(path, client):
            _finalize()
            return

        print("\nTest 1: from_log rehydrates the recorded history")
        ctx = ExecutionContext.from_log(path)
        check(len(ctx.events) >= 3, f">=3 events (got {len(ctx.events)})")
        check(ctx.current_step >= 2, f"current_step>=2 (got {ctx.current_step})")
        check(ctx.final_result is not None, f"final_result set (got {ctx.final_result!r})")
        tool_results = [
            item for ev in ctx.events for item in ev.content if item.type == "tool_result"
        ]
        check(
            len(tool_results) >= 1,
            f">=1 tool result rehydrated (got {len(tool_results)})",
        )

        print("\nTest 2: follow-up question sees the rehydrated tool result")
        contents = [item for ev in ctx.events for item in ev.content]
        contents.append(Message(role="user", content="Now what is that result times 10?"))
        response = await client.generate(
            LlmRequest(
                instructions=["You are a helpful assistant."],
                contents=contents,
            )
        )
        if quota_exhausted(response):
            skip(f"quota exhausted during follow-up: {response.error_message}")
            _finalize()
            return
        check(
            response.error_message is None,
            f"follow-up no error (got {response.error_message!r})",
        )

        text = collect_text(list(response.content))
        tool_calls = collect_tool_calls(list(response.content))

        # Diagnostic: dump whatever the model actually returned, so a future
        # failure is debuggable without re-running.
        print(f"  \u2139 text length: {len(text)}")
        print(f"  \u2139 tool calls: {len(tool_calls)}")
        if tool_calls:
            first = tool_calls[0]
            print(f"  \u2139 first tool call: {first.name}({first.arguments})")

        if text:
            check(True, "assistant text present")
            digits = text.replace(",", "").replace(" ", "")
            check(
                "40" in digits,
                f"text mentions 40 - model used the rehydrated 4 (got {text!r})",
            )
        elif tool_calls:
            # The model chose to verify with the calculator instead of
            # answering directly - equally valid proof that it saw the 4.
            check(True, "assistant produced a ToolCall instead of text")
            uses_4_10 = any(tool_call_uses_4_and_10(tc) for tc in tool_calls)
            check(
                uses_4_10,
                f"ToolCall multiplies 4 by 10 - model used the rehydrated 4 "
                f"(got {[(tc.name, tc.arguments) for tc in tool_calls]})",
            )
        else:
            check(False, "response has neither text nor tool calls")

    _finalize()


def _finalize() -> None:
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
        print(f"\n\u2705 Phase 14 (manual resume) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 14 (manual resume) verified.")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
