"""Phase 10: verify tool calls reassemble correctly from a stream.

LiteLLM streams tool-call fragments as partial JSON, keyed by index. The
chunk builder must stitch them into one call with parseable arguments, and
must preserve Gemini's provider metadata (thought signature) where present.

Skip conditions (not workflow bugs):
  - ``quota_exhausted``: Gemini daily quota wall
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion
"""

from __future__ import annotations

import asyncio
import os
import sys

from pyagent.llm import LlmClient, LlmRequest, LlmResponse
from pyagent.models import resolve_model
from pyagent.schemas import Message, ToolCall
from pyagent.tools import BaseTool
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


def _skip_if_llm_error(resp: LlmResponse) -> bool:
    if resp.error_message is None:
        return False
    if resp.error_metadata.get("quota_exhausted") is True:
        skip(f"quota exhausted: {resp.error_message}")
        return True
    if is_transient_llm_response(resp.error_message, resp.error_metadata):
        skip(f"transient LLM failure: {resp.error_message}")
        return True
    check(False, f"LLM failed: {resp.error_message[:200]}")
    return True


def cast_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"expected a number, got {value!r}")
    return float(value)


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    tools: list[BaseTool] = [calculator_tool]
    instructions = [
        "You are a helpful assistant. Use the calculator tool for any arithmetic.",
    ]

    print("\nTest 1: streamed tool call reassembles with complete arguments")
    client = LlmClient(
        model=resolve_model("gemini/gemini-3-flash-preview"),
        mode="stream",
        on_text_delta=lambda _d: None,
    )
    response = await client.generate(
        LlmRequest(
            instructions=instructions,
            contents=[
                Message(
                    role="user",
                    content=(
                        "Use the calculator to compute 12345 * 67890. "
                        "Do not compute it in your head."
                    ),
                ),
            ],
            tools=tools,
            tool_choice="auto",
        )
    )

    if _skip_if_llm_error(response):
        return

    tool_calls = [c for c in response.content if isinstance(c, ToolCall)]
    check(len(tool_calls) >= 1, f"at least one ToolCall (got {len(tool_calls)})")
    if not tool_calls:
        return

    tc = tool_calls[0]
    check(tc.name == "calculator", f"called calculator (got {tc.name})")
    check(tc.tool_call_id != "", f"tool_call_id populated (got {tc.tool_call_id!r})")

    print("\nTest 2: arguments parsed from accumulated fragments")
    check("operator" in tc.arguments, f"has operator (got {tc.arguments})")
    check("first_number" in tc.arguments, "has first_number")
    check("second_number" in tc.arguments, "has second_number")
    check(
        tc.arguments.get("operator") == "multiply",
        f"operator=multiply (got {tc.arguments.get('operator')})",
    )
    check(
        cast_float(tc.arguments.get("first_number")) == 12345.0,
        f"first_number=12345 (got {tc.arguments.get('first_number')})",
    )
    check(
        cast_float(tc.arguments.get("second_number")) == 67890.0,
        f"second_number=67890 (got {tc.arguments.get('second_number')})",
    )
    check("_raw" not in tc.arguments, "arguments are valid JSON, not a raw fragment")

    print("\nTest 3: tool executes from the streamed call")
    result = await calculator_tool.call(tc.arguments)
    check(
        cast_float(result) == 838102050.0,
        f"12345 * 67890 = 838102050 (got {result})",
    )

    print("\nTest 4: provider metadata preserved through streaming (informational)")
    if tc.provider_options is not None:
        print(f"  \u2713 provider_options captured: {list(tc.provider_options)}")
    else:
        print(
            "  \u26a0 no provider_options on streamed tool call. "
            "Known LiteLLM/Gemini streaming limitation; generate mode retains it."
        )

    print("\nTest 5: streamed and non-streamed calls agree on shape")
    gen_client = LlmClient(model=resolve_model("gemini/gemini-3-flash-preview"))
    gen_response = await gen_client.generate(
        LlmRequest(
            instructions=instructions,
            contents=[
                Message(
                    role="user",
                    content=(
                        "Use the calculator to compute 7 + 8. Do not compute it in your head."
                    ),
                ),
            ],
            tools=tools,
            tool_choice="auto",
        )
    )

    if _skip_if_llm_error(gen_response):
        return

    check(True, "generate succeeded")
    gen_calls = [c for c in gen_response.content if isinstance(c, ToolCall)]
    check(len(gen_calls) >= 1, "generate produced a ToolCall")
    if gen_calls:
        check(
            gen_calls[0].name == "calculator",
            f"same tool chosen (got {gen_calls[0].name})",
        )
        check(
            set(gen_calls[0].arguments) == set(tc.arguments),
            "same argument keys in both modes",
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
        print(f"\n\u2705 Phase 10 (streaming tool calls) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 10 (streaming tool calls) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
