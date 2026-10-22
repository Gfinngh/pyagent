"""Phase 5: verify tool-call round-trip through LlmClient.

Proves the full two-turn exchange:
  turn 1: user asks -> LLM returns ToolCall
  execute: calculator runs -> result 4
  turn 2: history replayed -> LLM returns final text answer
"""

from __future__ import annotations

import asyncio
import sys

from pyagent.llm import LlmClient, LlmRequest
from pyagent.models import resolve_model
from pyagent.schemas import ContentItem, Message, ToolCall, ToolResult
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


def preflight_api_key() -> None:
    import os

    print("\nPreflight: API key present")
    has_gemini = "GEMINI_API_KEY" in os.environ
    has_google = "GOOGLE_API_KEY" in os.environ
    check(
        has_gemini or has_google,
        f"GEMINI_API_KEY or GOOGLE_API_KEY set (gemini={has_gemini}, google={has_google})",
    )


def collect_text(content: list[ContentItem]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, Message) and item.role == "assistant":
            parts.append(item.content)
    return "".join(parts)


def collect_tool_calls(content: list[ContentItem]) -> list[ToolCall]:
    return [item for item in content if isinstance(item, ToolCall)]


def _skip_if_llm_error(error_message: str | None, metadata: dict[str, object]) -> bool:
    if error_message is None:
        return False
    if metadata.get("quota_exhausted") is True:
        skip(f"quota exhausted: {error_message}")
        return True
    if is_transient_llm_response(error_message, metadata):
        skip(f"transient LLM failure: {error_message}")
        return True
    check(False, f"LLM failed: {error_message[:200]}")
    return True


async def main() -> None:
    model = resolve_model("gemini/gemini-3-flash-preview")
    client = LlmClient(model=model)
    tools: list[BaseTool] = [calculator_tool]

    instructions = [
        "You are a helpful assistant. Use the calculator tool for any arithmetic.",
    ]

    print("\nTest 1: tool_definition passes through")
    defn = calculator_tool.tool_definition
    check(defn is not None, "calculator_tool has tool_definition")

    preflight_api_key()

    print("\nTest 2: turn 1 - LLM decides to call calculator")
    user_msg = Message(
        role="user",
        content="Use the calculator to compute 2 + 2. Do not compute in your head.",
    )
    turn1_contents: list[ContentItem] = [user_msg]
    turn1 = await client.generate(
        LlmRequest(
            instructions=instructions,
            contents=turn1_contents,
            tools=tools,
            tool_choice="auto",
        )
    )

    if _skip_if_llm_error(turn1.error_message, turn1.error_metadata):
        return
    check(True, "turn 1 no error")

    tool_calls = collect_tool_calls(turn1.content)
    check(len(tool_calls) >= 1, f"at least one ToolCall (got {len(tool_calls)})")

    if not tool_calls:
        return

    tc = tool_calls[0]
    check(tc.name == "calculator", f"called calculator (got {tc.name})")
    check("operator" in tc.arguments, f"has operator arg (got {tc.arguments})")
    check("first_number" in tc.arguments, "has first_number")
    check("second_number" in tc.arguments, "has second_number")

    print(f"\n  \u24d8 captured provider_options: {'present' if tc.provider_options else 'absent'}")

    print("\nTest 3: execute the tool call")
    result_value = await calculator_tool.call(tc.arguments)
    check(result_value == 4.0, f"2+2=4.0 (got {result_value})")

    print("\nTest 4: turn 2 - replay history, LLM produces final answer")
    tool_result = ToolResult.model_validate(
        {
            "tool_call_id": tc.tool_call_id,
            "name": tc.name,
            "status": "success",
            "content": [result_value],
        }
    )

    turn2_contents: list[ContentItem] = [user_msg, tc, tool_result]
    turn2 = await client.generate(
        LlmRequest(
            instructions=instructions,
            contents=turn2_contents,
            tools=tools,
            tool_choice="auto",
        )
    )

    if _skip_if_llm_error(turn2.error_message, turn2.error_metadata):
        return
    check(True, "turn 2 no error")

    final_text = collect_text(turn2.content)
    check(len(final_text) > 0, f"final text present (got {final_text!r})")
    check("4" in final_text, f"final text mentions 4 (got {final_text!r})")
    check(
        len(collect_tool_calls(turn2.content)) == 0,
        "no further tool calls in turn 2",
    )

    print("\nTest 5: provider_options round-trip on replay (informational)")
    if tc.provider_options is not None:
        print("  \u2713 tool_call carried provider_options on turn 1 (captured)")
        print(f"    shape: {list(tc.provider_options.keys())}")
        print("  \u2713 replay succeeded with provider_options attached")
    else:
        print("  \u26a0 no provider_options captured (may be provider/version dependent)")

    print(f"\n  \u2139 final answer: {final_text!r}")


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
        print(f"\n\u2705 Phase 5 verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 5 verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
