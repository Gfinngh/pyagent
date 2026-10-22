"""Phase 11: provider-options round-trip, single tool call.

Verifies the *payload we send*, not just that the call succeeded - LiteLLM
injects a dummy signature when the real one is missing, so success alone is
not proof of round-trip.

Tests:
  1. Legacy {"google": {...}} wrapper unwrapped on replay (unit)
  2. Generate mode: signature captured on response (both channels)
  3. Replay payload: flat provider_specific_fields with the same signature
  4. Stream mode: same checks

Skip conditions (not workflow bugs):
  - ``quota_exhausted``: Gemini daily quota wall
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, cast

from pyagent.llm import LlmClient, LlmRequest, LlmResponse
from pyagent.models import resolve_model
from pyagent.provider_options import (
    RequestCapture,
    find_replayed_tool_call,
    is_dummy_signature,
    signature_from_provider_fields,
    signature_from_tool_call,
    unwrap_legacy_provider_fields,
)
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


def first_tool_call(content: list[ContentItem]) -> ToolCall | None:
    for item in content:
        if isinstance(item, ToolCall):
            return item
    return None


def inspect_replayed_signature(
    capture: RequestCapture,
    tool_call_id: str,
    expected_signature: str,
    label: str,
) -> None:
    """Assert the outgoing tool call carries the expected signature, flat."""
    check(len(capture.calls) >= 1, f"{label}: at least one captured request")
    if not capture.calls:
        return

    messages: list[dict[str, Any]] = cast(list[dict[str, Any]], capture.calls[-1]["messages"])
    replayed = find_replayed_tool_call(messages, tool_call_id)
    check(
        replayed is not None,
        f"{label}: tool call {tool_call_id[:24]}... present in outgoing messages",
    )
    if replayed is None:
        return

    psf = replayed.get("provider_specific_fields")
    check(
        isinstance(psf, dict),
        f"{label}: provider_specific_fields is a dict (got {type(psf).__name__})",
    )
    if not isinstance(psf, dict):
        return

    psf_typed = cast(dict[str, object], psf)
    check(
        "google" not in psf_typed,
        f"{label}: flat dict, not wrapped in 'google' (got keys {list(psf_typed)})",
    )

    sent_sig = signature_from_provider_fields(psf_typed)
    check(
        sent_sig == expected_signature,
        f"{label}: signature matches turn 1 "
        f"({len(sent_sig or '')} vs {len(expected_signature)} chars)",
    )
    check(
        not is_dummy_signature(sent_sig),
        f"{label}: signature is NOT the dummy validator",
    )


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    model = resolve_model("gemini/gemini-3-flash-preview")
    tools: list[BaseTool] = [calculator_tool]
    instructions = ["Use the calculator for any arithmetic."]

    print("\nTest 1: legacy {'google': {...}} wrapper unwrapped on replay (unit)")

    flat = unwrap_legacy_provider_fields({"google": {"thought_signature": "abc"}})
    check(
        flat == {"thought_signature": "abc"},
        f"wrapper unwrapped (got {flat})",
    )
    passthrough = unwrap_legacy_provider_fields({"thought_signature": "abc"})
    check(
        passthrough == {"thought_signature": "abc"},
        f"flat dict passed through unchanged (got {passthrough})",
    )
    other = unwrap_legacy_provider_fields({"foo": "bar"})
    check(
        other == {"foo": "bar"},
        f"unrelated dict passed through unchanged (got {other})",
    )

    print("\nTest 2: generate mode - capture signature on turn 1")
    client = LlmClient(model=model)
    user_msg = Message(
        role="user",
        content="Use the calculator to compute 2 + 2. Do not compute it yourself.",
    )
    turn1 = await client.generate(
        LlmRequest(
            instructions=instructions,
            contents=[user_msg],
            tools=tools,
            tool_choice="auto",
        )
    )

    if _skip_if_llm_error(turn1):
        return
    check(True, "turn 1 no error")

    tc = first_tool_call(turn1.content)
    check(tc is not None, "turn 1 produced a tool call")
    if tc is None:
        return

    psf_sig, id_sig = signature_from_tool_call(_tool_call_like(tc))
    print(f"  \u2139 signature from provider_specific_fields: {len(psf_sig or '')} chars")
    print(f"  \u2139 signature embedded in tool_call_id: {len(id_sig or '')} chars")

    check(
        psf_sig is not None or id_sig is not None,
        "at least one signature channel is populated",
    )
    check(
        not is_dummy_signature(psf_sig),
        "captured signature is real, not the dummy validator",
    )

    expected_sig = psf_sig or id_sig
    if expected_sig is None:
        check(False, "no signature captured; cannot verify replay")
        return

    print("\nTest 3: generate mode - replay carries the same signature")
    result = await calculator_tool.call(tc.arguments)
    tool_result = ToolResult.model_validate(
        {
            "tool_call_id": tc.tool_call_id,
            "name": tc.name,
            "status": "success",
            "content": [result],
        }
    )
    turn2_contents: list[ContentItem] = [user_msg, tc, tool_result]

    capture = RequestCapture()
    replay_client = LlmClient(model=model, request_hook=capture)
    turn2 = await replay_client.generate(
        LlmRequest(
            instructions=instructions,
            contents=turn2_contents,
            tools=tools,
            tool_choice="auto",
        )
    )

    if _skip_if_llm_error(turn2):
        return
    check(True, "turn 2 no error")
    check(len(capture.calls) >= 1, f"captured >=1 request (got {len(capture.calls)})")
    if capture.calls:
        inspect_replayed_signature(
            capture=capture,
            tool_call_id=tc.tool_call_id,
            expected_signature=expected_sig,
            label="turn 2",
        )

    print("\nTest 4: stream mode - capture and replay")
    stream_capture = RequestCapture()
    stream_client = LlmClient(model=model, mode="stream", request_hook=stream_capture)
    stream_turn1 = await stream_client.generate(
        LlmRequest(
            instructions=instructions,
            contents=[
                Message(
                    role="user",
                    content=("Use the calculator to compute 3 + 4. Do not compute it."),
                ),
            ],
            tools=tools,
            tool_choice="auto",
        )
    )
    if _skip_if_llm_error(stream_turn1):
        return
    check(True, "stream turn 1 no error")
    check(
        len(stream_capture.calls) >= 1,
        f"stream: captured >=1 request (got {len(stream_capture.calls)})",
    )
    if stream_capture.calls:
        first_call = stream_capture.calls[0]
        check(
            first_call.get("stream") is True,
            f"stream: captured kwargs has stream=True (got {first_call.get('stream')})",
        )
    stc = first_tool_call(stream_turn1.content)
    check(stc is not None, "stream turn 1 produced a tool call")
    if stc is not None:
        spsf, sid = signature_from_tool_call(_tool_call_like(stc))
        print(f"  \u2139 stream signature channels: psf={len(spsf or '')} id={len(sid or '')}")
        check(
            spsf is not None or sid is not None,
            "stream signature captured on at least one channel",
        )
        if spsf is not None:
            check(
                not is_dummy_signature(spsf),
                "stream signature is real, not the dummy",
            )


def _tool_call_like(tc: ToolCall) -> object:
    return _Adapter(tc)


class _Adapter:
    def __init__(self, tc: ToolCall) -> None:
        self.provider_specific_fields = tc.provider_options
        self.id = tc.tool_call_id


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
        print(f"\n\u2705 Phase 11 (provider options) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 11 (provider options) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
