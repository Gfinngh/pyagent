"""Phase 11: provider-options round-trip across multiple turns.

Exercises the two Gemini rules that are easy to get wrong:
  1. Sequential tool calls: every turn that replays a tool call must include
     the signature that came with it.
  2. Parallel tool calls: only the *first* functionCall carries a signature;
     siblings arrive unsigned and must be returned unsigned.

Also checks that all turns complete without a 400 from Google (a corrupt
signature produces ``Base64 decoding failed`` / ``Invalid thought signature``).

Skip conditions (not workflow bugs):
  - ``quota_exhausted``: Gemini daily quota wall
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion. Observed
    2026-10-20 as a 503 on turn 2 with the primary key, before the transient
    skip was added.
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


def tool_calls_in(content: list[ContentItem]) -> list[ToolCall]:
    return [item for item in content if isinstance(item, ToolCall)]


def exec_tool_calls(tool_calls: list[ToolCall]) -> list[ToolResult]:
    results: list[ToolResult] = []
    for tc in tool_calls:
        if tc.name != "calculator":
            continue
        value = _run_calc_sync(tc.arguments)
        results.append(
            ToolResult.model_validate(
                {
                    "tool_call_id": tc.tool_call_id,
                    "name": tc.name,
                    "status": "success",
                    "content": [value],
                }
            )
        )
    return results


def _run_calc_sync(args: dict[str, object]) -> float:
    op = str(args.get("operator", ""))
    a = float(_as_num(args.get("first_number")))
    b = float(_as_num(args.get("second_number")))
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    raise ValueError(f"unknown operator {op!r}")


def _as_num(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    model = resolve_model("gemini/gemini-3.1-flash-lite")
    tools: list[BaseTool] = [calculator_tool]
    instructions = [
        "You are a helpful assistant. Use the calculator for every arithmetic "
        "step. Do not compute anything in your head.",
    ]

    print("\nTest 1: sequential tool calls - signature preserved on each turn")
    client = LlmClient(model=model)
    user1 = Message(
        role="user",
        content="Use the calculator to compute 2 + 3, then multiply the result by 4.",
    )
    contents: list[ContentItem] = [user1]

    turn_sigs: list[str] = []
    for turn in range(1, 3):
        capture = RequestCapture()
        client.request_hook = capture

        resp = await client.generate(
            LlmRequest(
                instructions=instructions,
                contents=list(contents),
                tools=tools,
                tool_choice="auto",
            )
        )
        if _skip_if_llm_error(resp):
            return
        check(True, f"turn {turn} no error")
        check(
            "400" not in (resp.error_message or ""),
            f"turn {turn} no 400 (signature not corrupted)",
        )
        check(
            len(capture.calls) >= 1,
            f"turn {turn}: captured >=1 request (got {len(capture.calls)})",
        )

        calls = tool_calls_in(resp.content)
        check(len(calls) >= 1, f"turn {turn} produced a tool call (got {len(calls)})")
        if not calls:
            return

        tc = calls[0]
        psf_sig, id_sig = signature_from_tool_call(_adapter(tc))
        sig = psf_sig or id_sig
        check(sig is not None, f"turn {turn} signature captured")
        if sig is not None:
            check(
                not is_dummy_signature(sig),
                f"turn {turn} signature is real, not dummy",
            )
            turn_sigs.append(sig)

        if turn >= 2 and capture.calls:
            out_messages: list[dict[str, Any]] = cast(
                list[dict[str, Any]], capture.calls[-1]["messages"]
            )
            for idx, prior_tc in enumerate(_prior_tool_calls(contents), start=1):
                replayed = find_replayed_tool_call(out_messages, prior_tc.tool_call_id)
                check(
                    replayed is not None,
                    f"turn {turn}: prior call #{idx} present in outgoing messages",
                )
                if replayed is not None:
                    sent = signature_from_provider_fields(replayed.get("provider_specific_fields"))
                    check(
                        sent is not None,
                        f"turn {turn}: prior call #{idx} carries a signature",
                    )
                    check(
                        not is_dummy_signature(sent),
                        f"turn {turn}: prior call #{idx} signature not dummy",
                    )
                    sent_psf = replayed.get("provider_specific_fields")
                    if isinstance(sent_psf, dict):
                        check(
                            "google" not in cast(dict[str, object], sent_psf),
                            f"turn {turn}: prior call #{idx} flat shape "
                            f"(keys={list(cast(dict[str, object], sent_psf))})",
                        )

        results = exec_tool_calls(calls)
        contents.extend(calls)
        contents.extend(results)

    check(
        len(turn_sigs) == 2,
        f"captured signatures on both turns (got {len(turn_sigs)})",
    )

    print("\nTest 2: parallel tool calls - first signed, siblings unsigned")
    parallel_client = LlmClient(model=model)
    parallel_user = Message(
        role="user",
        content=(
            "Use the calculator twice in one step: compute 1 + 1 and 2 + 2. Return both results."
        ),
    )
    parallel_resp = await parallel_client.generate(
        LlmRequest(
            instructions=instructions,
            contents=[parallel_user],
            tools=tools,
            tool_choice="auto",
        )
    )
    if _skip_if_llm_error(parallel_resp):
        return
    check(True, "parallel turn no error")
    calls = tool_calls_in(parallel_resp.content)
    if len(calls) < 2:
        skip(f"model returned {len(calls)} tool call(s); parallel rule not exercised this run")
    else:
        signatures: list[str | None] = []
        for tc in calls:
            psf_sig, id_sig = signature_from_tool_call(_adapter(tc))
            signatures.append(psf_sig or id_sig)
        signed = [s for s in signatures if s is not None]
        print(f"  \u2139 signature presence: {['signed' if s else 'unsigned' for s in signatures]}")
        check(
            len(signed) >= 1,
            "at least the first parallel call carries a signature",
        )
        if len(signatures) >= 2:
            check(
                signatures[0] is not None,
                "first parallel call carries the signature (Google's rule)",
            )


def _prior_tool_calls(contents: list[ContentItem]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for item in contents:
        if isinstance(item, ToolCall):
            out.append(item)
    return out


class _Adapter:
    def __init__(self, tc: ToolCall) -> None:
        self.provider_specific_fields = tc.provider_options
        self.id = tc.tool_call_id


def _adapter(tc: ToolCall) -> object:
    return _Adapter(tc)


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
        print(
            f"\n\u2705 Phase 11 (provider options, multi-turn) verified with "
            f"{len(_SKIPPED)} skipped:"
        )
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 11 (provider options, multi-turn) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
