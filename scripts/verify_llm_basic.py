"""Phase 4: verify basic LLM generate (text only, no tools, no retry).

Scope note: this script tests the Phase 4 surface - plain generate, no tools,
no streaming. Stream mode was unimplemented when Phase 4 was verified and had
a placeholder assertion "stream mode returns an error". Phase 10 implemented
streaming, so that placeholder no longer holds. Streaming is now covered by
``verify_stream.py`` and ``verify_stream_toolcall.py``; this script does not
duplicate those.
"""

from __future__ import annotations

import asyncio
import os
import sys

from pyagent.llm import LlmClient, LlmRequest
from pyagent.models import resolve_model
from pyagent.schemas import Message
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


def _skip_if_llm_error(error_message: str | None, metadata: dict[str, object]) -> bool:
    """Record and return True if the response should skip the suite."""
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


def preflight_api_key() -> None:
    print("\nPreflight: API key present in environment")
    has_gemini = "GEMINI_API_KEY" in os.environ
    has_google = "GOOGLE_API_KEY" in os.environ
    check(
        has_gemini or has_google,
        f"GEMINI_API_KEY or GOOGLE_API_KEY set "
        f"(gemini={has_gemini}, google={has_google}). "
        f"If both are False, add a key to /pyagent/.env",
    )


async def main() -> None:
    print("\nTest 1: resolve_model passes gemini/ ids through")
    m = resolve_model("gemini/gemini-3-flash-preview")
    check(m == "gemini/gemini-3-flash-preview", f"passthrough (got {m})")

    print("\nTest 2: resolve_model rejects unknown prefix")
    try:
        resolve_model("openai/gpt-4")
    except ValueError as e:
        check("unsupported" in str(e).lower(), f"ValueError: {e}")
    else:
        check(False, "expected ValueError for openai/ prefix")

    print("\nTest 3: LlmRequest defaults")
    req = LlmRequest()
    check(req.instructions == [], "instructions default []")
    check(req.contents == [], "contents default []")
    check(req.tools == [], "tools default []")
    check(req.tool_choice is None, "tool_choice default None")
    check(req.mode is None, "mode default None")

    preflight_api_key()

    print("\nTest 4: real Gemini call (plain text)")
    client = LlmClient(model=resolve_model("gemini/gemini-3-flash-preview"))
    request = LlmRequest(
        instructions=["You are a helpful assistant. Reply briefly."],
        contents=[
            Message(
                role="user",
                content="What is 2 + 2? Reply with just the number.",
            ),
        ],
    )
    response = await client.generate(request)

    if _skip_if_llm_error(response.error_message, response.error_metadata):
        return

    check(True, "no error from LLM call")
    check(
        len(response.content) >= 1,
        f"at least one content item (got {len(response.content)})",
    )

    text = "".join(
        item.content
        for item in response.content
        if isinstance(item, Message) and item.role == "assistant"
    )
    check("4" in text, f"response contains '4' (got {text!r})")
    check(
        response.usage_metadata.get("input_tokens") is not None,
        f"usage input_tokens present (got {response.usage_metadata})",
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
        print(f"\n\u2705 Phase 4 verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 4 verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
