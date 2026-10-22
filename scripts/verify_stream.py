"""Phase 10: verify streaming text (no tools).

Asserts that stream mode returns the same LlmResponse shape as generate:
accumulated content, usage, no error. Delta count is *informational* — some
providers buffer the whole response and return one chunk, which is still a
valid "stream" call.

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


def collect_text(content: list[object]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, Message) and item.role == "assistant":
            parts.append(item.content)
    return "".join(parts)


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    deltas: list[str] = []

    print("\nTest 1: stream mode returns accumulated text")
    client = LlmClient(
        model=resolve_model("gemini/gemini-3-flash-preview"),
        mode="stream",
        on_text_delta=deltas.append,
    )
    request = LlmRequest(
        instructions=["You are a helpful assistant."],
        contents=[
            Message(
                role="user",
                content="Count from 1 to 5, one number per line, nothing else.",
            ),
        ],
    )
    response = await client.generate(request)

    if _skip_if_llm_error(response):
        return
    check(True, "no error")

    text = collect_text(list(response.content))
    check(len(text) > 0, f"accumulated text non-empty (got {text!r})")

    print("\nTest 2: on_text_delta fired (count is informational)")
    check(len(deltas) >= 1, f"at least one delta fired (got {len(deltas)})")
    if len(deltas) > 1:
        print(f"  \u2139 {len(deltas)} deltas — tokens streamed incrementally")
    else:
        print(
            "  \u2139 1 delta — provider buffered the response into a single "
            "chunk. Valid streaming behavior; not a failure."
        )
    check(
        "".join(deltas) == text,
        f"delta concatenation equals final text ({len(''.join(deltas))} vs {len(text)} chars)",
    )

    print("\nTest 3: usage metadata present")
    check(
        response.usage_metadata.get("input_tokens") is not None,
        f"input_tokens present (got {response.usage_metadata})",
    )
    check(
        response.usage_metadata.get("output_tokens") is not None,
        "output_tokens present",
    )
    print(f"  \u2139 usage: {response.usage_metadata}")

    await asyncio.sleep(0.5)

    print("\nTest 4: stream content matches generate content shape")
    gen_client = LlmClient(model=resolve_model("gemini/gemini-3-flash-preview"))
    gen_response = await gen_client.generate(request)

    if _skip_if_llm_error(gen_response):
        return

    check(True, "generate also succeeded")
    if gen_response.content and response.content:
        check(
            type(gen_response.content[0]).__name__ == type(response.content[0]).__name__,
            "both modes produce the same content-item type",
        )
    else:
        check(
            False,
            f"cannot compare shapes "
            f"(stream={len(response.content)} items, "
            f"gen={len(gen_response.content)} items)",
        )
    check(
        gen_response.error_metadata == {},
        f"generate success carries no error metadata (got {gen_response.error_metadata})",
    )
    check(
        response.error_metadata == {},
        f"stream success carries no error metadata (got {response.error_metadata})",
    )

    print("\nTest 5: per-request mode override beats client default")
    generate_default = LlmClient(
        model=resolve_model("gemini/gemini-3-flash-preview"),
        mode="generate",
    )
    override_response = await generate_default.generate(
        LlmRequest(
            instructions=["Be brief."],
            contents=[Message(role="user", content="Say the word: ping")],
            mode="stream",
        )
    )
    if _skip_if_llm_error(override_response):
        return
    check(True, "per-request stream override succeeded")
    override_text = collect_text(list(override_response.content))
    check(
        len(override_text) > 0,
        f"override produced text (got {override_text!r})",
    )

    print("\nTest 6: mid-stream failure is retried cleanly")
    bad_client = LlmClient(
        model="gemini/definitely-not-a-real-model-xyz",
        mode="stream",
        max_retries=3,
    )
    bad = await bad_client.generate(request)
    check(bad.error_message is not None, "stream error surfaced")
    check(
        bad.error_metadata.get("attempts") == 1,
        f"terminal 404, one attempt (got {bad.error_metadata.get('attempts')})",
    )
    check(bad.content == [], f"no partial content leaked (got {bad.content})")


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
        print(f"\n\u2705 Phase 10 (streaming text) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 10 (streaming text) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
