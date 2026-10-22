"""Phase 12: fallback chain - real LiteLLM, broken first entry.

Chain: [nonexistent-model, gemini-3-flash-preview]. The first entry produces a
404 (terminal), but the test overrides `should_fallback` to advance on ANY
non-aborted error, so the chain still reaches the real model. Asserts the
success annotation and that the second entry actually answered.

Skip conditions (not workflow bugs):
  - ``quota_exhausted``: Gemini daily quota wall
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion
"""

from __future__ import annotations

import asyncio
import os
import sys

from pyagent.fallback import FallbackClient, FallbackEntry
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
    return False


def request() -> LlmRequest:
    return LlmRequest(
        instructions=["Reply briefly."],
        contents=[Message(role="user", content="Say hi in one word.")],
    )


async def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    print("\nTest 1: broken first entry -> real model answers")
    broken = LlmClient(
        model="gemini/definitely-not-a-real-model-xyz",
        max_retries=0,
        timeout_ms=10_000,
    )
    real = LlmClient(
        model=resolve_model("gemini/gemini-3.1-flash-lite"),
        max_retries=0,
        timeout_ms=30_000,
    )

    fallback_events: list[dict[str, object]] = []
    client = FallbackClient(
        chain=[
            FallbackEntry(label="broken", client=broken),
            FallbackEntry(label="real", client=real),
        ],
        # Override the default policy so the 404 (which is terminal) still
        # advances the chain. Test-only override: production uses the default,
        # where 404 is refused.
        should_fallback=lambda r: r.error_metadata.get("aborted") is not True,
        on_fallback=fallback_events.append,
    )

    resp = await client.generate(request())

    if _skip_if_llm_error(resp):
        return

    check(resp.error_message is None, f"recovered (got {resp.error_message!r})")
    if resp.error_message is not None:
        return

    text = "".join(
        item.content
        for item in resp.content
        if isinstance(item, Message) and item.role == "assistant"
    )
    check(len(text) > 0, f"assistant text present (got {text!r})")
    check(
        resp.error_metadata.get("fallback_used") is True,
        "fallback_used=True",
    )
    check(
        resp.error_metadata.get("fallback_from") == ["broken"],
        f"fallback_from=['broken'] (got {resp.error_metadata.get('fallback_from')})",
    )
    check(
        resp.error_metadata.get("fallback_landed_on") == "real",
        f"landed_on='real' (got {resp.error_metadata.get('fallback_landed_on')})",
    )

    print("\nTest 2: on_fallback fired for the advance")
    check(len(fallback_events) >= 1, f">=1 event (got {len(fallback_events)})")
    if fallback_events:
        ev = fallback_events[0]
        check(ev.get("from") == "broken", f"from='broken' (got {ev.get('from')})")
        check(ev.get("to") == "real", f"to='real' (got {ev.get('to')})")
        print(f"  \u2139 reason: {str(ev.get('reason', ''))[:100]}")

    print("\nTest 3: default policy would NOT have advanced a 404")
    from pyagent.fallback import default_fallback_policy

    broken_resp = await broken.generate(request())
    if _skip_if_llm_error(broken_resp):
        return
    check(
        broken_resp.error_message is not None,
        "broken entry fails as expected",
    )
    check(
        default_fallback_policy(broken_resp) is False,
        "default policy refuses to advance a 404 (terminal)",
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
        print(f"\n\u2705 Phase 12 (fallback, real) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 12 (fallback, real) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
