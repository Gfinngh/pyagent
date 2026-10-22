"""Phase 9: real provider errors through the real LiteLLM path.

A genuine quota exhaustion cannot be forced on demand (it would consume the
day's budget), so this exercises the two terminal provider failures that are
deterministically reproducible: an invalid API key and an unknown model. Both
must fail on attempt 1 with ``retryable=False`` - the behaviour that keeps us
from burning quota on guaranteed failures.
"""

from __future__ import annotations

import asyncio
import sys

from pyagent.llm import LlmClient, LlmRequest
from pyagent.models import resolve_model
from pyagent.schemas import Message

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def request() -> LlmRequest:
    return LlmRequest(
        instructions=["Reply briefly."],
        contents=[Message(role="user", content="Say hi.")],
    )


async def main() -> None:
    model = resolve_model("gemini/gemini-3.1-flash-lite")

    print("\nTest 1: invalid API key - terminal on attempt 1")
    client = LlmClient(
        model=model,
        max_retries=3,
        timeout_ms=30_000,
        api_key="not-a-real-key",
    )
    resp = await client.generate(request())

    check(resp.error_message is not None, f"error returned (got {resp.error_message!r})")
    check(
        resp.error_metadata.get("attempts") == 1,
        f"attempts=1 (got {resp.error_metadata.get('attempts')})",
    )
    check(
        resp.error_metadata.get("retryable") is False,
        f"retryable=False (got {resp.error_metadata.get('retryable')})",
    )
    check(resp.error_metadata.get("retry_exhausted") is not True, "not marked exhausted")
    print(f"  \u2139 status_code={resp.error_metadata.get('status_code')}")
    print(f"  \u2139 provider_code={resp.error_metadata.get('provider_code')}")
    print(f"  \u2139 provider_status={resp.error_metadata.get('provider_status')}")
    print(f"  \u2139 message={resp.error_message!r}")

    print("\nTest 2: nonexistent model - terminal 404, attempt 1")
    client_404 = LlmClient(
        model="gemini/definitely-not-a-real-model-xyz",
        max_retries=3,
        timeout_ms=30_000,
    )
    resp_404 = await client_404.generate(request())

    check(resp_404.error_message is not None, "error returned")
    check(
        resp_404.error_metadata.get("attempts") == 1,
        f"attempts=1 (got {resp_404.error_metadata.get('attempts')})",
    )
    check(
        resp_404.error_metadata.get("retryable") is False,
        f"retryable=False (got {resp_404.error_metadata.get('retryable')})",
    )
    print(f"  \u2139 status_code={resp_404.error_metadata.get('status_code')}")
    print(f"  \u2139 message={resp_404.error_message!r}")

    print("\nTest 3: both failures returned metadata, not raised")
    check("status_code" in resp.error_metadata, "invalid-key metadata has status_code")
    check("status_code" in resp_404.error_metadata, "404 metadata has status_code")
    check("provider_data" in resp.error_metadata, "provider_data captured for diagnosis")


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 9 (terminal provider errors) verified.")


if __name__ == "__main__":
    run()
