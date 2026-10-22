"""Phase 9: verify the retry loop and error classification.

Uses a scripted fake transport (no network). ``backoff_ms`` is patched to a
small override so the suite stays fast; Test 7 uses a large override so the
deadline check trips on attempt 1. The real backoff curve is asserted
separately in Test 11.
"""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from typing import Any

import pyagent.llm as llm_mod
from pyagent.llm import LlmClient, LlmRequest, backoff_ms, classify_error
from pyagent.schemas import Message

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLMError(Exception):
    """Mirrors the attribute surface our classifier sniffs."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
        body: object = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        if status_code is None:
            self.response = None
        else:
            captured = body
            self.response = SimpleNamespace(
                status_code=status_code,
                headers=dict(headers or {}),
                json=lambda: captured,
            )


def ok_response(text: str = "ok") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class ScriptedTransport:
    """Pops one behaviour per call: an Exception is raised, anything else returned."""

    def __init__(self, behaviours: list[object]) -> None:
        self.behaviours = list(behaviours)
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> Any:
        _ = kwargs
        self.calls += 1
        if not self.behaviours:
            raise AssertionError(f"unexpected extra call #{self.calls}")
        nxt = self.behaviours.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def request() -> LlmRequest:
    return LlmRequest(
        instructions=["be brief"],
        contents=[Message(role="user", content="hi")],
    )


async def with_transport(
    behaviours: list[object],
    *,
    max_retries: int = 3,
    timeout_ms: int = 60_000,
    backoff_override: float = 1.0,
) -> tuple[ScriptedTransport, Any]:
    transport = ScriptedTransport(behaviours)
    original_acompletion = llm_mod.acompletion
    original_backoff = llm_mod.backoff_ms
    llm_mod.acompletion = transport
    llm_mod.backoff_ms = lambda attempt: backoff_override
    try:
        client = LlmClient(
            model="gemini/gemini-3.1-flash-lite",
            max_retries=max_retries,
            timeout_ms=timeout_ms,
        )
        response = await client.generate(request())
        return transport, response
    finally:
        llm_mod.acompletion = original_acompletion
        llm_mod.backoff_ms = original_backoff


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def main() -> None:
    print("\nTest 1: success on first attempt - no retries")
    transport, resp = await with_transport([ok_response("hello")])
    check(resp.error_message is None, "no error")
    check(transport.calls == 1, f"one call (got {transport.calls})")
    check(resp.content[0].type == "message", "content is a message")

    print("\nTest 2: two 500s then success - retries then recovers")
    transport, resp = await with_transport(
        [
            FakeLLMError("boom", status_code=500),
            FakeLLMError("boom", status_code=500),
            ok_response("ok"),
        ]
    )
    check(resp.error_message is None, "no error after recovery")
    check(transport.calls == 3, f"three calls (got {transport.calls})")

    print("\nTest 3: always 503, retries exhausted")
    transport, resp = await with_transport(
        [FakeLLMError("unavailable", status_code=503) for _ in range(10)],
        max_retries=2,
    )
    check(resp.error_message is not None, "error returned")
    check(transport.calls == 3, f"3 attempts = max_retries+1 (got {transport.calls})")
    check(resp.error_metadata.get("retry_exhausted") is True, "retry_exhausted=True")
    check(
        resp.error_metadata.get("attempts") == 3,
        f"attempts=3 (got {resp.error_metadata.get('attempts')})",
    )
    check(resp.error_metadata.get("retryable") is True, "retryable=True")
    check(resp.error_metadata.get("status_code") == 503, "status_code=503")

    print("\nTest 4: 401 auth - terminal on attempt 1")
    transport, resp = await with_transport(
        [FakeLLMError("unauthorized", status_code=401)],
        max_retries=3,
    )
    check(resp.error_message is not None, "error returned")
    check(transport.calls == 1, f"one attempt only (got {transport.calls})")
    check(resp.error_metadata.get("retryable") is False, "retryable=False")
    check(resp.error_metadata.get("retry_exhausted") is not True, "not marked exhausted")
    check(
        "Authentication" in (resp.error_message or ""), f"auth wording (got {resp.error_message!r})"
    )

    print("\nTest 5: 429 RESOURCE_EXHAUSTED, long retryDelay - terminal quota")
    quota_body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"retryDelay": "3600s"}],
        }
    }
    transport, resp = await with_transport(
        [FakeLLMError("quota", status_code=429, body=quota_body)],
        max_retries=3,
    )
    check(transport.calls == 1, f"one attempt only (got {transport.calls})")
    check(resp.error_metadata.get("quota_exhausted") is True, "quota_exhausted=True")
    check(resp.error_metadata.get("retryable") is False, "retryable=False")
    check(
        resp.error_metadata.get("provider_status") == "RESOURCE_EXHAUSTED", "provider_status parsed"
    )
    check(
        resp.error_metadata.get("retry_after_ms") == 3_600_000,
        f"retry_after_ms=3600000 (got {resp.error_metadata.get('retry_after_ms')})",
    )

    print("\nTest 6: 429 with short Retry-After header - retries then succeeds")
    transport, resp = await with_transport(
        [
            FakeLLMError("slow down", status_code=429, headers={"retry-after": "0.05"}),
            ok_response("recovered"),
        ]
    )
    check(resp.error_message is None, "recovered after retry")
    check(transport.calls == 2, f"two calls (got {transport.calls})")

    classified = classify_error(
        FakeLLMError("slow down", status_code=429, headers={"retry-after": "0.05"})
    )
    check(
        classified.metadata.get("retry_after_ms") == 50,
        f"header parsed to 50ms (got {classified.metadata.get('retry_after_ms')})",
    )
    check(classified.metadata.get("rate_limited") is True, "rate_limited=True")

    print("\nTest 7: deadline hit before retry - fail fast")
    # backoff_override=500ms exceeds timeout_ms=50ms, so the deadline check
    # trips on attempt 1 without ever sleeping.
    started = time.monotonic()
    transport, resp = await with_transport(
        [FakeLLMError("boom", status_code=500)],
        max_retries=5,
        timeout_ms=50,
        backoff_override=500.0,
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    check(transport.calls == 1, f"one attempt only (got {transport.calls})")
    check(
        resp.error_metadata.get("retry_reason") == "timeoutBeforeRetry",
        f"reason (got {resp.error_metadata.get('retry_reason')})",
    )
    check("would exceed timeout" in (resp.error_message or ""), "message mentions timeout")
    check(elapsed_ms < 1000, f"failed fast, no sleeping (took {elapsed_ms:.0f}ms)")

    print("\nTest 8: LiteLLM 403/429 mis-map reclassified as quota")
    mis_mapped = FakeLLMError(
        "GeminiException - Please retry in 18.403470473s.",
        status_code=403,
        body={"error": {"status": "RESOURCE_EXHAUSTED"}},
    )
    classified = classify_error(mis_mapped)
    check(
        classified.metadata.get("status_code") == 429,
        f"403 re-mapped to 429 (got {classified.metadata.get('status_code')})",
    )
    check(classified.metadata.get("rate_limited") is True, "rate_limited=True")
    check(classified.metadata.get("retry_after_ms") is not None, "retry delay mined from text")

    print("\nTest 9: API_KEY_INVALID is terminal even with 400")
    classified = classify_error(
        FakeLLMError(
            'GeminiException - {"reason": "API_KEY_INVALID", "message": "API key not valid."}',
            status_code=400,
        )
    )
    check(classified.metadata.get("retryable") is False, "retryable=False")
    check(
        classified.metadata.get("provider_code") == "API_KEY_INVALID",
        f"code (got {classified.metadata.get('provider_code')})",
    )
    check("Authentication" in classified.message, f"auth wording (got {classified.message!r})")

    print("\nTest 10: unknown exception - terminal, no retries")
    transport, resp = await with_transport(
        [RuntimeError("something odd")],
        max_retries=3,
    )
    check(transport.calls == 1, f"one attempt (got {transport.calls})")
    check(resp.error_metadata.get("retryable") is False, "retryable=False")
    check("something odd" in (resp.error_message or ""), "original message preserved")

    print("\nTest 11: backoff curve and jitter bounds")
    for attempt in (1, 2, 3):
        samples = [backoff_ms(attempt) for _ in range(200)]
        nominal = 500 * (2 ** (attempt - 1))
        low, high = nominal * 0.75, nominal * 1.25
        check(min(samples) >= low * 0.99, f"attempt {attempt} min >= {low:.0f}ms")
        check(max(samples) <= high * 1.01, f"attempt {attempt} max <= {high:.0f}ms")
    check(backoff_ms(20) <= 8_000 * 1.25, "cap holds at 8s (+jitter)")

    print("\nTest 12: timeout_ms=0 disables the deadline")
    transport, resp = await with_transport(
        [FakeLLMError("boom", status_code=500), ok_response("fine")],
        timeout_ms=0,
    )
    check(resp.error_message is None, "retried past a 500 with no deadline")
    check(transport.calls == 2, f"two calls (got {transport.calls})")


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 9 (retry + classification) verified.")


if __name__ == "__main__":
    run()
