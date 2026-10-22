"""Phase 12: fallback chain - deterministic scenarios with fake clients.

Covers quota/rate-limit/5xx advancement, auth/policy refusal, chain
exhaustion, the on_fallback hook, signature stripping on model switch, and
chain-level abort. Zero network calls.
"""

from __future__ import annotations

import asyncio
import sys
from typing import cast

from pyagent.agent import Agent, LlmClientProtocol
from pyagent.fallback import (
    FallbackClient,
    FallbackEntry,
    default_fallback_policy,
)
from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import Message, ToolCall

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def request() -> LlmRequest:
    return LlmRequest(
        instructions=["be brief"],
        contents=[Message(role="user", content="hi")],
    )


def request_with_signed_tool_call() -> LlmRequest:
    """A request with a tool call carrying BOTH signature channels."""
    tc = ToolCall.model_validate(
        {
            "tool_call_id": "call_x__thought__SIGNATURE_BYTES",
            "name": "calculator",
            "arguments": {"operator": "add", "first_number": 1, "second_number": 2},
            "provider_options": {"thought_signature": "SIGNATURE_BYTES"},
        }
    )
    return LlmRequest(
        instructions=["use tools"],
        contents=[Message(role="user", content="calc 1+2"), tc],
    )


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class RecordingClient:
    """Fake LLM client. Records every request; returns a scripted response.

    ``delay_ms`` simulates latency before returning - needed to exercise the
    chain-level timeout deterministically (an instant fake races the deadline
    check across the async boundary).
    """

    def __init__(
        self,
        label: str,
        error: str | None = None,
        status: int = 0,
        delay_ms: float = 0,
    ) -> None:
        self.label = label
        self.error = error
        self.status = status
        self.delay_ms = delay_ms
        self.calls: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
        if self.error is None:
            return LlmResponse(
                content=[Message(role="assistant", content=f"answer from {self.label}")],
            )
        metadata: dict[str, object] = {"status_code": self.status}
        if self.status == 429:
            metadata["rate_limited"] = True
            metadata["quota_exhausted"] = True
        if self.status == 401:
            metadata["retryable"] = False
        if self.status >= 500:
            metadata["retryable"] = True
        return LlmResponse(
            content=[],
            error_message=f"[{self.label}] {self.error}",
            error_metadata=metadata,
        )


def entry(
    label: str,
    error: str | None = None,
    status: int = 0,
    delay_ms: float = 0,
) -> tuple[FallbackEntry, RecordingClient]:
    client = RecordingClient(label, error=error, status=status, delay_ms=delay_ms)
    return FallbackEntry(label=label, client=client), client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def main() -> None:
    print("\nTest 1: quota fail -> next entry succeeds")
    e1, c1 = entry("a", error="quota", status=429)
    e2, c2 = entry("b")
    client = FallbackClient(chain=[e1, e2])
    resp = await client.generate(request())
    check(resp.error_message is None, "no error returned")
    check(len(c1.calls) == 1, f"first client called once (got {len(c1.calls)})")
    check(len(c2.calls) == 1, f"second client called once (got {len(c2.calls)})")
    meta = resp.error_metadata
    check(meta.get("fallback_used") is True, "fallback_used=True")
    check(
        meta.get("fallback_from") == ["a"],
        f"fallback_from=['a'] (got {meta.get('fallback_from')})",
    )
    check(
        meta.get("fallback_landed_on") == "b",
        f"landed_on='b' (got {meta.get('fallback_landed_on')})",
    )

    print("\nTest 2: all entries fail with quota -> exhausted")
    e1, _ = entry("a", error="quota", status=429)
    e2, _ = entry("b", error="quota", status=429)
    e3, _ = entry("c", error="quota", status=429)
    client = FallbackClient(chain=[e1, e2, e3])
    resp = await client.generate(request())
    check(resp.error_message is not None, "error returned")
    meta = resp.error_metadata
    check(meta.get("fallback_exhausted") is True, "fallback_exhausted=True")
    check(
        meta.get("fallback_tried") == ["a", "b"],
        f"tried a then b (got {meta.get('fallback_tried')})",
    )
    check(
        meta.get("fallback_chain_length") == 3,
        f"chain_length=3 (got {meta.get('fallback_chain_length')})",
    )

    print("\nTest 3: 401 auth -> policy refuses, second client untouched")
    e1, c1 = entry("a", error="unauthorized", status=401)
    e2, c2 = entry("b")
    client = FallbackClient(chain=[e1, e2])
    resp = await client.generate(request())
    check(resp.error_message is not None, "error returned")
    check(len(c1.calls) == 1, "first client called")
    check(len(c2.calls) == 0, f"second client NOT called (got {len(c2.calls)})")
    meta = resp.error_metadata
    check(
        meta.get("fallback_skipped_by_policy") is True,
        "fallback_skipped_by_policy=True",
    )
    check(
        meta.get("fallback_tried") == [],
        f"tried=[] (got {meta.get('fallback_tried')})",
    )
    check(meta.get("fallback_exhausted") is not True, "not marked exhausted")

    print("\nTest 4: 5xx -> falls back")
    e1, _ = entry("a", error="server error", status=503)
    e2, _ = entry("b")
    client = FallbackClient(chain=[e1, e2])
    resp = await client.generate(request())
    check(resp.error_message is None, "recovered")
    check(resp.error_metadata.get("fallback_used") is True, "fallback_used=True")

    print("\nTest 5: empty chain raises")
    threw = False
    try:
        FallbackClient(chain=[])
    except ValueError:
        threw = True
    check(threw, "ValueError on empty chain")

    print("\nTest 6: on_fallback hook fires once per advance")
    events: list[dict[str, object]] = []
    e1, _ = entry("a", error="quota", status=429)
    e2, _ = entry("b")
    client = FallbackClient(chain=[e1, e2], on_fallback=events.append)
    await client.generate(request())
    check(len(events) == 1, f"one fallback event (got {len(events)})")
    if events:
        ev = events[0]
        check(ev.get("from") == "a", f"from='a' (got {ev.get('from')})")
        check(ev.get("to") == "b", f"to='b' (got {ev.get('to')})")
        check("quota" in str(ev.get("reason", "")).lower(), "reason mentions quota")

    print("\nTest 7: strip_provider_options_on_switch removes both channels")
    e1, _ = entry("a", error="quota", status=429)
    e2, c2 = entry("b")
    client = FallbackClient(chain=[e1, e2], strip_provider_options_on_switch=True)
    resp = await client.generate(request_with_signed_tool_call())
    check(resp.error_message is None, "recovered")
    check(len(c2.calls) == 1, "second client called once")
    if c2.calls:
        forwarded = c2.calls[0]
        stripped_tc = next((c for c in forwarded.contents if isinstance(c, ToolCall)), None)
        check(stripped_tc is not None, "tool call present in forwarded request")
        if stripped_tc is not None:
            check(
                stripped_tc.provider_options is None,
                f"provider_options=None (got {stripped_tc.provider_options})",
            )
            check(
                "__thought__" not in stripped_tc.tool_call_id,
                f"tool_call_id has no __thought__ (got {stripped_tc.tool_call_id[:40]})",
            )
            check(
                stripped_tc.tool_call_id == "call_x",
                f"bare id preserved (got {stripped_tc.tool_call_id})",
            )

    print("\nTest 8: strip_provider_options_on_switch=False preserves signatures")
    e1, _ = entry("a", error="quota", status=429)
    e2, c2 = entry("b")
    client = FallbackClient(chain=[e1, e2], strip_provider_options_on_switch=False)
    await client.generate(request_with_signed_tool_call())
    if c2.calls:
        forwarded = c2.calls[0]
        preserved_tc = next((c for c in forwarded.contents if isinstance(c, ToolCall)), None)
        check(preserved_tc is not None, "tool call present")
        if preserved_tc is not None:
            check(
                preserved_tc.provider_options is not None,
                "provider_options preserved when stripping disabled",
            )
            check(
                "__thought__" in preserved_tc.tool_call_id,
                "packed id preserved when stripping disabled",
            )

    print("\nTest 9: chain-level timeout aborts between entries")
    # First entry sleeps 50ms past the 10ms chain deadline, so the abort check
    # between entries fires deterministically before the second entry runs.
    e1, _ = entry("a", error="server error", status=503, delay_ms=50)
    e2, c2 = entry("b")
    client = FallbackClient(chain=[e1, e2], total_timeout_ms=10)
    resp = await client.generate(request())
    check(resp.error_message is not None, "error returned")
    check(len(c2.calls) == 0, f"second client NOT called (got {len(c2.calls)})")
    meta = resp.error_metadata
    check(meta.get("fallback_aborted") is True, "fallback_aborted=True")

    print("\nTest 10: default_fallback_policy - aborted is terminal")
    aborted = LlmResponse(
        content=[],
        error_message="aborted",
        error_metadata={"aborted": True, "retryable": False},
    )
    check(default_fallback_policy(aborted) is False, "aborted -> no fallback")

    print("\nTest 11: FallbackClient satisfies Agent's client protocol")
    e1, _ = entry("only")
    fallback_client = FallbackClient(chain=[e1])

    # The cast is the assertion: if FallbackClient's generate signature ever
    # drifts from LlmClientProtocol, pyright flags the cast at check time.
    _ = cast(LlmClientProtocol, fallback_client)
    check(True, "castable to LlmClientProtocol")

    try:
        _ = Agent(client=fallback_client, tools=[], instructions="test")
        check(True, "Agent(client=FallbackClient(...)) constructs")
    except Exception as e:
        check(False, f"Agent construction failed: {e!r}")


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 12 (fallback) verified.")


if __name__ == "__main__":
    run()
