"""Multi-model fallback client.

Wraps an ordered chain of LLM clients. On a terminal failure where the
fallback policy says "try the next one", re-issues the SAME request to the
next client. Drop-in for ``Agent.client``.

Tracing: ``llm.fallback`` span wraps the whole chain. Nested ``llm.generate``
spans (from each ``LlmClient.generate`` call) become children, so a trace
shows which entry answered or that all were exhausted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from pyagent.llm import LlmRequest, LlmResponse
from pyagent.provider_options import split_tool_call_id
from pyagent.schemas import ContentItem, ToolCall
from pyagent.tracing import SpanHandle, span

FallbackHook = Callable[[dict[str, object]], None]


class LlmClientProtocol(Protocol):
    async def generate(self, request: LlmRequest) -> LlmResponse: ...


@dataclass
class FallbackEntry:
    label: str
    client: LlmClientProtocol


def default_fallback_policy(response: LlmResponse) -> bool:
    """Fall back on provider/model-scoped errors, not on user or key problems.

    Retryable (advance the chain):
      - ``quota_exhausted``: free-tier quota is per-model
      - ``rate_limited``: rate limits are usually per-model
      - ``status_code == 408``: per-model timeout; a different model often
        succeeds even when one is slow.
      - ``status_code >= 500``: provider-side; another model is cheap to try.

    Terminal (do not advance):
      - ``aborted``, auth (401/403), bad requests (400/422), not-found (404),
        anything unmatched (conservative).
    """
    if response.error_message is None:
        return False
    meta = response.error_metadata

    if meta.get("aborted") is True:
        return False
    if meta.get("quota_exhausted") is True:
        return True
    if meta.get("rate_limited") is True:
        return True
    status = meta.get("status_code")
    if isinstance(status, int) and not isinstance(status, bool):
        if status == 408:
            return True
        if status >= 500:
            return True
    return False


def strip_provider_options(request: LlmRequest) -> LlmRequest:
    """Return a copy of ``request`` with both signature channels stripped."""
    new_contents: list[ContentItem] = []
    for item in request.contents:
        if isinstance(item, ToolCall):
            bare_id, _ = split_tool_call_id(item.tool_call_id)
            if item.provider_options is not None or bare_id != item.tool_call_id:
                new_contents.append(
                    item.model_copy(
                        update={
                            "tool_call_id": bare_id,
                            "provider_options": None,
                        }
                    )
                )
            else:
                new_contents.append(item)
        else:
            new_contents.append(item)
    return LlmRequest(
        instructions=list(request.instructions),
        contents=new_contents,
        tools=list(request.tools),
        tool_choice=request.tool_choice,
        mode=request.mode,
    )


@dataclass
class FallbackClient:
    chain: list[FallbackEntry]
    total_timeout_ms: int = 0
    strip_provider_options_on_switch: bool = False
    should_fallback: Callable[[LlmResponse], bool] = default_fallback_policy
    on_fallback: FallbackHook | None = None

    def __post_init__(self) -> None:
        if not self.chain:
            raise ValueError("FallbackClient: chain must not be empty")

    async def generate(self, request: LlmRequest) -> LlmResponse:
        with span(
            "llm.fallback",
            **{"fallback.chain_length": len(self.chain)},
        ) as h:
            response = await self._generate_chain(request)
            _record_fallback_span(h, response)
            return response

    async def _generate_chain(self, request: LlmRequest) -> LlmResponse:
        deadline_at: float | None = (
            time.monotonic() + self.total_timeout_ms / 1000 if self.total_timeout_ms > 0 else None
        )

        tried: list[str] = []
        last_failure: LlmResponse | None = None
        stopped_reason: str = "empty"

        for i, entry in enumerate(self.chain):
            is_first = i == 0
            effective_request = (
                strip_provider_options(request)
                if (not is_first and self.strip_provider_options_on_switch)
                else request
            )

            response = await entry.client.generate(effective_request)

            if response.error_message is None:
                stopped_reason = "success"
                if tried:
                    return response.model_copy(
                        update={
                            "error_metadata": {
                                **response.error_metadata,
                                "fallback_used": True,
                                "fallback_from": list(tried),
                                "fallback_landed_on": entry.label,
                            }
                        }
                    )
                return response

            last_failure = response

            has_next = i < len(self.chain) - 1
            if not has_next:
                stopped_reason = "exhausted"
                break

            if deadline_at is not None and time.monotonic() >= deadline_at:
                stopped_reason = "aborted"
                break

            if not self.should_fallback(response):
                stopped_reason = "policy"
                break

            next_entry = self.chain[i + 1]
            tried.append(entry.label)
            if self.on_fallback is not None:
                self.on_fallback(
                    {
                        "from": entry.label,
                        "to": next_entry.label,
                        "reason": response.error_message or "",
                        "metadata": dict(response.error_metadata),
                    }
                )

        if last_failure is not None:
            annotated = dict(last_failure.error_metadata)
            annotated["fallback_chain_length"] = len(self.chain)
            annotated["fallback_tried"] = list(tried)
            if stopped_reason == "exhausted":
                annotated["fallback_exhausted"] = True
            elif stopped_reason == "policy":
                annotated["fallback_skipped_by_policy"] = True
            elif stopped_reason == "aborted":
                annotated["fallback_aborted"] = True
            return last_failure.model_copy(update={"error_metadata": annotated})

        return LlmResponse(
            content=[],
            error_message="FallbackClient: no clients configured.",
            error_metadata={"fallback_chain_length": len(self.chain)},
            usage_metadata={},
        )


def _count_list_attr(value: object) -> int:
    """Return ``len(value)`` if it's a list, else 0.

    ``error_metadata`` values are ``object``; ``isinstance(value, list)``
    narrows to ``list[Unknown]`` under pyright strict, which rejects ``len``
    on a partially-unknown element type. Cast the item type explicitly.
    """
    if isinstance(value, list):
        return len(cast(list[object], value))
    return 0


def _record_fallback_span(h: SpanHandle, response: LlmResponse) -> None:
    """Attach final fallback attributes to the ``llm.fallback`` span."""
    meta = response.error_metadata
    if response.error_message is None:
        if meta.get("fallback_used") is True:
            h.set("fallback.used", True)
            h.set("fallback.landed_on", meta.get("fallback_landed_on"))
            h.set(
                "fallback.tried_count",
                _count_list_attr(meta.get("fallback_from")),
            )
        return
    h.fail(response.error_message)
    if meta.get("fallback_exhausted") is True:
        h.set("fallback.exhausted", True)
    if meta.get("fallback_skipped_by_policy") is True:
        h.set("fallback.skipped_by_policy", True)
    if meta.get("fallback_aborted") is True:
        h.set("fallback.aborted", True)
    h.set(
        "fallback.tried_count",
        _count_list_attr(meta.get("fallback_tried")),
    )
