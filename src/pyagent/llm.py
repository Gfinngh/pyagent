"""LLM client over LiteLLM.

Phase 11: provider-options round-trip. Two corrections vs Phase 10:

1. ``provider_options`` now mirrors LiteLLM's flat ``provider_specific_fields``
   dict verbatim. Earlier builds wrapped it as ``{"google": {...}}`` — the
   AI-SDK convention — which LiteLLM does not recognise. LiteLLM *silently
   injects a dummy signature* when the real one is missing, which caused
   multi-turn calls to succeed while losing reasoning state.
2. ``_attach_tool_call`` unwraps legacy ``{"google": {...}}`` shapes from older
   event logs, so JSONL resumes remain replayable.

Tracing: every ``generate`` call opens an ``llm.generate`` span with
``gen_ai.*`` semantic attributes (system, request model) and, on completion,
usage tokens or error metadata. When tracing is off, the ``span`` context
manager is a no-op.

Streaming / retry / classification behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from litellm import acompletion
from pydantic import BaseModel, Field

from pyagent.provider_options import unwrap_legacy_provider_fields
from pyagent.schemas import ContentItem, Message, ToolCall, ToolResult
from pyagent.tools import BaseTool, MultipartContent
from pyagent.tracing import SpanHandle, span

ToolChoiceSetting = Literal["auto", "none", "required"]
LlmMode = Literal["generate", "stream"]

RequestHook = Callable[[str, list[dict[str, Any]], dict[str, Any]], None]

_TERMINAL_PROVIDER_CODES: frozenset[str] = frozenset(
    {"insufficient_quota", "QUOTA_EXCEEDED", "API_KEY_INVALID"}
)
_SHORT_RETRY_MS = 60_000
_MAX_PROVIDER_DATA_CHARS = 4_000


@dataclass
class LlmRequest:
    """One LLM call. Fields mirror the TypeScript LlmRequest."""

    instructions: list[str] = field(default_factory=list[str])
    contents: list[ContentItem] = field(default_factory=list[ContentItem])
    tools: list[BaseTool] = field(default_factory=list[BaseTool])
    tool_choice: ToolChoiceSetting | None = None
    mode: LlmMode | None = None


class LlmResponse(BaseModel):
    """Result of an LLM call. Errors are returned in ``error_message`` - the
    agent converts them to exceptions at its boundary (Phase 6)."""

    content: list[ContentItem] = Field(default_factory=list[ContentItem])
    error_message: str | None = None
    error_metadata: dict[str, object] = Field(default_factory=dict[str, object])
    usage_metadata: dict[str, object] = Field(default_factory=dict[str, object])


@dataclass
class ErrorClassification:
    """Structured diagnosis of a failed LLM call."""

    message: str
    metadata: dict[str, object]


class LlmClient:
    """LiteLLM wrapper owning retries, deadlines, streaming, and classification."""

    model: str
    mode: LlmMode
    max_retries: int
    timeout_ms: int
    on_text_delta: Callable[[str], None] | None
    request_hook: RequestHook | None
    config: dict[str, Any]

    def __init__(
        self,
        *,
        model: str,
        mode: LlmMode = "generate",
        max_retries: int = 3,
        timeout_ms: int = 60_000,
        on_text_delta: Callable[[str], None] | None = None,
        request_hook: RequestHook | None = None,
        **config: Any,
    ) -> None:
        self.model = model
        self.mode = mode
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms
        self.on_text_delta = on_text_delta
        self.request_hook = request_hook
        self.config = dict(config)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate with retries and deadlines, wrapped in a trace span."""
        mode = request.mode or self.mode
        with span(
            "llm.generate",
            **{
                "gen_ai.system": "litellm",
                "gen_ai.request.model": self.model,
                "llm.mode": mode,
                "llm.tool_count": len(request.tools),
                "llm.tool_choice": request.tool_choice,
            },
        ) as h:
            response = await self._generate_with_retries(request, mode)
            _record_llm_response_span(h, response)
            return response

    async def _generate_with_retries(self, request: LlmRequest, mode: LlmMode) -> LlmResponse:
        deadline_at = time.monotonic() + self.timeout_ms / 1000 if self.timeout_ms > 0 else None
        total_attempts = self.max_retries + 1
        last: ErrorClassification | None = None
        attempt = 0

        while attempt < total_attempts:
            attempt += 1
            try:
                if mode == "stream":
                    return await self._attempt_streaming(request)
                return await self._attempt_generating(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                classified = classify_error(exc)
                last = classified

                if classified.metadata.get("retryable") is not True:
                    return _failure_response(classified, attempt)

                if attempt >= total_attempts:
                    break

                wait_ms = _retry_after_from(classified)
                if wait_ms is None:
                    wait_ms = backoff_ms(attempt)

                if deadline_at is not None and time.monotonic() + wait_ms / 1000 >= deadline_at:
                    return _failure_response(
                        ErrorClassification(
                            message=(
                                f"LLM request would exceed timeout "
                                f"({self.timeout_ms}ms) before retry. "
                                f"{classified.message}"
                            ),
                            metadata={
                                **classified.metadata,
                                "retry_exhausted": True,
                                "attempts": attempt,
                                "retry_reason": "timeoutBeforeRetry",
                            },
                        ),
                        attempt,
                    )

                await asyncio.sleep(wait_ms / 1000)

        return _failure_response(
            last
            or ErrorClassification(
                message="LLM request failed for an unknown reason.",
                metadata={"retryable": False},
            ),
            attempt,
            retry_exhausted=True,
        )

    # ------------------------------------------------------------------
    # Request kwargs shared by both modes
    # ------------------------------------------------------------------

    def _base_kwargs(self, request: LlmRequest) -> dict[str, Any]:
        messages = self._build_prompt(request)
        tools = self._build_tools(request.tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.config,
        }
        if tools is not None:
            kwargs["tools"] = tools
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice

        kwargs["num_retries"] = 0
        if self.timeout_ms > 0:
            kwargs.setdefault("timeout", self.timeout_ms / 1000)
        return kwargs

    def _fire_request_hook(self, kwargs: dict[str, Any]) -> None:
        if self.request_hook is None:
            return
        raw_messages = kwargs.get("messages", [])
        if not isinstance(raw_messages, list):
            return
        self.request_hook(
            self.model,
            cast(list[dict[str, Any]], raw_messages),
            kwargs,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    async def _attempt_generating(self, request: LlmRequest) -> LlmResponse:
        kwargs = self._base_kwargs(request)
        self._fire_request_hook(kwargs)
        response = await acompletion(**kwargs)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    async def _attempt_streaming(self, request: LlmRequest) -> LlmResponse:
        kwargs = self._base_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        self._fire_request_hook(kwargs)

        stream = cast(AsyncIterator[Any], await acompletion(**kwargs))

        chunks: list[Any] = []
        async for chunk in stream:
            chunks.append(chunk)
            for piece in _text_deltas(chunk):
                if self.on_text_delta is not None:
                    self.on_text_delta(piece)

        if not chunks:
            raise RuntimeError("stream produced no chunks")

        rebuilt = _rebuild_from_chunks(chunks)
        if rebuilt is not None:
            return self._parse_response(rebuilt)

        text = "".join(piece for chunk in chunks for piece in _text_deltas(chunk))
        content: list[ContentItem] = []
        if text:
            content.append(Message(role="assistant", content=text))
        return LlmResponse(
            content=content,
            error_message=None,
            error_metadata={"stream_builder_unavailable": True},
            usage_metadata=_usage_from_chunks(chunks),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, request: LlmRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": "\n\n".join(request.instructions),
                }
            )

        for item in request.contents:
            if isinstance(item, Message):
                messages.append({"role": item.role, "content": item.content})
            elif isinstance(item, ToolCall):
                self._attach_tool_call(messages, item)
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.tool_call_id,
                        "content": tool_result_output(item),
                    }
                )

        return messages

    def _attach_tool_call(
        self,
        messages: list[dict[str, Any]],
        item: ToolCall,
    ) -> None:
        tc_dict: dict[str, Any] = {
            "id": item.tool_call_id,
            "type": "function",
            "function": {
                "name": item.name,
                "arguments": json.dumps(item.arguments, default=str),
            },
        }
        if item.provider_options:
            tc_dict["provider_specific_fields"] = unwrap_legacy_provider_fields(
                item.provider_options
            )

        last = messages[-1] if messages else None
        if last is not None and last.get("role") == "assistant":
            existing = last.get("tool_calls")
            if isinstance(existing, list):
                cast(list[Any], existing).append(tc_dict)
            else:
                last["tool_calls"] = [tc_dict]
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc_dict],
                }
            )

    def _build_tools(self, tools: list[BaseTool]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        result: list[dict[str, Any]] = []
        for t in tools:
            defn = t.tool_definition
            if defn is not None:
                result.append(defn)
        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LlmResponse:
        content: list[ContentItem] = []

        choices = getattr(response, "choices", None)
        if not choices:
            return LlmResponse(
                content=[],
                error_message="provider returned no choices",
                error_metadata={"raw_type": type(response).__name__},
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            return LlmResponse(
                content=[],
                error_message="provider response missing 'message'",
                error_metadata={"raw_type": type(first_choice).__name__},
            )

        raw_text = getattr(message, "content", None)
        if isinstance(raw_text, str) and raw_text:
            content.append(Message(role="assistant", content=raw_text))
        elif raw_text:
            content.append(Message(role="assistant", content=str(raw_text)))

        raw_tool_calls: object = getattr(message, "tool_calls", None)
        for tc in _as_list(raw_tool_calls):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            raw_args = getattr(fn, "arguments", "{}") or "{}"
            try:
                parsed_args = json.loads(raw_args)
                if not isinstance(parsed_args, dict):
                    parsed_args = {"_value": parsed_args}
            except json.JSONDecodeError:
                parsed_args = {"_raw": raw_args}

            psf = getattr(tc, "provider_specific_fields", None)
            provider_options: dict[str, object] | None = None
            if isinstance(psf, dict):
                provider_options = cast(dict[str, object], psf)

            content.append(
                ToolCall(
                    tool_call_id=str(getattr(tc, "id", "")),
                    name=str(getattr(fn, "name", "")),
                    arguments=cast(dict[str, object], parsed_args),
                    provider_options=provider_options,
                )
            )

        return LlmResponse(
            content=content,
            error_message=None,
            error_metadata={},
            usage_metadata=self._extract_usage(response),
        )

    def _extract_usage(self, response: Any) -> dict[str, object]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }


# ---------------------------------------------------------------------------
# Span recording helpers
# ---------------------------------------------------------------------------


def _record_llm_response_span(h: SpanHandle, response: LlmResponse) -> None:
    """Attach final attributes to the ``llm.generate`` span."""
    if response.error_message is not None:
        h.fail(response.error_message)
        meta = response.error_metadata
        h.set("llm.error_type", meta.get("error_type"))
        h.set("llm.status_code", meta.get("status_code"))
        h.set("llm.retry_exhausted", meta.get("retry_exhausted"))
        h.set("llm.quota_exhausted", meta.get("quota_exhausted"))
        h.set("llm.rate_limited", meta.get("rate_limited"))
        h.set("llm.attempts", meta.get("attempts"))
        return
    usage = response.usage_metadata
    h.set("gen_ai.usage.input_tokens", usage.get("input_tokens"))
    h.set("gen_ai.usage.output_tokens", usage.get("output_tokens"))
    h.set("llm.content_items", len(response.content))


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _resolve_stream_chunk_builder() -> Callable[[list[Any]], Any] | None:
    try:
        import litellm as litellm_pkg
    except ImportError:
        return None

    for candidate in (
        getattr(litellm_pkg, "stream_chunk_builder", None),
        getattr(getattr(litellm_pkg, "main", None), "stream_chunk_builder", None),
    ):
        if callable(candidate):
            return cast(Callable[[list[Any]], Any], candidate)
    return None


def _rebuild_from_chunks(chunks: list[Any]) -> Any | None:
    builder = _resolve_stream_chunk_builder()
    if builder is None:
        return None
    try:
        return builder(chunks)
    except Exception:
        return None


def _text_deltas(chunk: Any) -> list[str]:
    pieces: list[str] = []
    choices = getattr(chunk, "choices", None)
    for choice in _as_list(choices):
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            pieces.append(content)
    return pieces


def _usage_from_chunks(chunks: list[Any]) -> dict[str, object]:
    for chunk in reversed(chunks):
        usage = getattr(chunk, "usage", None)
        if usage is None:
            continue
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
    return {}


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def backoff_ms(attempt: int) -> float:
    base = min(500.0 * (2 ** (attempt - 1)), 8_000.0)
    jitter = base * 0.25 * (random.random() * 2 - 1)
    return max(1.0, base + jitter)


def _retry_after_from(classified: ErrorClassification) -> float | None:
    raw = classified.metadata.get("retry_after_ms")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if raw < 0:
        return None
    return float(raw)


def _failure_response(
    classified: ErrorClassification,
    attempts: int,
    retry_exhausted: bool = False,
) -> LlmResponse:
    metadata: dict[str, object] = {**classified.metadata, "attempts": attempts}
    if retry_exhausted:
        metadata["retry_exhausted"] = True
    return LlmResponse(
        content=[],
        error_message=classified.message,
        error_metadata=metadata,
        usage_metadata={},
    )


# ---------------------------------------------------------------------------
# LiteLLM exception resolution
# ---------------------------------------------------------------------------


def _litellm_exception_map() -> dict[str, type[BaseException]]:
    result: dict[str, type[BaseException]] = {}
    try:
        import litellm.exceptions as lex
    except ImportError:
        return result
    for name in (
        "Timeout",
        "APIConnectionError",
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "ContextWindowExceededError",
    ):
        cls = getattr(lex, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            result[name] = cls
    return result


_LITELLM_EXCEPTIONS: dict[str, type[BaseException]] = _litellm_exception_map()


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_error(error: BaseException) -> ErrorClassification:
    status = _get_status_code(error)
    response = getattr(error, "response", None)
    body = _get_body(error, response)
    headers = _get_headers(error, response)
    raw_message = _get_message(error)

    retry_after_ms = _parse_retry_after(headers)
    if retry_after_ms is None:
        retry_after_ms = _parse_retry_delay(body)
    if retry_after_ms is None:
        retry_after_ms = _parse_retry_delay_from_text(raw_message)

    provider_status = _extract_provider_status(body) or _status_from_text(raw_message)
    provider_code = _extract_provider_code(body) or _code_from_text(raw_message)

    if status == 403 and (
        provider_status == "RESOURCE_EXHAUSTED"
        or "RESOURCE_EXHAUSTED" in raw_message
        or "quota" in raw_message.lower()
    ):
        status = 429

    retryable = _default_retryable(error, status)

    if provider_code in _TERMINAL_PROVIDER_CODES:
        retryable = False

    if provider_status == "RESOURCE_EXHAUSTED":
        short = retry_after_ms is not None and retry_after_ms <= _SHORT_RETRY_MS
        if not short:
            retryable = False

    is_auth_problem = (
        status in (401, 403)
        or provider_code == "API_KEY_INVALID"
        or "api key not valid" in raw_message.lower()
    )

    message = _build_message(
        status=status,
        retryable=retryable,
        retry_after_ms=retry_after_ms,
        provider_code=provider_code,
        provider_status=provider_status,
        raw_message=raw_message,
        is_auth_problem=is_auth_problem,
    )

    return ErrorClassification(
        message=message,
        metadata={
            "status_code": status,
            "provider_status": provider_status,
            "provider_code": provider_code,
            "retryable": retryable,
            "rate_limited": status == 429,
            "quota_exhausted": status == 429 and not retryable,
            "retry_after_ms": retry_after_ms,
            "error_type": type(error).__name__,
            "provider_data": _safe_provider_data(body),
        },
    )


def _default_retryable(error: BaseException, status: int | None) -> bool:
    retryable_names = ("Timeout", "APIConnectionError")
    retryable_classes = tuple(
        _LITELLM_EXCEPTIONS[n] for n in retryable_names if n in _LITELLM_EXCEPTIONS
    )
    if retryable_classes and isinstance(error, retryable_classes):
        return True

    terminal_names = (
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "ContextWindowExceededError",
    )
    terminal_classes = tuple(
        _LITELLM_EXCEPTIONS[n] for n in terminal_names if n in _LITELLM_EXCEPTIONS
    )
    if terminal_classes and isinstance(error, terminal_classes):
        return False

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return True

    if status is None:
        return False
    return status == 429 or status >= 500


def _build_message(
    *,
    status: int | None,
    retryable: bool,
    retry_after_ms: float | None,
    provider_code: str | None,
    provider_status: str | None,
    raw_message: str,
    is_auth_problem: bool,
) -> str:
    short_raw = _truncate(raw_message, 300)

    if is_auth_problem:
        return (
            f"Authentication/authorization failed ({status or 'no status'}). "
            "Check the API key and permissions."
        )

    if status == 429:
        if not retryable:
            reason = provider_code or provider_status or "quota"
            return (
                f"Provider quota exhausted (429, {reason}). Not retryable - check billing/limits."
            )
        if retry_after_ms is not None:
            return f"Rate limited by provider (429). Retry after ~{_format_ms(retry_after_ms)}."
        return "Rate limited by provider (429)."

    if status is not None and status >= 500:
        return f"Provider server error ({status})."

    if status is not None:
        return f"Provider API error ({status}): {short_raw}"

    return f"Provider API call failed: {short_raw}"


# ---------------------------------------------------------------------------
# Attribute / payload extraction
# ---------------------------------------------------------------------------


def _get_status_code(error: BaseException) -> int | None:
    raw = getattr(error, "status_code", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    response = getattr(error, "response", None)
    raw = getattr(response, "status_code", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _get_message(error: BaseException) -> str:
    raw = getattr(error, "message", None)
    if isinstance(raw, str) and raw:
        return raw
    return str(error)


def _get_headers(error: BaseException, response: object) -> dict[str, str]:
    for source in (
        getattr(error, "litellm_response_headers", None),
        getattr(response, "headers", None),
    ):
        if not source:
            continue
        try:
            return {str(k).lower(): str(v) for k, v in dict(source).items()}
        except (TypeError, ValueError):
            continue
    return {}


def _get_body(error: BaseException, response: object) -> object:
    body = getattr(error, "body", None)
    if body is not None:
        return body
    json_fn = getattr(response, "json", None)
    if callable(json_fn):
        try:
            return json_fn()
        except Exception:
            return None
    return None


def _extract_provider_status(body: object) -> str | None:
    data = _as_dict(body)
    if data is None:
        return None
    err = _as_dict(data.get("error"))
    if err is None:
        return None
    status = err.get("status")
    return status if isinstance(status, str) else None


def _extract_provider_code(body: object) -> str | None:
    data = _as_dict(body)
    if data is None:
        return None
    err = _as_dict(data.get("error"))
    if err is None:
        return None
    for key in ("code", "type", "status"):
        value = err.get(key)
        if isinstance(value, str):
            return value
    return None


def _status_from_text(text: str) -> str | None:
    return "RESOURCE_EXHAUSTED" if "RESOURCE_EXHAUSTED" in text else None


def _code_from_text(text: str) -> str | None:
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "API_KEY_INVALID"
    if "insufficient_quota" in text:
        return "insufficient_quota"
    return None


# ---------------------------------------------------------------------------
# Retry-after parsing
# ---------------------------------------------------------------------------

_RETRY_IN_TEXT = re.compile(r"retry\s+in\s+([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_IN_TEXT = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"')


def _parse_retry_after(headers: dict[str, str]) -> float | None:
    ms_raw = headers.get("retry-after-ms")
    if ms_raw is not None:
        try:
            return max(0.0, float(ms_raw))
        except ValueError:
            pass

    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw) * 1000)
    except ValueError:
        pass

    parsed = _parse_http_date(raw)
    if parsed is not None:
        return max(0.0, (parsed - time.time()) * 1000)
    return None


def _parse_http_date(raw: str) -> float | None:
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(raw).timestamp()
    except (TypeError, ValueError):
        return None


def _parse_retry_delay(body: object) -> float | None:
    data = _as_dict(body)
    if data is None:
        return None
    err = _as_dict(data.get("error"))
    if err is None:
        return None
    details = err.get("details")
    if not isinstance(details, list):
        return None
    for entry in cast(list[Any], details):
        detail = _as_dict(entry)
        if detail is None:
            continue
        delay = detail.get("retryDelay")
        if isinstance(delay, str):
            parsed = _parse_seconds_string(delay)
            if parsed is not None:
                return parsed
    return None


def _parse_retry_delay_from_text(text: str) -> float | None:
    for pattern in (_RETRY_DELAY_IN_TEXT, _RETRY_IN_TEXT):
        match = pattern.search(text)
        if match:
            try:
                return max(0.0, float(match.group(1)) * 1000)
            except ValueError:
                continue
    return None


def _parse_seconds_string(raw: str) -> float | None:
    match = re.match(r"^([\d.]+)s$", raw.strip())
    if not match:
        return None
    try:
        return max(0.0, float(match.group(1)) * 1000)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(cast(list[Any], value))
    return []


def _safe_provider_data(body: object) -> object:
    if body is None:
        return None
    try:
        serialized = json.dumps(body, default=str)
    except (TypeError, ValueError):
        return str(body)
    if len(serialized) > _MAX_PROVIDER_DATA_CHARS:
        return {"truncated": True, "preview": serialized[:_MAX_PROVIDER_DATA_CHARS]}
    return body


def _format_ms(ms: float) -> str:
    if ms < 1000:
        return f"{round(ms)}ms"
    return f"{ms / 1000:.1f}s"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"


# ---------------------------------------------------------------------------
# Tool-result normalization
# ---------------------------------------------------------------------------


def tool_result_output(item: ToolResult) -> str:
    values = item.content

    if len(values) == 1 and isinstance(values[0], MultipartContent):
        return _flatten_multipart(values[0])

    if len(values) <= 1:
        return _stringify(values[0]) if values else ""

    parts: list[str] = []
    for v in values:
        s = _stringify(v)
        if s:
            parts.append(s)
    return "\n".join(parts)


def _flatten_multipart(mc: MultipartContent) -> str:
    parts: list[str] = []
    for part in mc.parts:
        if part.type == "text":
            parts.append(part.text)
        elif part.type == "image":
            parts.append(f"[image: {part.mime_type}, {len(part.data)} bytes base64]")
        elif part.type == "resource":
            header = f"[Resource: {part.uri}"
            if part.mime_type:
                header += f" ({part.mime_type})"
            header += "]"
            if part.text is not None:
                body = part.text
            elif part.blob is not None:
                body = f"<binary blob omitted - {len(part.blob)} base64 chars>"
            else:
                body = "<empty resource>"
            parts.append(f"{header}\n{body}")
        elif part.type == "structured":
            parts.append(json.dumps(part.value, default=str, indent=2))
    return "\n".join(parts)


def _stringify(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, MultipartContent):
        return _flatten_multipart(v)
    try:
        return json.dumps(v, default=str)
    except TypeError:
        return str(v)
