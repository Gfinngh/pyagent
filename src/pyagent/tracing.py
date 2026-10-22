"""Tracing facade over OpenTelemetry.

Call sites deal with a small, typed surface (``span()`` yielding a
``SpanHandle`` with ``set()``, ``error()``, ``fail()`` methods) and never
import from ``opentelemetry`` directly. This isolates pyright from OTel's
loose typing and keeps the instrumentation readable.

Modes
-----
- **Disabled** (default): ``init_tracing()`` returns ``None`` and every
  ``span()`` call is a no-op.
- **OTLP exporter**: set ``OTEL_EXPORTER_OTLP_ENDPOINT`` +
  ``OTEL_EXPORTER_OTLP_HEADERS`` (standard OTel env vars), *or*
  ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` + ``LANGFUSE_HOST``
  (derived to Langfuse's OTLP endpoint with Basic Auth).
- **In-memory** (tests): ``init_tracing(in_memory=True)`` swaps in an
  ``InMemorySpanExporter`` that records spans in process for assertions.

Init is explicit and idempotent. Call ``init_tracing()`` once at program
start (``main.py``, each workflow's ``main()``).

Provider ownership
------------------
``span()`` obtains its tracer from ``_provider.get_tracer(...)`` — our own
reference to the active provider. It does NOT go through
``otel_trace.get_tracer()``, which resolves through the OTel global. OTel
allows ``set_tracer_provider`` only *once per process*; a second call is
silently rejected with a warning, and the global keeps returning tracers
from the old (potentially shut-down) provider. In the typical app lifecycle
init runs once so both approaches agree; but tests that shutdown and re-init
would hit stale tracers unless we bypass the global, which is what this
module does.

``otel_trace.set_tracer_provider`` is called once per process (guarded by
``_global_set``) so third-party code that resolves the tracer through OTel's
global still cooperates. Repeated init does not re-call it.

Module state lives under lowercase names (``_provider``, ``_initialized``,
``_in_memory_exporter``, ``_global_set``). Pyright treats uppercase
module-level names as constants and rejects reassignment; lowercase globals
can be mutated from functions with ``global``.

``SpanHandle`` is public because downstream helpers (``llm._record_*_span``,
``fallback._record_*_span``) type their arguments with it. Call sites that
only use ``with span(...) as h:`` never need to reference the type directly.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode

_TRACER_NAME: str = "pyagent"

_initialized: bool = False
_provider: TracerProvider | None = None
_in_memory_exporter: Any | None = None
_global_set: bool = False


# ---------------------------------------------------------------------------
# Init / shutdown
# ---------------------------------------------------------------------------


def init_tracing(
    *, in_memory: bool = False, service_name: str = "pyagent"
) -> TracerProvider | None:
    """Initialize tracing. Idempotent.

    Returns the active ``TracerProvider``, or ``None`` if tracing is disabled
    (no OTLP endpoint configured and ``in_memory=False``). Safe to call
    multiple times; the first call wins.

    If a prior ``shutdown_tracing`` ran, a fresh provider is created. The OTel
    global tracer provider is set only on the *first* init of the process; a
    subsequent re-init skips that global call (which OTel would reject) and
    our own ``span()`` uses ``_provider`` directly anyway.
    """
    global _initialized, _provider, _in_memory_exporter, _global_set

    if _initialized:
        return _provider

    processor: Any
    if in_memory:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        in_mem: InMemorySpanExporter = InMemorySpanExporter()
        _in_memory_exporter = in_mem
        processor = SimpleSpanProcessor(in_mem)
    else:
        env_exporter = _build_exporter_from_env()
        if env_exporter is None:
            _initialized = True
            return None
        processor = BatchSpanProcessor(env_exporter)

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    provider.add_span_processor(processor)

    # OTel's global tracer provider can be set exactly once per process. On
    # subsequent re-inits, keep our module-level ``_provider`` current (which
    # ``span()`` uses directly) and skip the global call.
    if not _global_set:
        try:
            otel_trace.set_tracer_provider(provider)
            _global_set = True
        except Exception:
            # Best-effort; ``span()`` doesn't depend on the global.
            pass

    _provider = provider
    _initialized = True
    return provider


def shutdown_tracing() -> None:
    """Flush and shut down the current provider; return to no-op mode.

    The in-memory exporter (if any) is retained for post-shutdown inspection,
    but ``span()`` calls stop producing new spans.
    """
    global _initialized, _provider

    if _provider is not None:
        _provider.shutdown()
        _provider = None
    _initialized = False


def in_memory_spans() -> list[ReadableSpan]:
    """Return recorded spans when ``init_tracing(in_memory=True)`` was used.

    Empty list otherwise. Call this after the span's ``with`` block exits;
    in-memory mode uses ``SimpleSpanProcessor`` so spans appear synchronously.
    """
    if _in_memory_exporter is None:
        return []
    return list(_in_memory_exporter.get_finished_spans())


# ---------------------------------------------------------------------------
# Span handle
# ---------------------------------------------------------------------------


@dataclass
class SpanHandle:
    """Thin wrapper over ``Span`` whose methods are safe when tracing is off."""

    span: Span | None

    def set(self, key: str, value: object) -> None:
        """Set a span attribute. No-op when tracing is disabled."""
        if self.span is None or value is None:
            return
        self.span.set_attribute(key, _coerce(value))

    def error(self, exc: BaseException) -> None:
        """Record the exception and mark the span as errored."""
        if self.span is None:
            return
        self.span.record_exception(exc)
        self.span.set_status(Status(StatusCode.ERROR, str(exc)[:300]))

    def fail(self, message: str) -> None:
        """Mark the span as errored without an exception object.

        For code paths that *return* an error rather than raise one — e.g.
        ``LlmClient.generate`` puts failures into ``LlmResponse.error_message``
        instead of raising, so the tracing wrapper must record failure status
        explicitly.
        """
        if self.span is None:
            return
        self.span.set_status(Status(StatusCode.ERROR, message[:300]))


@contextmanager
def span(name: str, **attrs: object) -> Generator[SpanHandle]:
    """Create a child of the current active span.

    When tracing is disabled (``init_tracing`` never called), yields a handle
    wrapping ``None`` — every method is a no-op and the context manager still
    works. Call sites never need ``if handle is not None``.

    The tracer is obtained from the module's ``_provider`` directly rather
    than through ``otel_trace.get_tracer()``. The global lookup returns stale
    tracers after shutdown + re-init (OTel only allows a single global
    provider per process); using ``_provider`` keeps spans flowing to the
    live exporter across re-initializations.
    """
    if _provider is None:
        yield SpanHandle(None)
        return

    tracer = _provider.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is None:
                continue
            s.set_attribute(k, _coerce(v))
        yield SpanHandle(s)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce(v: object) -> Any:
    """Coerce to an OTel-acceptable attribute value (str/int/float/bool)."""
    if isinstance(v, (str, bool, int, float)):
        return v
    return str(v)


def _build_exporter_from_env() -> SpanExporter | None:
    """Build an OTLP/HTTP exporter from env vars, or return None.

    Supports standard OTel env vars first, then Langfuse env vars as a
    convenience (deriving the Langfuse OTLP endpoint and Basic Auth header).
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")

    if endpoint is None:
        public = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret = os.environ.get("LANGFUSE_SECRET_KEY")
        if public and secret:
            host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            endpoint = f"{host.rstrip('/')}/api/public/otel"
            encoded = base64.b64encode(f"{public}:{secret}".encode()).decode()
            headers = f"Authorization=Basic {encoded}"

    if endpoint is None:
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    traces_endpoint = f"{endpoint.rstrip('/')}/v1/traces"
    return OTLPSpanExporter(
        endpoint=traces_endpoint,
        headers=_parse_headers(headers),
    )


def _parse_headers(raw: str) -> dict[str, str] | None:
    """Parse ``OTEL_EXPORTER_OTLP_HEADERS`` format: ``k1=v1,k2=v2``."""
    if not raw:
        return None
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        k = key.strip()
        if k:
            out[k] = value.strip()
    return out or None
