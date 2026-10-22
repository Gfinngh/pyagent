"""Provider-options (Gemini thought signature) helpers.

Two channels LiteLLM uses for the signature:
  1. ``tool_calls[].provider_specific_fields.thought_signature`` - flat dict
  2. ``tool_calls[].id`` with ``__thought__<base64 signature>`` appended
     (ids run 843-4423 chars; normalising the id corrupts the signature)

``RequestCapture`` plugs into ``LlmClient(request_hook=...)`` to snapshot every
outgoing request right before ``acompletion``. This is the ground truth for
"did our replay actually carry the signature?".

Why not a LiteLLM ``CustomLogger``: LiteLLM rejects duck-typed callbacks
(``_CallbackNotDispatchable``) and drops async callbacks in short-lived scripts
(BerriAI/litellm#16486). Capturing at our own boundary is deterministic and
sees exactly the messages/kwargs we hand to LiteLLM.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
from typing import Any, cast

_THOUGHT_SEPARATOR = "__thought__"
_DUMMY_SENTINEL = b"skip_thought_signature_validator"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def _snapshot(value: object) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


# ---------------------------------------------------------------------------
# Signature inspection
# ---------------------------------------------------------------------------


def split_tool_call_id(tool_call_id: str) -> tuple[str, str | None]:
    """Split a LiteLLM tool-call id into (bare_id, embedded_signature)."""
    if _THOUGHT_SEPARATOR in tool_call_id:
        bare, _, sig = tool_call_id.partition(_THOUGHT_SEPARATOR)
        return bare, sig or None
    return tool_call_id, None


def signature_from_provider_fields(psf: object) -> str | None:
    """Extract ``thought_signature`` from a ``provider_specific_fields`` value."""
    typed = _as_dict(psf)
    if typed is None:
        return None
    value = typed.get("thought_signature")
    return value if isinstance(value, str) and value else None


def signature_from_tool_call(tc: object) -> tuple[str | None, str | None]:
    """Return ``(from_provider_fields, from_id)`` for one tool call.

    Either side may be ``None``; LiteLLM populates at least one for Gemini 3.
    """
    psf_sig = signature_from_provider_fields(getattr(tc, "provider_specific_fields", None))
    tc_id = getattr(tc, "id", "") or ""
    _, id_sig = split_tool_call_id(str(tc_id))
    return psf_sig, id_sig


def is_dummy_signature(signature: str | None) -> bool:
    """True if the value is LiteLLM's ``skip_thought_signature_validator``.

    That sentinel is injected whenever a real signature is missing. Accepting
    it as proof of round-trip is how silent degradation hides.
    """
    if not signature:
        return False
    if signature == base64.b64encode(_DUMMY_SENTINEL).decode("ascii"):
        return True
    # Tolerate URL-safe / unpadded variants.
    try:
        padded = signature + "=" * (-len(signature) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return False
    return _DUMMY_SENTINEL in decoded


# ---------------------------------------------------------------------------
# Replay-shape helpers
# ---------------------------------------------------------------------------


def unwrap_legacy_provider_fields(
    provider_options: dict[str, object],
) -> dict[str, object]:
    """Return the flat ``provider_specific_fields`` dict LiteLLM expects.

    Legacy event logs (written before Phase 11) wrapped the flat dict in
    ``{"google": {...}}`` following the AI-SDK convention. Unwrap that so a
    resumed log still replays correctly. Everything else passes through
    unchanged.
    """
    if set(provider_options) == {"google"}:
        inner = provider_options.get("google")
        if isinstance(inner, dict):
            return cast(dict[str, object], inner)
    return provider_options


def find_replayed_tool_call(
    messages: list[dict[str, Any]],
    tool_call_id: str,
) -> dict[str, Any] | None:
    """Locate an assistant tool-call entry by id in a messages array."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        raw_calls = msg.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for entry in cast(list[Any], raw_calls):
            entry_dict = _as_dict(entry)
            if entry_dict is not None and entry_dict.get("id") == tool_call_id:
                return entry_dict
    return None


# ---------------------------------------------------------------------------
# Request capture (used by verification scripts)
# ---------------------------------------------------------------------------


@dataclass
class RequestCapture:
    """Snapshot every outgoing request handed to ``acompletion``.

    Plug into ``LlmClient(request_hook=capture)``. Fires synchronously once per
    attempt, right before the LLM call. No dependency on LiteLLM's own callback
    dispatch - which rejects duck-typed handlers and drops async callbacks in
    short-lived scripts (BerriAI/litellm#16486).

    Attributes:
        calls: One entry per attempt, each shaped
            ``{"model": str, "messages": [...], "kwargs": {...}, "stream": bool}``.
    """

    calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def __call__(
        self,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> None:
        self.calls.append(
            {
                "model": model,
                "messages": _snapshot(messages),
                "kwargs": _snapshot({k: v for k, v in kwargs.items() if k != "messages"}),
                "stream": bool(kwargs.get("stream", False)),
            }
        )

    def reset(self) -> None:
        self.calls.clear()

    def last(self) -> dict[str, Any] | None:
        return self.calls[-1] if self.calls else None
