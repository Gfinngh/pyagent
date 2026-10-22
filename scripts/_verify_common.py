"""Shared verification helpers.

Underscore prefix keeps this out of the ``verify-*`` task list. Imported by
other scripts.

Two variants because verify scripts fall into two shapes:

- ``is_transient_subprocess_failure(proc)`` for scripts that run the workflow
  or main.py as a subprocess and inspect ``proc.returncode`` and stderr.
- ``is_transient_llm_response(error_message, metadata)`` for scripts that call
  ``Agent.run`` / ``LlmClient.generate`` directly and catch ``LlmCallError``.

Both answer the same question: "did the provider flake, or did our code
break?". Transient failures are distinguishable by their specific markers -
408 timeouts, 5xx server errors, and retry exhaustion after the client's own
retry loop. A real code bug (schema violation, prompt injection, tool
mismatch) produces different error messages.

``terminal_error_line`` pulls the actionable ``Agent failed:`` line from a
subprocess stderr so skip messages show the terminal failure, not the
fallback chain's intermediate warnings.
"""

from __future__ import annotations

import re
import subprocess

_TRANSIENT_STATUS_RE = re.compile(r"\b(?:408|5\d\d)\b")

_TRANSIENT_STDERR_MARKERS: tuple[str, ...] = (
    "retry_exhausted",
    "connection timed out",
    "litellm.timeout",
    "timeout passed",
    "would exceed timeout",
    "provider server error",
    "serviceunavailableerror",
    "internal server error",
)


def is_transient_subprocess_failure(
    proc: subprocess.CompletedProcess[str],
) -> bool:
    """Detect provider-side flakes from a subprocess run's stderr.

    Return ``True`` when the subprocess exited 1 AND its stderr contains an
    LLM-layer transient marker. A workflow's own error (missing file, schema
    validation, permission denial) exits 1 too but does not produce these
    markers, so it will not be skipped.
    """
    if proc.returncode != 1:
        return False
    s = proc.stderr.lower()
    if "agent failed:" not in s:
        return False
    for marker in _TRANSIENT_STDERR_MARKERS:
        if marker in s:
            return True
    return bool(_TRANSIENT_STATUS_RE.search(s))


def is_transient_llm_response(
    error_message: str | None,
    metadata: dict[str, object],
) -> bool:
    """Detect provider-side flakes from a direct LLM call's error metadata.

    Same rules as ``is_transient_subprocess_failure`` but operating on the
    structured data a caught ``LlmCallError`` carries. ``aborted`` is
    explicitly terminal - a caller's own cancellation should not be silently
    skipped as a provider flake.
    """
    if error_message is None:
        return False
    if metadata.get("aborted") is True:
        return False
    if metadata.get("retry_exhausted") is True:
        return True
    status = metadata.get("status_code")
    return bool(
        isinstance(status, int)
        and not isinstance(status, bool)
        and (status == 408 or status >= 500)
    )


def terminal_error_line(stderr: str, limit: int = 400) -> str:
    """Extract the ``Agent failed:`` line from a subprocess stderr.

    main.py and the workflows print fallback-chain warnings first, then the
    terminal failure. Truncating stderr from the head therefore loses the
    actionable part. This helper finds the last ``agent failed:`` line
    (case-insensitive) and returns it trimmed to ``limit``. Falls back to the
    tail of stderr when no such line exists - defensive against a subprocess
    that died before the agent ran.
    """
    for line in reversed(stderr.strip().splitlines()):
        if "agent failed:" in line.lower():
            return line[:limit]
    return stderr.strip()[-limit:]
