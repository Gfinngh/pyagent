"""Append-only JSONL event log.

One JSON object per line, appended and never rewritten: cheap to write, cheap
to grep, safe to tail from another process. The log is the durable form of an
:class:`~pyagent.schemas.AgentEvent` stream, so a crashed run can be replayed.

Write safety
------------
``O_APPEND`` writes are atomic only up to ``PIPE_BUF`` (~4 KiB on Linux/macOS).
An ``AgentEvent`` carrying a Gemini thought signature is ~0.5-1.5 KiB, so a
single ``write()`` of ``json.dumps(event) + "\\n"`` is one syscall and stays
under the limit. A ``threading.Lock`` serialises concurrent appends from the
same process. Both together mean a line is never interleaved mid-write.

``flush()`` always runs so another process tailing the file sees the event.
``fsync()`` is opt-in (``fsync=True``): it costs ~1-10 ms per event on SSD,
which is wasted when the only consumer is the same process. Turn it on when a
crash must not lose the last event.

Read tolerance
--------------
- Blank lines are skipped.
- An unterminated *final* line is a crashed writer: the event was never fully
  written, so it is reported to stderr and skipped.
- A malformed line anywhere else raises with its line number. Silent skipping
  would hide real corruption on the resume path.

Lazy open
---------
The file is created on the first ``append``, not at construction. ``close()``
releases the handle; a later ``append`` reopens it. This matches the TypeScript
``JsonlEventLog`` and makes the log usable across a signal-driven shutdown.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Protocol, TextIO

from pyagent.schemas import AgentEvent


class EventLog(Protocol):
    """Structural protocol for event persistence sinks.

    Concrete implementations (``JsonlEventLog``, in-memory fakes,
    ``NullEventLog``) only need ``append`` and ``close``.
    """

    def append(self, event: AgentEvent) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# No-op sink
# ---------------------------------------------------------------------------


class NullEventLog:
    """Sink that discards every event. Useful to opt out conditionally."""

    def append(self, event: AgentEvent) -> None:
        _ = event

    def close(self) -> None:
        return None

    def __enter__(self) -> NullEventLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ---------------------------------------------------------------------------
# JSONL sink
# ---------------------------------------------------------------------------


class JsonlEventLog:
    """Append-only JSONL writer for :class:`AgentEvent` streams.

    Each event is serialised as ``event.model_dump_json()`` plus a newline and
    written in a single ``write()`` call under a per-instance lock. The file is
    opened lazily on the first ``append`` and reopened transparently after a
    ``close``.
    """

    path: Path
    fsync: bool
    _file: TextIO | None
    _lock: threading.Lock

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._file = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _ensure_open(self) -> TextIO:
        if self._file is None:
            # `newline="\n"` stops Python translating "\n" to "\r\n" on
            # Windows, keeping one record per line on every platform.
            self._file = self.path.open("a", encoding="utf-8", newline="\n")
        return self._file

    def append(self, event: AgentEvent) -> None:
        payload = event.model_dump_json() + "\n"
        with self._lock:
            fh = self._ensure_open()
            fh.write(payload)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> JsonlEventLog:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_event_log(path: str | Path) -> list[AgentEvent]:
    """Read a JSONL event log back into validated ``AgentEvent`` objects.

    Streams the file line by line rather than loading it whole, so a long log
    does not have to fit in memory twice.

    An unterminated final line is treated as a truncated write: it was never
    completed, so it is skipped with a stderr warning. Any other malformed line
    raises ``ValueError`` naming the file and line number - the resume path
    must not silently lose events.
    """
    p = Path(path)
    events: list[AgentEvent] = []

    with p.open("r", encoding="utf-8", newline="") as fh:
        for line_no, raw in enumerate(fh, start=1):
            terminated = raw.endswith("\n")
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                payload: object = json.loads(line)
                events.append(AgentEvent.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                if not terminated:
                    print(
                        f"[event_log] skipping truncated final line {line_no} of {p}: {exc}",
                        file=sys.stderr,
                    )
                    break
                raise ValueError(f"Event log {p}: invalid entry on line {line_no}: {exc}") from exc

    return events


def iter_event_log(path: str | Path) -> Iterator[AgentEvent]:
    """Lazy variant of :func:`read_event_log` for very large logs.

    Yields events as they are parsed. Same tolerance rules: blank lines are
    skipped, an unterminated final line is dropped with a warning, and any
    other malformed line raises.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as fh:
        for line_no, raw in enumerate(fh, start=1):
            terminated = raw.endswith("\n")
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                payload: object = json.loads(line)
                yield AgentEvent.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                if not terminated:
                    print(
                        f"[event_log] skipping truncated final line {line_no} of {p}: {exc}",
                        file=sys.stderr,
                    )
                    return
                raise ValueError(f"Event log {p}: invalid entry on line {line_no}: {exc}") from exc
