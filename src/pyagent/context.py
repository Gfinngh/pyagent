"""Execution context: central storage for a single agent run.

The ``events`` list IS the conversation: flattening every event's content
produces the flat conversation sent to the LLM. No separate message history
to keep in sync.

``ExecutionContext.from_log(path)`` rehydrates a context from a JSONL event
log. ``current_step`` and ``final_result`` are *derived* from the events rather
than persisted alongside them, which removes any chance of the counter
disagreeing with the history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from pyagent.event_log import EventLog, read_event_log
from pyagent.schemas import AgentEvent, Message, ToolCall, ToolResult

__all__ = ["EventLog", "ExecutionContext"]


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[AgentEvent] = field(default_factory=list[AgentEvent])
    current_step: int = 0
    state: dict[str, object] = field(default_factory=dict[str, object])
    final_result: str | dict[str, object] | None = None
    log: EventLog | None = None

    def add_event(self, event: AgentEvent) -> None:
        """Append to the in-memory list and forward to the log if set."""
        self.events.append(event)
        if self.log is not None:
            self.log.append(event)

    def increment_step(self) -> None:
        self.current_step += 1

    @classmethod
    def from_log(
        cls,
        path: str | Path,
        *,
        output_tool_name: str | None = None,
    ) -> ExecutionContext:
        """Rehydrate a context from a JSONL event log.

        ``current_step`` counts agent events that carry an assistant message or
        a tool call - exactly what ``Agent.step`` emits per iteration.
        ``final_result`` is extracted from the last event if that event looks
        like a final answer; otherwise it stays ``None``, which is how
        ``Agent.resume`` knows the run was interrupted.

        Args:
            path: JSONL file written by :class:`JsonlEventLog`.
            output_tool_name: ``"final_answer"`` when the original run used
                structured output. Without it, a completed structured run
                would look like an unfinished free-form one.
        """
        events = read_event_log(path)
        if not events:
            return cls()

        return cls(
            execution_id=events[0].execution_id,
            events=events,
            current_step=_count_steps(events),
            final_result=_detect_final_result(events, output_tool_name),
        )


# ---------------------------------------------------------------------------
# Derivation helpers (mirror Agent.step / _is_final_response)
# ---------------------------------------------------------------------------


def _count_steps(events: list[AgentEvent]) -> int:
    """Count events an ``Agent.step`` would have produced.

    One step yields one response event holding an assistant message, tool
    calls, or both. The follow-up tool-result event is part of the same step,
    so it is not counted again.
    """
    count = 0
    for event in events:
        for item in event.content:
            if isinstance(item, ToolCall) or (
                isinstance(item, Message) and item.role == "assistant"
            ):
                count += 1
                break
    return count


def _detect_final_result(
    events: list[AgentEvent],
    output_tool_name: str | None,
) -> str | dict[str, object] | None:
    """Extract the final answer from the last event, or ``None`` if unfinished."""
    if not events:
        return None
    last = events[-1]

    if output_tool_name is not None:
        for item in last.content:
            if (
                isinstance(item, ToolResult)
                and item.name == output_tool_name
                and item.status == "success"
                and item.content
            ):
                return _coerce_final_result(item.content[0])
        return None

    # Free-form: final only when the last event carries no tool activity.
    if any(isinstance(item, (ToolCall, ToolResult)) for item in last.content):
        return None
    for item in last.content:
        if isinstance(item, Message) and item.role == "assistant":
            return item.content
    return None


def _coerce_final_result(value: object) -> str | dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # ``isinstance(value, dict)`` narrows to ``dict[Unknown, Unknown]`` because
    # the key/value types are erased through ``object``. Cast documents the
    # intent and gives pyright a concrete ``dict[str, object]``.
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return str(value)
