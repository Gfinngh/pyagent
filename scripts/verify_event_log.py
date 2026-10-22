"""Phase 13: JSONL event log - write, read, rehydrate.

Covers round-trip fidelity, append accumulation, log forwarding from
``ExecutionContext.add_event``, ``from_log`` derivation for the three resume
states, structured output, empty logs, reopen-after-close, and the two
malformed-line tolerance rules. Zero network calls.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.event_log import JsonlEventLog, NullEventLog, read_event_log
from pyagent.schemas import AgentEvent, Message, ToolCall, ToolResult

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def user_event(text: str, execution_id: str = "e1") -> AgentEvent:
    return AgentEvent(
        execution_id=execution_id,
        author="user",
        content=[Message(role="user", content=text)],
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        print("\nTest 1: round-trip a single event")
        path = root / "single.jsonl"
        log = JsonlEventLog(path)
        ev = user_event("hi")
        log.append(ev)
        log.close()
        back = read_event_log(path)
        check(len(back) == 1, f"one event read back (got {len(back)})")
        if back:
            check(back[0].execution_id == "e1", "execution_id preserved")
            check(back[0].author == "user", "author preserved")
            check(back[0].id == ev.id, "id preserved")
            check(back[0].timestamp == ev.timestamp, "timestamp preserved")
            first = back[0].content[0]
            check(isinstance(first, Message), f"content[0] is Message (got {type(first).__name__})")
            if isinstance(first, Message):
                check(first.content == "hi", f"message text preserved (got {first.content!r})")

        print("\nTest 2: multiple appends accumulate")
        path = root / "multi.jsonl"
        with JsonlEventLog(path) as log:
            for i in range(3):
                log.append(user_event(f"m{i}"))
        check(len(read_event_log(path)) == 3, f"three events (got {len(read_event_log(path))})")

        print("\nTest 3: add_event forwards to the log")
        path = root / "forward.jsonl"
        log = JsonlEventLog(path)
        ctx = ExecutionContext(log=log)
        ctx.add_event(user_event("x", execution_id=ctx.execution_id))
        log.close()
        check(len(ctx.events) == 1, "in-memory list updated")
        check(len(read_event_log(path)) == 1, "persisted to disk")

        print("\nTest 4: from_log derives current_step (tool then answer)")
        path = root / "state_c.jsonl"
        eid = "state-c"
        with JsonlEventLog(path) as log:
            log.append(user_event("2+2", execution_id=eid))
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[ToolCall(tool_call_id="t1", name="calculator", arguments={})],
                )
            )
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[
                        ToolResult(
                            tool_call_id="t1", name="calculator", status="success", content=[4]
                        )
                    ],
                )
            )
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[Message(role="assistant", content="4")],
                )
            )
        ctx = ExecutionContext.from_log(path)
        check(ctx.execution_id == eid, f"execution_id preserved (got {ctx.execution_id})")
        check(len(ctx.events) == 4, f"four events (got {len(ctx.events)})")
        check(ctx.current_step == 2, f"current_step=2 (got {ctx.current_step})")
        check(ctx.final_result == "4", f"final_result='4' (got {ctx.final_result!r})")

        print("\nTest 5: from_log on interrupted run (ends with tool_result)")
        path = root / "state_d.jsonl"
        eid = "state-d"
        with JsonlEventLog(path) as log:
            log.append(user_event("2+2", execution_id=eid))
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[ToolCall(tool_call_id="t1", name="calculator", arguments={})],
                )
            )
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[
                        ToolResult(
                            tool_call_id="t1", name="calculator", status="success", content=[4]
                        )
                    ],
                )
            )
        ctx = ExecutionContext.from_log(path)
        check(ctx.current_step == 1, f"current_step=1 (got {ctx.current_step})")
        check(ctx.final_result is None, f"final_result=None (got {ctx.final_result!r})")

        print("\nTest 6: from_log with structured output (final_answer)")
        path = root / "structured.jsonl"
        eid = "structured"
        with JsonlEventLog(path) as log:
            log.append(user_event("12*7", execution_id=eid))
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[
                        ToolCall(
                            tool_call_id="t1",
                            name="final_answer",
                            arguments={"output": {"answer": 84}},
                        )
                    ],
                )
            )
            log.append(
                AgentEvent(
                    execution_id=eid,
                    author="agent",
                    content=[
                        ToolResult(
                            tool_call_id="t1",
                            name="final_answer",
                            status="success",
                            content=[{"answer": 84}],
                        )
                    ],
                )
            )
        ctx = ExecutionContext.from_log(path, output_tool_name="final_answer")
        check(ctx.current_step == 1, f"current_step=1 (got {ctx.current_step})")
        check(
            isinstance(ctx.final_result, dict) and ctx.final_result.get("answer") == 84,
            f"final_result={{'answer': 84}} (got {ctx.final_result!r})",
        )

        print("\nTest 7: from_log on an empty log")
        path = root / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        ctx = ExecutionContext.from_log(path)
        check(len(ctx.events) == 0, "no events")
        check(ctx.current_step == 0, "current_step=0")
        check(ctx.final_result is None, "final_result=None")

        print("\nTest 8: reopen after close")
        path = root / "reopen.jsonl"
        log = JsonlEventLog(path)
        log.append(user_event("a", execution_id="e8"))
        log.close()
        log.append(user_event("b", execution_id="e8"))
        log.close()
        check(len(read_event_log(path)) == 2, f"two events (got {len(read_event_log(path))})")

        print("\nTest 9: truncated final line is skipped with a warning")
        path = root / "truncated.jsonl"
        with JsonlEventLog(path) as log:
            log.append(user_event("complete", execution_id="e9"))
        with path.open("a", encoding="utf-8", newline="") as fh:
            fh.write('{"id": "partial", "execution_id": "e9"')  # no newline
        back = read_event_log(path)
        check(len(back) == 1, f"only the complete event survives (got {len(back)})")

        print("\nTest 10: malformed interior line raises with line number")
        path = root / "corrupt.jsonl"
        with JsonlEventLog(path) as log:
            log.append(user_event("first", execution_id="e10"))
        with path.open("a", encoding="utf-8", newline="") as fh:
            fh.write("not json\n")
        with path.open("a", encoding="utf-8", newline="") as fh:
            fh.write(user_event("third", execution_id="e10").model_dump_json() + "\n")
        raised: ValueError | None = None
        try:
            read_event_log(path)
        except ValueError as exc:
            raised = exc
        check(raised is not None, "ValueError raised")
        if raised is not None:
            check("line 2" in str(raised), f"message names line 2 (got {raised})")

        print("\nTest 11: NullEventLog discards silently")
        null = NullEventLog()
        null.append(user_event("ignored"))
        null.close()
        check(True, "append/close are no-ops")

    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Phase 13 (event log) verified.")


if __name__ == "__main__":
    main()
