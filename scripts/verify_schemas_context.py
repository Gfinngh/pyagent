"""Phase 2: verify schemas and context."""

from __future__ import annotations

import sys
import time
from typing import cast

from pyagent.context import ExecutionContext
from pyagent.schemas import AgentEvent, Message, ToolCall, ToolResult


def check(cond: object, msg: str) -> None:
    if not cond:
        print(f"  \u2717 {msg}")
        sys.exit(1)
    print(f"  \u2713 {msg}")


class FakeLog:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.closed = False

    def append(self, event: AgentEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def main() -> None:
    print("\nTest 1: Message")
    msg = Message(role="user", content="hello")
    check(msg.type == "message", "type defaults to 'message'")
    check(msg.role == "user", "role preserved")
    check(msg.content == "hello", "content preserved")
    check(
        msg.model_dump() == {"type": "message", "role": "user", "content": "hello"},
        "model_dump shape correct",
    )

    print("\nTest 2: ToolCall")
    tc = ToolCall(tool_call_id="t1", name="calc", arguments={"a": 1, "b": 2})
    check(tc.type == "tool_call", "type defaults correctly")
    check(tc.provider_options is None, "provider_options defaults to None")

    tc_meta = ToolCall(
        tool_call_id="t2",
        name="calc",
        arguments={},
        provider_options={"google": {"thoughtSignature": "sig"}},
    )
    check(tc_meta.provider_options is not None, "provider_options accepted")
    if tc_meta.provider_options is not None:
        google = tc_meta.provider_options.get("google")
        check(isinstance(google, dict), "google key holds a dict")
        if isinstance(google, dict):
            google_typed = cast(dict[str, object], google)
            check(
                google_typed.get("thoughtSignature") == "sig",
                "nested signature preserved",
            )

    print("\nTest 3: ToolResult")
    tr = ToolResult.model_validate(
        {
            "tool_call_id": "t1",
            "name": "calc",
            "status": "success",
            "content": [4],
        }
    )
    check(tr.type == "tool_result", "type defaults correctly")
    check(tr.content == [4], "content preserved")

    print("\nTest 4: AgentEvent defaults")
    t0 = time.time()
    ev = AgentEvent(execution_id="e1", author="user", content=[msg])
    check(len(ev.id) == 36, f"id is uuid4 (got {ev.id})")
    check(ev.timestamp >= t0, "timestamp auto-set")
    check(len(ev.content) == 1, "content list has one item")

    print("\nTest 5: discriminated union round-trip")
    mixed = AgentEvent(
        execution_id="e2",
        author="agent",
        content=[
            Message(role="assistant", content="thinking"),
            ToolCall(tool_call_id="tc1", name="calc", arguments={"op": "add"}),
            ToolResult.model_validate(
                {
                    "tool_call_id": "tc1",
                    "name": "calc",
                    "status": "success",
                    "content": [42],
                }
            ),
        ],
    )
    dumped = mixed.model_dump()
    restored = AgentEvent.model_validate(dumped)
    check(len(restored.content) == 3, "3 content items restored")
    for i, item in enumerate(restored.content):
        if isinstance(item, Message):
            check(item.role == "assistant", f"content[{i}] is assistant Message")
        elif isinstance(item, ToolCall):
            check(item.name == "calc", f"content[{i}] is ToolCall named calc")
        else:
            check(item.status == "success", f"content[{i}] is success ToolResult")
    check(restored.id == mixed.id, "id preserved across round-trip")
    check(restored.timestamp == mixed.timestamp, "timestamp preserved")

    print("\nTest 6: JSON round-trip")
    json_str = mixed.model_dump_json()
    check('"type":"message"' in json_str, "JSON has type discriminator")
    restored_json = AgentEvent.model_validate_json(json_str)
    tool_call_item = restored_json.content[1]
    check(isinstance(tool_call_item, ToolCall), "JSON round-trip preserves ToolCall")
    if isinstance(tool_call_item, ToolCall):
        check(tool_call_item.name == "calc", "nested field preserved")

    print("\nTest 7: ExecutionContext")
    ctx = ExecutionContext()
    check(len(ctx.execution_id) == 36, "execution_id is uuid4")
    check(ctx.current_step == 0, "current_step starts at 0")
    check(ctx.final_result is None, "final_result starts None")
    check(len(ctx.events) == 0, "events starts empty")

    ctx.add_event(ev)
    check(len(ctx.events) == 1, "add_event appends")
    check(ctx.events[0] is ev, "same event object")

    ctx.increment_step()
    ctx.increment_step()
    check(ctx.current_step == 2, "increment_step 2x -> 2")

    print("\nTest 8: ExecutionContext forwards to EventLog")
    fake_log = FakeLog()
    ctx_with_log = ExecutionContext(log=fake_log)
    ctx_with_log.add_event(ev)
    check(len(fake_log.events) == 1, "log.append called on add_event")
    check(fake_log.events[0] is ev, "same event object forwarded")

    print("\n\u2705 Phase 2 verified.")


if __name__ == "__main__":
    main()
