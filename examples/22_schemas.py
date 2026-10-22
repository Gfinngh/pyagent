"""22 — Schemas and content items.

The four models every run is built from:

    Message     a text turn (role: system | user | assistant)
    ToolCall    the model's request to run a tool
    ToolResult  the outcome of running it
    AgentEvent  one atomic occurrence: id, execution_id, timestamp, author, content

`ContentItem` is the discriminated union of the first three. Pydantic handles
serialisation; the `type` field is the discriminator.

    pixi run python examples/22_schemas.py
"""

from __future__ import annotations

import json

from examples._common import section
from pyagent.schemas import AgentEvent, Message, ToolCall, ToolResult


def main() -> None:
    section("1. build the four models")
    message = Message(role="assistant", content="Let me check that.")
    call = ToolCall(
        tool_call_id="call_1",
        name="calculator",
        arguments={"operator": "add", "first_number": 2, "second_number": 3},
        provider_options={"thought_signature": "sig_demo"},
    )
    result = ToolResult(
        tool_call_id="call_1",
        name="calculator",
        status="success",
        content=[5.0],
    )
    event = AgentEvent(
        execution_id="demo-run",
        author="agent",
        content=[message, call, result],
    )

    print(f"  event.id:          {event.id}")
    print(f"  event.execution_id {event.execution_id}")
    print(f"  event.author:      {event.author}")
    print(f"  event.content:     {[type(i).__name__ for i in event.content]}")

    section("2. JSON round-trip preserves the discriminator")
    payload = event.model_dump_json()
    print(payload[:200] + "…")

    reparsed = AgentEvent.model_validate(json.loads(payload))
    print(f"  content types after round-trip: {[i.type for i in reparsed.content]}")

    section("3. provider_options survive verbatim")
    reparsed_call = next(i for i in reparsed.content if isinstance(i, ToolCall))
    print(f"  tool_call_id:     {reparsed_call.tool_call_id}")
    print(f"  provider_options: {reparsed_call.provider_options}")

    section("notes")
    print("provider_options must be round-tripped verbatim on replay, or Gemini")
    print("multi-turn tool calls lose reasoning state. See examples/19.")


if __name__ == "__main__":
    main()
