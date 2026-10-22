"""24 — The request *is* the conversation.

`LlmRequest.contents` is the flattened `ExecutionContext.events` for that step.
There is no separate message history. This example records every outgoing
request with a scripted client and shows `contents` growing step by step.

    pixi run python examples/24_prompt_contents.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from examples._common import section
from examples._fakes import ScriptedClient, make_tool_call, text_response, tool_call_response
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.schemas import Message, ToolCall, ToolResult
from pyagent.tools import FunctionTool


class AddInput(BaseModel):
    a: float
    b: float


def add(args: AddInput, context: ExecutionContext | None = None) -> float:
    _ = context
    return args.a + args.b


def describe(contents) -> str:
    parts: list[str] = []
    for item in contents:
        if isinstance(item, Message):
            parts.append(f"{item.role}:{item.content[:18]!r}")
        elif isinstance(item, ToolCall):
            parts.append(f"call({item.name})")
        elif isinstance(item, ToolResult):
            parts.append(f"result({item.status})")
    return " → ".join(parts)


async def main() -> None:
    client = ScriptedClient(
        [
            tool_call_response([make_tool_call("c1", "add", {"a": 2, "b": 3})]),
            text_response("The answer is 5."),
        ]
    )
    agent = Agent(
        client=client,
        tools=[FunctionTool(name="add", description="add", schema=AddInput, func=add)],
        instructions="You are terse.",
        max_steps=3,
    )

    section("run")
    result = await agent.run("add 2 and 3")

    section("each outgoing request")
    for index, request in enumerate(client.requests, start=1):
        print(f"  step {index}: {len(request.contents)} content item(s)")
        print(f"    {describe(request.contents)}")
        print(f"    tool_choice={request.tool_choice} tools={len(request.tools)}")

    section("final context")
    print(f"  events:       {len(result.context.events)}")
    print(f"  current_step: {result.context.current_step}")
    print(f"  output:       {result.output!r}")

    section("notes")
    print("Every step re-sends the whole conversation; that is what makes replay")
    print("and resume exact. Example 05 shows the persisted form of the same data.")


if __name__ == "__main__":
    asyncio.run(main())
