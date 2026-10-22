"""17 — Tracing.

`init_tracing(in_memory=True)` installs an in-process span exporter you can
inspect with `in_memory_spans()`. The agent emits a span tree:

    agent.run
    ├── agent.step          one per loop iteration
    │   ├── llm.generate
    │   └── tool.call       one per tool execution
    └── ...

    pixi run python examples/17_tracing.py
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from pydantic import BaseModel

from examples._common import section
from examples._fakes import ScriptedClient, make_tool_call, text_response, tool_call_response
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool
from pyagent.tracing import in_memory_spans, init_tracing, shutdown_tracing


class AddInput(BaseModel):
    a: float
    b: float


def add(args: AddInput, context: ExecutionContext | None = None) -> float:
    _ = context
    return args.a + args.b


def print_tree(spans) -> None:
    by_parent: dict[str | None, list] = defaultdict(list)
    for s in spans:
        parent = s.parent.span_id if s.parent is not None else None
        by_parent[parent].append(s)

    def walk(parent: str | None, depth: int) -> None:
        for s in sorted(by_parent[parent], key=lambda x: x.start_time or 0):
            duration_ms = ((s.end_time or 0) - (s.start_time or 0)) / 1_000_000
            attrs = dict(list(s.attributes.items())[:3]) if s.attributes else {}
            print(f"{'  ' * depth}{s.name} ({duration_ms:.1f} ms) {attrs}")
            walk(s.context.span_id, depth + 1)

    walk(None, 0)


async def main() -> None:
    shutdown_tracing()
    init_tracing(in_memory=True)

    agent = Agent(
        client=ScriptedClient(
            [
                tool_call_response([make_tool_call("c1", "add", {"a": 2, "b": 3})]),
                text_response("The answer is 5."),
            ]
        ),
        tools=[
            FunctionTool(
                name="add",
                description="Add two numbers",
                schema=AddInput,
                func=add,
            )
        ],
        name="demo",
        max_steps=3,
    )
    await agent.run("add 2 and 3")

    section("span tree")
    spans = in_memory_spans()
    print_tree(spans)
    print(f"\ntotal spans: {len(spans)}")

    section("status")
    for s in spans:
        print(f"  {s.name:<16} status={s.status.status_code.name}")

    shutdown_tracing()


if __name__ == "__main__":
    asyncio.run(main())
