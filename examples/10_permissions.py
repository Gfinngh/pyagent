"""10 — The two-lock permission model.

Every tool carries a tier: `read`, `mutate`, or `dangerous`.

  read       always allowed
  mutate     requires a confirm_tool_call callback returning True
  dangerous  requires allow_dangerous=True AND a callback returning True

Uses a deterministic scripted client so the outcome is reproducible and costs no
quota — the point is the mechanism, not the model.

    pixi run python examples/10_permissions.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from examples._common import collect_tool_results, print_trace, section
from examples._fakes import ScriptedClient, make_tool_call, text_response, tool_call_response
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool


class Empty(BaseModel):
    """No arguments."""


def _tool(name: str, permission: str):
    def impl(args: Empty, context: ExecutionContext | None = None) -> str:
        _ = args, context
        return f"{name} ran"

    return FunctionTool(
        name=name,
        description=f"{permission} tier tool",
        schema=Empty,
        func=impl,
        permission=permission,  # type: ignore[arg-type]
    )


def sequence() -> ScriptedClient:
    return ScriptedClient(
        [
            tool_call_response([make_tool_call("c1", "peek")]),
            tool_call_response([make_tool_call("c2", "toggle")]),
            tool_call_response([make_tool_call("c3", "detonate")]),
            text_response("done"),
        ]
    )


async def run_case(label: str, *, allow_dangerous: bool, confirm) -> None:
    section(label)
    agent = Agent(
        client=sequence(),
        tools=[_tool("peek", "read"), _tool("toggle", "mutate"), _tool("detonate", "dangerous")],
        allow_dangerous=allow_dangerous,
        confirm_tool_call=confirm,
        max_steps=4,
    )
    result = await agent.run("go")
    print_trace(result)
    print("  outcomes:")
    for r in collect_tool_results(result):
        print(f"    {r.name:<10} status={r.status}")


async def main() -> None:
    await run_case(
        "case 1: no callback, allow_dangerous=False", allow_dangerous=False, confirm=None
    )
    await run_case("case 2: allow_dangerous=True, no callback", allow_dangerous=True, confirm=None)
    await run_case("case 3: callback approves", allow_dangerous=True, confirm=lambda tc: True)
    await run_case("case 4: callback denies", allow_dangerous=True, confirm=lambda tc: False)

    section("in production")
    print("The host CLI exposes --approve (auto-yes) and --deny (auto-no).")
    print("Interactive mode prompts at the terminal; non-interactive defaults to deny.")


if __name__ == "__main__":
    asyncio.run(main())
