"""Verify the tracing facade primitives + agent span tree.

Phase 1 (primitives): no-op mode, idempotent init, attributes, error recording,
parent-child, shutdown.

Phase 2 (agent tree): patch ``acompletion`` so ``LlmClient`` runs a real scripted
exchange without a network call. Wire the client through a single-entry
``FallbackClient`` — the shape every workflow uses — so the full span tree is
exercised:

    agent.run
      agent.step (x2)
        llm.fallback
          llm.generate (one per attempt)
        tool.call (once, on step 1)
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

from opentelemetry.trace import StatusCode

import pyagent.llm as llm_mod
from pyagent.agent import Agent
from pyagent.fallback import FallbackClient, FallbackEntry
from pyagent.llm import LlmClient
from pyagent.tools import BaseTool
from pyagent.tracing import (
    in_memory_spans,
    init_tracing,
    shutdown_tracing,
    span,
)

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def _phase1() -> None:
    print("\nTest 1: no-op mode before init")
    with span("test.noop", **{"k": "v"}) as h:
        h.set("more", 1)
        h.error(RuntimeError("should not crash"))
    check(in_memory_spans() == [], "no spans recorded before init")

    print("\nTest 2: idempotent init")
    p1 = init_tracing(in_memory=True)
    p2 = init_tracing(in_memory=True)
    check(p1 is not None, "provider returned")
    check(p1 is p2, "second init returns same provider")

    print("\nTest 3: span with attributes")
    with span(
        "test.attrs",
        **{"k.string": "s", "k.int": 42, "k.bool": True},
    ) as h:
        h.set("k.more", "set-inside")
        h.set("k.none", None)
    spans = in_memory_spans()
    attrs_span = next(s for s in spans if s.name == "test.attrs")
    attributes = attrs_span.attributes or {}
    check(attributes.get("k.string") == "s", "string attribute")
    check(attributes.get("k.int") == 42, "int attribute")
    check(attributes.get("k.bool") is True, "bool attribute")
    check(attributes.get("k.more") == "set-inside", "set() attribute")
    check("k.none" not in attributes, "None attribute dropped")

    print("\nTest 4: error recording")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        with span("test.error") as h:
            h.error(exc)
    spans = in_memory_spans()
    err_span = next((s for s in spans if s.name == "test.error"), None)
    check(err_span is not None, "error span recorded")
    if err_span is not None:
        check(
            err_span.status.status_code == StatusCode.ERROR,
            f"status is ERROR (got {err_span.status.status_code})",
        )

    print("\nTest 5: parent-child relationship")
    with span("test.parent") as ph:
        ph.set("level", 1)
        with span("test.child") as ch:
            ch.set("level", 2)
    spans = in_memory_spans()
    parent = next(s for s in spans if s.name == "test.parent")
    child = next(s for s in spans if s.name == "test.child")
    child_parent = child.parent
    parent_context = parent.context
    check(child_parent is not None, "child has a parent context")
    check(parent_context is not None, "parent has a context")
    if child_parent is not None and parent_context is not None:
        check(
            child_parent.span_id == parent_context.span_id,
            "child's parent span_id matches parent",
        )

    print("\nTest 6: shutdown stops producing new spans")
    before = len(in_memory_spans())
    shutdown_tracing()
    with span("test.after-shutdown"):
        pass
    after = len(in_memory_spans())
    check(after == before, f"no new span after shutdown ({before} -> {after})")


# ---------------------------------------------------------------------------
# Phase 2: agent span tree
# ---------------------------------------------------------------------------


class _EchoTool(BaseTool):
    async def call(
        self,
        args: object | None = None,
        context: object | None = None,
    ) -> object:
        _ = (args, context)
        return "echo-result"


def _ok_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _tool_response(tool_name: str, args: dict[str, object], call_id: str) -> Any:
    tc = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=tool_name, arguments=json.dumps(args)),
        provider_specific_fields=None,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tc]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


async def _phase2() -> None:
    print("\nTest 7: agent.run span tree")

    # Reset tracing so phase 2's span list is clean
    shutdown_tracing()
    init_tracing(in_memory=True, service_name="pyagent-test")

    responses: list[Any] = [
        _tool_response("echo", {"msg": "hi"}, "tc1"),
        _ok_response("final answer"),
    ]

    async def fake_acompletion(**kwargs: Any) -> Any:
        return responses.pop(0)

    original = llm_mod.acompletion
    llm_mod.acompletion = fake_acompletion  # type: ignore[assignment]
    try:
        llm = LlmClient(
            model="gemini/gemini-3-flash-preview",
            max_retries=0,
            timeout_ms=5_000,
        )
        client = FallbackClient(
            chain=[
                FallbackEntry(label="primary", client=llm),
            ],
        )
        agent = Agent(
            client=client,
            tools=[_EchoTool(name="echo", description="echo")],
            instructions="be brief",
            max_steps=4,
            name="test-agent",
        )
        result = await agent.run("do something")
    finally:
        llm_mod.acompletion = original

    check(result.output == "final answer", f"agent output (got {result.output!r})")

    spans = in_memory_spans()
    names = [s.name for s in spans]

    check("agent.run" in names, f"agent.run recorded (got {names})")
    check("agent.step" in names, "agent.step recorded")
    check("llm.fallback" in names, "llm.fallback recorded")
    check("llm.generate" in names, "llm.generate recorded")
    check("tool.call" in names, "tool.call recorded")

    root = next(s for s in spans if s.name == "agent.run")
    steps = [s for s in spans if s.name == "agent.step"]
    fallbacks = [s for s in spans if s.name == "llm.fallback"]
    gens = [s for s in spans if s.name == "llm.generate"]
    tools = [s for s in spans if s.name == "tool.call"]

    check(len(steps) == 2, f"2 agent.step (got {len(steps)})")
    check(len(fallbacks) == 2, f"2 llm.fallback (got {len(fallbacks)})")
    check(len(gens) == 2, f"2 llm.generate (got {len(gens)})")
    check(len(tools) == 1, f"1 tool.call (got {len(tools)})")

    root_ctx = root.context
    if root_ctx is not None:
        for step in steps:
            sp = step.parent
            check(
                sp is not None and sp.span_id == root_ctx.span_id,
                "step is child of agent.run",
            )

    # llm.fallback is parent of llm.generate (each generate call lives under
    # its own fallback span).
    if fallbacks and gens:
        for fb in fallbacks:
            fb_ctx = fb.context
            if fb_ctx is None:
                continue
            children = [
                g for g in gens if g.parent is not None and g.parent.span_id == fb_ctx.span_id
            ]
            check(
                len(children) == 1,
                f"each llm.fallback has exactly 1 llm.generate child (got {len(children)})",
            )

    if gens:
        attrs = dict(gens[0].attributes or {})
        check(
            attrs.get("gen_ai.request.model") == "gemini/gemini-3-flash-preview",
            f"gen_ai.request.model set (got {attrs.get('gen_ai.request.model')!r})",
        )
        check(
            attrs.get("gen_ai.usage.input_tokens") == 10,
            f"input_tokens attribute (got {attrs.get('gen_ai.usage.input_tokens')!r})",
        )
        check(
            attrs.get("gen_ai.usage.output_tokens") == 5,
            f"output_tokens attribute (got {attrs.get('gen_ai.usage.output_tokens')!r})",
        )

    if tools:
        attrs = dict(tools[0].attributes or {})
        check(attrs.get("tool.name") == "echo", "tool.name attribute")
        check(attrs.get("tool.status") == "success", "tool.status attribute")

    if root_ctx is not None:
        root_attrs = dict(root.attributes or {})
        check(
            root_attrs.get("agent.name") == "test-agent",
            "agent.name on root span",
        )
        check(
            root_attrs.get("agent.steps_used") == 2,
            f"steps_used (got {root_attrs.get('agent.steps_used')})",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _phase1()
    asyncio.run(_phase2())


def _finalize() -> None:
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Tracing verified.")


def run() -> None:
    try:
        main()
    finally:
        _finalize()


if __name__ == "__main__":
    run()
