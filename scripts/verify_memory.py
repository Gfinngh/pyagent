"""Verify the SQLite memory store and its tools.

Deterministic, no network. Covers storage, search, upsert, permission tiers,
and one end-to-end ``Agent.run`` with a fake client that calls ``remember``
then ``recall`` then finishes.

Scenarios:

  1. ``MemoryStore`` opens lazily and persists across reopen.
  2. ``remember`` stores a record; ``list_all`` sees it.
  3. ``remember`` upserts by (namespace, key) — no duplicate row.
  4. ``search`` finds by keyword.
  5. ``search`` finds by tag.
  6. ``search`` respects namespace filter.
  7. ``search`` returns empty for unrelated queries.
  8. FTS5 special characters don't raise (query returns normally).
  9. ``forget`` removes a memory; second call returns False.
 10. Tool factory: 4 tools with expected names and permission tiers.
 11. Tools via ``.call``: remember / recall / forget / list_memories.
 12. ``format_memories_for_prompt`` renders recall results.
 13. ``Agent`` with memory tools: model calls remember then recall then
     finishes; both tool calls recorded, both results successful, and the
     memory is actually persisted. The ``Agent`` needs a
     ``confirm_tool_call`` callback because ``remember`` is
     ``permission="mutate"`` — without it, the permission check denies the
     call (correct behavior) and the memory never lands.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyagent.llm as llm_mod
from pyagent.agent import Agent
from pyagent.llm import LlmClient
from pyagent.memory import (
    MemoryStore,
    format_memories_for_prompt,
    make_memory_tools,
)

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def _phase_store(db: Path) -> None:
    # --- Test 1: lazy open + persistence across reopen ------------------
    print("\nTest 1: store opens lazily and persists across reopen")
    smoke_db = db.parent / "smoke.sqlite"
    store_smoke = MemoryStore(smoke_db)
    check(not smoke_db.exists(), "db file not created before first use")
    store_smoke.remember("smoke", "ok")
    check(smoke_db.exists(), f"db file created after first write at {smoke_db}")
    store_smoke.close()

    store_reopened = MemoryStore(smoke_db)
    reopened_records = store_reopened.list_all()
    check(
        len(reopened_records) == 1,
        f"reopened store sees the persisted memory (got {len(reopened_records)})",
    )
    if reopened_records:
        check(
            reopened_records[0].value == "ok",
            f"persisted value preserved (got {reopened_records[0].value!r})",
        )
    store_reopened.close()

    # --- Main store for remaining tests ---------------------------------
    store = MemoryStore(db)

    print("\nTest 2: remember stores a record")
    rec = store.remember("editor", "neovim", namespace="user")
    check(rec.key == "editor" and rec.value == "neovim", "record returned")
    all_mem = store.list_all()
    check(len(all_mem) == 1, f"one memory in store (got {len(all_mem)})")

    print("\nTest 3: remember upserts by (namespace, key)")
    store.remember("editor", "emacs", namespace="user")
    all_mem = store.list_all(namespace="user")
    check(len(all_mem) == 1, f"still one memory after upsert (got {len(all_mem)})")
    check(all_mem[0].value == "emacs", f"value updated (got {all_mem[0].value!r})")

    print("\nTest 4: search by keyword")
    store.remember("db", "postgres 16 on port 5432", namespace="project-alpha")
    results = store.search("postgres", limit=5)
    check(len(results) == 1, f"one match (got {len(results)})")
    check(results[0].namespace == "project-alpha", "matched correct namespace")

    print("\nTest 5: search by tag")
    store.remember(
        "flaky-test",
        "auth tests need longer timeout",
        namespace="project-alpha",
        tags=["ci", "auth"],
    )
    by_tag = store.search("ci", limit=5)
    check(
        any(r.key == "flaky-test" for r in by_tag),
        f"tag search finds tagged memory (got {[r.key for r in by_tag]})",
    )

    print("\nTest 6: search respects namespace filter")
    only_alpha = store.search("editor", namespace="project-alpha", limit=5)
    check(
        len(only_alpha) == 0,
        f"editor not in project-alpha (got {len(only_alpha)})",
    )
    only_user = store.search("editor", namespace="user", limit=5)
    check(len(only_user) == 1, f"editor in user (got {len(only_user)})")

    print("\nTest 7: search returns empty for unrelated query")
    nothing = store.search("xyzzy-nonexistent", limit=5)
    check(len(nothing) == 0, f"no matches (got {len(nothing)})")

    print("\nTest 8: FTS5 special characters don't raise")
    safe = store.search('foo*(bar)-"baz"', limit=5)
    check(
        len(safe) == 0,
        f"non-matching special-char query returns empty (got {len(safe)})",
    )

    print("\nTest 9: forget removes a memory")
    removed = store.forget("editor", namespace="user")
    check(removed is True, "first forget returns True")
    again = store.forget("editor", namespace="user")
    check(again is False, "second forget returns False")
    remaining = store.list_all(namespace="user")
    check(len(remaining) == 0, f"user namespace empty (got {len(remaining)})")

    print("\nTest 10: tool factory produces the four tools with correct perms")
    tools = make_memory_tools(store)
    names = sorted(t.name for t in tools)
    check(
        names == ["forget", "list_memories", "recall", "remember"],
        f"tool names (got {names})",
    )
    perms = {t.name: t.permission for t in tools}
    check(perms["remember"] == "mutate", "remember is mutate")
    check(perms["forget"] == "mutate", "forget is mutate")
    check(perms["recall"] == "read", "recall is read")
    check(perms["list_memories"] == "read", "list_memories is read")

    print("\nTest 11: tools via .call()")
    remember = next(t for t in tools if t.name == "remember")
    recall = next(t for t in tools if t.name == "recall")
    forget = next(t for t in tools if t.name == "forget")
    lst = next(t for t in tools if t.name == "list_memories")

    r1 = await_or(
        remember.call(
            {
                "key": "lang",
                "value": "python 3.12",
                "namespace": "user",
            }
        )
    )
    check("remembered" in str(r1), f"remember returns confirmation (got {r1!r})")

    r2 = await_or(recall.call({"query": "python", "limit": 5}))
    check("python" in str(r2).lower(), f"recall returns match (got {r2!r})")

    r3 = await_or(lst.call({"namespace": "user"}))
    check("lang" in str(r3), f"list returns key (got {r3!r})")

    r4 = await_or(forget.call({"key": "lang", "namespace": "user"}))
    check("forgot" in str(r4), f"forget returns confirmation (got {r4!r})")

    print("\nTest 12: format_memories_for_prompt")
    samples = store.search("postgres", limit=1)
    formatted = format_memories_for_prompt(samples)
    check(
        "Relevant prior memories" in formatted,
        f"header present (got {formatted[:60]!r})",
    )

    store.close()


def await_or(x: Any) -> Any:
    """Handle both sync and async returns uniformly (tools are async)."""
    if asyncio.iscoroutine(x):
        return asyncio.run(x)
    return x


# ---------------------------------------------------------------------------
# Phase 2: agent end-to-end with memory tools
# ---------------------------------------------------------------------------


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


def _ok_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


async def _phase_agent(db: Path) -> None:
    print("\nTest 13: agent with memory tools end-to-end")

    store = MemoryStore(db)
    tools = make_memory_tools(store)

    responses: list[Any] = [
        _tool_response(
            "remember",
            {"key": "water-boils-at", "value": "100 degrees C", "namespace": "facts"},
            "tc1",
        ),
        _tool_response(
            "recall",
            {"query": "water", "namespace": "facts"},
            "tc2",
        ),
        _ok_response("stored and confirmed"),
    ]

    async def fake_acompletion(**kwargs: Any) -> Any:
        return responses.pop(0)

    original = llm_mod.acompletion
    llm_mod.acompletion = fake_acompletion  # type: ignore[assignment]
    try:
        llm = LlmClient(
            model="gemini/gemini-3.1-flash-lite",
            max_retries=0,
            timeout_ms=5_000,
        )
        agent = Agent(
            client=llm,
            tools=tools,
            instructions="be brief",
            max_steps=4,
            name="test-memory-agent",
            # `remember` is permission="mutate"; without a confirm callback
            # the permission check denies the call before it runs.
            confirm_tool_call=lambda tc: True,
        )
        result = await agent.run("remember that water boils at 100C")
    finally:
        llm_mod.acompletion = original

    check(
        result.output == "stored and confirmed",
        f"agent output (got {result.output!r})",
    )

    tool_calls = [
        c
        for ev in result.context.events
        for c in ev.content
        if getattr(c, "type", "") == "tool_call"
    ]
    check(len(tool_calls) == 2, f"2 tool calls (got {len(tool_calls)})")
    called_names = [getattr(c, "name", "?") for c in tool_calls]
    check("remember" in called_names, f"remember called (got {called_names})")
    check("recall" in called_names, f"recall called (got {called_names})")

    tool_results = [
        c
        for ev in result.context.events
        for c in ev.content
        if getattr(c, "type", "") == "tool_result"
    ]
    check(len(tool_results) == 2, f"2 tool results (got {len(tool_results)})")
    # Status check is the key signal: a denial produces a ToolResult with
    # status="error" and the tool never runs, so counting results alone is
    # not enough to know the tools actually executed.
    for r in tool_results:
        check(
            getattr(r, "status", "?") == "success",
            f"result for {getattr(r, 'name', '?')} is success (got {getattr(r, 'status', '?')!r})",
        )

    stored = store.list_all(namespace="facts")
    check(
        any(m.key == "water-boils-at" for m in stored),
        f"memory stored during run (got {[m.key for m in stored]})",
    )

    store.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "memory.sqlite"
        _phase_store(db)
        db2 = Path(tmp) / "memory2.sqlite"
        asyncio.run(_phase_agent(db2))


def _finalize() -> None:
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 Memory verified.")


def run() -> None:
    try:
        main()
    finally:
        _finalize()


if __name__ == "__main__":
    run()
