"""21 — Memory namespaces, tags, and prompt injection.

Namespaces isolate memories by topic. Same `(namespace, key)` upserts; different
namespaces never collide. `format_memories_for_prompt` is the convention a
workflow uses for auto-recall: search, then prepend to `Agent.instructions`.

    pixi run python examples/21_memory_namespaces.py
"""

from __future__ import annotations

from pathlib import Path

from examples._common import ensure_dir, section
from pyagent.memory import MemoryStore, format_memories_for_prompt

DB = Path(".examples/memory-namespaces.sqlite")


def main() -> None:
    ensure_dir(DB.parent)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{DB}{suffix}").unlink(missing_ok=True)

    with MemoryStore(DB) as store:
        section("populate two namespaces")
        store.remember("preferred-editor", "neovim", namespace="user", tags=["tooling"])
        store.remember("timezone", "UTC+9", namespace="user", tags=["profile"])
        store.remember("deadline", "March 15", namespace="project-alpha", tags=["milestone"])
        store.remember("reviewer", "Alice", namespace="project-alpha")
        for rec in store.list_all():
            print(f"  [{rec.namespace}] {rec.key} = {rec.value}  tags={rec.tags}")

        section("search across all namespaces")
        for rec in store.search("Alice"):
            print(f"  [{rec.namespace}] {rec.key}")

        section("search restricted to one namespace")
        for rec in store.search("March", namespace="project-alpha"):
            print(f"  [{rec.namespace}] {rec.key}: {rec.value}")
        print(f"  same query on 'user': {len(store.search('March', namespace='user'))} hits")

        section("upsert — same key, same namespace")
        store.remember("timezone", "UTC+8", namespace="user", tags=["profile"])
        print(f"  user/timezone = {store.search('UTC', namespace='user')[0].value}")
        print(f"  total rows in 'user': {len(store.list_all(namespace='user'))}")

        section("tag match — FTS indexes the tags column")
        for rec in store.search("milestone"):
            print(f"  [{rec.namespace}] {rec.key}")

        section("auto-recall pattern for a workflow")
        matches = store.search("neovim")
        injected = format_memories_for_prompt(matches)
        print(injected if injected else "(nothing to inject)")
        print("\nA workflow prepends that string to Agent(instructions=...).")


if __name__ == "__main__":
    main()
