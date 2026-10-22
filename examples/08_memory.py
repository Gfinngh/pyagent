"""08 — Memory store.

SQLite + FTS5 key/value store with keyword search, driven directly. Shows the
four store operations and the four tools the model uses.

    pixi run python examples/08_memory.py

Re-running is safe: the database is deleted first.
"""

from __future__ import annotations

from pathlib import Path

from examples._common import ensure_dir, section
from pyagent.memory import MemoryStore, make_memory_tools

EXAMPLES_DIR = Path(".examples")
DB_PATH = EXAMPLES_DIR / "memory.sqlite"


def main() -> None:
    ensure_dir(EXAMPLES_DIR)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{DB_PATH}{suffix}").unlink(missing_ok=True)

    with MemoryStore(DB_PATH) as store:
        print(f"db: {DB_PATH}")

        section("remember")
        store.remember(
            key="favorite-color",
            value="My favorite color is blue.",
            namespace="user",
            tags=["preference"],
        )
        store.remember(
            key="home-city",
            value="I live in Kyoto.",
            namespace="user",
            tags=["preference"],
        )
        store.remember(
            key="todo",
            value="Read about FTS5 ranking this week.",
            namespace="notes",
            tags=["learning"],
        )
        for rec in store.list_all():
            print(f"  [{rec.namespace}] {rec.key}")

        section('search("Kyoto")')
        for rec in store.search("Kyoto"):
            print(f"  [{rec.namespace}] {rec.key}: {rec.value}")

        section('search("ranking")')
        for rec in store.search("ranking"):
            print(f"  [{rec.namespace}] {rec.key}: {rec.value}")

        section('search("preference")  — matches the tag')
        for rec in store.search("preference"):
            print(f"  [{rec.namespace}] {rec.key}")

        section('search("Kyoto", namespace="notes")  — filtered')
        print(f"  results: {len(store.search('Kyoto', namespace='notes'))}")

        section('forget("todo", namespace="notes")')
        print(f"  removed: {store.forget('todo', namespace='notes')}")
        print(f"  remaining in notes: {len(store.list_all(namespace='notes'))}")

        section("tools exposed to the model")
        for tool in make_memory_tools(store):
            print(f"  {tool.name:<15} [{tool.permission}]")


if __name__ == "__main__":
    main()
