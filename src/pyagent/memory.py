"""SQLite-backed memory store and the tools that expose it.

Primitives only: ``sqlite3`` from the stdlib. The store owns the schema,
search ranking, and lifecycle. The tools are thin adapters the model drives.

Why SQLite
----------
- Stdlib. No install, no service, no dependency.
- Single-file. Backup, inspect, share = one file (``workspace/memory.sqlite``).
- FTS5 full-text search. Real ranking, not LIKE shortcuts.
- WAL mode. Concurrent readers, safe writer.

Not doing
---------
- Embeddings/vector search: adds an embedding model + index for what a
  personal agent can do with FTS5 for most queries.
- Cloud memory services (Mem0, Zep): 180-800ms per op and a network round
  trip for what should be a 2ms local read.
- Auto-recall at agent start: a workflow-level decision. ``Agent`` stays
  clean; a workflow that wants auto-recall calls ``store.search()`` itself
  and prepends ``format_memories_for_prompt(...)`` to ``instructions``.

FTS5 query notes
----------------
The ``MATCH`` operator does NOT accept table aliases — ``WHERE f MATCH ?``
fails with ``no such column: f``. Queries must reference the FTS table by
its real name: ``WHERE memories_fts MATCH ?``. The hidden ``rank`` column is
ordered with an unqualified ``ORDER BY rank`` (``memories`` has no column of
that name, so resolution is unambiguous).

Layout
------
One database file per workflow (path passed by the workflow). Namespaces
isolate memories by topic — ``user``, ``project-alpha``, etc. Same
namespace + key overwrites (upsert). Keys are unique per namespace.

Concurrency
-----------
The connection is single-threaded. Async ``tool.call`` returns to the event
loop between calls, but the actual SQLite work is a fast sync operation that
blocks the loop for ~1-5ms. Safe for the parallel-tool feature: sync SQLite
calls are atomic from the event loop's perspective (no ``await`` inside the
critical section), so two parallel memory tools never interleave mid-query.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from pyagent.context import ExecutionContext
from pyagent.tools import FunctionTool

DEFAULT_NAMESPACE = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(namespace, key)
);

CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    key, value, tags,
    content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, key, value, tags)
    VALUES (new.rowid, new.key, new.value, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, value, tags)
    VALUES ('delete', old.rowid, old.key, old.value, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, value, tags)
    VALUES ('delete', old.rowid, old.key, old.value, old.tags);
    INSERT INTO memories_fts(rowid, key, value, tags)
    VALUES (new.rowid, new.key, new.value, new.tags);
END;
"""


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """One stored memory."""

    id: str
    namespace: str
    key: str
    value: str
    tags: list[str]
    created_at: float
    updated_at: float


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MemoryStore:
    """SQLite-backed memory store. Lazy open, explicit close."""

    path: Path
    _conn: sqlite3.Connection | None

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = None

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path))
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> MemoryStore:
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember(
        self,
        key: str,
        value: str,
        namespace: str = DEFAULT_NAMESPACE,
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        """Insert or update a memory. Upsert on (namespace, key)."""
        now = time.time()
        tags_str = ",".join(tags or [])
        conn = self._ensure_open()

        existing = conn.execute(
            "SELECT id, created_at FROM memories WHERE namespace=? AND key=?",
            (namespace, key),
        ).fetchone()

        if existing:
            memory_id: str = existing["id"]
            created_at: float = existing["created_at"]
            conn.execute(
                "UPDATE memories SET value=?, tags=?, updated_at=? WHERE id=?",
                (value, tags_str, now, memory_id),
            )
        else:
            memory_id = str(uuid.uuid4())
            created_at = now
            conn.execute(
                "INSERT INTO memories "
                "(id, namespace, key, value, tags, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, namespace, key, value, tags_str, created_at, now),
            )
        conn.commit()

        return MemoryRecord(
            id=memory_id,
            namespace=namespace,
            key=key,
            value=value,
            tags=list(tags or []),
            created_at=created_at,
            updated_at=now,
        )

    def forget(self, key: str, namespace: str = DEFAULT_NAMESPACE) -> bool:
        """Delete one memory. Returns True if a row was removed."""
        conn = self._ensure_open()
        cur = conn.execute(
            "DELETE FROM memories WHERE namespace=? AND key=?",
            (namespace, key),
        )
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """FTS5 search with optional namespace filter. Best-ranked first.

        ``MATCH`` is applied to the real FTS table name (``memories_fts``),
        not an alias — FTS5's parser does not honour table aliases in the
        ``MATCH`` operator. ``ORDER BY rank`` uses FTS5's hidden rank column;
        unqualified because only the FTS table has such a column.
        """
        if not query.strip():
            return []
        conn = self._ensure_open()
        fts = _fts_query(query)

        if namespace is not None:
            rows = conn.execute(
                "SELECT m.* FROM memories m "
                "JOIN memories_fts ON memories_fts.rowid = m.rowid "
                "WHERE memories_fts MATCH ? AND m.namespace = ? "
                "ORDER BY rank LIMIT ?",
                (fts, namespace, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT m.* FROM memories m "
                "JOIN memories_fts ON memories_fts.rowid = m.rowid "
                "WHERE memories_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts, limit),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list_all(
        self,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Enumerate memories, most-recently-updated first."""
        conn = self._ensure_open()
        if namespace is not None:
            rows = conn.execute(
                "SELECT * FROM memories WHERE namespace=? ORDER BY updated_at DESC LIMIT ?",
                (namespace, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_matches(matches: list[MemoryRecord]) -> str:
    """Format search results for the model. Used by the ``recall`` tool."""
    if not matches:
        return "No matching memories."
    lines = [f"{len(matches)} match(es):"]
    for m in matches:
        lines.append(f"- [{m.namespace}] {m.key}: {m.value}")
        if m.tags:
            lines.append(f"  tags: {', '.join(m.tags)}")
    return "\n".join(lines)


def format_memories_for_prompt(matches: list[MemoryRecord]) -> str:
    """Format recall results for injection into ``Agent.instructions``.

    The convention a workflow uses for auto-recall: call ``store.search()``
    with the user's input, then prepend this to the instructions. Empty
    string when there's nothing to inject.
    """
    if not matches:
        return ""
    lines = ["Relevant prior memories:"]
    for m in matches:
        tags = f" [{', '.join(m.tags)}]" if m.tags else ""
        lines.append(f"- {m.namespace}/{m.key}{tags}: {m.value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _RememberInput(BaseModel):
    key: str = Field(
        description="Short identifier, e.g. 'user-preferred-editor' or 'trip-kyoto-dates'."
    )
    value: str = Field(description="The information to store.")
    namespace: str = Field(
        default=DEFAULT_NAMESPACE,
        description="Group/namespace for related memories (e.g. 'user', 'project-alpha').",
    )
    tags: list[str] = Field(
        default_factory=list[str],
        description="Optional tags for later filtering.",
    )


class _RecallInput(BaseModel):
    query: str = Field(description="What to search for. Natural language is fine.")
    namespace: str | None = Field(
        default=None,
        description="Restrict to one namespace; omit to search all.",
    )
    limit: int = Field(default=5, description="Maximum number of matches to return.")


class _ForgetInput(BaseModel):
    key: str = Field(description="Key of the memory to delete.")
    namespace: str = Field(
        default=DEFAULT_NAMESPACE,
        description="Namespace containing the memory.",
    )


class _ListInput(BaseModel):
    namespace: str | None = Field(
        default=None,
        description="Restrict to one namespace; omit to list all.",
    )
    limit: int = Field(default=20, description="Maximum number of memories to list.")


def _make_remember_tool(store: MemoryStore) -> FunctionTool:
    def _remember(args: _RememberInput, context: ExecutionContext | None = None) -> str:
        _ = context
        rec = store.remember(
            key=args.key,
            value=args.value,
            namespace=args.namespace,
            tags=args.tags,
        )
        return f"remembered {rec.namespace}/{rec.key} (id={rec.id[:8]}…)"

    return FunctionTool(
        name="remember",
        description=(
            "Store a piece of information for later recall. Upserts by "
            "(namespace, key): if the key exists in that namespace, its "
            "value is replaced."
        ),
        schema=_RememberInput,
        func=_remember,
        permission="mutate",
        timeout_ms=5_000,
    )


def _make_recall_tool(store: MemoryStore) -> FunctionTool:
    def _recall(args: _RecallInput, context: ExecutionContext | None = None) -> str:
        _ = context
        matches = store.search(
            query=args.query,
            namespace=args.namespace,
            limit=args.limit,
        )
        return format_matches(matches)

    return FunctionTool(
        name="recall",
        description=(
            "Search stored memories for the given query. Returns the "
            "highest-ranked matches across all namespaces, or restricted "
            "to one namespace if specified."
        ),
        schema=_RecallInput,
        func=_recall,
        permission="read",
        timeout_ms=5_000,
    )


def _make_forget_tool(store: MemoryStore) -> FunctionTool:
    def _forget(args: _ForgetInput, context: ExecutionContext | None = None) -> str:
        _ = context
        removed = store.forget(key=args.key, namespace=args.namespace)
        if removed:
            return f"forgot {args.namespace}/{args.key}"
        return f"no such memory: {args.namespace}/{args.key}"

    return FunctionTool(
        name="forget",
        description="Delete a stored memory by key within a namespace.",
        schema=_ForgetInput,
        func=_forget,
        permission="mutate",
        timeout_ms=5_000,
    )


def _make_list_tool(store: MemoryStore) -> FunctionTool:
    def _list(args: _ListInput, context: ExecutionContext | None = None) -> str:
        _ = context
        memories = store.list_all(namespace=args.namespace, limit=args.limit)
        if not memories:
            return "No memories stored."
        lines = [f"{len(memories)} memory(ies):"]
        for m in memories:
            lines.append(f"- [{m.namespace}] {m.key}: {m.value}")
        return "\n".join(lines)

    return FunctionTool(
        name="list_memories",
        description=(
            "Enumerate stored memories, most recently updated first. "
            "Restrict to one namespace or list all."
        ),
        schema=_ListInput,
        func=_list,
        permission="read",
        timeout_ms=5_000,
    )


def make_memory_tools(store: MemoryStore) -> list[FunctionTool]:
    """Return the four memory tools, all sharing one store.

    Attach to an ``Agent`` alongside other tools:

        store = MemoryStore("workspace/memory.sqlite")
        agent = Agent(
            client=client,
            tools=real_tools + make_memory_tools(store),
            ...
        )

    The workflow owns the store's lifecycle (open, close). See
    ``MemoryStore.__enter__`` / ``__exit__`` for context-manager use.
    """
    return [
        _make_remember_tool(store),
        _make_recall_tool(store),
        _make_forget_tool(store),
        _make_list_tool(store),
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fts_query(raw: str) -> str:
    """Convert a free-form query into a safe FTS5 MATCH expression.

    Whitespace-separated terms are AND-ed implicitly. Each term is wrapped
    in double quotes so FTS5 treats it as a literal phrase — this shields
    ``MATCH`` from special characters (``*``, ``(``, ``:``, ``-``) that
    would otherwise raise a ``sqlite3.OperationalError``. Embedded quotes
    are doubled, per FTS5's escape convention.
    """
    terms = [t for t in raw.split() if t]
    if not terms:
        return '""'
    return " ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    tags_raw = str(row["tags"] or "")
    tags = [t for t in tags_raw.split(",") if t] if tags_raw else []
    return MemoryRecord(
        id=str(row["id"]),
        namespace=str(row["namespace"]),
        key=str(row["key"]),
        value=str(row["value"]),
        tags=tags,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )
