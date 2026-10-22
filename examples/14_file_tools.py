"""14 — Workspace file tools.

`list_files`, `read_file`, and `write_file` confine every path to
`WORKSPACE_ROOT`. Absolute paths and `..` traversal are rejected before any
filesystem access. Hidden path segments are skipped when listing.

    pixi run python examples/14_file_tools.py
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from examples._common import extract_text, mcp_with_workspace, section
from pyagent.client import McpToolError, ToolOverrides

WORKSPACE = Path(".examples/workspace-files")


async def main() -> None:
    # Fresh workspace so the output is reproducible.
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "hello.txt").write_text("hello world\n", encoding="utf-8")
    (WORKSPACE / "notes.md").write_text("# Notes\n\nsome notes\n", encoding="utf-8")
    (WORKSPACE / "subdir").mkdir()
    (WORKSPACE / "subdir" / "nested.txt").write_text("nested\n", encoding="utf-8")

    overrides = {
        "write_file": ToolOverrides(permission="mutate", timeout_ms=10_000),
    }

    async with mcp_with_workspace(WORKSPACE, overrides=overrides) as tools:
        by_name = {t.name: t for t in tools}

        section('list_files(pattern="**/*")')
        raw = await by_name["list_files"].call({"pattern": "**/*"})
        print(extract_text(raw))

        section('read_file(path="hello.txt")')
        raw = await by_name["read_file"].call({"path": "hello.txt"})
        print(extract_text(raw))

        section('write_file(path="created.txt", content="written\\n")')
        raw = await by_name["write_file"].call({"path": "created.txt", "content": "written\n"})
        print(extract_text(raw))
        raw = await by_name["read_file"].call({"path": "created.txt"})
        print(extract_text(raw))

        section('write_file(path="deep/nested/new.txt")  — parents auto-created')
        raw = await by_name["write_file"].call({"path": "deep/nested/new.txt", "content": "x\n"})
        print(extract_text(raw))

        section("path confinement")
        for bad in ("/etc/passwd", "../escape.txt", "subdir/../../escape.txt"):
            try:
                await by_name["read_file"].call({"path": bad})
                print(f"  {bad:<28} → UNEXPECTEDLY ALLOWED")
            except McpToolError as exc:
                print(f"  {bad:<28} → rejected: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
