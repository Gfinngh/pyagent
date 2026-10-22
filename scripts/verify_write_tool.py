"""Verify the write_file MCP tool: success path and sandbox rejections.

Five deterministic scenarios (fake workspace, no LLM):

  1. ``write_file`` is discovered in the tool list
  2. Simple write to ``greeting.txt`` succeeds; content matches
  3. Nested write to ``sub/nested.txt`` succeeds (auto-creates dirs)
  4. ``../escape.txt`` is rejected; no file escapes the workspace
  5. Absolute ``/tmp/abs.txt`` is rejected; no file created outside

``WORKSPACE_ROOT`` is set before ``McpConnection.connect()`` so the spawned
MCP server inherits it via the default ``env=dict(os.environ)``. It is
restored after the test to avoid leaking into subsequent runs in the same
process.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from pyagent.client import McpConnection, McpServerParams

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def server_params() -> McpServerParams:
    return McpServerParams(command=sys.executable, args=["-m", "pyagent.server"])


async def _run(workspace: Path, escape_dir: Path) -> None:
    async with McpConnection(server_params()) as conn:
        # --------------------------------------------------------------
        print("\nTest 1: write_file discovered")
        tools = await conn.list_tools()
        names = sorted(t.name for t in tools)
        check("write_file" in names, f"write_file present (got {names})")

        # --------------------------------------------------------------
        print("\nTest 2: simple write succeeds")
        result = await conn.call_tool(
            "write_file",
            {"path": "greeting.txt", "content": "hello from pyagent"},
        )
        check(not result.is_error, f"write did not error (error={result.is_error})")
        target = workspace / "greeting.txt"
        check(target.exists(), f"file exists at {target}")
        if target.exists():
            check(
                target.read_text(encoding="utf-8") == "hello from pyagent",
                f"content matches (got {target.read_text(encoding='utf-8')!r})",
            )

        # --------------------------------------------------------------
        print("\nTest 3: nested write auto-creates directories")
        result = await conn.call_tool(
            "write_file",
            {"path": "sub/nested.txt", "content": "nested content"},
        )
        check(not result.is_error, "nested write did not error")
        nested = workspace / "sub" / "nested.txt"
        check(nested.exists(), f"nested file exists at {nested}")
        if nested.exists():
            check(
                nested.read_text(encoding="utf-8") == "nested content",
                "nested content matches",
            )

        # --------------------------------------------------------------
        print("\nTest 4: '..' traversal rejected, no file escapes")
        result = await conn.call_tool(
            "write_file",
            {"path": "../escape.txt", "content": "should not exist"},
        )
        check(result.is_error, f"traversal is_error=True (got {result.is_error})")
        escaped = escape_dir / "escape.txt"
        check(
            not escaped.exists(),
            f"no file created outside workspace (checked {escaped})",
        )

        # --------------------------------------------------------------
        print("\nTest 5: absolute path rejected, no file created")
        abs_target = Path(tempfile.gettempdir()) / "pyagent_abs_should_not_exist.txt"
        if abs_target.exists():
            abs_target.unlink()
        result = await conn.call_tool(
            "write_file",
            {"path": str(abs_target), "content": "should not exist"},
        )
        check(result.is_error, f"absolute is_error=True (got {result.is_error})")
        check(
            not abs_target.exists(),
            f"absolute path file not created (checked {abs_target})",
        )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        escape_dir = Path(tmp)
        original = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = str(workspace)
        try:
            await _run(workspace, escape_dir)
        finally:
            if original is None:
                os.environ.pop("WORKSPACE_ROOT", None)
            else:
                os.environ["WORKSPACE_ROOT"] = original


def run() -> None:
    asyncio.run(main())
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 write_file sandbox verified.")


if __name__ == "__main__":
    run()
