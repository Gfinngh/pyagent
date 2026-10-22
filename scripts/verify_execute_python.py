"""Verify execute_python: Pyodide/WASM sandbox mechanics and permissions.

Skips if Deno is unavailable or if the Pyodide warm-up fails (first use needs
network). The sandbox itself runs with ``--cached-only --allow-read=<npm-cache>``
— no runtime network, no env, no subprocess, no FFI, no writes.

Scenarios:

  1.  Tool discovered; declared `permission="dangerous"`.
  2.  Basic execution: print(2+2) → "4".
  3.  stderr captured.
  4.  Non-zero exit surfaces traceback.
  5.  Network is blocked: socket.create_connection → OSError.
  6.  Filesystem is blocked: open('/etc/passwd') → OSError.
  7.  Timeout: while True: pass → "timed out".
  8.  Output truncation at 16 KB.
  9.  Permission tier: no allow_dangerous → denied.
 10.  Permission tier: allow_dangerous + confirm=True → allowed.
 11.  Permission tier: allow_dangerous + confirm=False → denied.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from typing import Any

from pyagent.agent import Agent
from pyagent.client import (
    McpConnection,
    McpServerParams,
    McpToolError,
    ToolOverrides,
    load_mcp_tools,
)
from pyagent.context import ExecutionContext
from pyagent.schemas import ToolCall, ToolResult
from pyagent.server import sandbox_is_ready, sandbox_script_path
from pyagent.tools import MultipartContent, RemoteTool, TextPart

_FAILURES: list[str] = []
_SKIPPED: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def skip(msg: str) -> None:
    _SKIPPED.append(msg)
    print(f"  \u26a0 SKIP: {msg}")


def _deno_available() -> bool:
    return shutil.which("deno") is not None


def _server_params() -> McpServerParams:
    return McpServerParams(command=sys.executable, args=["-m", "pyagent.server"])


def _first_text(content: list[object]) -> str:
    if not content:
        return ""
    part = content[0]
    if isinstance(part, MultipartContent):
        return "\n".join(p.text for p in part.parts if isinstance(p, TextPart))
    return str(part)


async def _run_tool(tool: RemoteTool, args: dict[str, object]) -> tuple[bool, str]:
    try:
        out = await tool.call(args)
    except McpToolError as exc:
        return True, str(exc)
    if isinstance(out, MultipartContent):
        return False, _first_text([out])
    return False, str(out)


async def _phase_sandbox() -> None:
    print("\nSetup: connect to MCP server")
    async with McpConnection(_server_params()) as conn:
        # `execute_python` is declared `permission="dangerous"` in
        # `main.py`'s `TOOL_OVERRIDES`. The Python MCP server does not
        # advertise permissions, so `load_mcp_tools` defaults every
        # tool to `read` unless a `ToolOverrides` is supplied.
        #
        # Note: the mapping values must be `ToolOverrides` **instances**,
        # not plain dicts. `client.py` reads `overrides.permission` and
        # `overrides.timeout_ms` as attributes.
        overrides: dict[str, ToolOverrides] = {
            "execute_python": ToolOverrides(
                permission="dangerous",
                timeout_ms=15_000,
            ),
        }
        tools = await load_mcp_tools(conn, overrides=overrides)
        by_name = {t.name: t for t in tools}
        check("execute_python" in by_name, "execute_python discovered")
        ex = by_name.get("execute_python")
        if ex is None:
            return

        print("\nTest 1: tool is permission='dangerous'")
        check(ex.permission == "dangerous", f"permission={ex.permission!r}")

        print("\nTest 2: basic execution")
        raised, out = await _run_tool(ex, {"code": "print(2 + 2)"})
        check(not raised, f"no error raised (got {out[:200]!r})")
        check("4" in out, f"output contains '4' (got {out!r})")

        print("\nTest 3: stderr is captured")
        raised, out = await _run_tool(ex, {"code": "import sys; sys.stderr.write('to-stderr\\n')"})
        check(not raised, "no error raised")
        check("to-stderr" in out, f"stderr text present (got {out!r})")

        print("\nTest 4: non-zero exit surfaces traceback")
        raised, out = await _run_tool(ex, {"code": "raise ValueError('boom-in-sandbox')"})
        check(not raised, "tool itself does not raise")
        check(
            "boom-in-sandbox" in out,
            f"traceback surfaced (got {out[:200]!r})",
        )

        print("\nTest 5: network is blocked")
        raised, out = await _run_tool(
            ex,
            {
                "code": (
                    "import socket\n"
                    "try:\n"
                    "    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
                    "    print('CONNECTED')\n"
                    "except OSError as e:\n"
                    "    print(f'blocked: {e}')\n"
                )
            },
        )
        check(not raised, "no error raised")
        check("CONNECTED" not in out, f"must not connect (got {out!r})")

        print("\nTest 6: filesystem is blocked")
        raised, out = await _run_tool(
            ex,
            {
                "code": (
                    "try:\n"
                    "    open('/etc/passwd').read()\n"
                    "    print('READABLE')\n"
                    "except OSError as e:\n"
                    "    print(f'blocked: {e}')\n"
                )
            },
        )
        check(not raised, "no error raised")
        check("READABLE" not in out, f"fs must not be readable (got {out!r})")

        print("\nTest 7: timeout is enforced")
        raised, out = await _run_tool(ex, {"code": "while True: pass", "timeout_s": 2})
        check(not raised, "tool itself does not raise")
        check("timed out" in out.lower(), f"timeout marker (got {out[-200:]!r})")

        print("\nTest 8: output is truncated at 16 KB")
        raised, out = await _run_tool(ex, {"code": "print('x' * 100_000)"})
        check(not raised, "no error raised")
        check("truncated" in out.lower(), f"truncation marker (got {out[-120:]!r})")


def _confirm_yes(tc: ToolCall) -> bool:
    _ = tc
    return True


def _confirm_no(tc: ToolCall) -> bool:
    _ = tc
    return False


async def _phase_permissions() -> None:
    from pyagent.server import execute_python as _server_execute_python

    async def _stub_call(args: dict[str, object]) -> object:
        code = str(args.get("code", ""))
        raw_timeout = args.get("timeout_s", 5)
        timeout_s = int(raw_timeout) if isinstance(raw_timeout, (int, str)) else 5
        out = await _server_execute_python(code, timeout_s)
        return MultipartContent(parts=[TextPart(text=str(out))])

    tool = RemoteTool(
        name="execute_python",
        description="stub",
        input_schema={"type": "object"},
        call=_stub_call,
        permission="dangerous",
        timeout_ms=15_000,
    )

    class _NullClient:
        async def generate(self, request: Any) -> Any:
            _ = request
            raise RuntimeError("should not be called")

    async def _agent_act(
        allow_dangerous: bool,
        confirm: Any,
    ) -> list[ToolResult]:
        agent = Agent(
            client=_NullClient(),
            tools=[tool],
            allow_dangerous=allow_dangerous,
            confirm_tool_call=confirm,
        )
        tc = ToolCall.model_validate(
            {
                "tool_call_id": "tc-perm",
                "name": "execute_python",
                "arguments": {"code": "print('ran')"},
            }
        )
        return await agent.act(ExecutionContext(), [tc])

    print("\nTest 9: no allow_dangerous → denied")
    results = await _agent_act(False, _confirm_yes)
    check(len(results) == 1, "one result")
    check(results[0].status == "error", f"denied (got {results[0].status})")
    msg = str(results[0].content[0]).lower()
    check("dangerous" in msg, f"message names 'dangerous' (got {msg!r})")

    print("\nTest 10: allow_dangerous + confirm=True → allowed")
    results = await _agent_act(True, _confirm_yes)
    check(len(results) == 1, "one result")
    check(results[0].status == "success", f"allowed (got {results[0].status})")
    if results[0].status == "success":
        text = _first_text(list(results[0].content))
        check("ran" in text, f"container ran code (got {text!r})")

    print("\nTest 11: allow_dangerous + confirm=False → denied")
    results = await _agent_act(True, _confirm_no)
    check(len(results) == 1, "one result")
    check(results[0].status == "error", f"denied (got {results[0].status})")
    msg = str(results[0].content[0]).lower()
    check("denied" in msg, f"message names 'denied' (got {msg!r})")


async def _main() -> None:
    if not _deno_available():
        skip("Deno is not installed — skipping execute_python verification")
        return

    sandbox_script = sandbox_script_path()
    if not sandbox_script.exists():
        skip(f"sandbox script not found: {sandbox_script}")
        return

    try:
        await asyncio.to_thread(sandbox_is_ready)
    except Exception as exc:
        skip(f"Pyodide warm-up failed: {exc}")
        return

    await _phase_sandbox()
    await _phase_permissions()


def main() -> None:
    asyncio.run(_main())


def _finalize(*, completed: bool) -> None:
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        if _SKIPPED:
            print(f"\n\u26a0 {len(_SKIPPED)} skipped:")
            for s in _SKIPPED:
                print(f"  - {s}")
        sys.exit(1)
    if not completed:
        print("\n❌ verify script crashed before completing")
        return

    if _SKIPPED:
        print(f"\n\u2705 execute_python verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 execute_python verified.")


def run() -> None:
    completed = False
    try:
        main()
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
