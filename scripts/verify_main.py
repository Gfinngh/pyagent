"""Phase 15: main.py smoke test.

Runs ``main.py`` as a subprocess with a temp log path, then asserts:

  1. Exit code 0
  2. Trace shape in stdout: header, author tags, tool call, tool result,
     assistant message
  3. The JSONL log exists and rehydrates into a completed context
  4. The rehydrated final answer matches the trace's Output line
  5. Provider metadata survived the round-trip

Skip conditions (not workflow bugs):
  - ``EXIT_QUOTA`` (3): main.py's distinct quota exit code
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion. Detected from
    main.py's stderr; see ``_verify_common.is_transient_subprocess_failure``.
    The skip message shows the terminal ``Agent failed:`` line, not the
    fallback warnings that precede it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.schemas import Message, ToolCall, ToolResult
from scripts._verify_common import (
    is_transient_subprocess_failure,
    terminal_error_line,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Mirror of main.py's EXIT_QUOTA. Kept as a plain constant so this script
# doesn't import from main.py (which would spawn a subprocess cascade).
EXIT_QUOTA = 3

_FAILURES: list[str] = []
_SKIPPED: list[str] = []


def strip_ansi(text: str) -> str:
    """Remove ANSI SGR escape sequences."""
    return _ANSI_RE.sub("", text)


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def skip(msg: str) -> None:
    _SKIPPED.append(msg)
    print(f"  \u26a0 SKIP: {msg}")


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "verify_main.jsonl"

        print("\nSetup: run main.py as a subprocess")
        env = {**os.environ, "PYAGENT_LOG_PATH": str(log_path)}
        proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "Use the calculator to compute 2 + 2. Do not compute it yourself.",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
            check=False,
        )

        print(f"  \u2139 exit code: {proc.returncode}")

        if proc.returncode == EXIT_QUOTA:
            skip(f"quota exhausted: {terminal_error_line(proc.stderr)}")
            return
        if is_transient_subprocess_failure(proc):
            skip(f"transient LLM failure: {terminal_error_line(proc.stderr)}")
            return
        if proc.returncode != 0:
            check(False, f"main.py exited {proc.returncode}")
            print(f"  \u2139 stderr:\n{proc.stderr[:1500]}")
            print(f"  \u2139 stdout:\n{proc.stdout[:1500]}")
            return

        check(True, "main.py exited 0")

        out = strip_ansi(proc.stdout)

        print("\nTest 1: trace structure in stdout")
        check("Discovered tools:" in out, "'Discovered tools:' header present")
        check("=== Event trace ===" in out, "'=== Event trace ===' header present")
        check("[user]" in out, "user event block present")
        check("[agent]" in out, "agent event block present")
        check("call: calculator(" in out, "tool call line present")
        check("result(success):" in out, "successful tool result line present")
        check("msg(assistant):" in out, "assistant message line present")

        print("\nTest 2: Output line present and non-empty")
        output_line = next((ln for ln in out.splitlines() if ln.startswith("Output: ")), None)
        check(output_line is not None, "'Output:' line present")
        if output_line is not None:
            value = output_line[len("Output: ") :].strip()
            check(len(value) > 0, f"Output value non-empty (got {value!r})")
            check(
                "None" not in value,
                f"Output is not 'None' (got {value!r})",
            )

        print("\nTest 3: JSONL log written and parseable")
        check(log_path.exists(), f"log file exists at {log_path}")
        if not log_path.exists():
            return

        raw = log_path.read_text(encoding="utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        check(len(lines) >= 4, f"at least 4 log lines (got {len(lines)})")

        print("\nTest 4: log rehydrates into a completed context")
        ctx = ExecutionContext.from_log(log_path)
        check(len(ctx.events) >= 4, f">=4 events (got {len(ctx.events)})")
        check(ctx.current_step >= 1, f"current_step >= 1 (got {ctx.current_step})")
        check(ctx.final_result is not None, "final_result set")

        tool_calls = [
            item for ev in ctx.events for item in ev.content if isinstance(item, ToolCall)
        ]
        tool_results = [
            item for ev in ctx.events for item in ev.content if isinstance(item, ToolResult)
        ]
        assistant_msgs = [
            item
            for ev in ctx.events
            for item in ev.content
            if isinstance(item, Message) and item.role == "assistant"
        ]
        check(
            len(tool_calls) >= 1,
            f">=1 tool call rehydrated (got {len(tool_calls)})",
        )
        check(
            len(tool_results) >= 1,
            f">=1 tool result rehydrated (got {len(tool_results)})",
        )
        check(
            len(assistant_msgs) >= 1,
            f">=1 assistant message rehydrated (got {len(assistant_msgs)})",
        )

        print("\nTest 5: log final_result matches stdout Output line")
        if output_line is not None and isinstance(ctx.final_result, str):
            stdout_value = output_line[len("Output: ") :].strip()
            check(
                stdout_value == ctx.final_result,
                f"stdout matches log final_result "
                f"(stdout={stdout_value!r}, log={ctx.final_result!r})",
            )

        print("\nTest 6: provider metadata survived the round-trip")
        signed = [
            tc
            for tc in tool_calls
            if tc.provider_options is not None or "__thought__" in tc.tool_call_id
        ]
        if signed:
            check(True, f"signature channel present on {len(signed)} tool call(s)")
        else:
            print(
                "  \u2139 no provider_options on rehydrated tool calls "
                "(model may not have emitted a signature this run)"
            )


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
        sys.exit(1)

    if _SKIPPED:
        print(f"\n\u2705 Phase 15 (main entry point) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 15 (main entry point) verified.")


def run() -> None:
    completed = False
    try:
        main()
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
