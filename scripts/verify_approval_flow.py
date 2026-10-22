"""Verify --approve / --deny end-to-end through main.py.

Two scenarios (both use a real LLM, so skip conditions apply):

  1. ``main.py --approve "<prompt to write>"`` -> ``write_file`` runs; the
     target file exists with the exact content; the log has a success
     ``ToolResult`` for ``write_file``.
  2. ``main.py --deny "<prompt to write>"`` -> ``write_file`` is denied; no
     file is created; the log has an error ``ToolResult`` for ``write_file``
     whose message mentions "denied".

If the model refuses to call ``write_file`` on a given run (returns text
instead of a tool call), that scenario is skipped - not a failure of the
approval wiring.

Skip conditions (not workflow bugs):
  - ``EXIT_QUOTA`` (3): main.py's distinct quota exit code
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion. Detected via
    ``_verify_common.is_transient_subprocess_failure``. The skip message
    shows the terminal ``Agent failed:`` line, not the fallback warnings.

Timing is printed per scenario and as a total. The suite runs ``main.py``
twice, each with at least two LLM calls; a slow or flaky provider shows up as
60s+ per scenario. That is expected behaviour, not a bug.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.schemas import Message, ToolCall, ToolResult
from scripts._verify_common import (
    is_transient_subprocess_failure,
    terminal_error_line,
)

EXIT_QUOTA = 3

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


def _handle_phase_result(proc: subprocess.CompletedProcess[str], phase_label: str) -> bool:
    """Return True if we should stop (skip or fail); False if phase succeeded."""
    if proc.returncode == 0:
        return False
    if proc.returncode == EXIT_QUOTA:
        skip(f"quota exhausted during {phase_label}: {terminal_error_line(proc.stderr)}")
        return True
    if is_transient_subprocess_failure(proc):
        skip(f"transient LLM failure during {phase_label}: {terminal_error_line(proc.stderr)}")
        return True
    check(False, f"{phase_label} exited {proc.returncode}")
    print(f"  \u2139 stderr:\n{proc.stderr[:1500]}")
    print(f"  \u2139 stdout:\n{proc.stdout[:1500]}")
    return True


def run_main(
    args: list[str],
    env_overrides: dict[str, str],
    cwd: Path,
) -> tuple[subprocess.CompletedProcess[str], float]:
    """Run main.py as a subprocess; return (proc, elapsed_seconds)."""
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env_overrides},
        check=False,
    )
    return proc, time.monotonic() - t0


def write_calls_and_results(
    ctx: ExecutionContext,
) -> tuple[list[ToolCall], list[ToolResult]]:
    calls: list[ToolCall] = []
    results: list[ToolResult] = []
    for ev in ctx.events:
        for item in ev.content:
            if isinstance(item, ToolCall) and item.name == "write_file":
                calls.append(item)
            elif isinstance(item, ToolResult) and item.name == "write_file":
                results.append(item)
    return calls, results


PROMPT = (
    "Use the write_file tool to write the exact text 'hello from pyagent' "
    "to a file named 'greeting.txt' inside the workspace. "
    "Do not answer with text; make the tool call."
)


def _main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    total_start = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        target = workspace / "greeting.txt"

        # ---------------------------------------------------------------
        print("\nTest 1: --approve runs write_file and creates the file")
        approve_log = Path(tmp) / "approve.jsonl"
        env = {
            "WORKSPACE_ROOT": str(workspace),
            "PYAGENT_LOG_PATH": str(approve_log),
        }
        proc, elapsed = run_main(["--approve", PROMPT], env, repo_root)
        print(f"  \u2139 exit code: {proc.returncode}")
        print(f"  \u2139 elapsed:   {elapsed:.1f}s")

        if _handle_phase_result(proc, "--approve run"):
            return

        check(True, "--approve exited 0")
        ctx_a = ExecutionContext.from_log(approve_log)
        calls_a, results_a = write_calls_and_results(ctx_a)

        if not calls_a:
            assistant_msgs = [
                item
                for ev in ctx_a.events
                for item in ev.content
                if isinstance(item, Message) and item.role == "assistant"
            ]
            msg_preview = assistant_msgs[-1].content[:120] if assistant_msgs else ""
            skip(
                "model did not call write_file on --approve run "
                f"(last assistant message: {msg_preview!r})"
            )
        else:
            check(True, "model called write_file")
            check(target.exists(), f"file exists at {target}")
            if target.exists():
                content = target.read_text(encoding="utf-8")
                check(
                    content == "hello from pyagent",
                    f"content matches (got {content!r})",
                )
            check(
                len(results_a) >= 1,
                f"at least one write_file ToolResult (got {len(results_a)})",
            )
            if results_a:
                check(
                    results_a[0].status == "success",
                    f"ToolResult.status=success (got {results_a[0].status})",
                )

        if target.exists():
            target.unlink()

        # ---------------------------------------------------------------
        print("\nTest 2: --deny blocks write_file; no file created")
        deny_log = Path(tmp) / "deny.jsonl"
        env_deny = {
            "WORKSPACE_ROOT": str(workspace),
            "PYAGENT_LOG_PATH": str(deny_log),
        }
        proc, elapsed = run_main(["--deny", PROMPT], env_deny, repo_root)
        print(f"  \u2139 exit code: {proc.returncode}")
        print(f"  \u2139 elapsed:   {elapsed:.1f}s")

        if _handle_phase_result(proc, "--deny run"):
            return

        check(True, "--deny exited 0")
        ctx_d = ExecutionContext.from_log(deny_log)
        calls_d, results_d = write_calls_and_results(ctx_d)

        if not calls_d:
            skip(
                "model did not call write_file on --deny run "
                "(cannot verify denial without a call attempt)"
            )
        else:
            check(True, "model called write_file")
            check(
                not target.exists(),
                f"file NOT created on deny (checked {target})",
            )
            errors = [r for r in results_d if r.status == "error"]
            check(
                len(errors) >= 1,
                f"at least one error ToolResult (got {len(errors)})",
            )
            if errors:
                msg = str(errors[0].content[0]).lower()
                check(
                    "denied" in msg,
                    f"message mentions 'denied' (got {errors[0].content[0]!r})",
                )

    print(f"\n  \u2139 total elapsed: {time.monotonic() - total_start:.1f}s")


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
        print(f"\n\u2705 Approval flow verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Approval flow verified.")


def run() -> None:
    completed = False
    try:
        _main()
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
