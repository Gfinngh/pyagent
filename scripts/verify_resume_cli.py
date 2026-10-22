"""Phase 17: main.py --resume CLI smoke test.

Four-step flow, all via subprocess so the real CLI is exercised:

  1. Fresh run produces log A (completed, has final_result).
  2. ``main.py --resume <logA>`` on a completed run is a no-op: exit 0,
     message mentions "already complete", and NO new log is written.
  3. ``main.py --resume <logA> "follow-up"`` continues the conversation:
     exit 0, new log B written, more events than A, different final_result.
  4. Log A remains byte-for-byte unchanged (immutable source).

Skip conditions (not workflow bugs):
  - ``EXIT_QUOTA`` (3): main.py's distinct quota exit code
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion. Detected via
    ``_verify_common.is_transient_subprocess_failure``. The skip message
    shows the terminal ``Agent failed:`` line, not the fallback warnings.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pyagent.context import ExecutionContext
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
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, **env_overrides},
        check=False,
    )


def _main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log_a = tmp_path / "run_a.jsonl"
        log_b = tmp_path / "run_b.jsonl"

        # ---------------------------------------------------------------
        print("\nSetup: fresh run produces log A")
        proc_a = run_main(
            ["Use the calculator to compute 2 + 2. Do not compute it yourself."],
            {"PYAGENT_LOG_PATH": str(log_a)},
            repo_root,
        )
        if _handle_phase_result(proc_a, "fresh run"):
            return
        check(proc_a.returncode == 0, f"fresh run exit 0 (got {proc_a.returncode})")
        check(log_a.exists(), "log A exists")
        if not log_a.exists():
            return

        ctx_a = ExecutionContext.from_log(log_a)
        check(ctx_a.final_result is not None, "log A is completed")
        events_a = len(ctx_a.events)
        check(events_a >= 4, f"log A has >=4 events (got {events_a})")

        # Byte-for-byte snapshot of log A for the immutability check at the end
        a_bytes_before = log_a.read_bytes()

        # ---------------------------------------------------------------
        print("\nTest 1: resume without follow-up on completed run is a no-op")
        proc_noop = run_main(
            ["--resume", str(log_a)],
            {"PYAGENT_LOG_PATH": str(log_b)},
            repo_root,
        )
        if _handle_phase_result(proc_noop, "resume no-op"):
            return
        check(
            proc_noop.returncode == 0,
            f"no-op exit 0 (got {proc_noop.returncode})",
        )
        noop_out = strip_ansi(proc_noop.stdout).lower()
        check(
            "already complete" in noop_out,
            f"message says 'already complete' (got {noop_out[:120]!r})",
        )
        check(not log_b.exists(), "no new log written on no-op")

        # ---------------------------------------------------------------
        print("\nTest 2: resume with follow-up continues the conversation")
        proc_resume = run_main(
            ["--resume", str(log_a), "Now what is that result times 10?"],
            {"PYAGENT_LOG_PATH": str(log_b)},
            repo_root,
        )
        if _handle_phase_result(proc_resume, "resume continue"):
            return
        check(
            proc_resume.returncode == 0,
            f"resume exit 0 (got {proc_resume.returncode})",
        )
        check(log_b.exists(), "log B exists")
        if not log_b.exists():
            return

        ctx_b = ExecutionContext.from_log(log_b)
        check(
            len(ctx_b.events) > events_a,
            f"log B has more events ({len(ctx_b.events)} vs {events_a})",
        )
        check(ctx_b.final_result is not None, "log B has final_result")
        check(
            ctx_b.final_result != ctx_a.final_result,
            f"log B final_result differs from A "
            f"(A={ctx_a.final_result!r}, B={ctx_b.final_result!r})",
        )

        # ---------------------------------------------------------------
        print("\nTest 3: stdout Output matches log B final_result")
        out_b = strip_ansi(proc_resume.stdout)
        output_line = next((ln for ln in out_b.splitlines() if ln.startswith("Output: ")), None)
        check(output_line is not None, "'Output:' line present in resume stdout")
        if output_line is not None and isinstance(ctx_b.final_result, str):
            stdout_value = output_line[len("Output: ") :].strip()
            check(
                stdout_value == ctx_b.final_result,
                f"stdout matches log B final_result "
                f"(stdout={stdout_value!r}, log={ctx_b.final_result!r})",
            )

        # ---------------------------------------------------------------
        print("\nTest 4: log A unchanged (immutable source)")
        a_bytes_after = log_a.read_bytes()
        check(
            a_bytes_before == a_bytes_after,
            "log A bytes identical before and after resume",
        )
        ctx_a2 = ExecutionContext.from_log(log_a)
        check(
            len(ctx_a2.events) == events_a,
            f"log A still {events_a} events (got {len(ctx_a2.events)})",
        )
        check(
            ctx_a2.final_result == ctx_a.final_result,
            f"log A final_result unchanged (got {ctx_a2.final_result!r})",
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
        return

    if _SKIPPED:
        print(f"\n\u2705 Phase 17 (resume CLI) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 17 (resume CLI) verified.")


def run() -> None:
    completed = False
    try:
        _main()
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
