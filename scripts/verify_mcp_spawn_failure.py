"""Verify main.py handles MCP spawn failure cleanly.

Runs ``main.py`` with ``PYAGENT_MCP_MODULE=nonexistent_module_xyz``, which
points the MCP server spawn at a module that doesn't exist. The subprocess
starts Python, which exits immediately with "No module named ...". Expected
behaviour:

  - main.py exits with code 1 (EXIT_FAILURE), not with a raw traceback
  - stderr contains "MCP server failed to start"
  - no log file is written (no events were recorded)
  - the whole run completes in well under the 10s connect cap
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_FAILURES: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        _FAILURES.append(msg)
        print(f"  \u2717 {msg}")
    else:
        print(f"  \u2713 {msg}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "spawn_failure.jsonl"

        print("\nSetup: run main.py with a broken MCP module")
        env = {
            **os.environ,
            "PYAGENT_MCP_MODULE": "nonexistent_module_xyz",
            "PYAGENT_LOG_PATH": str(log_path),
        }
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "main.py", "test prompt"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )
        elapsed = time.monotonic() - t0
        print(f"  \u2139 exit code: {proc.returncode}")
        print(f"  \u2139 elapsed:   {elapsed:.1f}s")

        print("\nTest 1: exits with EXIT_FAILURE, not a crash")
        check(proc.returncode == 1, f"exit code 1 (got {proc.returncode})")

        print("\nTest 2: stderr contains a clean error message")
        stderr_lower = proc.stderr.lower()
        check(
            "mcp server failed to start" in stderr_lower,
            f"stderr names the failure (got {proc.stderr[:200]!r})",
        )
        check(
            "traceback" not in stderr_lower,
            "stderr has no raw traceback",
        )
        check(
            "nonexistent_module_xyz" in proc.stderr or "no module named" in stderr_lower,
            f"stderr hints at the missing module (got {proc.stderr[:200]!r})",
        )

        print("\nTest 3: no log file written (no events recorded)")
        check(
            not log_path.exists() or log_path.stat().st_size == 0,
            f"log is absent or empty (checked {log_path})",
        )

        print("\nTest 4: fails fast (well under the 10s connect cap)")
        check(elapsed < 8.0, f"elapsed under 8s (got {elapsed:.1f}s)")

        print("\nTest 5: no tools were discovered")
        out_lower = proc.stdout.lower()
        check(
            "discovered tools" not in out_lower,
            "no tool-discovery line printed",
        )


def run() -> None:
    main()
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n\u2705 MCP spawn failure handled cleanly.")


if __name__ == "__main__":
    run()
