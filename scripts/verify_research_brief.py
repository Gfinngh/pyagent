"""Verify the research brief workflow end-to-end.

Skips on quota or transient LLM failure.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.schemas import ToolCall
from scripts._verify_common import is_transient_subprocess_failure

EXIT_QUOTA = 3
STEPS_CAP = 6

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


def _count_urls(text: str) -> int:
    return len(set(re.findall(r"https?://[^\s)\]>\"']+", text)))


def _count_citation_markers(text: str) -> int:
    m = re.search(
        r"##\s*Key Facts\s*\n(.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return 0
    return len(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", m.group(1)))


def _numbered_sources(text: str) -> int:
    m = re.search(
        r"##\s*Sources\s*\n(.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return 0
    return len(re.findall(r"^\s*\d+\.\s+\[", m.group(1), re.MULTILINE))


def _generated_timestamp(text: str) -> datetime | None:
    m = re.search(r"\*\*Generated\*\*:\s*(\S+)", text)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    topic = "health benefits of drinking water"

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "briefs"

        print("\nSetup: run the workflow as a subprocess")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyagent.workflows.research_brief",
                topic,
                "--output-dir",
                str(output_dir),
                "--steps",
                str(STEPS_CAP),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=400,
            check=False,
        )
        print(f"  \u2139 exit code: {proc.returncode}")

        if proc.returncode == EXIT_QUOTA:
            skip(f"quota exhausted: {proc.stderr.strip()[:200]}")
            _finalize()
            return
        if is_transient_subprocess_failure(proc):
            skip(f"transient LLM failure: {proc.stderr.strip()[:200]}")
            _finalize()
            return
        if proc.returncode != 0:
            check(False, f"workflow exited {proc.returncode}")
            print(f"  \u2139 stderr:\n{proc.stderr[:1500]}")
            print(f"  \u2139 stdout:\n{proc.stdout[:1500]}")
            _finalize()
            return

        check(True, "workflow exited 0")

        print("\nTest 1: brief file created inside per-run folder")
        run_folders = sorted(p for p in output_dir.iterdir() if p.is_dir())
        check(
            len(run_folders) == 1,
            f"exactly one per-run folder (got {len(run_folders)})",
        )
        if not run_folders:
            _finalize()
            return
        run_folder = run_folders[0]
        print(f"  \u2139 run folder: {run_folder.name}")

        brief = run_folder / "brief.md"
        check(brief.exists(), f"brief.md exists at {brief}")
        if not brief.exists():
            _finalize()
            return

        print("\nTest 2: required markdown sections present")
        content = brief.read_text(encoding="utf-8")
        check(len(content) > 100, f"content has body (got {len(content)} chars)")
        check("## Summary" in content, "## Summary section present")
        check("## Key Facts" in content, "## Key Facts section present")
        check("## Sources" in content, "## Sources section present")
        check("## Open Questions" in content, "## Open Questions section present")
        check(
            any(line.startswith("# ") for line in content.splitlines()),
            "top-level # heading present",
        )

        print("\nTest 3: at least 3 source URLs")
        n_urls = _count_urls(content)
        print(f"  \u2139 distinct URLs: {n_urls}")
        check(n_urls >= 3, f"at least 3 distinct URLs (got {n_urls})")

        print("\nTest 4: Sources section uses numbered entries")
        n_numbered = _numbered_sources(content)
        print(f"  \u2139 numbered source lines: {n_numbered}")
        check(
            n_numbered >= 3,
            f"at least 3 numbered source entries (got {n_numbered})",
        )

        print("\nTest 5: Key Facts carries citation markers")
        n_markers = _count_citation_markers(content)
        print(f"  \u2139 [N] markers in Key Facts: {n_markers}")
        check(
            n_markers >= 2,
            f"at least 2 [N] citation markers in Key Facts (got {n_markers})",
        )

        print("\nTest 6: log file created and parseable")
        log_path = run_folder / "run.log.jsonl"
        check(log_path.exists(), "run.log.jsonl exists")
        if not log_path.exists():
            _finalize()
            return

        ctx = ExecutionContext.from_log(log_path)
        check(len(ctx.events) >= 4, f">=4 events (got {len(ctx.events)})")
        check(ctx.final_result is not None, "log has final_result")

        print("\nTest 7: agent used search_web and write_file")
        calls: list[ToolCall] = [
            item for ev in ctx.events for item in ev.content if isinstance(item, ToolCall)
        ]
        search_calls = [c for c in calls if c.name == "search_web"]
        write_calls = [c for c in calls if c.name == "write_file"]
        check(
            len(search_calls) >= 1,
            f">=1 search_web call (got {len(search_calls)})",
        )
        check(
            len(write_calls) >= 1,
            f">=1 write_file call (got {len(write_calls)})",
        )
        print(
            f"  \u2139 tool calls: search_web x{len(search_calls)}, write_file x{len(write_calls)}"
        )

        print("\nTest 8: max_steps respected")
        check(
            ctx.current_step <= STEPS_CAP,
            f"current_step <= {STEPS_CAP} (got {ctx.current_step})",
        )

        print("\nTest 9: 'Generated' timestamp is real (not fabricated)")
        gen_ts = _generated_timestamp(content)
        if gen_ts is None:
            check(False, "**Generated**: line has ISO timestamp")
        else:
            age_seconds = (datetime.now(UTC) - gen_ts).total_seconds()
            print(f"  \u2139 timestamp: {gen_ts.isoformat()} (age {age_seconds:.0f}s)")
            check(
                0 <= age_seconds <= 3600,
                f"timestamp is recent, within last hour (age {age_seconds:.0f}s)",
            )

        print("\nTest 10: brief does not contain obvious fabrication markers")
        check(
            "example.com" not in content.lower(),
            "brief has no 'example.com' placeholder URLs",
        )


def _finalize() -> None:
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        if _SKIPPED:
            print(f"\n\u26a0 {len(_SKIPPED)} skipped:")
            for s in _SKIPPED:
                print(f"  - {s}")
        sys.exit(1)
    if _SKIPPED:
        print(f"\n\u2705 Research brief workflow verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Research brief workflow verified.")


if __name__ == "__main__":
    main()
