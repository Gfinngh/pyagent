"""Verify the iterative brief workflow end-to-end.

Skips on quota or transient LLM failure.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.schemas import ToolCall
from scripts._verify_common import is_transient_subprocess_failure

EXIT_QUOTA = 3
STEPS_CAP = 8
TOPIC = "health benefits of drinking water"
REFINE = "Add a section on criticisms of drinking water advice"

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


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pyagent.workflows.iterative_brief", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=400,
        check=False,
    )


def _count_markers(text: str) -> int:
    m = re.search(
        r"##\s*Key Facts\s*\n(.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return 0
    return len(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", m.group(1)))


def _count_sources(text: str) -> int:
    m = re.search(
        r"##\s*Sources\s*\n(.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return 0
    return len(re.findall(r"^\s*\d+\.\s+\[", m.group(1), re.MULTILINE))


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "briefs-iter"

        # --------------------------------------------------------------
        print("\nSetup phase 1: run iterative_brief for v1")
        proc_v1 = _run(
            [TOPIC, "--output-dir", str(out_dir), "--steps", str(STEPS_CAP)],
            repo_root,
        )
        print(f"  \u2139 exit code: {proc_v1.returncode}")

        if proc_v1.returncode == EXIT_QUOTA:
            skip(f"quota exhausted on v1: {proc_v1.stderr.strip()[:200]}")
            _finalize()
            return
        if is_transient_subprocess_failure(proc_v1):
            skip(f"transient LLM failure on v1: {proc_v1.stderr.strip()[:200]}")
            _finalize()
            return
        if proc_v1.returncode != 0:
            check(False, f"v1 phase exited {proc_v1.returncode}")
            print(f"  \u2139 stderr:\n{proc_v1.stderr[:1500]}")
            print(f"  \u2139 stdout:\n{proc_v1.stdout[:1500]}")
            _finalize()
            return

        check(True, "v1 phase exited 0")

        print("\nTest 1: v1 artifacts exist inside a v1 folder")
        v1_folders = sorted(p for p in out_dir.iterdir() if p.is_dir() and "-v1-" in p.name)
        check(len(v1_folders) == 1, f"exactly one v1 folder (got {len(v1_folders)})")
        if not v1_folders:
            _finalize()
            return
        v1_folder = v1_folders[0]
        v1_md = v1_folder / "brief.md"
        v1_log = v1_folder / "run.log.jsonl"
        print(f"  \u2139 v1 folder: {v1_folder.name}")
        check(v1_md.exists(), "v1 brief.md exists")
        check(v1_log.exists(), "v1 run.log.jsonl exists")
        if not (v1_md.exists() and v1_log.exists()):
            _finalize()
            return

        print("\nTest 2: v1 has required structure")
        v1_content = v1_md.read_text(encoding="utf-8")
        check("## Summary" in v1_content, "v1 has ## Summary")
        check("## Key Facts" in v1_content, "v1 has ## Key Facts")
        check("## Sources" in v1_content, "v1 has ## Sources")
        check("## Open Questions" in v1_content, "v1 has ## Open Questions")
        v1_markers = _count_markers(v1_content)
        v1_sources = _count_sources(v1_content)
        print(f"  \u2139 v1: {len(v1_content)} chars, {v1_markers} markers, {v1_sources} sources")
        check(v1_markers >= 2, f"v1 has >=2 citation markers (got {v1_markers})")
        check(v1_sources >= 3, f"v1 has >=3 sources (got {v1_sources})")

        v1_bytes_before = v1_md.read_bytes()
        v1_ctx = ExecutionContext.from_log(v1_log)
        v1_event_count = len(v1_ctx.events)

        # --------------------------------------------------------------
        print("\nSetup phase 2: run iterative_brief --refine for v2")
        proc_v2 = _run(
            [
                TOPIC,
                "--refine",
                REFINE,
                "--output-dir",
                str(out_dir),
                "--steps",
                str(STEPS_CAP),
            ],
            repo_root,
        )
        print(f"  \u2139 exit code: {proc_v2.returncode}")

        if proc_v2.returncode == EXIT_QUOTA:
            skip(f"quota exhausted on v2: {proc_v2.stderr.strip()[:200]}")
            _finalize()
            return
        if is_transient_subprocess_failure(proc_v2):
            skip(f"transient LLM failure on v2: {proc_v2.stderr.strip()[:200]}")
            _finalize()
            return
        if proc_v2.returncode != 0:
            check(False, f"v2 phase exited {proc_v2.returncode}")
            print(f"  \u2139 stderr:\n{proc_v2.stderr[:1500]}")
            print(f"  \u2139 stdout:\n{proc_v2.stdout[:1500]}")
            _finalize()
            return

        check(True, "v2 phase exited 0")

        print("\nTest 3: v2 artifacts exist inside a v2 folder")
        v2_folders = sorted(p for p in out_dir.iterdir() if p.is_dir() and "-v2-" in p.name)
        check(len(v2_folders) == 1, f"exactly one v2 folder (got {len(v2_folders)})")
        if not v2_folders:
            _finalize()
            return
        v2_folder = v2_folders[0]
        v2_md = v2_folder / "brief.md"
        v2_log = v2_folder / "run.log.jsonl"
        print(f"  \u2139 v2 folder: {v2_folder.name}")
        check(v2_md.exists(), "v2 brief.md exists")
        check(v2_log.exists(), "v2 run.log.jsonl exists")
        if not (v2_md.exists() and v2_log.exists()):
            _finalize()
            return

        print("\nTest 4: v1 folder unchanged (immutable source)")
        v1_bytes_after = v1_md.read_bytes()
        check(
            v1_bytes_before == v1_bytes_after,
            "v1 bytes identical before and after refinement",
        )
        v1_ctx_after = ExecutionContext.from_log(v1_log)
        check(
            len(v1_ctx_after.events) == v1_event_count,
            f"v1 log still {v1_event_count} events (got {len(v1_ctx_after.events)})",
        )

        print("\nTest 5: v2 differs from v1")
        v2_content = v2_md.read_text(encoding="utf-8")
        check(v1_content != v2_content, "v2 content differs from v1")
        delta = len(v2_content) - len(v1_content)
        print(f"  \u2139 v2: {len(v2_content)} chars ({delta:+d} vs v1)")
        check(
            abs(delta) >= 100,
            f"v2 differs in length by >=100 chars (got {delta:+d})",
        )

        print("\nTest 6: v2 preserves required structure")
        check("## Summary" in v2_content, "v2 has ## Summary")
        check("## Key Facts" in v2_content, "v2 has ## Key Facts")
        check("## Sources" in v2_content, "v2 has ## Sources")
        check("## Open Questions" in v2_content, "v2 has ## Open Questions")

        print("\nTest 7: v2 mentions the refinement subject")
        check(
            "critic" in v2_content.lower(),
            "v2 contains 'critic' (refinement was about criticisms)",
        )

        print("\nTest 8: v2 sources not fewer than v1")
        v2_sources = _count_sources(v2_content)
        print(f"  \u2139 v2: {v2_sources} sources (v1 had {v1_sources})")
        check(
            v2_sources >= v1_sources,
            f"v2 sources >= v1 sources (v1={v1_sources}, v2={v2_sources})",
        )

        print("\nTest 9: v2 log is superset of v1 log")
        v2_ctx = ExecutionContext.from_log(v2_log)
        check(
            len(v2_ctx.events) > v1_event_count,
            f"v2 has more events than v1 (v1={v1_event_count}, v2={len(v2_ctx.events)})",
        )
        v1_ids = [ev.id for ev in v1_ctx.events]
        v2_ids = [ev.id for ev in v2_ctx.events]
        check(
            v2_ids[: len(v1_ids)] == v1_ids,
            "v2 log begins with exactly v1's event ids",
        )

        print("\nTest 10: v2 log shows a write to the v2 folder")
        write_calls: list[ToolCall] = [
            item
            for ev in v2_ctx.events
            for item in ev.content
            if isinstance(item, ToolCall) and item.name == "write_file"
        ]
        check(
            len(write_calls) >= 1,
            f"at least one write_file call (got {len(write_calls)})",
        )
        v2_paths: list[str] = [
            str(c.arguments.get("path", ""))
            for c in write_calls
            if "v2-" in str(c.arguments.get("path", ""))
        ]
        check(
            len(v2_paths) >= 1,
            f"a write targeted a v2 folder (paths seen: "
            f"{[c.arguments.get('path') for c in write_calls]})",
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
        print(f"\n\u2705 Iterative brief workflow verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Iterative brief workflow verified.")


if __name__ == "__main__":
    main()
