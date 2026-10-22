"""Verify the notes digest workflow end-to-end.

Skips on quota or transient LLM failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.schemas import ToolCall
from pyagent.workflows.notes_digest import NoteDigest
from scripts._verify_common import is_transient_subprocess_failure

EXIT_QUOTA = 3
STEPS_CAP = 10

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


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "ws"

        print("\nSetup: run the workflow with --create-sample")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyagent.workflows.notes_digest",
                "--workspace",
                str(workspace),
                "--steps",
                str(STEPS_CAP),
                "--max-files",
                "3",
                "--create-sample",
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

        print("\nTest 1: artifacts created inside per-run folder")
        digests_dir = workspace / "digests"
        run_folders = sorted(p for p in digests_dir.iterdir() if p.is_dir())
        check(
            len(run_folders) == 1,
            f"exactly one per-run folder (got {len(run_folders)})",
        )
        if not run_folders:
            _finalize()
            return
        run_folder = run_folders[0]
        print(f"  \u2139 run folder: {run_folder.name}")

        json_path = run_folder / "digest.json"
        md_path = run_folder / "digest.md"
        log_path = run_folder / "run.log.jsonl"
        check(json_path.exists(), "digest.json exists")
        check(md_path.exists(), "digest.md exists")
        check(log_path.exists(), "run.log.jsonl exists")
        if not (json_path.exists() and md_path.exists() and log_path.exists()):
            _finalize()
            return

        print("\nTest 2: JSON validates against NoteDigest schema")
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        try:
            digest = NoteDigest.model_validate(raw)
            check(True, "schema valid")
        except Exception as exc:
            check(False, f"schema validation failed: {exc}")
            _finalize()
            return

        print("\nTest 3: files_analyzed non-empty and real")
        check(
            len(digest.files_analyzed) >= 1,
            f">=1 file analyzed (got {len(digest.files_analyzed)})",
        )
        for f in digest.files_analyzed:
            p = workspace / f
            check(p.is_file(), f"analyzed file exists: {f}")
            if p.is_file() and not f.startswith("notes/"):
                check(False, f"analyzed path is under notes/: {f}")

        print("\nTest 4: markdown digest has required sections")
        md_content = md_path.read_text(encoding="utf-8")
        check("# Notes Digest" in md_content, "# Notes Digest heading present")
        check("## Summary" in md_content, "## Summary present")
        check("## Files Analyzed" in md_content, "## Files Analyzed present")
        check("## Top TODOs" in md_content, "## Top TODOs present")
        check("## Decisions" in md_content, "## Decisions present")
        check("## Open Questions" in md_content, "## Open Questions present")
        check(len(md_content) > 200, f"content substantial ({len(md_content)} chars)")

        print("\nTest 5: log shows list_files + read_file + final_answer")
        ctx = ExecutionContext.from_log(log_path)
        calls = [item for ev in ctx.events for item in ev.content if isinstance(item, ToolCall)]
        names = [c.name for c in calls]
        check("list_files" in names, f"list_files called (names={names})")
        check(
            names.count("read_file") >= 1,
            f"read_file called (got {names.count('read_file')})",
        )
        check(
            "final_answer" in names,
            f"final_answer called (names={names})",
        )

        print("\nTest 6: max_steps respected")
        check(
            ctx.current_step <= STEPS_CAP,
            f"current_step <= {STEPS_CAP} (got {ctx.current_step})",
        )

        print("\nTest 7: todo source_files are in files_analyzed")
        valid = set(digest.files_analyzed)
        for i, item in enumerate(digest.top_todos):
            check(
                item.source_file in valid,
                f"todo[{i}].source_file={item.source_file!r} in files_analyzed",
            )

        print("\nTest 8: digest has at least one TODO or decision")
        total = len(digest.top_todos) + len(digest.decisions) + len(digest.open_questions)
        check(
            total >= 1,
            f"at least one extracted item across todos/decisions/questions (got {total})",
        )

        print(
            f"\n  \u2139 summary: {len(digest.files_analyzed)} files, "
            f"{len(digest.top_todos)} TODOs, "
            f"{len(digest.decisions)} decisions, "
            f"{len(digest.open_questions)} open questions"
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
        print(f"\n\u2705 Notes digest workflow verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Notes digest workflow verified.")


if __name__ == "__main__":
    main()
