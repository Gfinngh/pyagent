"""Verify research_brief auto-recall + auto-store memory behavior.

Two subprocess runs, same topic, same workspace:

  1. **Run 1 (cold)**: no prior memory for the topic.
     - stdout contains ``Memory: no prior research found``.
     - stdout contains ``Memory: stored summary at briefs/<slug>``.
     - memory.sqlite exists with one record (``key=<slug>``).
  2. **Run 2 (warm)**: same topic, prior memory exists.
     - stdout contains ``Memory: found 1 prior brief(s)``.
     - stdout contains ``Memory: stored summary at briefs/<slug>`` again.
     - memory.sqlite still has exactly one record for this slug (upsert;
       ``updated_at >= created_at``).
     - the brief.md is well-formed (structure preserved through recall).

Quota-tolerant: both runs skip together if the LLM is unavailable. The
whole point is end-to-end via the CLI, so no fake clients here — real runs
or skip.

Not asserted: whether the model *used* the injected memory. That's a model
decision, not a workflow property. The workflow's job is to inject; the
model's job is to read the instructions. This test verifies the injection
happened, not the model's response.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pyagent.memory import MemoryStore
from scripts._verify_common import is_transient_subprocess_failure

EXIT_QUOTA = 3
TOPIC = "health benefits of drinking water"
TOPIC_SLUG = "health-benefits-of-drinking-water"

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
        [sys.executable, "-m", "pyagent.workflows.research_brief", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=400,
        check=False,
    )


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
        ws = Path(tmp) / "ws"
        out_dir = ws / "briefs"
        memory_path = ws / "memory.sqlite"

        args = [
            TOPIC,
            "--output-dir",
            str(out_dir),
            "--steps",
            "6",
        ]

        # Declared before the try so pyright can prove it's bound when the
        # final assertion in Test 6 references it. Overwritten inside
        # ``if all_mem:`` if a record exists.
        cold_updated_at: float = 0.0

        # ---------------------------------------------------------------
        print("\nSetup phase 1: cold run (no prior memory)")
        proc1 = _run(args, repo_root)
        print(f"  \u2139 exit code: {proc1.returncode}")

        if proc1.returncode == EXIT_QUOTA:
            skip(f"quota exhausted on cold run: {proc1.stderr.strip()[:200]}")
            _finalize()
            return
        if is_transient_subprocess_failure(proc1):
            skip(f"transient LLM failure on cold run: {proc1.stderr.strip()[:200]}")
            _finalize()
            return
        if proc1.returncode != 0:
            check(False, f"cold run exited {proc1.returncode}")
            print(f"  \u2139 stderr:\n{proc1.stderr[:1500]}")
            print(f"  \u2139 stdout:\n{proc1.stdout[:1500]}")
            _finalize()
            return

        check(True, "cold run exited 0")

        print("\nTest 1: cold run reports no prior memory")
        check(
            "Memory: no prior research found" in proc1.stdout,
            f"stdout shows no prior memory (got {proc1.stdout[:400]!r})",
        )

        print("\nTest 2: cold run stored the summary")
        check(
            f"Memory: stored summary at briefs/{TOPIC_SLUG}" in proc1.stdout,
            f"stdout shows stored summary (got {proc1.stdout[-400:]!r})",
        )
        check(
            memory_path.exists(),
            f"memory.sqlite exists at {memory_path}",
        )

        print("\nTest 3: memory has exactly one record for this slug")
        store = MemoryStore(memory_path)
        try:
            all_mem = store.list_all(namespace="briefs")
            check(
                len(all_mem) == 1,
                f"one memory record after cold run (got {len(all_mem)})",
            )
            if all_mem:
                check(
                    all_mem[0].key == TOPIC_SLUG,
                    f"record key is slug (got {all_mem[0].key!r})",
                )
                check(
                    len(all_mem[0].value) > 0,
                    "record value (summary) is non-empty",
                )
                check(
                    TOPIC in all_mem[0].tags,
                    f"record tags include topic (got {all_mem[0].tags})",
                )
                cold_updated_at = all_mem[0].updated_at
        finally:
            store.close()

        # ---------------------------------------------------------------
        print("\nSetup phase 2: warm run (prior memory present)")
        proc2 = _run(args, repo_root)
        print(f"  \u2139 exit code: {proc2.returncode}")

        if proc2.returncode == EXIT_QUOTA:
            skip(f"quota exhausted on warm run: {proc2.stderr.strip()[:200]}")
            _finalize()
            return
        if is_transient_subprocess_failure(proc2):
            skip(f"transient LLM failure on warm run: {proc2.stderr.strip()[:200]}")
            _finalize()
            return
        if proc2.returncode != 0:
            check(False, f"warm run exited {proc2.returncode}")
            print(f"  \u2139 stderr:\n{proc2.stderr[:1500]}")
            print(f"  \u2139 stdout:\n{proc2.stdout[:1500]}")
            _finalize()
            return

        check(True, "warm run exited 0")

        print("\nTest 4: warm run reports prior memory found")
        check(
            "Memory: found 1 prior brief(s)" in proc2.stdout,
            f"stdout shows prior memory found (got {proc2.stdout[:400]!r})",
        )

        print("\nTest 5: warm run stored summary again (upsert)")
        check(
            f"Memory: stored summary at briefs/{TOPIC_SLUG}" in proc2.stdout,
            f"stdout shows stored summary (got {proc2.stdout[-400:]!r})",
        )

        print("\nTest 6: memory still has one record (upsert, not append)")
        store = MemoryStore(memory_path)
        try:
            all_mem = store.list_all(namespace="briefs")
            check(
                len(all_mem) == 1,
                f"one memory record after warm run (got {len(all_mem)})",
            )
            if all_mem:
                check(
                    all_mem[0].key == TOPIC_SLUG,
                    f"same record updated (got {all_mem[0].key!r})",
                )
                check(
                    all_mem[0].updated_at >= cold_updated_at,
                    f"updated_at advanced (cold={cold_updated_at}, warm={all_mem[0].updated_at})",
                )
        finally:
            store.close()

        print("\nTest 7: warm run still produces a well-formed brief")
        run_folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
        check(
            len(run_folders) == 2,
            f"two run folders (cold + warm) (got {len(run_folders)})",
        )
        warm_folder = run_folders[-1]
        warm_brief = warm_folder / "brief.md"
        check(warm_brief.exists(), "warm brief.md exists")
        if warm_brief.exists():
            content = warm_brief.read_text(encoding="utf-8")
            check("## Summary" in content, "warm brief has ## Summary")
            check("## Key Facts" in content, "warm brief has ## Key Facts")
            check("## Sources" in content, "warm brief has ## Sources")
            check("## Open Questions" in content, "warm brief has ## Open Questions")
            check(
                _count_sources(content) >= 3,
                f"warm brief has >=3 sources (got {_count_sources(content)})",
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
        print(f"\n\u2705 Brief memory verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Brief memory verified.")


if __name__ == "__main__":
    main()
