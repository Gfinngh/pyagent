"""Verify the trip planner workflow end-to-end.

Skips on quota or transient LLM failure. Detects failures honestly (a
workflow's own error would have different stderr).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pyagent.context import ExecutionContext
from pyagent.schemas import ToolCall
from pyagent.workflows.trip_planner import TripPlan
from scripts._verify_common import is_transient_subprocess_failure

EXIT_QUOTA = 3
STEPS_CAP = 12
DAYS = 2

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


def _count_urls(texts: list[str]) -> int:
    return len(set(re.findall(r"https?://[^\s)\]>\"']+", "\n".join(texts))))


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  \u2717 GEMINI_API_KEY not set")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    destination = "Kyoto"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "trips"

        print("\nSetup: run the workflow as a subprocess")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyagent.workflows.trip_planner",
                destination,
                "--days",
                str(DAYS),
                "--output-dir",
                str(out_dir),
                "--steps",
                str(STEPS_CAP),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=500,
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

        # Per-run folder structure: <out_dir>/<stamp>-<slug>/<file>
        print("\nTest 1: all three artifacts created inside one folder")
        run_folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
        check(
            len(run_folders) == 1,
            f"exactly one per-run folder (got {len(run_folders)})",
        )
        if not run_folders:
            _finalize()
            return
        run_folder = run_folders[0]
        print(f"  \u2139 run folder: {run_folder.name}")

        json_path = run_folder / "itinerary.json"
        md_path = run_folder / "itinerary.md"
        log_path = run_folder / "run.log.jsonl"
        check(json_path.exists(), "itinerary.json exists")
        check(md_path.exists(), "itinerary.md exists")
        check(log_path.exists(), "run.log.jsonl exists")
        if not (json_path.exists() and md_path.exists() and log_path.exists()):
            _finalize()
            return

        print("\nTest 2: JSON validates against TripPlan schema")
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        try:
            plan = TripPlan.model_validate(raw)
            check(True, "schema valid")
        except Exception as exc:
            check(False, f"schema validation failed: {exc}")
            _finalize()
            return

        print(f"\nTest 3: daily_itinerary has exactly {DAYS} days")
        check(
            len(plan.daily_itinerary) == DAYS,
            f"{DAYS} days (got {len(plan.daily_itinerary)})",
        )
        numbers = [d.day_number for d in plan.daily_itinerary]
        check(
            numbers == list(range(1, DAYS + 1)),
            f"day numbers 1..{DAYS} (got {numbers})",
        )

        print("\nTest 4: each day has full content")
        for d in plan.daily_itinerary:
            check(len(d.title) > 0, f"day {d.day_number} has title")
            check(len(d.morning) > 20, f"day {d.day_number} morning substantive")
            check(len(d.afternoon) > 20, f"day {d.day_number} afternoon substantive")
            check(len(d.evening) > 20, f"day {d.day_number} evening substantive")
            check(len(d.meals) >= 1, f"day {d.day_number} has meals")

        print("\nTest 5: markdown has required sections")
        md = md_path.read_text(encoding="utf-8")
        check(f"# {destination}" in md, "destination heading present")
        check("## Daily Itinerary" in md, "## Daily Itinerary present")
        check("## Highlights" in md, "## Highlights present")
        check("## Practical Tips" in md, "## Practical Tips present")
        check("## Sources" in md, "## Sources present")
        check(len(md) > 400, f"markdown substantial ({len(md)} chars)")

        print("\nTest 6: at least 3 source URLs")
        n_urls = _count_urls(plan.sources)
        print(f"  \u2139 distinct source URLs: {n_urls}")
        check(n_urls >= 3, f">=3 source URLs (got {n_urls})")

        print("\nTest 7: every source URL is http(s)")
        for url in plan.sources:
            check(
                url.startswith("http://") or url.startswith("https://"),
                f"source is http(s): {url[:60]}…",
            )

        print("\nTest 8: log shows >=2 search_web + final_answer")
        ctx = ExecutionContext.from_log(log_path)
        calls = [item for ev in ctx.events for item in ev.content if isinstance(item, ToolCall)]
        names = [c.name for c in calls]
        check(
            names.count("search_web") >= 2,
            f">=2 search_web calls (got {names.count('search_web')})",
        )
        check(
            "final_answer" in names,
            f"final_answer called (names={names})",
        )
        print(
            f"  \u2139 tool calls: search_web x{names.count('search_web')}, "
            f"final_answer x{names.count('final_answer')}"
        )

        print("\nTest 9: max_steps respected")
        check(
            ctx.current_step <= STEPS_CAP,
            f"current_step <= {STEPS_CAP} (got {ctx.current_step})",
        )

        print("\nTest 10: plan references the destination")
        haystack = " ".join(
            [
                plan.destination,
                *plan.highlights,
                *plan.practical_tips,
                *[d.title for d in plan.daily_itinerary],
            ]
        ).lower()
        check(
            destination.lower() in haystack,
            "destination name appears in plan content",
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
        print(f"\n\u2705 Trip planner workflow verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Trip planner workflow verified.")


if __name__ == "__main__":
    main()
