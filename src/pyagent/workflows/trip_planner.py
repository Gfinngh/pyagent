"""Trip planner workflow.

Takes a destination and a trip length, produces a structured day-by-day
itinerary with attractions, meals, tips, and sources.

Output (one folder per run)::

    <output-dir>/<stamp>-<slug>/
        itinerary.json      structured TripPlan
        itinerary.md        human-readable itinerary
        run.log.jsonl       event log

On failure, the folder is deleted by default. Pass ``--keep-on-failure`` to
preserve it for debugging (contains partial artifacts and the event log up to
the failure point).

Exit codes:
    0   success
    1   failure
    2   usage error
    3   LLM quota exhausted
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

from pydantic import BaseModel, Field  # noqa: E402

from pyagent.agent import Agent, LlmCallError  # noqa: E402
from pyagent.client import (  # noqa: E402
    McpConnection,
    McpServerParams,
    ToolOverrides,
    load_mcp_tools,
)
from pyagent.context import ExecutionContext  # noqa: E402
from pyagent.event_log import JsonlEventLog  # noqa: E402
from pyagent.fallback import FallbackClient, FallbackEntry  # noqa: E402
from pyagent.llm import LlmClient  # noqa: E402
from pyagent.models import resolve_model  # noqa: E402
from pyagent.tracing import init_tracing  # noqa: E402

DEFAULT_OUTPUT_DIR = "workspace/trips"
DEFAULT_DAYS = 3
DEFAULT_BUDGET = "moderate"
DEFAULT_STEPS = 12

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_QUOTA = 3

REQUIRED_TOOLS: frozenset[str] = frozenset({"search_web"})

TOOL_OVERRIDES: dict[str, ToolOverrides] = {
    "search_web": ToolOverrides(permission="read", timeout_ms=20_000),
}


class DayPlan(BaseModel):
    """One day of the itinerary."""

    day_number: int = Field(description="1-indexed day number.")
    title: str = Field(description="Short theme, e.g. 'Arrival & central Kyoto'.")
    morning: str = Field(description="Morning activity, 1-2 sentences.")
    afternoon: str = Field(description="Afternoon activity, 1-2 sentences.")
    evening: str = Field(description="Evening activity, 1-2 sentences.")
    meals: list[str] = Field(
        default_factory=list[str],
        description="2-3 specific meal recommendations, each naming a place.",
    )


class TripPlan(BaseModel):
    """Structured day-by-day itinerary."""

    destination: str = Field(description="The destination as given.")
    duration_days: int = Field(description="Number of days in the trip.")
    budget_estimate: str = Field(
        description="Short description of expected daily cost for the chosen tier."
    )
    daily_itinerary: list[DayPlan] = Field(
        description="Exactly duration_days entries, numbered 1..N sequentially."
    )
    highlights: list[str] = Field(
        default_factory=list[str],
        description="4-6 top attractions across the trip, one per line.",
    )
    practical_tips: list[str] = Field(
        default_factory=list[str],
        description="3-5 actionable tips (transit, timing, etiquette).",
    )
    sources: list[str] = Field(
        default_factory=list[str],
        description="3+ URLs returned by search_web, copied verbatim.",
    )


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    if not s:
        return "trip"
    if len(s) <= max_len:
        return s
    trunc = s[:max_len]
    last = trunc.rfind("-")
    if last > 0:
        trunc = trunc[:last]
    return trunc.rstrip("-") or "trip"


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(exit_code: int, message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def _build_llm_client() -> FallbackClient:
    chain = [
        FallbackEntry(
            label="3.1-lite",
            client=LlmClient(
                model=resolve_model("gemini/gemini-3.1-flash-lite"),
                max_retries=1,
                timeout_ms=60_000,
            ),
        ),
        FallbackEntry(
            label="3.5-lite",
            client=LlmClient(
                model=resolve_model("gemini/gemini-3.5-flash-lite"),
                max_retries=1,
                timeout_ms=60_000,
            ),
        ),
        FallbackEntry(
            label="3.6-flash",
            client=LlmClient(
                model=resolve_model("gemini/gemini-3.6-flash"),
                max_retries=0,
                timeout_ms=60_000,
            ),
        ),
    ]
    return FallbackClient(
        chain=chain,
        strip_provider_options_on_switch=True,
    )


def _instructions(destination: str, days: int, budget: str) -> str:
    return f"""You are a travel planning assistant creating a {days}-day itinerary
for {destination}.

Budget tier: {budget}

Process:
1. Research the destination. Call search_web 4-6 times with different angles:
   - "{destination} top attractions"
   - "{destination} best restaurants local cuisine"
   - "{destination} public transportation getting around"
   - "{destination} day trips nearby"
   - "{destination} safety tips best time to visit"
   Use max_results=5 on each call. You do not need to call every angle; stop
   when you have enough distinct sources to plan a well-grounded trip.
2. Synthesize the research into a structured {days}-day itinerary.
3. Call final_answer with the structured TripPlan.

Requirements:
- daily_itinerary MUST have exactly {days} entries, day_number 1..{days}
  sequentially with no gaps or repeats.
- Each day: title (short theme), morning / afternoon / evening (each 1-2
  sentences naming SPECIFIC places), meals (2-3 specific place suggestions).
- Do NOT repeat the same attraction across multiple days.
- If a day includes a day trip (e.g. a nearby town), that day trip is the
  only activity for that day. Day trips consume the full day.
- highlights: 4-6 top attractions that appear in the daily itinerary.
- practical_tips: 3-5 actionable tips grounded in what you found.
- budget_estimate: short description matching the {budget} tier.
- sources: 3+ URLs actually returned by search_web. Copy URLs verbatim from
  the search results. Do NOT invent URLs.

Rules:
- Do NOT fabricate places, restaurants, or URLs. Only recommend things you
  found via search_web. If a search failed, work with what you have and note
  the limitation in practical_tips.
- Every day must be grounded in at least one source URL.
"""


def _render_markdown(plan: TripPlan, generated_at: str) -> str:
    lines: list[str] = []
    lines.append(f"# {plan.destination} — {plan.duration_days}-Day Itinerary\n")
    lines.append(f"**Generated**: {generated_at}\n")
    lines.append(f"**Budget**: {plan.budget_estimate}\n")

    lines.append(f"## Daily Itinerary ({len(plan.daily_itinerary)} days)\n")
    for day in plan.daily_itinerary:
        lines.append(f"### Day {day.day_number} — {day.title}\n")
        lines.append(f"- **Morning:** {day.morning}")
        lines.append(f"- **Afternoon:** {day.afternoon}")
        lines.append(f"- **Evening:** {day.evening}")
        if day.meals:
            lines.append("- **Meals:**")
            for meal in day.meals:
                lines.append(f"  - {meal}")
        lines.append("")

    lines.append(f"## Highlights ({len(plan.highlights)})\n")
    if plan.highlights:
        for h in plan.highlights:
            lines.append(f"- {h}")
    else:
        lines.append("(none recorded)")
    lines.append("")

    lines.append(f"## Practical Tips ({len(plan.practical_tips)})\n")
    if plan.practical_tips:
        for t in plan.practical_tips:
            lines.append(f"- {t}")
    else:
        lines.append("(none recorded)")
    lines.append("")

    lines.append(f"## Sources ({len(plan.sources)})\n")
    if plan.sources:
        for i, url in enumerate(plan.sources, 1):
            lines.append(f"{i}. {url}")
    else:
        lines.append("(none recorded)")

    return "\n".join(lines) + "\n"


def _cleanup_or_keep(output_folder: Path, succeeded: bool, keep_on_failure: bool) -> None:
    if succeeded:
        return
    if keep_on_failure:
        print(
            f"kept failed run for debugging: {output_folder}",
            file=sys.stderr,
        )
        return
    shutil.rmtree(output_folder, ignore_errors=True)
    print(f"cleaned up: {output_folder}", file=sys.stderr)


async def _run(
    destination: str,
    days: int,
    budget: str,
    output_dir: Path,
    max_steps: int,
    keep_on_failure: bool,
) -> int:
    output_dir = output_dir.resolve()
    workspace_root = output_dir.parent
    slug = f"{_slug(destination)}-{days}d"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S-%f")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_folder = output_dir / f"{stamp}-{slug}"
    json_path = output_folder / "itinerary.json"
    md_path = output_folder / "itinerary.md"
    log_path = output_folder / "run.log.jsonl"

    _out(f"Destination: {destination}")
    _out(f"Days:        {days}")
    _out(f"Budget:      {budget}")
    _out(f"Workspace:   {workspace_root}")
    _out(f"Output:      {output_folder}")
    _out(f"Max steps:   {max_steps}")

    output_folder.mkdir(parents=True, exist_ok=False)
    succeeded = False
    try:
        llm_client = _build_llm_client()
        server_params = McpServerParams(
            command=sys.executable,
            args=["-m", "pyagent.server"],
            env={**os.environ, "WORKSPACE_ROOT": str(workspace_root)},
        )

        log = JsonlEventLog(log_path)
        ctx = ExecutionContext(log=log)

        try:
            async with McpConnection(server_params) as conn:
                all_tools = await load_mcp_tools(conn, overrides=TOOL_OVERRIDES)
                tools = [t for t in all_tools if t.name in REQUIRED_TOOLS]
                discovered = {t.name for t in tools}
                missing = REQUIRED_TOOLS - discovered
                if missing:
                    return _err(
                        EXIT_FAILURE,
                        f"required tools not discovered: {sorted(missing)}",
                    )
                _out(f"Tools:       {sorted(discovered)}")

                agent = Agent(
                    client=llm_client,
                    tools=tools,
                    instructions=_instructions(destination, days, budget),
                    max_steps=max_steps,
                    output_type=TripPlan,
                )

                _out("\nPlanning...")
                start = time.monotonic()
                try:
                    result = await agent.run(f"Plan a {days}-day trip to {destination}.", ctx)
                except LlmCallError as exc:
                    if exc.metadata.get("quota_exhausted") is True:
                        return _err(EXIT_QUOTA, f"quota exhausted: {exc}")
                    return _err(EXIT_FAILURE, f"agent failed: {exc}")
                finally:
                    elapsed = time.monotonic() - start
                    _out(f"\nCompleted in {elapsed:.1f}s")
        finally:
            log.close()

        raw_output = result.output
        if not isinstance(raw_output, dict):
            return _err(
                EXIT_FAILURE,
                f"agent output was not a structured dict (got {type(raw_output).__name__})",
            )

        try:
            plan = TripPlan.model_validate(raw_output)
        except Exception as exc:
            print(f"raw final_answer output:\n{raw_output!r}", file=sys.stderr)
            return _err(
                EXIT_FAILURE,
                f"final_answer did not validate against TripPlan: {exc}",
            )

        if len(plan.daily_itinerary) != days:
            return _err(
                EXIT_FAILURE,
                f"expected {days} days in daily_itinerary, got {len(plan.daily_itinerary)}",
            )
        numbers = [d.day_number for d in plan.daily_itinerary]
        if numbers != list(range(1, days + 1)):
            return _err(
                EXIT_FAILURE,
                f"day numbers must be 1..{days} sequentially, got {numbers}",
            )

        json_path.write_text(json.dumps(plan.model_dump(), indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(plan, generated_at), encoding="utf-8")

        _out(f"\nTrip JSON:   {json_path}")
        _out(f"Trip MD:     {md_path}")
        _out(f"Log:         {log_path}")
        _out(
            f"Summary:     {len(plan.daily_itinerary)} days, "
            f"{len(plan.highlights)} highlights, "
            f"{len(plan.practical_tips)} tips, "
            f"{len(plan.sources)} sources"
        )
        succeeded = True
        return EXIT_OK
    finally:
        _cleanup_or_keep(output_folder, succeeded, keep_on_failure)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyagent-trip-planner",
        description="Produce a structured day-by-day trip itinerary.",
    )
    parser.add_argument("destination", help="Destination city or region")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days in the trip (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--budget",
        default=DEFAULT_BUDGET,
        help=f"Budget tier or amount, free-form (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Maximum agent steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--keep-on-failure",
        action="store_true",
        help=(
            "Preserve the run folder on failure (partial artifacts + log) "
            "for debugging. Default: delete the folder."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    init_tracing(service_name="pyagent-trip-planner")
    args = _parse_args(sys.argv[1:])
    if not args.destination.strip():
        return _err(EXIT_USAGE, "destination must not be empty")
    if args.days < 1:
        return _err(EXIT_USAGE, "--days must be >= 1")
    if args.steps < 1:
        return _err(EXIT_USAGE, "--steps must be >= 1")
    return asyncio.run(
        _run(
            args.destination.strip(),
            args.days,
            args.budget,
            Path(args.output_dir),
            args.steps,
            args.keep_on_failure,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
