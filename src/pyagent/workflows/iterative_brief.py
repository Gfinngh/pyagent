"""Iterative brief workflow.

Two-phase research brief with refinement.

Output (one folder per version)::

    <workspace>/briefs-iter/<stamp>-v1-<slug>/
        brief.md
        run.log.jsonl

    <workspace>/briefs-iter/<stamp>-v2-<slug>/
        brief.md
        run.log.jsonl

On a fresh run failure, the v1 folder is deleted by default. On a refine
failure, only the new v(N+1) folder is deleted; prior versions are always
preserved (they succeeded). Pass ``--keep-on-failure`` to preserve the failed
folder for debugging.

Exit codes:
    0   success
    1   failure
    2   usage error (bad args, no prior version for --refine)
    3   LLM quota exhausted
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

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

DEFAULT_OUTPUT_DIR = "workspace/briefs-iter"
DEFAULT_STEPS = 8

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_QUOTA = 3

REQUIRED_TOOLS: frozenset[str] = frozenset({"search_web", "write_file"})

TOOL_OVERRIDES: dict[str, ToolOverrides] = {
    "search_web": ToolOverrides(permission="read", timeout_ms=15_000),
    "write_file": ToolOverrides(permission="mutate", timeout_ms=10_000),
}


def _slug(topic: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", topic.strip().lower()).strip("-")
    if not s:
        return "untitled"
    if len(s) <= max_len:
        return s
    trunc = s[:max_len]
    last = trunc.rfind("-")
    if last > 0:
        trunc = trunc[:last]
    return trunc.rstrip("-") or "untitled"


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(exit_code: int, message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return exit_code


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


def _find_latest_version(output_dir: Path, slug: str) -> tuple[int, Path] | None:
    """Find the folder for the highest vN of the given slug.

    Folder name format: ``<stamp>-v<N>-<slug>``.
    """
    if not output_dir.exists():
        return None
    suffix_pattern = re.compile(rf"-v(\d+)-{re.escape(slug)}$")
    versions: list[tuple[int, Path]] = []
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        m = suffix_pattern.search(entry.name)
        if m:
            versions.append((int(m.group(1)), entry))
    if not versions:
        return None
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0]


def _fresh_instructions(topic: str, output_rel: str, generated_at: str) -> str:
    return f"""You are a research assistant producing a structured markdown brief.

Topic: {topic}

Process:
1. Call search_web 2-3 times with different query angles (overview, recent
   developments, expert perspective). Use max_results=5 on each call.
2. Synthesize the results into a well-organized brief with numbered citations.
3. Write the brief using write_file with EXACTLY this path argument (a
   relative path; do not add or remove directory components): {output_rel}

Markdown structure (required):
# {topic}

**Generated**: {generated_at}

## Summary
2-4 sentences summarizing what the brief covers.

## Key Facts
- 5-10 bullet points, each a concrete finding.
- One fact per bullet.
- End each bullet with a citation marker matching the numbered Sources list.

## Sources
1. [Title](URL)
2. [Title](URL)
- Number each source to match the [N] markers in Key Facts.
- Copy each URL verbatim from a search_web result. Do NOT invent URLs.

## Open Questions
- 2-4 questions that remain unanswered.

Rules:
- Do NOT fabricate facts or URLs. Only cite what search_web returned.
- Every [N] citation in Key Facts MUST correspond to a numbered Sources entry.
- After the write succeeds, respond with exactly: "Brief saved"
"""


def _refine_instructions(topic: str, prior_rel: str, new_rel: str) -> str:
    return f"""You are revising a research brief you previously wrote to {prior_rel}.
The original brief is visible in your conversation history - look at the
write_file tool call content from earlier in this conversation.

You will receive a refinement instruction in the next user message.
1. If the revision needs new information, call search_web to gather it
   (max_results=5). Otherwise work from existing sources.
2. Revise the brief according to the instruction.
3. Write the revised version using write_file with EXACTLY this path argument
   (a relative path; do not add or remove directory components): {new_rel}

Preserve:
- The full markdown structure: # {topic}, ## Summary, ## Key Facts,
  ## Sources, ## Open Questions.
- All existing [N] citation markers in Key Facts, keeping them linked to
  numbered Sources entries.
- All existing source URLs. Do not remove sources that still support the brief.

Rules:
- If new information is needed, gather via search_web, then add new sources
  with new [N] markers.
- Do NOT fabricate facts or URLs. New facts must come from search_web results.
- Do NOT delete existing valid facts. Add or refine.
- Every [N] citation in Key Facts MUST correspond to a numbered Sources entry.
- After the write succeeds, respond with exactly: "Brief revised"
"""


async def _run_fresh(
    topic: str,
    slug: str,
    output_dir: Path,
    max_steps: int,
    keep_on_failure: bool,
) -> int:
    output_dir = output_dir.resolve()
    workspace_root = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S-%f")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    existing = _find_latest_version(output_dir, slug)
    if existing is not None:
        existing_version, existing_folder = existing
        return _err(
            EXIT_USAGE,
            f"v{existing_version} already exists at {existing_folder}; "
            f"pass --refine to build on it",
        )

    output_folder = output_dir / f"{stamp}-v1-{slug}"
    brief_md = output_folder / "brief.md"
    brief_log = output_folder / "run.log.jsonl"
    brief_rel = f"{output_dir.name}/{output_folder.name}/brief.md"

    _out("Mode:        fresh (v1)")
    _out(f"Topic:       {topic}")
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

        log = JsonlEventLog(brief_log)
        ctx = ExecutionContext(log=log)

        try:
            async with McpConnection(server_params) as conn:
                all_tools = await load_mcp_tools(conn, overrides=TOOL_OVERRIDES)
                tools = [t for t in all_tools if t.name in REQUIRED_TOOLS]
                missing = REQUIRED_TOOLS - {t.name for t in tools}
                if missing:
                    return _err(
                        EXIT_FAILURE,
                        f"required tools not discovered: {sorted(missing)}",
                    )
                _out(f"Tools:       {sorted(t.name for t in tools)}")

                agent = Agent(
                    client=llm_client,
                    tools=tools,
                    instructions=_fresh_instructions(topic, brief_rel, generated_at),
                    max_steps=max_steps,
                    confirm_tool_call=lambda tc: True,
                )

                _out("\nResearching (v1)...")
                start = time.monotonic()
                try:
                    result = await agent.run(
                        f"Research this topic and save the brief: {topic}", ctx
                    )
                except LlmCallError as exc:
                    if exc.metadata.get("quota_exhausted") is True:
                        return _err(EXIT_QUOTA, f"quota exhausted: {exc}")
                    return _err(EXIT_FAILURE, f"agent failed: {exc}")
                finally:
                    _out(f"\nCompleted in {time.monotonic() - start:.1f}s")

                _out(f"Agent output: {result.output}")
        finally:
            log.close()

        if not brief_md.exists():
            return _err(EXIT_FAILURE, f"brief not created at {brief_md}")

        _out(f"\nBrief saved: {brief_md}")
        _out(f"Log:         {brief_log}")
        succeeded = True
        return EXIT_OK
    finally:
        _cleanup_or_keep(output_folder, succeeded, keep_on_failure)


async def _run_refine(
    topic: str,
    slug: str,
    output_dir: Path,
    refinement: str,
    max_steps: int,
    keep_on_failure: bool,
) -> int:
    output_dir = output_dir.resolve()
    workspace_root = output_dir.parent

    found = _find_latest_version(output_dir, slug)
    if found is None:
        return _err(
            EXIT_USAGE,
            f"no prior brief for slug {slug!r} in {output_dir}; run without --refine first",
        )

    current_version, prior_folder = found
    prior_log_path = prior_folder / "run.log.jsonl"
    prior_md_path = prior_folder / "brief.md"

    if not prior_log_path.exists():
        return _err(
            EXIT_FAILURE,
            f"prior version folder is missing run.log.jsonl: {prior_folder}",
        )

    next_version = current_version + 1
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S-%f")
    output_folder = output_dir / f"{stamp}-v{next_version}-{slug}"
    new_log_path = output_folder / "run.log.jsonl"
    new_md_path = output_folder / "brief.md"

    prior_rel = f"{output_dir.name}/{prior_folder.name}/brief.md"
    new_rel = f"{output_dir.name}/{output_folder.name}/brief.md"

    _out(f"Mode:        refine (v{current_version} -> v{next_version})")
    _out(f"Topic:       {topic}")
    _out(f"Prior:       {prior_md_path}")
    _out(f"Refinement:  {refinement}")
    _out(f"Workspace:   {workspace_root}")
    _out(f"Output:      {output_folder}")
    _out(f"Max steps:   {max_steps}")

    prior_ctx = ExecutionContext.from_log(prior_log_path)
    _out(f"Prior events: {len(prior_ctx.events)}")

    output_folder.mkdir(parents=True, exist_ok=False)
    succeeded = False
    try:
        llm_client = _build_llm_client()
        server_params = McpServerParams(
            command=sys.executable,
            args=["-m", "pyagent.server"],
            env={**os.environ, "WORKSPACE_ROOT": str(workspace_root)},
        )

        async with McpConnection(server_params) as conn:
            all_tools = await load_mcp_tools(conn, overrides=TOOL_OVERRIDES)
            tools = [t for t in all_tools if t.name in REQUIRED_TOOLS]
            missing = REQUIRED_TOOLS - {t.name for t in tools}
            if missing:
                return _err(
                    EXIT_FAILURE,
                    f"required tools not discovered: {sorted(missing)}",
                )
            _out(f"Tools:       {sorted(t.name for t in tools)}")

            new_log = JsonlEventLog(new_log_path)
            for ev in prior_ctx.events:
                new_log.append(ev)
            prior_ctx.log = new_log

            try:
                agent = Agent(
                    client=llm_client,
                    tools=tools,
                    instructions=_refine_instructions(topic, prior_rel, new_rel),
                    max_steps=max_steps,
                    confirm_tool_call=lambda tc: True,
                )

                _out(f"\nRefining (v{current_version} -> v{next_version})...")
                start = time.monotonic()
                try:
                    result = await agent.resume(prior_ctx, refinement)
                except LlmCallError as exc:
                    if exc.metadata.get("quota_exhausted") is True:
                        return _err(EXIT_QUOTA, f"quota exhausted: {exc}")
                    return _err(EXIT_FAILURE, f"agent failed: {exc}")
                finally:
                    _out(f"\nCompleted in {time.monotonic() - start:.1f}s")

                _out(f"Agent output: {result.output}")
            finally:
                new_log.close()

        if not new_md_path.exists():
            return _err(EXIT_FAILURE, f"revised brief not created at {new_md_path}")

        _out(f"\nRevised brief: {new_md_path}")
        _out(f"Log:           {new_log_path}")
        succeeded = True
        return EXIT_OK
    finally:
        _cleanup_or_keep(output_folder, succeeded, keep_on_failure)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyagent-iterative-brief",
        description="Iteratively refine a research brief (fresh then --refine).",
    )
    parser.add_argument("topic", help="Research topic")
    parser.add_argument(
        "--refine",
        metavar="INSTRUCTION",
        default=None,
        help=(
            "Refinement instruction. Reads the latest existing version and "
            "writes the next one. Omit for a fresh v1."
        ),
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
            "Preserve the failed run folder (partial artifacts + log) for "
            "debugging. Prior successful versions are always preserved "
            "regardless. Default: delete only the failed folder."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    init_tracing(service_name="pyagent-iterative-brief")
    args = _parse_args(sys.argv[1:])
    if not args.topic.strip():
        return _err(EXIT_USAGE, "topic must not be empty")
    if args.steps < 1:
        return _err(EXIT_USAGE, "--steps must be >= 1")

    slug = _slug(args.topic)
    output_dir = Path(args.output_dir)

    if args.refine is None:
        return asyncio.run(
            _run_fresh(
                args.topic,
                slug,
                output_dir,
                args.steps,
                args.keep_on_failure,
            )
        )

    if not args.refine.strip():
        return _err(EXIT_USAGE, "--refine must not be empty")
    return asyncio.run(
        _run_refine(
            args.topic,
            slug,
            output_dir,
            args.refine.strip(),
            args.steps,
            args.keep_on_failure,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
