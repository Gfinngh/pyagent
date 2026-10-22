"""Notes digest workflow.

Reads markdown notes from ``workspace/notes/``, produces a structured digest
of TODOs, decisions, and open questions, and writes two artifacts.

Output (one folder per run)::

    <workspace>/digests/<stamp>/
        digest.json         structured NoteDigest data
        digest.md           human-readable digest
        run.log.jsonl       event log

On failure, the folder is deleted by default. Pass ``--keep-on-failure`` to
preserve it for debugging.

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
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

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

DEFAULT_WORKSPACE = "workspace"
DEFAULT_STEPS = 10
DEFAULT_MAX_FILES = 3

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_QUOTA = 3

REQUIRED_TOOLS: frozenset[str] = frozenset({"list_files", "read_file"})

TOOL_OVERRIDES: dict[str, ToolOverrides] = {
    "list_files": ToolOverrides(permission="read", timeout_ms=5_000),
    "read_file": ToolOverrides(permission="read", timeout_ms=5_000),
}


class ActionItem(BaseModel):
    """One actionable TODO extracted from a note."""

    text: str = Field(description="The action item, verbatim or lightly paraphrased.")
    source_file: str = Field(description="Which note file this came from.")
    priority: Literal["high", "medium", "low"] = Field(
        default="medium", description="Priority if stated or inferable."
    )


class NoteDigest(BaseModel):
    """Structured digest of a directory of markdown notes."""

    files_analyzed: list[str] = Field(description="Relative paths of the notes that were read.")
    summary: str = Field(description="2-4 sentence overview of what the notes collectively cover.")
    top_todos: list[ActionItem] = Field(
        default_factory=list[ActionItem],
        description="Up to 5 highest-value action items across all notes.",
    )
    decisions: list[str] = Field(
        default_factory=list[str],
        description="Decisions already made, one per entry.",
    )
    open_questions: list[str] = Field(
        default_factory=list[str],
        description="Unanswered questions surfaced by the notes.",
    )


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


_SAMPLE_NOTES: dict[str, str] = {
    "standup.md": """# Standup — 2026-10-20

## Discussion
The auth migration is behind schedule. Sam flagged that the OAuth flow
breaks when sessions persist across tab closes.

## Decisions
- Use Postgres for the new billing service (not DynamoDB).
- Defer the mobile app rewrite to Q1.

## TODOs
- Finish the auth migration by Friday.
- Review Sam's OAuth fix PR (#247).
- Schedule the post-mortem for last week's outage.

## Open questions
- Do we need to support legacy SAML providers after Q4?
- Which CDN do we pick for the new static assets?
""",
    "reading-log.md": """# Reading Log — 2026-10-18

Just finished chapter 4 of "Designing Data-Intensive Applications".

## Takeaways
- Event sourcing is a real tradeoff, not a free lunch.
- The log-structured merge tree section is clearer than the B-tree one.

## TODOs
- Write a summary of chapter 4 for the team wiki.
- Read chapter 5 (replication) by next Sunday.

## Open questions
- Should we adopt a streaming-first architecture for the events pipeline?
- Does Kafka fit our latency budget?
""",
    "project-alpha.md": """# Project Alpha — 2026-10-15

## Status
Alpha is feature-complete for the MVP. Beta waiting on legal review.

## Decisions
- Ship the MVP without the reporting dashboard.
- Roll out to 20% of users first, then 100% after 2 weeks.

## TODOs
- Prepare the release notes for the beta announcement.
- Ask legal about the GDPR checklist for EU rollout.
- Set up the canary deployment pipeline.

## Open questions
- What's the budget for cloud costs during the beta period?
- Do we need a separate privacy policy for the EU?
""",
}


def _create_sample_notes(notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _SAMPLE_NOTES.items():
        (notes_dir / name).write_text(content, encoding="utf-8")


def _instructions(notes_rel: str, max_files: int) -> str:
    return f"""You are analyzing a collection of markdown notes and producing a
structured digest.

Process:
1. Call list_files with pattern "{notes_rel}/*.md" to see what notes exist.
2. Read each note using read_file. Read at most {max_files} notes; prioritize
   the most recently-dated filenames.
3. Call final_answer with the structured digest matching the required schema.

Files you analyze must come from the list_files result. Use the exact relative
paths returned by list_files when calling read_file and when setting
files_analyzed.

Extraction rules:
- top_todos: at most 5 items. Each item needs text, source_file, priority.
  Only include items that are genuinely actionable ("review X", "ask Y").
  Do not include "read chapter N" style goals that are more like intentions.
- decisions: only decisions already made, not proposals or open questions.
- open_questions: only questions the notes explicitly leave unanswered.
- files_analyzed: the paths you actually read, in the order you read them.
- summary: 2-4 sentences. Do not repeat the TODOs; synthesize the picture.

Rules:
- Do NOT invent content. If the notes are sparse, return small lists rather
  than padding them.
- source_file MUST match one of the entries in files_analyzed.
- priority should be "high" if the note says "urgent"/"by Friday"/"blocking";
  "low" if clearly optional; "medium" otherwise.
"""


def _render_markdown(digest: NoteDigest, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Notes Digest\n")
    lines.append(f"**Generated**: {generated_at}\n")

    lines.append("## Summary\n")
    lines.append(digest.summary + "\n")

    lines.append(f"## Files Analyzed ({len(digest.files_analyzed)})\n")
    if digest.files_analyzed:
        for f in digest.files_analyzed:
            lines.append(f"- `{f}`")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"## Top TODOs ({len(digest.top_todos)})\n")
    if digest.top_todos:
        for i, item in enumerate(digest.top_todos, 1):
            lines.append(
                f"{i}. **[{item.priority.upper()}]** {item.text} *(from `{item.source_file}`)*"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"## Decisions ({len(digest.decisions)})\n")
    if digest.decisions:
        for d in digest.decisions:
            lines.append(f"- {d}")
    else:
        lines.append("(none recorded)")
    lines.append("")

    lines.append(f"## Open Questions ({len(digest.open_questions)})\n")
    if digest.open_questions:
        for q in digest.open_questions:
            lines.append(f"- {q}")
    else:
        lines.append("(none recorded)")

    return "\n".join(lines) + "\n"


async def _run(
    workspace: Path,
    max_steps: int,
    max_files: int,
    create_sample: bool,
    keep_on_failure: bool,
) -> int:
    workspace = workspace.resolve()
    notes_dir = workspace / "notes"
    digests_dir = workspace / "digests"
    notes_rel = "notes"

    if not notes_dir.exists() or not any(notes_dir.glob("*.md")):
        if create_sample:
            _create_sample_notes(notes_dir)
            _out(f"Created sample notes in {notes_dir}")
        else:
            return _err(
                EXIT_USAGE,
                f"no markdown notes in {notes_dir}; drop some files there or pass --create-sample",
            )

    digests_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S-%f")

    output_folder = digests_dir / stamp
    json_path = output_folder / "digest.json"
    md_path = output_folder / "digest.md"
    log_path = output_folder / "run.log.jsonl"

    _out(f"Workspace:  {workspace}")
    _out(f"Notes:      {notes_dir}")
    _out(f"Output:     {output_folder}")
    _out(f"Max files:  {max_files}")
    _out(f"Max steps:  {max_steps}")

    output_folder.mkdir(parents=True, exist_ok=False)
    succeeded = False
    try:
        llm_client = _build_llm_client()
        server_params = McpServerParams(
            command=sys.executable,
            args=["-m", "pyagent.server"],
            env={**os.environ, "WORKSPACE_ROOT": str(workspace)},
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
                _out(f"Tools:      {sorted(discovered)}")

                agent = Agent(
                    client=llm_client,
                    tools=tools,
                    instructions=_instructions(notes_rel, max_files),
                    max_steps=max_steps,
                    output_type=NoteDigest,
                )

                _out("\nDigesting notes...")
                start = time.monotonic()
                try:
                    result = await agent.run(
                        f"Analyze the notes in {notes_rel}/ and produce a digest.",
                        ctx,
                    )
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
                f"agent output was not a structured dict "
                f"(got {type(raw_output).__name__}: {raw_output!r})",
            )

        try:
            digest = NoteDigest.model_validate(raw_output)
        except Exception as exc:
            print(f"raw final_answer output:\n{raw_output!r}", file=sys.stderr)
            return _err(
                EXIT_FAILURE,
                f"final_answer did not validate against NoteDigest: {exc}",
            )

        valid_files = set(digest.files_analyzed)
        for item in digest.top_todos:
            if item.source_file not in valid_files:
                print(
                    f"warning: todo references unknown file "
                    f"{item.source_file!r}; files_analyzed="
                    f"{sorted(valid_files)}",
                    file=sys.stderr,
                )

        json_path.write_text(json.dumps(digest.model_dump(), indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_markdown(digest, generated_at), encoding="utf-8")

        _out(f"\nDigest JSON: {json_path}")
        _out(f"Digest MD:   {md_path}")
        _out(f"Log:         {log_path}")
        _out(
            f"Summary:     {len(digest.files_analyzed)} files, "
            f"{len(digest.top_todos)} TODOs, "
            f"{len(digest.decisions)} decisions, "
            f"{len(digest.open_questions)} open questions"
        )
        succeeded = True
        return EXIT_OK
    finally:
        _cleanup_or_keep(output_folder, succeeded, keep_on_failure)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyagent-notes-digest",
        description="Digest a directory of markdown notes into a structured summary.",
    )
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        default=DEFAULT_WORKSPACE,
        help=(f"Workspace root containing notes/ and digests/ (default: {DEFAULT_WORKSPACE})"),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Maximum agent steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"Maximum number of notes to read (default: {DEFAULT_MAX_FILES})",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Seed sample notes if the notes/ directory is empty",
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
    init_tracing(service_name="pyagent-notes-digest")
    args = _parse_args(sys.argv[1:])
    if args.steps < 1:
        return _err(EXIT_USAGE, "--steps must be >= 1")
    if args.max_files < 1:
        return _err(EXIT_USAGE, "--max-files must be >= 1")
    return asyncio.run(
        _run(
            Path(args.workspace),
            args.steps,
            args.max_files,
            args.create_sample,
            args.keep_on_failure,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
