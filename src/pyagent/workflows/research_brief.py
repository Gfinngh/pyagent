"""Research brief workflow.

Takes a topic, produces a structured markdown brief written to disk.

Output (one folder per run)::

    <output-dir>/<stamp>-<slug>/
        brief.md            the markdown brief
        run.log.jsonl       event log

On failure, the folder is deleted by default. Pass ``--keep-on-failure`` to
preserve it for debugging.

Memory (T2, SQLite + FTS5)
--------------------------
The workflow owns continuity across runs:

- **Auto-recall on entry**: ``MemoryStore.search(topic, namespace="briefs")``
  runs before the agent starts. Matching prior briefs (same or related
  topics) are formatted and appended to the model's ``instructions``, so the
  model sees what it already researched. No tool call required.
- **Auto-store on success**: after ``brief.md`` is written, the ``## Summary``
  section is extracted and upserted into memory under ``key=slug``. Re-running
  the same topic replaces its memory (upsert); running a related topic adds a
  new record.

Memory lives at ``<output-dir>.parent / "memory.sqlite"`` — ``workspace/memory.sqlite``
for default invocations, ``/tmp/xxx/memory.sqlite`` for tests using
``--output-dir /tmp/xxx/briefs``. Shared across runs in the same workspace.

Memory is secondary. If the store fails to open, search, or write, the failure
is logged to stderr and the brief proceeds normally. The brief is the primary
artifact; memory is a convenience.

No memory tools are attached to the agent. The model doesn't call ``remember``
or ``recall`` — the workflow owns the lifecycle, which is why this pattern
scales without prompt-engineering "remember to remember."

Exit codes:
    0   success
    1   failure
    2   usage error
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
from pyagent.memory import MemoryStore, format_memories_for_prompt  # noqa: E402
from pyagent.models import resolve_model  # noqa: E402
from pyagent.schemas import ToolResult  # noqa: E402
from pyagent.tools import MultipartContent, TextPart  # noqa: E402
from pyagent.tracing import init_tracing  # noqa: E402

DEFAULT_OUTPUT_DIR = "workspace/briefs"
DEFAULT_STEPS = 8
MEMORY_NAMESPACE = "briefs"
MEMORY_RECALL_LIMIT = 3

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
    s = re.sub(r"[^a-z0-9]+", "-", topic.strip().lower())
    s = s.strip("-")
    if not s:
        return "untitled"
    if len(s) <= max_len:
        return s
    truncated = s[:max_len]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > 0:
        truncated = truncated[:last_hyphen]
    return truncated.rstrip("-") or "untitled"


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(exit_code: int, message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _write_file_result_text(result: ToolResult) -> str:
    if not result.content:
        return "<empty>"
    part = result.content[0]
    if isinstance(part, MultipartContent):
        texts = [p.text for p in part.parts if isinstance(p, TextPart)]
        return " ".join(texts) if texts else "<non-text content>"
    return str(part)


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


def _extract_summary(brief_markdown: str) -> str:
    """Pull the ``## Summary`` section out of the brief. Empty if missing.

    Collapses whitespace and caps at 800 chars so a runaway summary does not
    bloat the memory record. The full brief remains the source of truth; the
    memory record is a search hint.
    """
    match = re.search(
        r"^##\s+Summary\s*\n(.*?)(?=^##\s|\Z)",
        brief_markdown,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        return ""
    text = " ".join(match.group(1).split())
    return text[:800]


def _memory_path(workspace_root: Path) -> Path:
    return workspace_root / "memory.sqlite"


def _build_prior_block(matches: list[object]) -> str:
    """Format prior memories into an instructions suffix for the model."""
    from pyagent.memory import MemoryRecord

    records: list[MemoryRecord] = [m for m in matches if isinstance(m, MemoryRecord)]
    if not records:
        return ""
    return (
        "\n\nPrior research you have done on this topic or related topics "
        "(from your own memory):\n"
        + format_memories_for_prompt(records)
        + "\n\nYou may reuse facts from these prior briefs if relevant. "
        "If the current topic builds on prior research, you do not need to "
        "search for the same facts again — cite them from the prior summary "
        "and focus new searches on what is new or different."
    )


def _instructions(topic: str, brief_rel: str, generated_at: str) -> str:
    return f"""You are a research assistant producing a structured markdown brief.

Topic: {topic}

Process:
1. Call search_web 2-3 times with different query angles (overview, recent
   developments, expert perspective). Use max_results=5 on each call so you
   have enough sources to synthesize from.
2. Synthesize the results into a well-organized brief with numbered citations.
3. Write the brief using write_file with EXACTLY this path argument (a
   relative path; do not add or remove directory components): {brief_rel}

Markdown structure (required):
# {topic}

**Generated**: {generated_at}

## Summary
2-4 sentences summarizing what the brief covers.

## Key Facts
- 5-10 bullet points, each a concrete finding.
- One fact per bullet.
- End each bullet with a citation marker matching the numbered Sources list.
  Example: "Participants lost an average of 9% of body weight over 12 weeks [1]."

## Sources
1. [Title](URL)
2. [Title](URL)
- Number each source (1., 2., 3., ...) to match the [N] markers in Key Facts.
- One source per line.
- Copy each URL verbatim from a search_web result. Do NOT invent URLs.

## Open Questions
- 2-4 questions that remain unanswered.

Rules:
- Do NOT fabricate facts or URLs. Only cite what search_web returned.
- Every [N] citation in Key Facts MUST correspond to a numbered entry in Sources.
- If search_web fails (e.g. missing TAVILY_API_KEY), write a brief whose
  Summary explains the limitation, with empty Key Facts and Sources.
- After the write succeeds, respond with exactly: "Brief saved"
"""


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


async def _run(
    topic: str,
    output_dir: Path,
    max_steps: int,
    keep_on_failure: bool,
) -> int:
    slug = _slug(topic)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S-%f")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_dir = output_dir.resolve()
    workspace_root = output_dir.parent
    output_folder = output_dir / f"{stamp}-{slug}"
    brief_rel = f"{output_dir.name}/{output_folder.name}/brief.md"
    brief_path = output_folder / "brief.md"
    log_path = output_folder / "run.log.jsonl"

    _out(f"Topic:       {topic}")
    _out(f"Workspace:   {workspace_root}")
    _out(f"Output:      {output_folder}")
    _out(f"Max steps:   {max_steps}")

    # Declared before the outer try so the finally clause below can always
    # reference it — pyright strict rejects a variable that's only assigned
    # inside a try that might jump to finally before assignment.
    memory_store: MemoryStore | None = None
    succeeded = False

    output_folder.mkdir(parents=True, exist_ok=False)
    try:
        # --- Memory: auto-recall on entry (degrade on failure) -----------
        prior_block = ""
        try:
            memory_store = MemoryStore(_memory_path(workspace_root))
            prior_matches = memory_store.search(
                topic,
                namespace=MEMORY_NAMESPACE,
                limit=MEMORY_RECALL_LIMIT,
            )
            prior_block = _build_prior_block(list(prior_matches))
            if prior_matches:
                _out(
                    f"Memory: found {len(prior_matches)} prior brief(s) — "
                    f"injected into instructions"
                )
            else:
                _out("Memory: no prior research found for this topic")
        except Exception as exc:
            _warn(f"memory recall skipped: {exc}")
            memory_store = None

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

                instructions = _instructions(topic, brief_rel, generated_at)
                instructions += prior_block

                agent = Agent(
                    client=llm_client,
                    tools=tools,
                    instructions=instructions,
                    max_steps=max_steps,
                    confirm_tool_call=lambda tc: True,
                )

                _out("\nResearching...")
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
                    elapsed = time.monotonic() - start
                    _out(f"\nCompleted in {elapsed:.1f}s")

                _out(f"Agent output: {result.output}")
        finally:
            log.close()

        if not brief_path.exists():
            write_results = [
                item
                for ev in ctx.events
                for item in ev.content
                if isinstance(item, ToolResult) and item.name == "write_file"
            ]
            if write_results and write_results[-1].status == "success":
                actual = _write_file_result_text(write_results[-1])
                return _err(
                    EXIT_FAILURE,
                    f"model wrote to unexpected path ({actual}); expected {brief_path}",
                )
            return _err(EXIT_FAILURE, f"brief not created at {brief_path}")

        _out(f"\nBrief saved to {brief_path}")
        _out(f"Log at {log_path}")

        # --- Memory: auto-store on success (degrade on failure) ----------
        if memory_store is not None:
            try:
                summary = _extract_summary(brief_path.read_text(encoding="utf-8"))
                if summary:
                    rec = memory_store.remember(
                        key=slug,
                        value=summary,
                        namespace=MEMORY_NAMESPACE,
                        tags=[topic],
                    )
                    _out(
                        f"Memory: stored summary at {MEMORY_NAMESPACE}/{rec.key} (id={rec.id[:8]}…)"
                    )
                else:
                    _warn("brief has no '## Summary' section; not stored")
            except Exception as exc:
                _warn(f"memory store skipped: {exc}")

        succeeded = True
        return EXIT_OK
    finally:
        if memory_store is not None:
            memory_store.close()
        _cleanup_or_keep(output_folder, succeeded, keep_on_failure)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyagent-research-brief",
        description="Produce a structured markdown research brief on a topic.",
    )
    parser.add_argument("topic", help="Research topic")
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the brief and log (default: {DEFAULT_OUTPUT_DIR})",
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
    init_tracing(service_name="pyagent-research-brief")
    args = _parse_args(sys.argv[1:])
    if not args.topic.strip():
        return _err(EXIT_USAGE, "topic must not be empty")
    if args.steps < 1:
        return _err(EXIT_USAGE, "--steps must be >= 1")
    return asyncio.run(
        _run(
            args.topic,
            Path(args.output_dir),
            args.steps,
            args.keep_on_failure,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
