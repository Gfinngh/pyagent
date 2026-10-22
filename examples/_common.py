"""Shared helpers for the examples.

Display and setup utilities only — no assertions, no test infrastructure.
Importing anything from ``pyagent`` already runs ``load_dotenv()``, so these
helpers do not load ``.env`` themselves.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from pyagent.agent import AgentResult
from pyagent.client import (
    McpConnection,
    McpServerParams,
    ToolOverrides,
    load_mcp_tools,
)
from pyagent.llm import LlmClient
from pyagent.models import resolve_model
from pyagent.schemas import Message, ToolCall, ToolResult
from pyagent.tools import MultipartContent, RemoteTool, TextPart

# A model id known to exist on current Gemini tiers. Override with PYAGENT_MODEL.
DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_llm_client(model: str | None = None, **kwargs: object) -> LlmClient:
    """Build a real LiteLLM-backed client for the configured model."""
    raw = model or os.environ.get("PYAGENT_MODEL") or DEFAULT_MODEL
    return LlmClient(model=resolve_model(raw), **kwargs)  # type: ignore[arg-type]


def extract_text(value: object) -> str:
    """Flatten a ``RemoteTool.call`` result to plain text.

    ``RemoteTool.call`` returns either a ``MultipartContent`` (one or more
    content blocks) or a single value. This helper produces a
    human-readable string suitable for ``print``.
    """
    if isinstance(value, MultipartContent):
        return "\n".join(p.text for p in value.parts if isinstance(p, TextPart))
    if isinstance(value, TextPart):
        return value.text
    if isinstance(value, str):
        return value
    return str(value)


@contextlib.asynccontextmanager
async def mcp_with_workspace(
    workspace: str | Path,
    overrides: dict[str, ToolOverrides] | None = None,
) -> AsyncIterator[list[RemoteTool]]:
    """Open an MCP session whose server sees ``workspace`` as WORKSPACE_ROOT.

    ``pyagent.server``'s ``_workspace_root()`` reads ``WORKSPACE_ROOT`` from
    the environment at every tool call. The subprocess inherits the parent
    environment by default, so setting ``os.environ`` around the subprocess
    spawn is sufficient to point the file tools at the desired directory.
    The original value is restored afterwards, even on error.
    """
    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    original = os.environ.get("WORKSPACE_ROOT")
    os.environ["WORKSPACE_ROOT"] = str(workspace_path)
    try:
        params = McpServerParams(
            command=sys.executable,
            args=["-m", "pyagent.server"],
        )
        async with McpConnection(params) as conn:
            yield await load_mcp_tools(conn, overrides=overrides)
    finally:
        if original is None:
            os.environ.pop("WORKSPACE_ROOT", None)
        else:
            os.environ["WORKSPACE_ROOT"] = original


def _flatten(value: object) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def print_trace(result: AgentResult) -> None:
    """Print every event in the run as a compact, human-readable trace."""
    for event in result.context.events:
        for item in event.content:
            if isinstance(item, Message):
                prefix = {"user": "→", "assistant": "←", "system": "·"}[item.role]
                print(f"{prefix} {item.content}")
            elif isinstance(item, ToolCall):
                print(f"  ↪ {item.name}({item.arguments})")
            elif isinstance(item, ToolResult):
                flat = " | ".join(_flatten(c) for c in item.content)
                print(f"  ↪ result [{item.status}]: {flat}")


def print_final(result: AgentResult) -> None:
    """Print the final output, handling the unfinished case."""
    if result.output is None:
        print("(no final output)")
        return
    if isinstance(result.output, str):
        print(result.output)
    else:
        import json

        print(json.dumps(result.output, indent=2, default=str))


def collect_tool_results(result: AgentResult) -> list[ToolResult]:
    """Every ToolResult from a run, in emission order."""
    return [
        item
        for event in result.context.events
        for item in event.content
        if isinstance(item, ToolResult)
    ]
