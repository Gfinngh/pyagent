"""Host entry point for the pyagent ReAct agent.

Usage:
    pixi run host                                # fresh run, default prompt
    pixi run host "What is 2 + 2?"              # custom prompt
    pixi run host --approve "Write to a.txt"    # auto-approve mutations
    pixi run host --deny "Write to a.txt"       # auto-deny mutations
    python main.py --resume runs/host-X.jsonl   # resume
    python main.py --resume runs/host-X.jsonl "next question"

Tracing: ``init_tracing()`` runs at program start. With no OTLP env vars set
it is a no-op; set ``OTEL_EXPORTER_OTLP_ENDPOINT`` or ``LANGFUSE_*`` to
export spans to a collector.

Log path precedence: --log <path> > PYAGENT_LOG_PATH env > runs/host-<timestamp>.jsonl

On ``--resume``, events are replayed into a NEW log; the source log is never
modified.

Exit codes: 0 success, 1 LLM/MCP failure, 2 usage error, 3 quota exhausted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

from pyagent.agent import Agent, ConfirmCallback, LlmCallError  # noqa: E402
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
from pyagent.schemas import Message, ToolCall, ToolResult  # noqa: E402
from pyagent.tracing import init_tracing  # noqa: E402

DEFAULT_PROMPT = "What is 2 + 2? Use the calculator."
DEFAULT_LOG_DIR = "runs"
DEFAULT_MCP_MODULE = "pyagent.server"
MCP_CONNECT_TIMEOUT_S = 10.0

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_QUOTA = 3

TOOL_OVERRIDES: dict[str, ToolOverrides] = {
    "search_web": ToolOverrides(permission="read", timeout_ms=15_000),
    "write_file": ToolOverrides(permission="mutate", timeout_ms=10_000),
    "execute_python": ToolOverrides(permission="dangerous", timeout_ms=15_000),
}


def _default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(DEFAULT_LOG_DIR) / f"host-{timestamp}.jsonl"


def _resolve_log_path(cli_arg: str | None) -> Path:
    if cli_arg is not None:
        return Path(cli_arg)
    env = os.environ.get("PYAGENT_LOG_PATH")
    if env:
        return Path(env)
    return _default_log_path()


def _resolve_mcp_module() -> str:
    return os.environ.get("PYAGENT_MCP_MODULE", DEFAULT_MCP_MODULE)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyagent",
        description="ReAct agent host (LiteLLM + pydantic + MCP).",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="...")
    parser.add_argument("--resume", metavar="PATH", default=None, help="...")
    parser.add_argument("--log", metavar="PATH", default=None, help="...")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--approve", action="store_true", help="...")
    approval.add_argument("--deny", action="store_true", help="...")
    return parser.parse_args(argv)


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(exit_code: int, message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _print_trace(context: ExecutionContext) -> None:
    _out("\n=== Event trace ===")
    for event in context.events:
        _out(f"\n[{event.author}] (id={event.id[:8]}…)")
        for item in event.content:
            if isinstance(item, Message):
                _out(f"  msg({item.role}): {item.content}")
            elif isinstance(item, ToolCall):
                _out(f"  call: {item.name}({json.dumps(item.arguments, default=str)})")
            elif isinstance(item, ToolResult):
                _out(f"  result({item.status}): {json.dumps(item.content, default=str)}")


def _report_llm_failure(exc: LlmCallError) -> int:
    print(f"Agent failed: {exc}", file=sys.stderr)
    print(f"  metadata: {exc.metadata}", file=sys.stderr)
    if exc.metadata.get("quota_exhausted") is True:
        return EXIT_QUOTA
    return EXIT_FAILURE


def _make_confirm_callback(mode: str) -> ConfirmCallback:
    async def _confirm(tc: ToolCall) -> bool:
        _out(
            f"\n[approval] tool '{tc.name}' with arguments {json.dumps(tc.arguments, default=str)}"
        )
        if mode == "yes":
            _out("  auto-approved (--approve)")
            return True
        if mode == "no":
            _out("  auto-denied (--deny)")
            return False
        if not sys.stdin.isatty():
            _out("  stdin is not a TTY; denying (use --approve for automation)")
            return False
        try:
            answer = await asyncio.to_thread(input, "  approve? [y/N]: ")
        except EOFError:
            _out("  stdin closed; denying")
            return False
        approved = answer.strip().lower() in {"y", "yes"}
        _out(f"  {'approved' if approved else 'denied'} by user")
        return approved

    return _confirm


def _build_agent(remote_tools: list[object], approval_mode: str) -> Agent:
    from pyagent.tools import BaseTool

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
    client = FallbackClient(
        chain=chain,
        strip_provider_options_on_switch=True,
        on_fallback=lambda info: _warn(
            f"model fallback: {info['from']} -> {info['to']} ({str(info['reason'])[:80]})"
        ),
    )
    tools = [t for t in remote_tools if isinstance(t, BaseTool)]
    return Agent(
        client=client,
        tools=tools,
        instructions="You are a helpful assistant. Use tools when appropriate.",
        max_steps=4,
        confirm_tool_call=_make_confirm_callback(approval_mode),
    )


async def _run_fresh(agent: Agent, prompt: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    log = JsonlEventLog(log_path)
    ctx = ExecutionContext(log=log)
    try:
        result = await agent.run(prompt, ctx)
    except LlmCallError as exc:
        log.close()
        return _report_llm_failure(exc)
    finally:
        log.close()

    _out(f"Output: {result.output}")
    _print_trace(result.context)
    _out(f"\nLog written to {log_path} ({len(result.context.events)} events)")
    return EXIT_OK


async def _run_resume(
    agent: Agent,
    ctx: ExecutionContext,
    source_path: Path,
    follow_up: str | None,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    log = JsonlEventLog(log_path)
    for event in ctx.events:
        log.append(event)
    ctx.log = log
    try:
        result = await agent.resume(ctx, follow_up)
    except LlmCallError as exc:
        log.close()
        return _report_llm_failure(exc)
    finally:
        log.close()

    _out(f"Output: {result.output}")
    _print_trace(result.context)
    _out(f"\nResumed from {source_path}")
    _out(f"Log written to {log_path} ({len(result.context.events)} events)")
    return EXIT_OK


async def _run(args: argparse.Namespace) -> int:
    log_path = _resolve_log_path(args.log)

    probed_ctx: ExecutionContext | None = None
    if args.resume is not None:
        source_path = Path(args.resume)
        if not source_path.exists():
            return _err(EXIT_USAGE, f"--resume file not found: {source_path}")
        probed_ctx = ExecutionContext.from_log(source_path)
        if not probed_ctx.events:
            return _err(EXIT_USAGE, f"--resume file is empty: {source_path}")
        if probed_ctx.final_result is not None and args.prompt is None:
            _out("Resumed run is already complete.")
            _out(f"  final_result: {probed_ctx.final_result}")
            _out("Pass a positional message to add a follow-up turn.")
            _out("No new log file written.")
            return EXIT_OK
        if source_path.resolve() == log_path.resolve():
            return _err(EXIT_USAGE, "--log must not point at the --resume source file")

    approval_mode = "yes" if args.approve else "no" if args.deny else "ask"
    if approval_mode == "yes":
        _warn("--approve is on; mutate tools will run without confirmation")

    server_params = McpServerParams(
        command=sys.executable,
        args=["-m", _resolve_mcp_module()],
    )

    conn = McpConnection(server_params)
    try:
        try:
            await asyncio.wait_for(conn.connect(), timeout=MCP_CONNECT_TIMEOUT_S)
        except TimeoutError:
            return _err(
                EXIT_FAILURE,
                f"MCP server failed to start within {MCP_CONNECT_TIMEOUT_S:.0f}s",
            )
        except Exception as exc:
            return _err(EXIT_FAILURE, f"MCP server failed to start: {exc}")

        try:
            remote_tools = await load_mcp_tools(conn, overrides=TOOL_OVERRIDES)
        except Exception as exc:
            return _err(EXIT_FAILURE, f"MCP tool discovery failed: {exc}")

        _out(f"Discovered tools: {sorted(t.name for t in remote_tools)}")
        agent = _build_agent(list(remote_tools), approval_mode)

        if probed_ctx is not None:
            return await _run_resume(
                agent,
                probed_ctx,
                Path(args.resume or ""),
                args.prompt,
                log_path,
            )
        return await _run_fresh(agent, args.prompt or DEFAULT_PROMPT, log_path)
    finally:
        await conn.close()


def main() -> int:
    init_tracing(service_name="pyagent-cli")
    args = _parse_args(sys.argv[1:])
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
