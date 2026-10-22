"""25 — Host CLI usage.

Prints `python main.py --help` and documents each mode. No LLM call.

    pixi run python examples/25_cli_modes.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from examples._common import section

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    section("python main.py --help")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    print(result.stdout.rstrip() or result.stderr.rstrip())

    section("common invocations")
    for cmd, note in [
        ('pixi run host "What is 2 + 2?"', "fresh run"),
        ('pixi run host --log runs/demo.jsonl "What is 2 + 2?"', "explicit log path"),
        ("python main.py --resume runs/demo.jsonl", "resume (no-op if complete)"),
        ('python main.py --resume runs/demo.jsonl "follow-up"', "resume with a follow-up"),
        ('pixi run host --approve "Write x to notes.txt"', "auto-approve mutations"),
        ('pixi run host --deny "Write x to notes.txt"', "auto-deny mutations"),
    ]:
        print(f"  {cmd}")
        print(f"    {note}")

    section("environment")
    for name, desc in [
        ("PYAGENT_MODEL", "default model id (keeps the LiteLLM prefix)"),
        ("PYAGENT_LOG_PATH", "default --log value"),
        ("PYAGENT_MCP_MODULE", "module launched as the MCP server"),
        ("WORKSPACE_ROOT", "root for the MCP file tools"),
        ("TAVILY_API_KEY", "required for search_web"),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTLP trace export"),
        ("LANGFUSE_HOST", "Langfuse OTLP endpoint"),
    ]:
        print(f"  {name:<28} {desc}")

    section("exit codes")
    print("  0  success")
    print("  1  LLM or MCP failure")
    print("  2  usage error")
    print("  3  quota exhausted")

    section("workflows")
    print('  pixi run brief "topic"                 research brief')
    print('  pixi run iterative-brief "topic" --refine "..."')
    print("  pixi run notes-digest --notes-dir notes/")
    print('  pixi run trip-planner "Kyoto" --days 3')


if __name__ == "__main__":
    main()
