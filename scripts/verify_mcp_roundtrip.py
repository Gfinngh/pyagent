"""Phase 8: verify MCP client — discovery, wrapping, error handling, agent loop.

Deterministic where possible (no LLM) plus one end-to-end agent run over the
remote calculator.

Skip conditions (not workflow bugs):
  - ``quota_exhausted``: Gemini daily quota wall
  - transient LLM failure: 408 timeout, 5xx, retry exhaustion
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from pyagent.agent import Agent, LlmCallError
from pyagent.client import (
    McpConnection,
    McpServerParams,
    McpToolError,
    load_mcp_tools,
    mcp_result_to_multipart,
)
from pyagent.context import ExecutionContext
from pyagent.llm import LlmClient
from pyagent.models import resolve_model
from pyagent.schemas import ToolCall
from pyagent.tools import (
    ImagePart,
    MultipartContent,
    ResourcePart,
    StructuredPart,
    TextPart,
)
from scripts._verify_common import is_transient_llm_response

_EXPECTED_TOOLS: tuple[str, ...] = (
    "calculator",
    "greet",
    "list_files",
    "read_file",
    "search_web",
    "write_file",
    "execute_python",
)

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


def _skip_if_llm_error(exc: LlmCallError) -> bool:
    if exc.metadata.get("quota_exhausted") is True:
        skip(f"quota exhausted: {exc}")
        return True
    if is_transient_llm_response(str(exc), exc.metadata):
        skip(f"transient LLM failure: {exc}")
        return True
    check(False, f"LLM failed: {exc}")
    return True


def text_parts(mc: MultipartContent) -> str:
    return "\n".join(p.text for p in mc.parts if isinstance(p, TextPart))


def server_params() -> McpServerParams:
    return McpServerParams(command=sys.executable, args=["-m", "pyagent.server"])


async def main() -> None:
    print("\nTest 1: connect, discover, wrap as RemoteTool")
    async with McpConnection(server_params()) as conn:
        tools = await load_mcp_tools(conn)
        names = sorted(t.name for t in tools)
        check(
            names == sorted(_EXPECTED_TOOLS),
            f"expected {sorted(_EXPECTED_TOOLS)} (got {names})",
        )
        check(
            all(t.tool_definition is not None for t in tools),
            "every tool has a tool_definition",
        )

        print("\nTest 2: remote greet returns MultipartContent with text")
        greet = next(t for t in tools if t.name == "greet")
        out = await greet.call({"name": "World"})
        check(
            isinstance(out, MultipartContent),
            f"returns MultipartContent (got {type(out).__name__})",
        )
        if isinstance(out, MultipartContent):
            check(
                len(out.parts) == 1,
                f"one part, no duplicate structured_content (got {len(out.parts)})",
            )
            check(
                text_parts(out) == "Hello, World.",
                f"text is 'Hello, World.' (got {text_parts(out)!r})",
            )

        print("\nTest 3: remote calculator add")
        calc = next(t for t in tools if t.name == "calculator")
        out = await calc.call({"operator": "add", "first_number": 2, "second_number": 2})
        check(isinstance(out, MultipartContent), "returns MultipartContent")
        if isinstance(out, MultipartContent):
            check(
                len(out.parts) == 1,
                f"one part, no duplicate structured_content (got {len(out.parts)})",
            )
            check(
                "4" in text_parts(out),
                f"result contains '4' (got {text_parts(out)!r})",
            )

        print("\nTest 4: divide by zero raises with prefix stripped")
        try:
            await calc.call({"operator": "divide", "first_number": 1, "second_number": 0})
        except McpToolError as e:
            msg = str(e)
            check(
                "divide by zero" in msg.lower(),
                f"message mentions divide by zero (got {msg!r})",
            )
            check(
                not msg.startswith("Error executing tool"),
                f"SDK prefix stripped (got {msg!r})",
            )
        else:
            check(False, "expected McpToolError")

        print("\nTest 5: Agent.act records tool error without crashing")
        agent = Agent(
            client=LlmClient(model=resolve_model("gemini/gemini-3.1-flash-lite")),
            tools=tools,
        )
        tc = ToolCall.model_validate(
            {
                "tool_call_id": "t-div",
                "name": "calculator",
                "arguments": {
                    "operator": "divide",
                    "first_number": 1,
                    "second_number": 0,
                },
            }
        )
        results = await agent.act(ExecutionContext(), [tc])
        check(len(results) == 1, "one ToolResult")
        check(results[0].status == "error", f"status=error (got {results[0].status})")
        err_text = str(results[0].content[0])
        check(
            "divide by zero" in err_text.lower(),
            f"error text preserved (got {err_text!r})",
        )
        check(
            not err_text.startswith("Error executing tool"),
            f"prefix stripped before reaching the agent (got {err_text!r})",
        )

        print("\nTest 6: end-to-end agent run over remote calculator")
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            skip("GEMINI_API_KEY not set (needed for Test 6)")
        else:
            try:
                result = await agent.run(
                    "Use the calculator to compute 12345 * 67890. Do not compute it yourself.",
                )
            except LlmCallError as e:
                # Record skip/failure in _SKIPPED/_FAILURES and return.
                # _finalize() runs once, from run()'s finally clause.
                _skip_if_llm_error(e)
                return
            else:
                check(
                    result.output is not None,
                    f"output not None (got {result.output!r})",
                )
                if isinstance(result.output, str):
                    digits = result.output.replace(",", "").replace(" ", "")
                    check(
                        "838102050" in digits,
                        f"output contains 838102050 (got {result.output!r})",
                    )
                else:
                    check(
                        False,
                        f"output is str (got {type(result.output).__name__})",
                    )

    print("\nTest 7: mcp_result_to_multipart — content blocks preserved,")
    print("         structured_content suppressed when content is non-empty")
    synthetic_with_content = CallToolResult(
        content=[
            TextContent(type="text", text="hello"),
            ImageContent(type="image", data="aGVsbG8=", mime_type="image/png"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///notes.txt",
                    mime_type="text/plain",
                    text="body text",
                ),
            ),
        ],
        structured_content={"answer": 42},
    )
    mc = mcp_result_to_multipart(synthetic_with_content)
    kinds = [p.type for p in mc.parts]
    check(
        kinds == ["text", "image", "resource"],
        f"part kinds — structured suppressed (got {kinds})",
    )
    image = next((p for p in mc.parts if isinstance(p, ImagePart)), None)
    check(image is not None and image.mime_type == "image/png", "image mime preserved")
    resource = next((p for p in mc.parts if isinstance(p, ResourcePart)), None)
    check(
        resource is not None and resource.text == "body text",
        "resource text preserved",
    )

    print("\nTest 7b: structured_content used when content is empty")
    synthetic_structured_only = CallToolResult(
        content=[],
        structured_content={"answer": 42},
    )
    mc = mcp_result_to_multipart(synthetic_structured_only)
    check(len(mc.parts) == 1, f"one part (got {len(mc.parts)})")
    structured = mc.parts[0]
    check(
        isinstance(structured, StructuredPart) and structured.value == {"answer": 42},
        "structured_content mapped when content empty",
    )

    print("\nTest 8: empty content + no structured_content yields one empty text part")
    mc = mcp_result_to_multipart(CallToolResult(content=[]))
    check(len(mc.parts) == 1 and mc.parts[0].type == "text", "single text placeholder")


def _finalize(*, completed: bool) -> None:
    if _FAILURES:
        print(f"\n\u274c {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        if _SKIPPED:
            print(f"\n\u26a0 {len(_SKIPPED)} skipped:")
            for s in _SKIPPED:
                print(f"  - {s}")
        sys.exit(1)
    if not completed:
        print("\n❌ verify script crashed before completing")
        return

    if _SKIPPED:
        print(f"\n\u2705 Phase 8 (MCP client) verified with {len(_SKIPPED)} skipped:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return
    print("\n\u2705 Phase 8 (MCP client) verified.")


def run() -> None:
    completed = False
    try:
        asyncio.run(main())
        completed = True
    finally:
        _finalize(completed=completed)


if __name__ == "__main__":
    run()
