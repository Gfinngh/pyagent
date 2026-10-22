"""MCP client: spawn servers, discover tools, wrap them as RemoteTool.

Mirrors ``client.ts``. Divergences (documented):
- ``McpConnection`` owns an ``AsyncExitStack`` so the subprocess and session
  close cleanly. It works both as an explicit object (``connect()`` /
  ``close()``, like the TS version) and as an async context manager.
- MCP SDK v2 field names are snake_case (``is_error``, ``input_schema``,
  ``structured_content``); the wire format is unchanged from v1.
- Tool error text arrives prefixed with ``"Error executing tool <name>: "``.
  We strip it so the LLM sees the raw message, matching the TypeScript client.
- Every content block kind is preserved (text, image, audio, resource link,
  embedded resource). Reading only ``.text`` would silently drop images and
  resources.
- MCPServer v2 auto-populates ``structured_content`` for *every* tool return,
  including plain strings - so ``content`` and ``structured_content`` often
  carry the same data. We append ``structured_content`` only when ``content``
  produced zero parts, to avoid duplicating what the LLM already sees.

Tool overrides
--------------
MCP tools are discovered at runtime and default to ``permission="read"``,
``timeout_ms=0``. ``load_mcp_tools(overrides=...)`` lets the caller set those
per tool by name, without touching ``RemoteTool`` internals. Tools not in the
overrides map keep the defaults, so existing call sites are unchanged.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
)

from pyagent.tools import (
    ImagePart,
    MultipartContent,
    PartSpec,
    Permission,
    RemoteTool,
    ResourcePart,
    StructuredPart,
    TextPart,
)

_ERROR_PREFIX_TEMPLATE = "Error executing tool {name}: "


@dataclass
class ToolOverrides:
    """Per-tool overrides applied at ``load_mcp_tools`` time.

    Fields mirror the ``RemoteTool`` constructor; anything not set falls back
    to the class default (``read`` permission, no timeout).
    """

    permission: Permission = "read"
    timeout_ms: int = 0


class McpToolError(RuntimeError):
    """Raised when an MCP tool reports ``is_error=True``.

    ``Agent.act`` catches this and records ``ToolResult(status="error")``, so
    the model sees the failure text and can correct itself.
    """


@dataclass
class McpServerParams:
    """How to launch an MCP server subprocess."""

    command: str
    args: list[str] = field(default_factory=list[str])
    env: dict[str, str] | None = None


class McpConnection:
    """An open stdio connection to an MCP server.

    Owns the subprocess and the ``ClientSession`` via an ``AsyncExitStack``,
    which unwinds both in the right order on ``close()``.
    """

    params: McpServerParams
    _stack: AsyncExitStack | None
    _session: ClientSession | None

    def __init__(self, params: McpServerParams) -> None:
        self.params = params
        self._stack = None
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("McpConnection is not connected; call connect() first.")
        return self._session

    async def connect(self) -> None:
        if self._session is not None:
            return

        stack = AsyncExitStack()
        try:
            server_params = StdioServerParameters(
                command=self.params.command,
                args=list(self.params.args),
                env=(self.params.env if self.params.env is not None else dict(os.environ)),
            )
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def __aenter__(self) -> McpConnection:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def list_tools(self) -> list[Tool]:
        result = await self.session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, object]) -> CallToolResult:
        return await self.session.call_tool(name, arguments)


async def create_mcp_client(params: McpServerParams) -> McpConnection:
    """Spawn the server and initialize a session. Caller must ``close()``."""
    conn = McpConnection(params)
    await conn.connect()
    return conn


async def load_mcp_tools(
    conn: McpConnection,
    overrides: dict[str, ToolOverrides] | None = None,
) -> list[RemoteTool]:
    """Discover tools via ``tools/list`` and wrap each as a ``RemoteTool``.

    ``overrides`` is an optional per-tool config map keyed by tool name. Tools
    not present use the defaults (``read`` permission, no timeout).
    """
    tools = await conn.list_tools()
    config = overrides or {}
    return [_create_remote_tool(t, conn, config.get(t.name)) for t in tools]


def _create_remote_tool(
    tool: Tool,
    conn: McpConnection,
    overrides: ToolOverrides | None,
) -> RemoteTool:
    name = tool.name
    description = tool.description or ""
    permission: Permission = overrides.permission if overrides else "read"
    timeout_ms = overrides.timeout_ms if overrides else 0

    async def _call(args: dict[str, object]) -> object:
        result = await conn.call_tool(name, args)
        if result.is_error:
            raise McpToolError(_error_text(result, name))
        return mcp_result_to_multipart(result)

    return RemoteTool(
        name=name,
        description=description,
        input_schema=tool.input_schema,
        call=_call,
        permission=permission,
        timeout_ms=timeout_ms,
    )


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


def mcp_result_to_multipart(result: CallToolResult) -> MultipartContent:
    """Normalize an MCP ``CallToolResult`` into typed content parts.

    Unknown block kinds are preserved as ``StructuredPart`` so nothing is
    silently dropped. ``structured_content`` is a fallback representation -
    only emitted when ``content`` produced zero parts, to avoid duplicating
    data the LLM already sees.
    """
    parts: list[PartSpec] = []

    for block in cast(list[object], result.content):
        if isinstance(block, TextContent):
            parts.append(TextPart(text=block.text))
        elif isinstance(block, ImageContent):
            parts.append(ImagePart(data=block.data, mime_type=block.mime_type))
        elif isinstance(block, EmbeddedResource):
            parts.append(_embedded_resource_part(block))
        elif isinstance(block, ResourceLink):
            parts.append(TextPart(text=f"[resource link: {block.name} <{block.uri}>]"))
        else:
            # Audio blocks and any block kind newer than this SDK version.
            parts.append(StructuredPart(value=_dump_block(block)))

    structured = getattr(result, "structured_content", None)
    if structured is not None and not parts:
        # MCPServer v2 populates structured_content alongside text content for
        # plain tool returns; appending it would duplicate the same data. Only
        # used when content was empty (e.g. structured-only third-party tools).
        parts.append(StructuredPart(value=structured))

    if not parts:
        parts.append(TextPart(text=""))

    return MultipartContent(parts=parts)


def _embedded_resource_part(block: EmbeddedResource) -> PartSpec:
    resource = block.resource
    if isinstance(resource, TextResourceContents):
        return ResourcePart(
            uri=str(resource.uri),
            mime_type=resource.mime_type,
            text=resource.text,
        )
    # pyright narrows resource to BlobResourceContents after the branch above.
    return ResourcePart(
        uri=str(resource.uri),
        mime_type=resource.mime_type,
        blob=resource.blob,
    )


def _dump_block(block: Any) -> object:
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump()
    return str(block)


# ---------------------------------------------------------------------------
# Error text
# ---------------------------------------------------------------------------


def _error_text(result: CallToolResult, tool_name: str) -> str:
    texts: list[str] = []
    for block in cast(list[object], result.content):
        if isinstance(block, TextContent):
            texts.append(block.text)

    raw = "\n".join(texts).strip()
    stripped = _strip_error_prefix(raw, tool_name)
    if stripped:
        return stripped
    return f"MCP tool '{tool_name}' returned an error."


def _strip_error_prefix(text: str, tool_name: str) -> str:
    """Remove the SDK's ``"Error executing tool <name>: "`` wrapper."""
    prefix = _ERROR_PREFIX_TEMPLATE.format(name=tool_name)
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


# ---------------------------------------------------------------------------
# Smoke entry point (``pixi run client``)
# ---------------------------------------------------------------------------


async def _smoke() -> None:
    import sys

    params = McpServerParams(command=sys.executable, args=["-m", "pyagent.server"])
    async with McpConnection(params) as conn:
        tools = await conn.list_tools()
        print(f"Discovered {len(tools)} tool(s): {sorted(t.name for t in tools)}")
        greet = await conn.call_tool("greet", {"name": "World"})
        print(f"greet -> {mcp_result_to_multipart(greet).model_dump()}")


if __name__ == "__main__":
    asyncio.run(_smoke())
