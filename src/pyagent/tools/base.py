"""Base tool classes and multi-part tool result types.

- ``BaseTool`` — abstract base; ``name`` / ``description`` / ``permission`` /
  ``concurrency_safe`` / ``timeout_ms`` / ``tool_definition`` plus ``call``.
- ``FunctionTool[S]`` — wraps a pydantic schema + function.
- ``RemoteTool`` — wraps an MCP-discovered tool.
- ``MultipartContent`` — container for multi-part tool results.

Timeouts
--------
``timeout_ms`` is advisory on the tool: ``Agent.act`` enforces it with
``asyncio.timeout``. Default ``0`` means no per-tool limit.

Permissions
-----------
``permission`` is an internal policy field, *not* exposed to the LLM. Three
levels:

- ``"read"`` (default) — safe; ``Agent.act`` runs the tool without asking.
- ``"mutate"`` — changes external state; requires a ``confirm_tool_call``
  callback on the agent.
- ``"dangerous"`` — irreversible; requires both ``Agent(..., allow_dangerous=True)``
  *and* a callback returning True. Two-lock model.

Concurrency
-----------
``concurrency_safe`` controls parallel tool execution. When the model emits
multiple tool calls in one response and *every* approved call in the batch is
concurrency-safe, ``Agent.act`` runs them via ``asyncio.gather``. If *any* call
in the batch is not safe, all run sequentially — mutations always see
consistent state, and mixed read/write batches cannot reorder.

Default ``None`` derives safety from permission: ``read`` is concurrency-safe,
``mutate`` and ``dangerous`` are not. Set explicitly when the derivation is
wrong (e.g. a read tool that mutates internal cache state and cannot run
concurrently with itself).
"""

from __future__ import annotations

import abc
import inspect
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, TypeVar, cast

from pydantic import BaseModel, Field

from pyagent.context import ExecutionContext

S = TypeVar("S", bound=BaseModel)

Permission = Literal["read", "mutate", "dangerous"]

_VALID_PERMISSIONS: frozenset[str] = frozenset({"read", "mutate", "dangerous"})

_STRIP_KEYS = ("$schema", "title")


def _clean_parameters(schema: dict[str, object]) -> dict[str, object]:
    """Strip pydantic-only noise ($schema, top-level title) from a JSON schema."""
    return {k: v for k, v in schema.items() if k not in _STRIP_KEYS}


def _validate_permission(value: str) -> Permission:
    if value not in _VALID_PERMISSIONS:
        raise ValueError(f"permission must be one of {sorted(_VALID_PERMISSIONS)}; got {value!r}")
    return cast(Permission, value)


class BaseTool(abc.ABC):
    """Abstract base for all tools."""

    name: str
    description: str
    permission: Permission
    concurrency_safe: bool | None
    timeout_ms: int
    _tool_definition: dict[str, object] | None

    def __init__(
        self,
        *,
        name: str = "",
        description: str = "",
        permission: Permission = "read",
        concurrency_safe: bool | None = None,
        timeout_ms: int = 0,
        tool_definition: dict[str, object] | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.description = description
        self.permission = _validate_permission(permission)
        self.concurrency_safe = concurrency_safe
        self.timeout_ms = max(0, int(timeout_ms))
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> dict[str, object] | None:
        return self._tool_definition

    @abc.abstractmethod
    async def call(
        self,
        args: object | None = None,
        context: ExecutionContext | None = None,
    ) -> object: ...


class FunctionTool(BaseTool):
    """Local tool backed by a pydantic schema and a Python function."""

    schema: type[BaseModel]
    func: Callable[[BaseModel, ExecutionContext | None], object | Awaitable[object]]

    def __init__(
        self,
        *,
        schema: type[S],
        func: Callable[[S, ExecutionContext | None], object | Awaitable[object]],
        name: str,
        description: str,
        permission: Permission = "read",
        concurrency_safe: bool | None = None,
        timeout_ms: int = 0,
    ) -> None:
        parameters = _clean_parameters(schema.model_json_schema())
        tool_def: dict[str, object] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        super().__init__(
            name=name,
            description=description,
            permission=permission,
            concurrency_safe=concurrency_safe,
            timeout_ms=timeout_ms,
            tool_definition=tool_def,
        )
        self.schema = schema
        self.func = cast(
            Callable[[BaseModel, ExecutionContext | None], object | Awaitable[object]],
            func,
        )

    async def call(
        self,
        args: object | None = None,
        context: ExecutionContext | None = None,
    ) -> object:
        parsed = self.schema.model_validate(args or {})
        result = self.func(parsed, context)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result


class RemoteTool(BaseTool):
    """MCP-discovered tool. Uses the server's JSON Schema verbatim."""

    call_fn: Callable[[dict[str, object]], object | Awaitable[object]]

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, object],
        call: Callable[[dict[str, object]], object | Awaitable[object]],
        permission: Permission = "read",
        concurrency_safe: bool | None = None,
        timeout_ms: int = 0,
    ) -> None:
        tool_def: dict[str, object] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": _clean_parameters(input_schema),
            },
        }
        super().__init__(
            name=name,
            description=description,
            permission=permission,
            concurrency_safe=concurrency_safe,
            timeout_ms=timeout_ms,
            tool_definition=tool_def,
        )
        self.call_fn = call

    async def call(
        self,
        args: object | None = None,
        context: ExecutionContext | None = None,
    ) -> object:
        _ = context
        args_dict = cast(dict[str, object], args) if isinstance(args, dict) else {}
        result = self.call_fn(args_dict)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result


# ---------------------------------------------------------------------------
# Multipart content
# ---------------------------------------------------------------------------


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    data: str
    mime_type: str


class ResourcePart(BaseModel):
    type: Literal["resource"] = "resource"
    uri: str
    mime_type: str | None = None
    text: str | None = None
    blob: str | None = None


class StructuredPart(BaseModel):
    type: Literal["structured"] = "structured"
    value: object


PartSpec = Annotated[
    TextPart | ImagePart | ResourcePart | StructuredPart,
    Field(discriminator="type"),
]


class MultipartContent(BaseModel):
    """Multi-part tool result."""

    parts: list[PartSpec]
