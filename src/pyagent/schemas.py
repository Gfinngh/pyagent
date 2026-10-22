"""Data schemas for the ReAct agent.

Pydantic v2 models mirroring the TypeScript zod schemas:
- Message: a text turn
- ToolCall: LLM's request to execute a tool
- ToolResult: result from tool execution
- AgentEvent: one atomic occurrence during agent execution
- ContentItem: discriminated union of Message | ToolCall | ToolResult
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A text message in the conversation."""

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool.

    ``provider_options`` carries opaque provider metadata (e.g. Gemini
    ``thoughtSignature``). Must be round-tripped verbatim on replay or
    multi-turn tool calls degrade.
    """

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict[str, object]
    provider_options: dict[str, object] | None = None


class ToolResult(BaseModel):
    """Result from executing a tool.

    ``content`` holds raw tool outputs — heterogeneous by design (str, int,
    dict, MultipartContent, …). ``object`` is the honest type here.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list[object]


ContentItem = Annotated[
    Message | ToolCall | ToolResult,
    Field(discriminator="type"),
]


class AgentEvent(BaseModel):
    """One atomic occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=time.time)
    author: str
    content: list[ContentItem] = Field(default_factory=list[ContentItem])
