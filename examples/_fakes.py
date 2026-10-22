"""Deterministic LLM client for examples that show agent-loop mechanics.

This is a *demo* fake, not test infrastructure: it replays a scripted sequence
so the reader can watch what the real agent loop does with a known input. No
assertions, no network, no quota.

The verification suite has its own scripted client under ``scripts/``. Keeping
them separate lets each evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import Message, ToolCall


@dataclass
class ScriptedClient:
    """Replays a fixed list of ``LlmResponse`` objects, one per ``generate``."""

    responses: list[LlmResponse]
    requests: list[LlmRequest] = field(default_factory=list)
    _cursor: int = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self._cursor >= len(self.responses):
            raise RuntimeError(
                f"ScriptedClient exhausted after {self._cursor} call(s); add more responses"
            )
        response = self.responses[self._cursor]
        self._cursor += 1
        return response


def text_response(text: str) -> LlmResponse:
    return LlmResponse(content=[Message(role="assistant", content=text)])


def tool_call_response(
    tool_calls: list[ToolCall],
    text: str | None = None,
) -> LlmResponse:
    content: list[object] = []
    if text is not None:
        content.append(Message(role="assistant", content=text))
    content.extend(tool_calls)
    return LlmResponse(content=content)  # type: ignore[arg-type]


def error_response(
    message: str,
    metadata: dict[str, object] | None = None,
) -> LlmResponse:
    return LlmResponse(error_message=message, error_metadata=metadata or {})


def make_tool_call(
    tool_call_id: str,
    name: str,
    arguments: dict[str, object] | None = None,
) -> ToolCall:
    return ToolCall(
        tool_call_id=tool_call_id,
        name=name,
        arguments=arguments or {},
    )
