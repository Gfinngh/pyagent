"""ReAct agent: the loop that strings LLM calls and tool executions.

Tracing: ``agent.run`` / ``agent.resume`` are root spans per invocation;
``agent.step`` is a child per loop iteration; ``tool.call`` is a child per
tool execution. Parallel tool calls (``asyncio.gather``) share the step span
as their common parent — gather copies contextvars, so each task sees the step
as the active span.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, Field, create_model

from pyagent.context import ExecutionContext
from pyagent.llm import LlmRequest, LlmResponse, ToolChoiceSetting
from pyagent.schemas import AgentEvent, ContentItem, Message, ToolCall, ToolResult
from pyagent.tools import BaseTool, FunctionTool
from pyagent.tracing import span

ConfirmCallback = Callable[[ToolCall], bool | Awaitable[bool]]


class LlmClientProtocol(Protocol):
    async def generate(self, request: LlmRequest) -> LlmResponse: ...


class LlmCallError(Exception):
    metadata: dict[str, object]

    def __init__(self, message: str, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class AgentTimeoutError(Exception):
    metadata: dict[str, object]

    def __init__(self, message: str, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


@dataclass
class AgentResult:
    output: str | dict[str, object] | None
    context: ExecutionContext


def _make_final_answer_tool(user_schema: type[BaseModel]) -> FunctionTool:
    schema: type[BaseModel] = create_model(
        "FinalAnswerInput",
        output=(user_schema, Field(..., description="The final structured answer.")),
    )

    def final_answer_func(
        args: BaseModel,
        context: ExecutionContext | None = None,
    ) -> object:
        _ = context
        dumped = args.model_dump()
        return dumped.get("output")

    return FunctionTool(
        name="final_answer",
        description="Return the final structured answer matching the required schema.",
        schema=schema,
        func=final_answer_func,
    )


def _denial_result(tc: ToolCall, reason: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tc.tool_call_id,
        name=tc.name,
        status="error",
        content=[reason],
    )


def _is_concurrency_safe(tool: BaseTool) -> bool:
    if tool.concurrency_safe is not None:
        return tool.concurrency_safe
    return tool.permission == "read"


class Agent:
    client: LlmClientProtocol
    instructions: str
    max_steps: int
    run_timeout_ms: int
    name: str
    output_type: type[BaseModel] | None
    output_tool_name: str | None
    tools: list[BaseTool]
    confirm_tool_call: ConfirmCallback | None
    allow_dangerous: bool

    def __init__(
        self,
        *,
        client: LlmClientProtocol,
        tools: Sequence[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 4,
        run_timeout_ms: int = 0,
        name: str = "agent",
        output_type: type[BaseModel] | None = None,
        confirm_tool_call: ConfirmCallback | None = None,
        allow_dangerous: bool = False,
    ) -> None:
        self.client = client
        self.instructions = instructions
        self.max_steps = max_steps
        self.run_timeout_ms = max(0, int(run_timeout_ms))
        self.name = name
        self.output_type = output_type
        self.output_tool_name = None
        self.tools = self._setup_tools(list(tools or []))
        self.confirm_tool_call = confirm_tool_call
        self.allow_dangerous = allow_dangerous

    def _setup_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        result = list(tools)
        if self.output_type is not None:
            result.append(_make_final_answer_tool(self.output_type))
            self.output_tool_name = "final_answer"
        return result

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str,
        context: ExecutionContext | None = None,
    ) -> AgentResult:
        ctx = context if context is not None else ExecutionContext()

        with span(
            "agent.run",
            **{
                "agent.name": self.name,
                "agent.max_steps": self.max_steps,
                "agent.tool_count": len(self.tools),
                "agent.has_output_type": self.output_type is not None,
                "execution.id": ctx.execution_id,
                "user.input_length": len(user_input),
            },
        ) as h:
            user_event = AgentEvent(
                execution_id=ctx.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            ctx.add_event(user_event)

            try:
                await self._run_with_deadline(lambda: self._loop(ctx), ctx)
            finally:
                h.set("agent.steps_used", ctx.current_step)
                h.set("agent.events_count", len(ctx.events))
                h.set("agent.final_result_present", ctx.final_result is not None)

            return AgentResult(output=ctx.final_result, context=ctx)

    async def resume(
        self,
        context: ExecutionContext,
        user_input: str | None = None,
    ) -> AgentResult:
        async def _body() -> None:
            last = context.events[-1] if context.events else None
            if last is not None and self._has_pending_tool_calls(last):
                pending = [item for item in last.content if isinstance(item, ToolCall)]
                recovery_results = await self.act(context, pending)
                context.add_event(
                    AgentEvent(
                        execution_id=context.execution_id,
                        author=self.name,
                        content=cast(list[ContentItem], recovery_results),
                    )
                )

            if user_input is not None:
                context.add_event(
                    AgentEvent(
                        execution_id=context.execution_id,
                        author="user",
                        content=[Message(role="user", content=user_input)],
                    )
                )
                context.final_result = None

            await self._loop(context)

        with span(
            "agent.resume",
            **{
                "agent.name": self.name,
                "agent.max_steps": self.max_steps,
                "execution.id": context.execution_id,
                "resume.has_user_input": user_input is not None,
                "resume.prior_events": len(context.events),
            },
        ) as h:
            try:
                await self._run_with_deadline(_body, context)
            finally:
                h.set("agent.steps_used", context.current_step)
                h.set("agent.events_count", len(context.events))
                h.set("agent.final_result_present", context.final_result is not None)

            return AgentResult(output=context.final_result, context=context)

    # ------------------------------------------------------------------
    # Deadline wrapper
    # ------------------------------------------------------------------

    async def _run_with_deadline(
        self,
        body: Callable[[], Awaitable[None]],
        context: ExecutionContext,
    ) -> None:
        if self.run_timeout_ms <= 0:
            await body()
            return

        deadline = asyncio.timeout(self.run_timeout_ms / 1000)
        try:
            async with deadline:
                await body()
        except TimeoutError:
            if deadline.expired():
                raise AgentTimeoutError(
                    f"agent exceeded {self.run_timeout_ms}ms run deadline",
                    {
                        "timeout_ms": self.run_timeout_ms,
                        "current_step": context.current_step,
                        "events_count": len(context.events),
                    },
                ) from None
            raise

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    async def _loop(self, context: ExecutionContext) -> None:
        while context.final_result is None and context.current_step < self.max_steps:
            await self.step(context)
            last_event = context.events[-1] if context.events else None
            if last_event is not None and self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)

    def _has_pending_tool_calls(self, event: AgentEvent) -> bool:
        has_calls = any(isinstance(item, ToolCall) for item in event.content)
        has_results = any(isinstance(item, ToolResult) for item in event.content)
        return has_calls and not has_results

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    async def step(self, context: ExecutionContext) -> None:
        with span(
            "agent.step",
            **{"step.number": context.current_step + 1},
        ) as h:
            request = self._prepare_llm_request(context)
            response = await self.think(request)

            if response.error_message is not None:
                raise LlmCallError(response.error_message, response.error_metadata)

            response_event = AgentEvent(
                execution_id=context.execution_id,
                author=self.name,
                content=list(response.content),
            )
            context.add_event(response_event)

            tool_calls = [c for c in response.content if isinstance(c, ToolCall)]
            if tool_calls:
                results = await self.act(context, tool_calls)
                tool_event = AgentEvent(
                    execution_id=context.execution_id,
                    author=self.name,
                    content=cast(list[ContentItem], results),
                )
                context.add_event(tool_event)

            h.set("agent.step.tool_calls", len(tool_calls))
            h.set("agent.step.has_message", any(isinstance(c, Message) for c in response.content))
            context.increment_step()

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------

    async def _check_permission(
        self,
        tool: BaseTool,
        tc: ToolCall,
    ) -> ToolResult | None:
        if tool.permission == "read":
            return None

        if tool.permission == "dangerous" and not self.allow_dangerous:
            return _denial_result(
                tc,
                f"Tool '{tool.name}' is marked dangerous; set "
                f"allow_dangerous=True on the Agent to enable it",
            )

        if self.confirm_tool_call is None:
            return _denial_result(
                tc,
                f"Tool '{tool.name}' (permission='{tool.permission}') "
                f"requires approval but no confirm_tool_call is configured "
                f"on the Agent",
            )

        try:
            raw = self.confirm_tool_call(tc)
            decision = await raw if inspect.isawaitable(raw) else raw
        except Exception as exc:
            return _denial_result(
                tc,
                f"Permission check for tool '{tool.name}' failed: {exc}",
            )

        if not decision:
            return _denial_result(
                tc,
                f"Tool '{tool.name}' was denied by the confirm_tool_call callback",
            )

        return None

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_one(
        self,
        tc: ToolCall,
        tool: BaseTool,
        context: ExecutionContext,
    ) -> ToolResult:
        with span(
            "tool.call",
            **{
                "tool.name": tc.name,
                "tool.call_id": tc.tool_call_id,
                "tool.permission": tool.permission,
            },
        ) as h:
            try:
                if tool.timeout_ms > 0:
                    async with asyncio.timeout(tool.timeout_ms / 1000):
                        output = await tool.call(tc.arguments, context)
                else:
                    output = await tool.call(tc.arguments, context)
                h.set("tool.status", "success")
                return ToolResult(
                    tool_call_id=tc.tool_call_id,
                    name=tc.name,
                    status="success",
                    content=[output],
                )
            except TimeoutError:
                h.fail(f"timeout after {tool.timeout_ms}ms")
                h.set("tool.status", "timeout")
                return ToolResult(
                    tool_call_id=tc.tool_call_id,
                    name=tc.name,
                    status="error",
                    content=[f"Tool '{tc.name}' timed out after {tool.timeout_ms}ms"],
                )
            except Exception as e:
                h.error(e)
                h.set("tool.status", "error")
                return ToolResult(
                    tool_call_id=tc.tool_call_id,
                    name=tc.name,
                    status="error",
                    content=[str(e)],
                )

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        tools_dict: dict[str, BaseTool] = {t.name: t for t in self.tools}

        # Phase 1: resolve + permission (sequential)
        slots: list[tuple[ToolCall, BaseTool] | ToolResult] = []
        for tc in tool_calls:
            tool = tools_dict.get(tc.name)
            if tool is None:
                slots.append(
                    ToolResult(
                        tool_call_id=tc.tool_call_id,
                        name=tc.name,
                        status="error",
                        content=[f"Tool '{tc.name}' not found"],
                    )
                )
                continue
            denial = await self._check_permission(tool, tc)
            if denial is not None:
                slots.append(denial)
                continue
            slots.append((tc, tool))

        # Phase 2: run
        executable = [s for s in slots if isinstance(s, tuple)]
        all_safe = all(_is_concurrency_safe(t) for _, t in executable) if executable else True

        if all_safe and len(executable) > 1:
            exec_results = await asyncio.gather(
                *(self._execute_one(tc, t, context) for tc, t in executable)
            )
            exec_iter = iter(exec_results)
        else:
            exec_iter = None

        # Phase 3: assemble in original order
        results: list[ToolResult] = []
        for slot in slots:
            if isinstance(slot, ToolResult):
                results.append(slot)
            elif exec_iter is not None:
                results.append(next(exec_iter))
            else:
                tc, tool = slot
                results.append(await self._execute_one(tc, tool, context))

        return results

    async def think(self, request: LlmRequest) -> LlmResponse:
        return await self.client.generate(request)

    # ------------------------------------------------------------------
    # Request building / response inspection
    # ------------------------------------------------------------------

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        flat: list[ContentItem] = []
        for event in context.events:
            flat.extend(event.content)

        tool_choice: ToolChoiceSetting | None = None
        if self.output_tool_name is not None:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"

        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    def _is_final_response(self, event: AgentEvent) -> bool:
        if self.output_tool_name is not None:
            return any(
                isinstance(item, ToolResult)
                and item.name == self.output_tool_name
                and item.status == "success"
                for item in event.content
            )

        has_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_calls and not has_results

    def _extract_final_result(
        self,
        event: AgentEvent,
    ) -> str | dict[str, object] | None:
        if self.output_tool_name is not None:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and len(item.content) > 0
                ):
                    return self._coerce_final_result(item.content[0])
            return None

        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _coerce_final_result(self, value: object) -> str | dict[str, object] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, BaseModel):
            return cast(dict[str, object], value.model_dump())
        if isinstance(value, dict):
            return cast(dict[str, object], value)
        return str(value)
