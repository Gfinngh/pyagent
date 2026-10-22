"""Greet tool — say hello to someone."""

from __future__ import annotations

from pydantic import BaseModel, Field

from pyagent.context import ExecutionContext
from pyagent.tools.base import FunctionTool


class GreetInput(BaseModel):
    name: str = Field(description="person's name")


def greet(args: GreetInput, context: ExecutionContext | None = None) -> str:
    _ = context
    return f"Hello, {args.name}."


greet_tool = FunctionTool(
    name="greet",
    description="greet a person",
    schema=GreetInput,
    func=greet,
)
