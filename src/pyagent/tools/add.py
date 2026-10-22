"""Add tool — sum two numbers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from pyagent.context import ExecutionContext
from pyagent.tools.base import FunctionTool


class AddInput(BaseModel):
    a: float = Field(description="First operand")
    b: float = Field(description="Second operand")


def add(args: AddInput, context: ExecutionContext | None = None) -> float:
    _ = context
    return args.a + args.b


add_tool = FunctionTool(
    name="add",
    description="Add two numbers",
    schema=AddInput,
    func=add,
)
