"""Calculator tool — basic arithmetic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pyagent.context import ExecutionContext
from pyagent.tools.base import FunctionTool


class CalculatorInput(BaseModel):
    operator: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Arithmetic operation to perform"
    )
    first_number: float = Field(description="First operand")
    second_number: float = Field(description="Second operand")


def calculator(args: CalculatorInput, context: ExecutionContext | None = None) -> float:
    _ = context
    if args.operator == "add":
        return args.first_number + args.second_number
    if args.operator == "subtract":
        return args.first_number - args.second_number
    if args.operator == "multiply":
        return args.first_number * args.second_number
    # Remaining possibility: divide
    if args.second_number == 0:
        raise ValueError("Cannot divide by zero")
    return args.first_number / args.second_number


calculator_tool = FunctionTool(
    name="calculator",
    description="Perform basic arithmetic operations",
    schema=CalculatorInput,
    func=calculator,
)
