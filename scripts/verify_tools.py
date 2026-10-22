"""Phase 3: verify tool construction, calling, and schema shapes."""

from __future__ import annotations

import asyncio
import sys
from typing import cast

from pydantic import ValidationError

from pyagent.context import ExecutionContext
from pyagent.tools import (
    FunctionTool,
    ImagePart,
    MultipartContent,
    RemoteTool,
    StructuredPart,
    TextPart,
)
from pyagent.tools.add import add_tool
from pyagent.tools.calculator import calculator_tool
from pyagent.tools.greet import greet_tool


def check(cond: object, msg: str) -> None:
    if not cond:
        print(f"  \u2717 {msg}")
        sys.exit(1)
    print(f"  \u2713 {msg}")


def as_dict(v: object) -> dict[str, object]:
    if not isinstance(v, dict):
        raise AssertionError(f"expected dict, got {type(v).__name__}")
    return cast(dict[str, object], v)


def as_str(v: object) -> str:
    if not isinstance(v, str):
        raise AssertionError(f"expected str, got {type(v).__name__}")
    return v


async def main() -> None:
    print("\nTest 1: calculator tool_definition shape")
    defn = calculator_tool.tool_definition
    check(defn is not None, "tool_definition present")
    if defn is not None:
        check(defn.get("type") == "function", "outer type=function")
        fn = as_dict(defn["function"])
        check(as_str(fn["name"]) == "calculator", "name=calculator")
        params = as_dict(fn["parameters"])
        check(params.get("type") == "object", "parameters.type=object")
        props = as_dict(params["properties"])
        check("operator" in props, "properties has operator")
        check("first_number" in props, "properties has first_number")
        check("second_number" in props, "properties has second_number")
        check("$schema" not in params, "no $schema in parameters")
        check("title" not in params, "no title in parameters")

    print("\nTest 2: calculator add")
    r = await calculator_tool.call({"operator": "add", "first_number": 2, "second_number": 2})
    check(r == 4, f"2+2=4 (got {r})")

    print("\nTest 3: calculator multiply (floats)")
    r = await calculator_tool.call({"operator": "multiply", "first_number": 12, "second_number": 7})
    check(r == 84, f"12*7=84 (got {r})")

    print("\nTest 4: calculator divide-by-zero raises")
    try:
        await calculator_tool.call({"operator": "divide", "first_number": 1, "second_number": 0})
    except ValueError as e:
        check("divide by zero" in str(e).lower(), f"ValueError: {e}")
    else:
        check(False, "expected ValueError")

    print("\nTest 5: calculator invalid args raises ValidationError")
    try:
        await calculator_tool.call({"operator": "add"})
    except ValidationError:
        check(True, "ValidationError on missing fields")
    else:
        check(False, "expected ValidationError")

    print("\nTest 6: add tool")
    r = await add_tool.call({"a": 2, "b": 3})
    check(r == 5, f"2+3=5 (got {r})")

    print("\nTest 7: greet tool")
    r = await greet_tool.call({"name": "World"})
    check(r == "Hello, World.", f'greet -> "Hello, World." (got {r!r})')

    print("\nTest 8: async function tool")

    async def async_double(args: object, context: ExecutionContext | None = None) -> float:
        _ = context
        return 42.0

    from pydantic import BaseModel

    class DoubleInput(BaseModel):
        x: float

    async_tool = FunctionTool(
        name="double",
        description="doubles a number (fake)",
        schema=DoubleInput,
        func=async_double,  # type: ignore[arg-type]
    )
    r = await async_tool.call({"x": 21})
    check(r == 42.0, f"async func awaited (got {r})")

    print("\nTest 9: RemoteTool definition + call")
    calls: list[dict[str, object]] = []

    def fake_call(args: dict[str, object]) -> object:
        calls.append(args)
        return {"echo": args}

    remote = RemoteTool(
        name="echo",
        description="echo args back",
        input_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
        call=fake_call,
    )
    rd = remote.tool_definition
    check(rd is not None, "remote tool_definition present")
    if rd is not None:
        fn = as_dict(rd["function"])
        check(as_str(fn["name"]) == "echo", "remote name=echo")
        params = as_dict(fn["parameters"])
        check("$schema" not in params, "remote params have no $schema")

    r = await remote.call({"msg": "hi"})
    check(r == {"echo": {"msg": "hi"}}, f"remote delegates to call_fn (got {r})")
    check(len(calls) == 1, "call_fn invoked once")
    check(calls[0] == {"msg": "hi"}, "args passed through")

    print("\nTest 10: RemoteTool async call_fn")

    async def async_call(args: dict[str, object]) -> object:
        return {"async": args}

    remote_async = RemoteTool(
        name="aecho",
        description="async echo",
        input_schema={"type": "object"},
        call=async_call,
    )
    r = await remote_async.call({"x": 1})
    check(r == {"async": {"x": 1}}, f"async call_fn awaited (got {r})")

    print("\nTest 11: MultipartContent round-trip")
    mc = MultipartContent(
        parts=[
            TextPart(text="see this:"),
            ImagePart(data="aGVsbG8=", mime_type="image/png"),
            StructuredPart(value={"answer": 42}),
        ]
    )
    check(len(mc.parts) == 3, "3 parts")
    check(mc.parts[0].type == "text", "part[0] is text")
    check(mc.parts[1].type == "image", "part[1] is image")
    check(mc.parts[2].type == "structured", "part[2] is structured")

    dumped = mc.model_dump()
    restored = MultipartContent.model_validate(dumped)
    check(len(restored.parts) == 3, "3 parts after round-trip")
    check(restored.parts[0].type == "text", "text part survives")
    check(restored.parts[1].type == "image", "image part survives")
    check(restored.parts[2].type == "structured", "structured part survives")

    print("\nTest 12: BaseTool abstract enforcement")
    try:
        # BaseTool has abstractmethod call — cannot instantiate
        from pyagent.tools import BaseTool

        BaseTool()  # type: ignore[abstract]
    except TypeError as e:
        check("abstract" in str(e).lower(), f"BaseTool cannot be instantiated: {e}")
    else:
        check(False, "expected TypeError instantiating BaseTool")

    print("\n\u2705 Phase 3 verified.")


if __name__ == "__main__":
    asyncio.run(main())
