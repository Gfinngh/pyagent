"""06 — Streaming.

`LlmClient(mode="stream", on_text_delta=...)` fires each chunk as it arrives and
also returns the complete text on the response. This demo drives the client
directly — the `Agent` layer uses non-streaming generation.

    pixi run python examples/06_streaming.py
"""

from __future__ import annotations

import asyncio

from examples._common import build_llm_client, section
from pyagent.llm import LlmRequest
from pyagent.schemas import Message


async def main() -> None:
    chunks: list[str] = []

    def on_delta(delta: str) -> None:
        chunks.append(delta)
        print(delta, end="")

    client = build_llm_client(mode="stream", on_text_delta=on_delta)

    section("stream")
    response = await client.generate(
        LlmRequest(
            instructions=["Write a haiku about the sea. Three short lines."],
            contents=[Message(role="user", content="Go.")],
        )
    )
    print()  # newline after the streamed text

    section("summary")
    print(f"deltas received: {len(chunks)}")
    print(f"error:           {response.error_message}")
    joined = "".join(item.content for item in response.content if isinstance(item, Message))
    print(f"final text:      {joined!r}")


if __name__ == "__main__":
    asyncio.run(main())
