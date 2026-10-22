"""09 — Fallback chain.

`FallbackClient` wraps an ordered chain of clients. On a retryable failure
(quota, rate limit, 408, ≥500) it re-issues the same request to the next entry.
This demo uses a scripted "quota wall" as the primary so the rotation is
deterministic on every run; the backup is a real Gemini call.

    pixi run python examples/09_fallback.py
"""

from __future__ import annotations

import asyncio

from examples._common import build_llm_client, section
from pyagent.fallback import FallbackClient, FallbackEntry
from pyagent.llm import LlmRequest, LlmResponse
from pyagent.schemas import Message


class SimulatedQuotaClient:
    """Always reports a quota wall. Real quota failures come from the provider."""

    async def generate(self, request: LlmRequest) -> LlmResponse:
        _ = request
        return LlmResponse(
            error_message="simulated: free-tier quota exhausted",
            error_metadata={
                "quota_exhausted": True,
                "status_code": 429,
            },
        )


async def main() -> None:
    backup = build_llm_client(max_retries=1)
    chain = FallbackClient(
        chain=[
            FallbackEntry(label="simulated-quota", client=SimulatedQuotaClient()),
            FallbackEntry(label=backup.model, client=backup),
        ],
    )

    section("chain")
    for entry in chain.chain:
        print(f"  {entry.label}")

    section("run")
    response = await chain.generate(
        LlmRequest(
            instructions=['Say "fallback worked" and nothing else.'],
            contents=[Message(role="user", content="Go.")],
        )
    )

    if response.error_message is not None:
        print(f"error: {response.error_message}")
    else:
        for item in response.content:
            if isinstance(item, Message):
                print(f"text: {item.content}")

    section("metadata")
    for key, value in response.error_metadata.items():
        print(f"  {key}: {value!r}")


if __name__ == "__main__":
    asyncio.run(main())
