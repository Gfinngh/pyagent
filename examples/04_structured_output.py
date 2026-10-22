"""04 — Structured output.

Constrain the model to return a JSON object matching a pydantic model instead of
free-form text. The agent synthesises a `final_answer` tool whose schema wraps
your model under `output`, and the result is `result.output` as a dict.

    pixi run python examples/04_structured_output.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from examples._common import build_llm_client, print_final, section
from pyagent.agent import Agent


class CityFacts(BaseModel):
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Approximate population")
    notable_landmark: str = Field(description="One well-known landmark")


async def main() -> None:
    agent = Agent(
        client=build_llm_client(),
        output_type=CityFacts,
        max_steps=4,
    )

    section("run")
    result = await agent.run("Give me basic facts about Tokyo as a structured object.")

    section("final output")
    print_final(result)

    if isinstance(result.output, dict):
        parsed = CityFacts.model_validate(result.output)
        print("\nvalidated against CityFacts:")
        print(parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
