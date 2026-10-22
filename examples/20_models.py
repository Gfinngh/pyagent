"""20 — Model id resolution.

LiteLLM routes on the provider prefix, so `resolve_model` validates the prefix
and returns the id **unchanged**. (The TypeScript port strips `gemini/` because
the AI SDK does not accept it — different provider, different rule.)

    pixi run python examples/20_models.py
"""

from __future__ import annotations

import os

from examples._common import section
from pyagent.models import resolve_model

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"


def main() -> None:
    section("1. accepted — provider prefix present")
    for model in ("gemini/gemini-3.5-flash-lite", "gemini/gemini-2.5-pro"):
        print(f"  {model:<32} → {resolve_model(model)}")

    section("2. rejected — prefix missing or unknown")
    for model in ("gpt-4", "claude-3", "gemini-3.5-flash-lite", ""):
        try:
            resolve_model(model)
            print(f"  {model!r:<32} → UNEXPECTEDLY ACCEPTED")
        except ValueError as exc:
            print(f"  {model!r:<32} → rejected: {exc}")

    section("3. what the examples use")
    print(f"  PYAGENT_MODEL: {os.environ.get('PYAGENT_MODEL', '(unset)')}")
    print(f"  fallback:      {DEFAULT_MODEL}")

    section("notes")
    print("Set PYAGENT_MODEL in .env to change the default for every example")
    print("and for `pixi run host`.")


if __name__ == "__main__":
    main()
