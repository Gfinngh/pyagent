"""19 — Gemini thought signatures.

Gemini 3 attaches a signature to every function call that must be echoed back on
the next turn. LiteLLM exposes it through two channels:

  1. ``tool_calls[].provider_specific_fields.thought_signature`` — a flat dict
  2. ``tool_calls[].id`` with ``__thought__<sig>`` appended

`provider_options.py` provides helpers for both, plus detection of LiteLLM's
``skip_thought_signature_validator`` sentinel — accepting that sentinel as proof
of a round-trip is how silent degradation hides.

    pixi run python examples/19_provider_signatures.py
"""

from __future__ import annotations

import base64

from examples._common import section
from pyagent.provider_options import (
    is_dummy_signature,
    signature_from_provider_fields,
    signature_from_tool_call,
    split_tool_call_id,
    unwrap_legacy_provider_fields,
)


class _FakeToolCall:
    def __init__(self, id_: str, psf: dict[str, object] | None = None) -> None:
        self.id = id_
        self.provider_specific_fields = psf


def main() -> None:
    section("1. split_tool_call_id")
    for raw in ("call_abc", "call_abc__thought__sig_xyz", "call_abc__thought__"):
        print(f"  {raw!r:<38} → {split_tool_call_id(raw)}")

    section("2. signature_from_provider_fields")
    print(f"  flat dict:  {signature_from_provider_fields({'thought_signature': 'sig'})!r}")
    print(f"  absent:     {signature_from_provider_fields({})!r}")
    print(f"  malformed:  {signature_from_provider_fields('nope')!r}")

    section("3. signature_from_tool_call — either channel")
    a = _FakeToolCall("call_1", {"thought_signature": "from-fields"})
    b = _FakeToolCall("call_2__thought__from-id")
    c = _FakeToolCall("call_3")
    for label, tc in (("fields", a), ("id suffix", b), ("neither", c)):
        print(f"  {label:<10} → {signature_from_tool_call(tc)}")

    section("4. is_dummy_signature")
    dummy = base64.b64encode(b"skip_thought_signature_validator").decode("ascii")
    for label, value in (
        ("real signature", "cmVhbA=="),
        ("dummy sentinel", dummy),
        ("empty", ""),
        ("none", None),
    ):
        print(f"  {label:<16} → {is_dummy_signature(value)}")

    section("5. unwrap_legacy_provider_fields")
    print(f"  flat:   {unwrap_legacy_provider_fields({'thought_signature': 'x'})}")
    print(f"  legacy: {unwrap_legacy_provider_fields({'google': {'thought_signature': 'x'}})}")

    section("notes")
    print("You do not call these directly; llm.py and fallback.py use them to")
    print("round-trip the signature across turns and to strip it on chain switch.")


if __name__ == "__main__":
    main()
