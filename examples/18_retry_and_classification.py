"""18 — Error classification and backoff.

`classify_error` maps a provider exception to a structured diagnosis;
`backoff_ms` produces the exponential delay with jitter. `default_fallback_policy`
decides whether a failure should advance the fallback chain.

This example runs entirely offline: no provider is contacted.

    pixi run python examples/18_retry_and_classification.py
"""

from __future__ import annotations

from examples._common import section
from pyagent.fallback import default_fallback_policy
from pyagent.llm import LlmResponse, backoff_ms


def response_with(meta: dict[str, object]) -> LlmResponse:
    return LlmResponse(error_message="x", error_metadata=meta)


def main() -> None:
    section("1. backoff_ms — exponential with jitter, capped")
    for attempt in range(1, 7):
        print(f"  attempt {attempt}: {backoff_ms(attempt)} ms")

    section("2. default_fallback_policy — which failures advance the chain")
    cases: list[tuple[str, dict[str, object]]] = [
        ("quota exhausted", {"quota_exhausted": True}),
        ("rate limited", {"rate_limited": True}),
        ("timeout 408", {"status_code": 408}),
        ("server 500", {"status_code": 500}),
        ("server 503", {"status_code": 503}),
        ("auth 401", {"status_code": 401}),
        ("auth 403", {"status_code": 403}),
        ("bad request 400", {"status_code": 400}),
        ("not found 404", {"status_code": 404}),
        ("aborted", {"aborted": True}),
        ("unknown", {}),
    ]
    for label, meta in cases:
        advance = default_fallback_policy(response_with(meta))
        print(f"  {label:<18} → {'fall back' if advance else 'terminal'}")

    section("3. success short-circuits")
    ok = LlmResponse(content=[])
    print(f"  no error → fall back: {default_fallback_policy(ok)}")

    section("notes")
    print("The retry loop inside LlmClient uses classify_error; the fallback policy")
    print("is a separate decision made one layer up by FallbackClient.")


if __name__ == "__main__":
    main()
