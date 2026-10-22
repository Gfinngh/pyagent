"""Probe all configured Gemini models for current availability.

One short prompt per model, no retries, per-model timeout reported. Used to
distinguish three failure modes that look alike:

  - Provider-wide outage (all models timeout) -> API key or account issue,
    or a genuine provider incident.
  - Model-level capacity (some models work, some timeout) -> pick working ones.
  - Timeout too tight (all timeout at 30s, succeed at 60s) -> raise the
    workflow timeout during peak load.

Run::

    pixi run python scripts/probe_models.py
    pixi run python scripts/probe_models.py --timeout 60
    pixi run python scripts/probe_models.py --models gemini/gemini-3.8-flash

Exit code is 0 if at least one model succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from pyagent.llm import LlmClient, LlmRequest
from pyagent.schemas import Message

MODELS: list[str] = [
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3-flash-preview",
]


async def _probe(model: str, timeout_ms: int) -> tuple[str, str, float]:
    """Return (model, status_label, latency_seconds)."""
    client = LlmClient(
        model=model,
        max_retries=0,
        timeout_ms=timeout_ms,
    )
    req = LlmRequest(
        instructions=["Reply with the single word: ok"],
        contents=[Message(role="user", content="ping")],
    )

    start = time.monotonic()
    resp = await client.generate(req)
    elapsed = time.monotonic() - start

    if resp.error_message is None:
        return model, "ok", elapsed

    meta = resp.error_metadata
    status_code = meta.get("status_code")
    error_type = str(meta.get("error_type", "unknown"))
    msg_lower = (resp.error_message or "").lower()

    if meta.get("quota_exhausted") is True:
        return model, "quota", elapsed
    if status_code == 408 or "timeout" in msg_lower or "timed out" in msg_lower:
        return model, f"timeout ({error_type})", elapsed
    if status_code is not None:
        return model, f"error {status_code} ({error_type})", elapsed
    return model, f"error ({error_type})", elapsed


async def main(models: list[str], timeout_ms: int) -> int:
    print(f"Probing {len(models)} models, {timeout_ms}ms timeout each\n")
    print(f"{'Model':40s}  {'Status':30s}  {'Latency':>8s}")
    print("-" * 84)

    results: list[tuple[str, str, float]] = []
    for m in models:
        try:
            results.append(await _probe(m, timeout_ms))
        except Exception as exc:
            results.append((m, f"raised: {type(exc).__name__}", 0.0))

    for m, status, latency in results:
        print(f"{m:40s}  {status:30s}  {latency:>7.1f}s")

    ok = [m for m, s, _ in results if s == "ok"]
    timeouts = [m for m, s, _ in results if s.startswith("timeout")]
    errors = [(m, s) for m, s, _ in results if s not in ("ok",) and not s.startswith("timeout")]

    print()
    print(f"Summary: {len(ok)} ok, {len(timeouts)} timeout, {len(errors)} error")
    if ok:
        print(f"  working:  {ok}")
    if timeouts:
        print(f"  timeouts: {timeouts}")
    if errors:
        print(f"  errors:   {errors}")

    return 0 if ok else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Gemini models.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30_000,
        help="Per-model timeout in ms (default 30000)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to probe (default: all configured)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args.models or MODELS, args.timeout)))
