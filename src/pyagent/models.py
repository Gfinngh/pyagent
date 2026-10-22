"""Model string resolution.

LiteLLM accepts LiteLLM-style model strings directly (e.g. "gemini/gemini-3-flash-preview"),
so resolution here is a validated pass-through. The prefix check catches typos early.

Divergence from TypeScript: ``resolveModel`` in TS stripped the ``gemini/`` prefix
and rebuilt an AI SDK ``LanguageModel``. In Python, LiteLLM routes on the prefix
itself, so the string passes through unchanged.
"""

from __future__ import annotations

_SUPPORTED_PREFIXES: tuple[str, ...] = ("gemini/",)


def resolve_model(model_id: str) -> str:
    """Validate a LiteLLM-style model id and return it unchanged.

    Raises ``ValueError`` if the prefix is not recognized — early failure is
    better than an opaque 404 from the provider.
    """
    for prefix in _SUPPORTED_PREFIXES:
        if model_id.startswith(prefix):
            return model_id
    raise ValueError(
        f"unsupported model prefix in {model_id!r}. Expected one of: {_SUPPORTED_PREFIXES}"
    )
