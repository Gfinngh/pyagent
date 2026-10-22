# Examples

Runnable, self-contained demonstrations of what `pyagent` can do. One
capability per file, human-readable output so you can see exactly what
happened.

These are **not** tests. For the verification suite see `scripts/verify_*.py`
and `pixi run python scripts/verify_<name>.py`.

## Prerequisites

```bash
pixi install
cp .env.example .env          # set GEMINI_API_KEY (and TAVILY_API_KEY)