"""Phase 1: confirm package imports, deps resolve, env vars present."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check(cond: object, msg: str) -> None:
    if not cond:
        print(f"  \u2717 {msg}")
        sys.exit(1)
    print(f"  \u2713 {msg}")


def main() -> None:
    print("\nTest 1: package imports")
    import pyagent

    check(hasattr(pyagent, "__version__"), "pyagent has __version__")
    check(pyagent.__version__ == "0.1.0", f"version 0.1.0 (got {pyagent.__version__})")

    from pyagent import tools

    check(tools.__name__ == "pyagent.tools", "pyagent.tools is a package")

    print("\nTest 2: dependencies")
    import litellm
    import mcp
    import pydantic

    check(pydantic.VERSION.startswith("2."), f"pydantic v2 (got {pydantic.VERSION})")
    check(hasattr(litellm, "acompletion"), "litellm.acompletion")
    check(hasattr(mcp, "ClientSession"), "mcp.ClientSession")

    print("\nTest 3: sitecustomize + rich")
    import sitecustomize

    project_root = Path(__file__).resolve().parent.parent
    expected = (project_root / "sitecustomize.py").resolve()
    actual = Path(sitecustomize.__file__).resolve()
    check(actual == expected, f"sitecustomize from project root (got {actual})")

    print("\nTest 4: env vars (informational)")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        print("  \u2713 GEMINI_API_KEY set")
    else:
        print("  \u26a0 GEMINI_API_KEY not set (LLM tests will fail later)")

    print("\n\u2705 Phase 1 verified.")


if __name__ == "__main__":
    main()
