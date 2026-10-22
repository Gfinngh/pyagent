"""15 — Python sandbox.

`execute_python` runs user Python inside Pyodide/WASM via Deno with no network,
no host filesystem, no subprocesses, and no environment. Requires Deno on PATH;
skips cleanly otherwise. The first call warms the cache (needs network once).

    pixi run python examples/15_execute_python.py
"""

from __future__ import annotations

import asyncio
import shutil

from examples._common import section
from pyagent.server import execute_python


async def main() -> None:
    if shutil.which("deno") is None:
        section("skipped")
        print("Deno is not on PATH. Install it with:")
        print("  curl -fsSL https://deno.land/install.sh | sh")
        return

    print("first call warms the Deno/npm cache (may take 30–60 s).")

    section("print(1 + 1)")
    print(f"  → {await execute_python('print(1 + 1)')}")

    section("multi-line")
    code = "for i in range(3):\n    print(i)\n"
    print(await execute_python(code))

    section("uncaught exception")
    print((await execute_python('raise ValueError("boom")'))[:300])

    section("network is blocked")
    print(
        await execute_python(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
            "    print('CONNECTED')\n"
            "except OSError as e:\n"
            "    print(f'blocked: {e}')\n"
        )
    )

    section("filesystem is blocked")
    print(
        await execute_python(
            "try:\n"
            "    open('/etc/passwd').read()\n"
            "    print('READABLE')\n"
            "except OSError as e:\n"
            "    print(f'blocked: {e}')\n"
        )
    )

    section("environment is blocked")
    print(await execute_python("import os\nprint(sorted(os.environ))"))

    section("notes")
    print("timeout_s is clamped to [1, 30]; output truncates at 16 KiB.")
    print("The tool is permission='dangerous' — two locks required in an agent.")


if __name__ == "__main__":
    asyncio.run(main())
