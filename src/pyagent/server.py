"""MCP server exposing calculator, greet, search_web, write_file, list_files,
read_file, execute_python over stdio.

Run with: ``python -m pyagent.server`` (or ``pixi run server``).

``execute_python`` runs untrusted, model-generated code in a Pyodide/WASM
sandbox via a Deno subprocess. The Deno process is launched with
``--cached-only`` and ``--allow-read=<npm-cache>`` only: no network, no env,
no subprocess, no FFI, no writes. Pyodide's Python has no syscalls — its
open()/socket() go through a virtual MEMFS that cannot reach the host — so
the executed code is isolated regardless of Deno's permission model. The
single read grant is consumed by Pyodide's JavaScript bootstrap, which needs
to load ``pyodide.asm.wasm`` and ``python_stdlib.zip`` from the npm cache.

Two-phase startup: on the first ``execute_python`` call, the server runs a
warm-up invocation with net + read + write on the npm cache so Pyodide's
runtime assets are present. The actual sandbox invocation never has net
access. If the warm-up fails (e.g., no network on first use), the tool
raises ``ToolError`` with a clear message; subsequent calls retry.

Docker is not required. Deno is required (``deno`` on PATH).

Standard-library imports live at module top; third-party imports sit inside
the stdout-redirect block so a missing dependency yields a clean ImportError.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Literal, cast

# Protect the JSON-RPC channel. Under stdio transport, stdout carries protocol
# frames only; anything printed at import time would corrupt the stream.
_real_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    import httpx
    from mcp.server import MCPServer
    from pydantic import Field

    _TOOLERROR_CANDIDATES: list[tuple[str, str]] = [
        ("mcp.server", "ToolError"),
        ("mcp.server.exceptions", "ToolError"),
        ("mcp.server.mcpserver.exceptions", "ToolError"),
        ("mcp.server.fastmcp.exceptions", "ToolError"),
        ("mcp.exceptions", "ToolError"),
        ("mcp.types", "ToolError"),
    ]
    ToolError: type[Exception]
    for _module_name, _attr in _TOOLERROR_CANDIDATES:
        try:
            _mod = importlib.import_module(_module_name)
            _candidate = getattr(_mod, _attr, None)
            if _candidate is not None and isinstance(_candidate, type):
                ToolError = _candidate
                break
        except ImportError:
            continue
    else:
        ToolError = Exception
finally:
    sys.stdout = _real_stdout

# Cache-location resolution is delegated to a small standalone module so
# it can be reused by verify_execute_python.py. See
# `_npm_cache.py` for the three-tier strategy (DENO_DIR → bare
# `deno info --json` → platform default).
from pyagent.tools._npm_cache import ensure_npm_cache

mcp = MCPServer("server")

# ---------------------------------------------------------------------------
# execute_python constants
# ---------------------------------------------------------------------------

SANDBOX_MAX_OUTPUT_BYTES = 16 * 1024
SANDBOX_DEFAULT_TIMEOUT_S = 10
SANDBOX_MAX_TIMEOUT_S = 30
SANDBOX_WARMUP_TIMEOUT_S = 180

# Memoized sandbox readiness: (ready, npm_cache_path). Populated on first
# execute_python call by _ensure_sandbox_ready.
_sandbox_ready: bool = False
_sandbox_npm_cache: str | None = None


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _workspace_root() -> Path:
    root = Path(os.environ.get("WORKSPACE_ROOT", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_in_workspace(path: str) -> Path:
    """Resolve ``path`` against WORKSPACE_ROOT, rejecting escapes."""
    root = _workspace_root()
    requested = Path(path)
    if requested.is_absolute():
        raise ToolError(f"path must be relative to the workspace; got absolute {path!r}")
    target = (root / requested).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"path {path!r} escapes the workspace root ({root})") from exc
    return target


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _sandbox_script() -> Path:
    """Path to scripts/sandbox.ts (project root, two levels above src/pyagent)."""
    return Path(__file__).resolve().parent.parent.parent / "scripts" / "sandbox.ts"


def sandbox_script_path() -> Path:
    """Public accessor for :func:`_sandbox_script`.

    Exists so verification scripts can assert that ``scripts/sandbox.ts``
    is present without reaching into a private helper.
    """
    return _sandbox_script()


def _deno_on_path() -> bool:
    return shutil.which("deno") is not None


def _run_deno_sync(args: list[str], *, stdin: bytes, timeout: float) -> tuple[int, bytes]:
    """Run a Deno subprocess synchronously. Returns (returncode, combined output)."""
    import subprocess

    try:
        proc = subprocess.run(
            ["deno", *args],
            input=stdin,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return -1, (exc.stdout or b"") + b"\n[timed out]"
    return proc.returncode, proc.stdout + proc.stderr


def _ensure_sandbox_ready() -> str:
    """Warm the Deno/npm cache and return the npm cache path.

    Three defects from the previous implementation are fixed here:

      1. ``deno install --entrypoint <script>`` no longer caches a local
         script's module graph in Deno 2 (``deno install`` is for CLI
         tools from jsr/npm). The correct primitive is ``deno run`` —
         it fetches and caches every transitive dependency before
         executing.

      2. ``deno info --json <script>`` returns a module dependency
         graph (``modules``, ``redirects``, ``roots``, ``totalModules``)
         and never includes ``npmCache``. Only **bare**
         ``deno info --json`` (no module argument) reports it. That
         mismatch was the direct cause of every ``execute_python`` call
         failing with "deno info did not report an npmCache path".
         Cache resolution is delegated to
         :func:`pyagent.tools._npm_cache.ensure_npm_cache`, which
         handles ``DENO_DIR``, bare ``deno info``, and the platform
         default.

      3. The warm-up ran with ``stdin=b""``. Empty input is not valid
         Python and does not exercise ``loadPyodide()``, so Pyodide's
         WASM assets were never cached. Sending ``b"pass"`` is
         strictly safer: it works whether ``sandbox.ts`` loads Pyodide
         eagerly or lazily.

    Raises ``ToolError`` with a specific message if any phase fails.
    Memoized — repeated calls return the cached path.
    """
    global _sandbox_ready, _sandbox_npm_cache

    if _sandbox_ready and _sandbox_npm_cache is not None:
        return _sandbox_npm_cache

    if not _deno_on_path():
        raise ToolError(
            "Deno is not installed. Install it with: curl -fsSL https://deno.land/install.sh | sh"
        )

    script = _sandbox_script()
    if not script.exists():
        raise ToolError(f"sandbox script not found: {script}")

    # Phase 1: resolve the npm cache path. This is the whole fix; the
    # rest of the function is the warm-up run.
    try:
        npm_cache_path = ensure_npm_cache()
    except Exception as exc:
        raise ToolError(f"failed to resolve npm cache: {exc}") from exc
    npm_cache = str(npm_cache_path)

    # Phase 2: warm-up run (net + read + write on the npm cache) so
    # Pyodide's WASM and stdlib assets land in the cache. Sending
    # b"pass" as stdin exercises loadPyodide() without doing anything
    # observable.
    warm_rc, warm_out = _run_deno_sync(
        [
            "run",
            "--allow-read",
            "--allow-net",
            f"--allow-write={npm_cache}",
            str(script),
        ],
        stdin=b"pass",
        timeout=SANDBOX_WARMUP_TIMEOUT_S,
    )
    if warm_rc != 0:
        raise ToolError(
            "Pyodide warm-up failed. First use requires network access. "
            f"Output: {warm_out.decode('utf-8', errors='replace')[:400]}"
        )

    _sandbox_ready = True
    _sandbox_npm_cache = npm_cache
    return npm_cache


def sandbox_is_ready() -> str:
    """Public accessor for :func:`_ensure_sandbox_ready`.

    Runs the warm-up if needed and returns the npm cache path. Raises
    ``ToolError`` on any warm-up failure; callers that want to skip
    cleanly should catch it.
    """
    return _ensure_sandbox_ready()


def _run_sandbox_sync(code: str, timeout_s: int) -> str:
    """Run ``code`` in the sandbox. Synchronous — call from asyncio.to_thread."""
    npm_cache = _ensure_sandbox_ready()

    rc, out = _run_deno_sync(
        [
            "run",
            "--cached-only",
            f"--allow-read={npm_cache}",
            str(_sandbox_script()),
        ],
        stdin=code.encode("utf-8"),
        timeout=timeout_s + 5,
    )
    text = out.decode("utf-8", errors="replace")
    if len(text.encode("utf-8")) > SANDBOX_MAX_OUTPUT_BYTES:
        text = (
            text.encode("utf-8")[:SANDBOX_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            + f"\n<truncated at {SANDBOX_MAX_OUTPUT_BYTES} bytes>"
        )
    if rc == -1:
        text += f"\n[timed out after {timeout_s}s]"
    return text if text else "<no output>"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def calculator(
    operator: Annotated[
        Literal["add", "subtract", "multiply", "divide"],
        Field(description="Arithmetic operation to perform"),
    ],
    first_number: Annotated[float, Field(description="First operand")],
    second_number: Annotated[float, Field(description="Second operand")],
) -> float:
    """Perform basic arithmetic operations"""
    if operator == "add":
        return first_number + second_number
    if operator == "subtract":
        return first_number - second_number
    if operator == "multiply":
        return first_number * second_number
    if second_number == 0:
        raise ToolError("Cannot divide by zero")
    return first_number / second_number


@mcp.tool()
def greet(
    name: Annotated[str, Field(description="person's name")],
) -> str:
    """greet a person"""
    return f"Hello, {name}."


@mcp.tool()
async def search_web(
    query: Annotated[str, Field(description="Search query string")],
    max_results: Annotated[
        int,
        Field(
            description=(
                "Maximum number of results to return. Default 5, sized for "
                "research synthesis; pass 1 for a single-source lookup."
            )
        ),
    ] = 5,
) -> str:
    """Search the web using the Tavily API."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ToolError("TAVILY_API_KEY not set - cannot search the web")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                },
            )
    except httpx.HTTPError as exc:
        raise ToolError(f"Tavily request failed: {exc}") from exc

    if resp.status_code != 200:
        raise ToolError(f"Tavily error: {resp.status_code} {resp.text[:300]}")

    raw: object = resp.json()
    if not isinstance(raw, dict):
        return "No results found."
    data = cast(dict[str, object], raw)

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return "No results found."
    results = cast(list[object], raw_results)
    if not results:
        return "No results found."

    parts: list[str] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        r = cast(dict[str, object], entry)
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"Title: {title}\nURL: {url}\nContent: {content}")
    return "\n\n".join(parts)


@mcp.tool()
async def list_files(
    pattern: Annotated[
        str,
        Field(
            description=(
                "Glob pattern to match, e.g. '*.md' or 'notes/*.md'. "
                "Empty string matches everything under the workspace."
            )
        ),
    ] = "",
    max_files: Annotated[int, Field(description="Maximum files to list (default 20)")] = 20,
) -> str:
    """List files in the sandboxed workspace that match a glob pattern."""
    root = _workspace_root()
    glob = pattern or "*"

    matches: list[Path] = []
    cap = max_files * 5
    try:
        for p in root.rglob(glob):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            matches.append(p)
            if len(matches) >= cap:
                break
    except OSError as exc:
        raise ToolError(f"list_files failed: {exc}") from exc

    matches.sort(key=lambda p: p.relative_to(root).as_posix())
    matches = matches[:max_files]

    if not matches:
        return f"No files matching {glob!r} under workspace."

    lines = [f"{len(matches)} file(s) matching {glob!r}:"]
    for p in matches:
        rel = p.relative_to(root).as_posix()
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        lines.append(f"  {rel} ({size} bytes)")
    return "\n".join(lines)


@mcp.tool()
async def read_file(
    path: Annotated[str, Field(description="Relative path inside the workspace")],
    max_bytes: Annotated[int, Field(description="Maximum bytes to read (default 8000)")] = 8000,
) -> str:
    """Read a file's text content from the sandboxed workspace."""
    target = _resolve_in_workspace(path)
    if not target.is_file():
        raise ToolError(f"file not found: {path}")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ToolError(f"stat failed for {path}: {exc}") from exc

    try:
        with target.open("rb") as f:
            data = await asyncio.to_thread(f.read, max_bytes)
    except OSError as exc:
        raise ToolError(f"read failed for {path}: {exc}") from exc

    truncated = size > len(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(
            f"file {path!r} is not valid UTF-8 text; binary files are not supported by read_file"
        ) from exc

    header_parts = [f"[{path}]", f"({size} bytes"]
    if truncated:
        header_parts.append(f"truncated to {max_bytes}")
    header = " ".join(header_parts) + ")\n"
    return header + text


@mcp.tool()
async def write_file(
    path: Annotated[
        str,
        Field(
            description=(
                "Relative path inside the workspace. "
                "Absolute paths and '..' traversal are rejected."
            )
        ),
    ],
    content: Annotated[str, Field(description="Text content to write")],
) -> str:
    """Write text to a file inside the sandboxed workspace."""
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(target.write_text, content, encoding="utf-8")
    rel = target.relative_to(_workspace_root()).as_posix()
    return f"wrote {len(content)} chars to {rel}"


@mcp.tool()
async def execute_python(
    code: Annotated[str, Field(description="Python source code to execute")],
    timeout_s: Annotated[
        int,
        Field(
            description=(
                f"Wall-clock timeout in seconds (default "
                f"{SANDBOX_DEFAULT_TIMEOUT_S}, max {SANDBOX_MAX_TIMEOUT_S})"
            )
        ),
    ] = SANDBOX_DEFAULT_TIMEOUT_S,
) -> str:
    """Execute Python in a Pyodide/WASM sandbox.

    The Python code runs inside WebAssembly with no syscalls — no network,
    no host filesystem, no subprocesses. Deno launches with ``--cached-only``
    and ``--allow-read`` on the npm cache only; everything else is denied.

    Requires Deno on PATH. The first call warms the Deno/npm cache (needs
    network once). After that, calls are fully offline.
    """
    if not code.strip():
        raise ToolError("code must not be empty")

    effective_timeout = max(1, min(timeout_s, SANDBOX_MAX_TIMEOUT_S))
    try:
        return await asyncio.to_thread(_run_sandbox_sync, code, effective_timeout)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"execute_python failed: {exc}") from exc


if __name__ == "__main__":
    mcp.run(transport="stdio")
