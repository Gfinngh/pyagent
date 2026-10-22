"""Resolve the Deno npm cache directory used by `execute_python`.

Why this exists: the previous code called `deno info --json <module>`,
which returns a *module dependency graph* (keys: `modules`, `redirects`,
`roots`, `totalModules`) and never includes `npmCache`. The
environment/config output that carries `npmCache` is only produced by
the **bare** invocation `deno info --json` with no module argument.
That mismatch is what caused `execute_python` to skip forever with an
`npmCache missing` error.

This module resolves the cache robustly across three cases:

    1. `DENO_DIR` is set → `<DENO_DIR>/npm`. This is Deno's own
       documented convention, and it is what the warm-up and the
       sandbox `--allow-read` / `--allow-write` flags must agree on.
    2. `deno` is on PATH and bare `deno info --json` succeeds → use
       the `npmCache` field, falling back to `denoDir + "/npm"` if
       that field is absent (older Deno versions omit it).
    3. Deno cannot be queried → use the platform default:
           Linux:   ~/.cache/deno/npm
           macOS:   ~/Library/Caches/deno/npm
           Windows: %LOCALAPPDATA%\\deno\\npm

The result is a `pathlib.Path` that callers can pass directly to
`--allow-read=` / `--allow-write=` in a Deno subprocess invocation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def resolve_npm_cache() -> Path:
    """Return the Deno npm cache directory.

    Never raises. Returns the best available guess when Deno cannot be
    queried; callers are expected to `mkdir(parents=True, exist_ok=True)`
    the parent before using the path in a Deno command.
    """
    # 1. Explicit env var wins. This is what the operator sees, and it
    #    is what the warm-up must populate for `--cached-only` to work.
    deno_dir = os.environ.get("DENO_DIR")
    if deno_dir:
        return Path(deno_dir) / "npm"

    # 2. Bare `deno info --json` — the one that actually has npmCache.
    try:
        result = subprocess.run(
            ["deno", "info", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            info = json.loads(result.stdout)
            npm_cache = info.get("npmCache")
            if isinstance(npm_cache, str) and npm_cache:
                return Path(npm_cache)
            deno_dir_field = info.get("denoDir")
            if isinstance(deno_dir_field, str) and deno_dir_field:
                return Path(deno_dir_field) / "npm"
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        OSError,
    ):
        pass

    # 3. Platform default. Matches Deno's own installation layout.
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "deno" / "npm"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "deno" / "npm"
    return Path.home() / ".cache" / "deno" / "npm"


def ensure_npm_cache() -> Path:
    """Resolve the npm cache path and ensure its parent directory exists.

    Call this before invoking Deno with `--allow-read=<cache>` or
    `--allow-write=<cache>`. Creating the parent (not the cache itself)
    keeps Deno free to populate the directory on first use.
    """
    cache = resolve_npm_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    return cache
