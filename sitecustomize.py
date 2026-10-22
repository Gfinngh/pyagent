"""Project-local rich setup. Loaded automatically by Python's site.py via
PYTHONPATH (set in pyproject.toml under [tool.pixi.activation.env]).

Same behavior as the user-site usercustomize.py, but scoped to this project
and Pixi environment. Safe no-op if rich isn't importable.

IMPORTANT: tracebacks go to stderr, never stdout. MCP stdio servers reserve
stdout exclusively for JSON-RPC frames; a traceback printed to stdout
corrupts the protocol stream and the client fails to parse it.

``force_terminal=None`` (the default) instead of ``True``: Rich then respects
``NO_COLOR`` and non-TTY streams. The Pixi env sets ``NO_COLOR=1``, so captured
subprocess output and piped output is plain text - essential for the verify
scripts that compare stdout to log content. Interactive terminals still get
colors because Rich detects the TTY.
"""

try:
    import builtins
    import sys

    from rich import inspect as rich_inspect
    from rich import pretty, traceback
    from rich.console import Console

    # Human-facing output: stdout. ``force_terminal=None`` lets Rich auto-detect.
    _stdout_console = Console(
        color_system="truecolor",
        width=150,
        soft_wrap=False,
    )

    # Diagnostics: stderr. Also auto-detects.
    _stderr_console = Console(
        color_system="truecolor",
        width=150,
        soft_wrap=False,
        stderr=True,
    )

    pretty.install(console=_stdout_console)
    traceback.install(console=_stderr_console, show_locals=False)

    def _inspect(obj: object, **kwargs: object) -> object:
        defaults: dict[str, object] = {"methods": True, "private": True}
        defaults.update(kwargs)
        return rich_inspect(obj, **defaults)  # type: ignore[arg-type]

    _orig_print = builtins.print

    def _print(*args: object, **kwargs: object) -> None:
        if kwargs.get("file") not in (None, sys.stdout):
            return _orig_print(*args, **kwargs)  # type: ignore[arg-type]
        kwargs.pop("file", None)
        return _stdout_console.print(*args, **kwargs)

    builtins.print = _print
    builtins.inspect = _inspect  # type: ignore[assignment]
except ImportError:
    pass
