# AGENTS.md

Orientation for an LLM reading this repository to generate study material,
handbook chapters, or guided walkthroughs.

## What this project is

A Python ReAct agent kernel over **LiteLLM + pydantic + MCP**. Five
primitives form the kernel, all under `src/pyagent/`:

1. **Event-sourced conversation** — `schemas.py`, `context.py`,
   `event_log.py`. `ExecutionContext.events` *is* the conversation:
   flattening each event's content produces the flat conversation sent to
   the LLM. `current_step` and `final_result` are **derived** from the
   events, never stored separately. Each `AgentEvent` carries `id`,
   `execution_id`, `timestamp`, `author`, and `content`.

2. **Tool abstraction** — `tools/base.py`. `BaseTool` is abstract;
   `FunctionTool` wraps a pydantic `BaseModel` schema plus a function;
   `RemoteTool` wraps an MCP-discovered tool. Three permission tiers:
   `read` (no callback), `mutate` (requires `confirm_tool_call`),
   `dangerous` (requires `allow_dangerous=True` **and** a callback
   returning `True`). Concurrency safety defaults to
   `permission == "read"`.

3. **LLM client** — `llm.py`, `provider_options.py`. LiteLLM wrapper
   owning retries, error classification (`classify_error`, `backoff_ms`),
   deadlines, streaming (`mode="stream"` + `on_text_delta`), prompt
   building, and the Gemini thought-signature round-trip.
   `LlmRequest.contents` is a `list[ContentItem]`; `LlmResponse.content`
   likewise.

4. **Agent loop** — `agent.py`. `Agent.run(user_input, context=None)` and
   `Agent.resume(context, user_input=None)`, both returning
   `AgentResult(output, context)`. Defaults: `max_steps=4`,
   `name="agent"`, `run_timeout_ms=0` (no deadline). Structured output is
   injected as a synthetic `final_answer` tool built from `output_type`.

5. **MCP boundary** — `client.py`, `server.py`. `McpConnection` owns the
   stdio subprocess through an `AsyncExitStack`; `load_mcp_tools(conn,
   overrides=…)` wraps each tool as a `RemoteTool`, applying
   `ToolOverrides` by tool name. `server.py` exposes seven tools,
   including the Deno/Pyodide sandbox (`execute_python`).

Everything else — `memory.py`, `tracing.py`, `fallback.py`, `models.py`,
`workflows/` — builds on those five.

## Repository layout

| Path | Contents |
|---|---|
| `src/pyagent/` | Kernel modules: `agent`, `client`, `context`, `event_log`, `fallback`, `llm`, `memory`, `models`, `provider_options`, `schemas`, `server`, `tracing` |
| `src/pyagent/tools/` | `base`, `add`, `calculator`, `greet`, `_npm_cache` |
| `src/pyagent/workflows/` | `research_brief`, `iterative_brief`, `notes_digest`, `trip_planner` |
| `scripts/` | `verify_*.py` (executable specification), `sandbox.ts` (Deno/Pyodide host), `probe_models.py`, `_verify_common.py` |
| `examples/` | 25 runnable demos, `_common.py`, `_fakes.py`, `README.md` |
| `main.py` | Host CLI |
| `README.md`, `AGENTS.md`, `LICENSE`, `.gitignore`, `.env.example` | Project metadata |

## Where to start reading

- **`examples/README.md`** — coverage map: every core part of the project
  mapped to the example(s) that demonstrate it. The natural chapter list
  for a handbook.
- **Real-LLM examples** — `01`–`06` and `09`. Use for "what an agent run
  looks like end-to-end". They consume quota.
- **Deterministic examples** — `07`, `08`, and `10`–`25`. Scripted LLM
  client, local SQLite, or pure local primitives. Identical output on
  every run, no quota. Best for study.
- **`scripts/verify_*.py`** — executable specifications. Each `check(…)`
  call is part of the behavioural contract. Reading one `verify_*.py`
  alongside its source module is the fastest path to understanding a
  primitive.

## Invariants that must not be broken

These are load-bearing. Every verification script asserts at least one.

- **Two-lock permission model.** `read` always runs. `mutate` requires a
  `confirm_tool_call` callback. `dangerous` requires both
  `allow_dangerous=True` **and** a callback returning true. The callback
  is not consulted when `allow_dangerous` is false.
- **Denied calls become error `ToolResult`s**, never exceptions. The
  denied tool is never invoked.
- **`ExecutionContext.events` is the conversation.** Derive, don't cache.
- **`current_step` counts events carrying an assistant message or a tool
  call.** The follow-up tool-result event belongs to the same step and is
  not counted again.
- **`from_log(path, output_tool_name=…)`.** Without `output_tool_name`, a
  completed structured run looks unfinished — this is deliberate and is
  what makes `Agent.resume` detect interruption correctly.
- **`resolve_model` validates but does not rewrite.** LiteLLM routes on
  the `gemini/` provider prefix, so the id passes through unchanged. (The
  TypeScript port strips the prefix because the AI SDK does not accept
  it — different provider, different rule.)
- **MCP tool permissions default to `read`.** The Python server does not
  advertise `_meta.permission`; callers pass `ToolOverrides` by tool name.
  `main.py`'s `TOOL_OVERRIDES` is the canonical example:
  `search_web` read / 15 000 ms, `write_file` mutate / 10 000 ms,
  `execute_python` dangerous / 15 000 ms.

## Conventions

- **No test framework.** Verification lives in `scripts/verify_*.py`, run
  via `pixi run verify-<name>` or `python scripts/verify_<name>.py`. Each
  is a standalone executable specification.
- **Every module has a matching verification script.** Adding a primitive
  means adding its `verify_*.py` in the same step.
- **`scripts/_verify_common.py`** provides `check`, `check_equal`, `skip`,
  `make_temp_dir`. Print `✓` / `✗` / `SKIP`; exit non-zero on failure.
- **Verify-script `_finalize` should be honest.** If a script's `run()`
  sets a `completed: bool` flag and passes it to `_finalize(completed=…)`,
  a crash before the first `check()` reports `❌ crashed before
  completing` instead of a misleading `✅ verified`. Not every script has
  been migrated to this pattern; `scripts/_migrate_finalize.py` shows the
  transformation.
- **Examples are demos, not tests.** They are not part of the verification
  sweep and are not run by any `verify:*` entry.
- **`pyagent/__init__.py` calls `load_dotenv()`.** Importing anything from
  the package loads `.env`. Examples do not need their own loader.
- **`examples/_common.py` provides `mcp_with_workspace`.** It sets
  `WORKSPACE_ROOT` for the duration of an MCP session and restores the
  original afterwards. Any example that spawns the server and wants the
  file tools confined to a specific directory should use it — otherwise
  the server sees the default workspace and the file tools miss the
  expected files.
- **`examples/_common.py` provides `extract_text`.** `RemoteTool.call`
  returns a `MultipartContent` (or a scalar); `extract_text` flattens it
  to a string for `print`. Do not print the raw `MultipartContent`
  representation in examples — it is not human-readable.

## Streaming (`llm.py`)

`LlmClient(mode="stream", on_text_delta=callback)` streams by iterating
`litellm.acompletion(stream=True)`. The response is rebuilt from the
accumulated chunks via `_rebuild_from_chunks` when possible; if the
rebuild fails, the chunks are concatenated into a plain text `Message`
and `error_metadata` carries `stream_builder_unavailable: True`. That
signal tells the caller structured fields (tool calls, usage) were not
preserved.

`on_text_delta` is called once per text chunk as it arrives. A single
streaming call can emit many deltas.

### Known LiteLLM/rich quirk

On some combinations of `litellm` and `rich`, LiteLLM's verbose logging
reaches `rich.console.Console.print(..., flush=…)`, which rejects
`flush`. A streaming call on such a version can raise
`TypeError: Console.print() got an unexpected keyword argument 'flush'`
from inside `acompletion`, and the error surfaces as a generic
`Provider API call failed: …` because there is no HTTP status attached.

The provider call itself was fine — the logging broke it.

If this appears:

- Set `LITELLM_LOG=ERROR` in `.env` (read before importing `litellm`).
- Or pin compatible versions of `litellm` and `rich` in
  `pyproject.toml`.

The project does not ship a compatibility shim for this. It is an
environment/version concern, not a source-code concern.

## Sandbox (`execute_python`)

- **Runs under Deno with `--cached-only` and `--allow-read=<npm-cache>`
  only.** No `--allow-net`, no `--allow-env`, no `--allow-run`, no
  `--allow-ffi`, no `--allow-write`. Pyodide's Python has no syscalls at
  all — its `open()` and `socket()` go through a virtual MEMFS that
  cannot reach the host.
- **Cache resolution is delegated to `tools/_npm_cache.py`.** Three
  tiers: `DENO_DIR` env var → **bare** `deno info --json` (the only form
  that reports `npmCache`; the with-module form returns a dependency
  graph and never includes it) → platform default. This is the fix for
  the long-standing "deno info did not report an npmCache path" failure.
- **Warm-up sends `b"pass"` as stdin**, not empty input. Empty input is
  not valid Python and would not exercise `loadPyodide()`, so the WASM
  assets would never land in the cache and the first real
  `--cached-only` call would fail. `sandbox.ts` reads raw Python source
  from stdin and executes it — `b"pass"` is the smallest valid program.
- **Output is truncated at 16 KiB with an explicit
  `[truncated: N more bytes]` marker.** The marker is load-bearing —
  `verify_execute_python.py`'s truncation test looks for it.
- **`execute_python` is `permission="dangerous"`.** `main.py`'s
  `TOOL_OVERRIDES` enforces this at the CLI; other callers pass the same
  override to `load_mcp_tools`.

## MCP permissions and `ToolOverrides`

The Python project's MCP server does not advertise a permission tier for
its tools — a deliberate difference from the TypeScript port, which adds
`_meta.permission` and a `trusted` flag on the connection. In Python:

- **The server exposes tools without permission metadata.**
- **`load_mcp_tools(conn, overrides=…)` applies `ToolOverrides` by tool
  name.** The default for any tool without an override is `read` with
  `timeout_ms=0`.
- **`main.py`'s `TOOL_OVERRIDES` is the canonical configuration.** It is
  the only place where `write_file` becomes a `mutate` tool and
  `execute_python` becomes `dangerous`. Any caller that spawns the MCP
  server and wants the two-lock model to apply must pass the same
  overrides.

## What this project deliberately does not have

Honest list — an LLM generating study material should not invent these
capabilities because they are present in the TypeScript port:

- **A test framework.** No pytest, no unittest.
- **CI workflows.** Verification needs provider keys.
- **Symlink re-containment in file tools.** `_resolve_in_workspace` does a
  lexical `relative_to` check only. A symlink inside the workspace that
  points outside would pass. The TypeScript port adds `realpath`
  re-containment; Python does not.
- **`_meta.permission` on MCP tools.** Permissions are supplied by
  `ToolOverrides`, not advertised by the server. No `trusted` flag on
  `McpConnection`.
- **Atomic artifact writes.** `notes_digest` and `trip_planner` write
  their `.json` / `.md` pair with plain `Path.write_text`. A crash
  between the two writes leaves one artifact written and one not.
- **Prompt caching.** `build_messages` rebuilds the full message list on
  every step. Fine at `max_steps=4`; not optimized.
- **Memory bounds and pruning.** `MemoryStore` has no size caps on key /
  value / tags and no `prune()` method.
- **Streaming event-log reader.** `read_event_log` reads the whole file
  into memory (`read_text().splitlines()`). Fine for typical runs; not
  optimized for very large logs.
- **Deno child env allowlist.** `_run_deno_sync` inherits the full parent
  environment. Pyodide's Python does not see `os.environ` the way a
  normal Python process would, but the Deno process itself does. The
  TypeScript port passes an allowlist; Python does not.
- **Signal cleanup on SIGINT.** The host CLI relies on `asyncio`
  cancellation. There is no dedicated `SIGINT` handler to close the MCP
  child and the event log explicitly before exit.
- **MCP reconnect.** `McpConnection.call_tool` does not retry after a
  subprocess death.
- **Tracing shutdown timeout.** `shutdown_tracing()` awaits the provider's
  `shutdown()` without a bound.
- **Auto-recall inside `Agent`.** Memory recall is a workflow-level
  decision; `research_brief` demonstrates the pattern.
- **Embeddings / vector search.** FTS5 covers a personal agent's needs.