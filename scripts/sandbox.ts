// scripts/sandbox.ts — run Python via Pyodide in Deno's WASM sandbox.
//
// Reads Python source from stdin, writes combined stdout/stderr to stdout.
//
// Two-phase usage (driven by server.py):
//   1. Warm-up: deno run --allow-read --allow-net --allow-write=<npm-cache> \
//                 scripts/sandbox.ts  <<< ""
//      Ensures Pyodide's runtime assets are present in the npm cache.
//   2. Sandbox: deno run --cached-only --allow-read=<npm-cache> \
//                 scripts/sandbox.ts  <<< "<user code>"
//      No net, no env, no run, no ffi, no write. Read-only to the npm cache
//      (needed by Pyodide's JS bootstrap, not by the executed Python).
//
// Security model:
//   - Deno-level: only read on the npm cache; everything else denied.
//   - WASM-level: Pyodide's Python has no syscalls. Its open()/socket() go
//     through Pyodide's virtual MEMFS, which cannot reach the host regardless
//     of Deno's permissions. The --allow-read grant is consumed by the
//     JavaScript bootstrap that loads the .wasm binary, not by user code.

import { loadPyodide } from "npm:pyodide@0.26.2";

const code = await new Response(Deno.stdin.readable).text();
const encoder = new TextEncoder();

const write = (s: string, err = false): void => {
  const bytes = encoder.encode(s);
  if (err) Deno.stderr.writeSync(bytes);
  else Deno.stdout.writeSync(bytes);
};

const pyodide = await loadPyodide({
  stdout: (text) => write(text + "\n"),
  stderr: (text) => write(text + "\n", true),
});

try {
  const result = await pyodide.runPythonAsync(code);
  if (result !== undefined && result !== null) {
    write(String(result) + "\n");
  }
} catch (e) {
  write((e instanceof Error ? e.message : String(e)) + "\n", true);
  Deno.exit(1);
}