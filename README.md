# pyagent

A Python ReAct agent over **LiteLLM + pydantic + MCP**. Event-sourced
conversation, three-tier tool permissions, parallel tool execution,
resumable runs, four opinionated workflows, SQLite+FTS5 memory,
OpenTelemetry tracing, and a Deno/Pyodide sandbox for untrusted code.

## Quick start

```bash
pixi install
cp .env.example .env          # set GEMINI_API_KEY (and TAVILY_API_KEY)
pixi run python main.py "What is 2 + 2? Use the calculator."