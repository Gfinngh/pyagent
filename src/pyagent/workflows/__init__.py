"""Task-specific workflows built on the pyagent core.

Each workflow is a self-contained module that:

- Configures the agent with task-specific instructions and a tool subset.
- Produces a concrete artifact (file, report, decision).
- Exits with a code the caller can branch on (0/1/2/3, matching main.py).

The general-purpose host is ``main.py``; workflows are opinionated launchers
for specific jobs. ``python -m pyagent.workflows.<name>`` runs one.

Design principle: the workflow orchestrates (parses args, computes paths,
wires the MCP connection), the agent reasons (searches, synthesizes, calls
tools). If the workflow has to nudge the model mid-run, the prompt is wrong.
"""
