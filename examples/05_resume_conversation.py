"""05 — Resume a conversation.

Event sourcing: the conversation is a persisted list of events. Run an agent
normally with a JSONL log attached, rehydrate the context from that log with
`ExecutionContext.from_log`, then call `agent.resume(...)` with a follow-up. The
resumed run sees the entire prior conversation.

    pixi run python examples/05_resume_conversation.py

Re-running is safe: the script deletes `.examples/resume-*.jsonl` first.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples._common import (
    build_llm_client,
    ensure_dir,
    print_final,
    print_trace,
    section,
)
from pyagent.agent import Agent
from pyagent.context import ExecutionContext
from pyagent.event_log import JsonlEventLog

EXAMPLES_DIR = Path(".examples")
RUN1_LOG = EXAMPLES_DIR / "resume-1.jsonl"
RUN2_LOG = EXAMPLES_DIR / "resume-2.jsonl"


async def main() -> None:
    ensure_dir(EXAMPLES_DIR)
    for p in (RUN1_LOG, RUN2_LOG):
        p.unlink(missing_ok=True)

    agent = Agent(client=build_llm_client(), max_steps=4)

    # ---- Run 1: fresh conversation, persisted to JSONL ----
    section("run 1 (fresh)")
    with JsonlEventLog(RUN1_LOG, fsync=True) as log:
        ctx = ExecutionContext(log=log)
        result1 = await agent.run('Say "first" and nothing else.', ctx)
    print_trace(result1)
    print_final(result1)

    # ---- Rehydrate from the log ----
    section(f"rehydrate from {RUN1_LOG}")
    rehydrated = ExecutionContext.from_log(RUN1_LOG)
    print(f"events:        {len(rehydrated.events)}")
    print(f"current_step:  {rehydrated.current_step}")
    print(f"final_result:  {rehydrated.final_result!r}")

    # ---- Run 2: resume with a follow-up ----
    section("run 2 (resumed)")
    with JsonlEventLog(RUN2_LOG, fsync=True) as log:
        rehydrated.log = log
        result2 = await agent.resume(
            rehydrated,
            user_input='Say "second" and nothing else.',
        )
    print_trace(result2)
    print_final(result2)
    print(f"\ntotal events after resume: {len(result2.context.events)}")


if __name__ == "__main__":
    asyncio.run(main())
