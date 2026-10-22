"""16 — Event log primitives.

The conversation is a JSONL file, one event per line. `JsonlEventLog` appends
atomically; `read_event_log` handles three edge cases:

  - blank lines                        → silently skipped
  - unterminated final line            → reported to stderr and skipped
  - malformed interior line            → raises, with the line number

    pixi run python examples/16_event_log.py
"""

from __future__ import annotations

import json
from pathlib import Path

from examples._common import ensure_dir, section
from pyagent.event_log import JsonlEventLog, read_event_log
from pyagent.schemas import AgentEvent, Message

DIR = Path(".examples/logs")


def ev(text: str) -> AgentEvent:
    return AgentEvent(
        execution_id="demo",
        author="assistant",
        content=[Message(role="assistant", content=text)],
    )


def main() -> None:
    ensure_dir(DIR)

    section("1. write + read round-trip")
    path = DIR / "basic.jsonl"
    with JsonlEventLog(path, fsync=True) as log:
        log.append(ev("first"))
        log.append(ev("second"))
    print(f"  events: {len(read_event_log(path))}")

    section("2. blank lines are skipped")
    path = DIR / "blank.jsonl"
    path.write_text(
        ev("a").model_dump_json() + "\n\n\n" + ev("b").model_dump_json() + "\n",
        encoding="utf-8",
    )
    print(f"  events: {len(read_event_log(path))} (expected 2)")

    section("3. truncated final line")
    path = DIR / "truncated.jsonl"
    path.write_text(
        ev("a").model_dump_json() + "\n" + '{"id":"partial","execu',
        encoding="utf-8",
    )
    print(f"  events: {len(read_event_log(path))} (expected 1; warning on stderr)")

    section("4. malformed interior line")
    path = DIR / "malformed.jsonl"
    path.write_text(
        ev("a").model_dump_json() + "\nnot-json\n" + ev("b").model_dump_json() + "\n",
        encoding="utf-8",
    )
    try:
        read_event_log(path)
        print("  unexpected: no error")
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}")

    section("5. reopen after close does not truncate")
    path = DIR / "reopen.jsonl"
    with JsonlEventLog(path) as log:
        log.append(ev("first"))
    with JsonlEventLog(path) as log:
        log.append(ev("second"))
    print(f"  events: {len(read_event_log(path))} (expected 2)")

    section("6. one record per line")
    print(json.dumps({"note": "each event is one json.dumps + newline"}))


if __name__ == "__main__":
    main()
