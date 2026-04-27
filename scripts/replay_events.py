"""
scripts/replay_events.py — read back the bus stream and print each event.

Usage
-----
    python scripts/replay_events.py [--stream signals] [--since 0]
                                    [--limit 1000] [--format json|pretty]

When ``EVENT_BUS_ENABLED=0`` the in-memory backend is used; the replay
tool is then only meaningful inside a single live process. With
``EVENT_BUS_ENABLED=1`` the tool reads from Redis Streams.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bus.event_bus import get_event_bus  # noqa: E402
from bus.events import Stream  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.replay_events",
        description="Replay events from the bus.",
    )
    parser.add_argument("--stream", default=Stream.SIGNALS)
    parser.add_argument("--since", default="0")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--format", choices=("json", "pretty"), default="pretty",
    )
    return parser


def replay(stream: str, since: str, limit: int, fmt: str) -> int:
    bus = get_event_bus()
    count = 0
    for msg_id, event in bus.replay(stream, since=since, limit=limit):
        if fmt == "json":
            line = json.dumps(
                {
                    "msg_id": msg_id,
                    "event_type": event.event_type,
                    "ts": event.ts,
                    "payload": event.payload,
                }
            )
            print(line)
        else:
            print(f"{msg_id}  {event.ts}  {event.event_type}  {event.payload}")
        count += 1
    if count == 0:
        print(f"# no events on stream {stream!r}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return replay(args.stream, args.since, args.limit, args.format)


if __name__ == "__main__":
    sys.exit(main())
