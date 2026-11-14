from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class EventJournal:
    def __init__(self, directory: Path, stream: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.stream = stream
        self.sequence = 0
        self.path = directory / f"{stream}-{self.pid}.jsonl"
        self.file = self.path.open("x", buffering=1)
        self.emit("open", {})

    def emit(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "schema": "pr-event-v1",
            "stream": self.stream,
            "pid": self.pid,
            "sequence": self.sequence,
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "data": data,
        }
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.sequence += 1

    def close(self) -> None:
        if not self.file.closed:
            self.emit("close", {})
            self.file.close()


def read_journal(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identity = None
    last_time = -1
    with path.open() as source:
        for sequence, line in enumerate(source):
            if not line.endswith("\n"):
                raise ValueError("truncated event record")
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != {
                "schema",
                "stream",
                "pid",
                "sequence",
                "monotonic_ns",
                "event",
                "data",
            }:
                raise ValueError("unknown event fields")
            current = record["stream"], record["pid"]
            if identity is None:
                identity = current
            if (
                record["schema"] != "pr-event-v1"
                or type(record["sequence"]) is not int
                or record["sequence"] != sequence
                or type(record["monotonic_ns"]) is not int
                or record["monotonic_ns"] < last_time
                or current != identity
                or not isinstance(record["event"], str)
                or not isinstance(record["data"], dict)
            ):
                raise ValueError("event identity, ordering or clock failure")
            if sequence == 0 and record["event"] != "open":
                raise ValueError("missing journal opening")
            if records and records[-1]["event"] == "close":
                raise ValueError("event after journal close")
            records.append(record)
            last_time = record["monotonic_ns"]
    if not records:
        raise ValueError("empty event journal")
    return records
