# -*- coding: utf-8 -*-
"""Append-only event and text logs for AFAC v2.0 runs.

- ``RunEventLog`` appends structured events to ``run_events.jsonl``, one JSON
  object per line with ``event_type``, ``timestamp``, and ``payload``.
- ``UnbufferedTextLog`` writes plain-text lines to ``unbuffered.log`` and
  flushes after every write so external watchers always see fresh output.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

EVENTS_FILENAME = "run_events.jsonl"
UNBUFFERED_FILENAME = "unbuffered.log"


class RunEventLog:
    """Append-only JSONL event writer for ``run_events.jsonl``."""

    def __init__(self, run_dir: str | Path, clock: Callable[[], float] = time.time) -> None:
        self.run_dir = Path(run_dir)
        self.clock = clock

    @property
    def path(self) -> Path:
        return self.run_dir / EVENTS_FILENAME

    def append(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event line and return the stored event record."""
        event = {
            "event_type": str(event_type),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock())),
            "payload": payload or {},
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
        return event

    def load(self) -> list[dict[str, Any]]:
        """Read back all appended events."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events


class UnbufferedTextLog:
    """Plain-text log writer that flushes after every write."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    @property
    def path(self) -> Path:
        return self.run_dir / UNBUFFERED_FILENAME

    def write(self, line: str) -> None:
        """Append one text line and flush immediately."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(str(line) + "\n")
            file.flush()
