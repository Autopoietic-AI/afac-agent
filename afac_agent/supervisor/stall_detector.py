# -*- coding: utf-8 -*-
"""Stall detection for long-running AFAC v2.0 runs.

Classifies progress freshness into three states based on the time elapsed
since the last recorded progress event:

- ``ok``: fresh progress within ``suspect_after_seconds``.
- ``suspected_stall``: no progress for ``suspect_after_seconds``.
- ``stalled``: no progress for ``stall_after_seconds``.

The clock is injectable so tests can advance time deterministically.
"""
from __future__ import annotations

import time
from typing import Callable

STALL_OK = "ok"
STALL_SUSPECTED = "suspected_stall"
STALL_STALLED = "stalled"


class StallDetector:
    """Classify run liveness from the last progress timestamp."""

    def __init__(
        self,
        suspect_after_seconds: float = 180.0,
        stall_after_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if stall_after_seconds < suspect_after_seconds:
            raise ValueError("stall_after_seconds must be >= suspect_after_seconds")
        self.suspect_after_seconds = float(suspect_after_seconds)
        self.stall_after_seconds = float(stall_after_seconds)
        self.clock = clock

    def status(self, last_progress_time: float, now: float | None = None) -> str:
        """Return the stall state for a given last-progress timestamp."""
        current = self.clock() if now is None else float(now)
        idle = current - float(last_progress_time)
        if idle >= self.stall_after_seconds:
            return STALL_STALLED
        if idle >= self.suspect_after_seconds:
            return STALL_SUSPECTED
        return STALL_OK
