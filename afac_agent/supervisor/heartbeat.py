# -*- coding: utf-8 -*-
"""Heartbeat state and writer for AFAC v2.0 runs.

``HeartbeatWriter`` maintains a single ``HeartbeatState`` and persists it to
``heartbeat.json`` inside the run directory. Writes are throttled to
``interval_seconds`` unless ``tick(force=True)`` is used, and are performed
atomically-ish (write to a sibling temp file, then ``os.replace``).

CPU activity is sampled with ``psutil.cpu_percent(interval=None)``. GPU
activity defaults to ``False``; a probe callable can be injected later via the
``gpu_probe`` constructor argument without changing the writer.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil

from ..research.event_store import json_dumps
from .stall_detector import StallDetector

HEARTBEAT_FILENAME = "heartbeat.json"


@dataclass
class HeartbeatState:
    """Snapshot of a run's liveness and progress."""

    run_id: str = ""
    task: str = ""
    status: str = "initializing"
    stage: str = ""
    substage: str = ""
    current_experiment: str = ""
    current_model: str = ""
    rounds_used: int = 0
    elapsed_seconds: float = 0.0
    remaining_seconds: float | None = None
    cpu_active: float = 0.0
    gpu_active: bool = False
    latest_metric: dict[str, Any] | None = None
    last_progress_time: float = 0.0
    next_checkpoint: str = ""
    estimated_completion: str = ""
    safe_resume_point: str = ""
    stall_status: str = "ok"


class HeartbeatWriter:
    """Throttled, injectable-clock writer for ``heartbeat.json``."""

    def __init__(
        self,
        run_dir: str | Path,
        interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
        gpu_probe: Callable[[], bool] | None = None,
        stall_detector: StallDetector | None = None,
        budget_seconds: float | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self.gpu_probe = gpu_probe
        self.stall_detector = stall_detector or StallDetector(clock=clock)
        self.budget_seconds = budget_seconds
        self.start_time = float(clock())
        now = float(clock())
        self.state = HeartbeatState(last_progress_time=now)
        self._last_write_time: float | None = None

    @property
    def path(self) -> Path:
        return self.run_dir / HEARTBEAT_FILENAME

    def update(self, **fields: Any) -> HeartbeatState:
        """Set fields on the heartbeat state; unknown fields raise."""
        for key, value in fields.items():
            if not hasattr(self.state, key):
                raise AttributeError(f"HeartbeatState has no field {key!r}")
            setattr(self.state, key, value)
        return self.state

    def mark_progress(self) -> None:
        """Record a progress event at the current clock time."""
        self.state.last_progress_time = float(self.clock())

    def _refresh_derived(self, now: float) -> None:
        state = self.state
        state.elapsed_seconds = max(0.0, now - self.start_time)
        if self.budget_seconds is not None:
            state.remaining_seconds = max(0.0, float(self.budget_seconds) - state.elapsed_seconds)
            state.estimated_completion = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + state.remaining_seconds)
            )
        state.cpu_active = float(psutil.cpu_percent(interval=None))
        state.gpu_active = bool(self.gpu_probe()) if self.gpu_probe is not None else False
        state.stall_status = self.stall_detector.status(state.last_progress_time, now=now)

    def tick(self, force: bool = False) -> bool:
        """Refresh derived fields and write ``heartbeat.json`` if due.

        Returns True when the file was (re)written.
        """
        now = float(self.clock())
        self._refresh_derived(now)
        due = (
            self._last_write_time is None
            or (now - self._last_write_time) >= self.interval_seconds
        )
        if not force and not due:
            return False
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.run_dir / (HEARTBEAT_FILENAME + ".tmp")
        tmp_path.write_text(json_dumps(asdict(self.state)) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.path)
        self._last_write_time = now
        return True
