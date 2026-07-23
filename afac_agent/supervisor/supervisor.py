# -*- coding: utf-8 -*-
"""RunSupervisor facade composing heartbeat, logs, renderers, and resume state.

One object per run; wires together:

- ``HeartbeatWriter`` (throttled ``heartbeat.json``)
- ``RunEventLog`` / ``UnbufferedTextLog`` (``run_events.jsonl`` / ``unbuffered.log``)
- ``StallDetector`` (liveness classification)
- ``ResumeManager`` (``resume_state.json`` checkpoints)
- ``STATUS.md`` / ``dashboard.html`` renderers
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .dashboard import write_dashboard
from .event_log import RunEventLog, UnbufferedTextLog
from .heartbeat import HeartbeatState, HeartbeatWriter
from .resume_manager import ResumeManager
from .stall_detector import StallDetector
from .status_renderer import write_status


class RunSupervisor:
    """Compose all run-supervision components behind a small facade."""

    def __init__(
        self,
        run_dir: str | Path,
        task: str,
        run_id: str,
        budget_seconds: float | None = None,
        interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
        gpu_probe: Callable[[], bool] | None = None,
        suspect_after_seconds: float = 180.0,
        stall_after_seconds: float = 600.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.clock = clock
        self.stall_detector = StallDetector(
            suspect_after_seconds=suspect_after_seconds,
            stall_after_seconds=stall_after_seconds,
            clock=clock,
        )
        self.heartbeat_writer = HeartbeatWriter(
            run_dir,
            interval_seconds=interval_seconds,
            clock=clock,
            gpu_probe=gpu_probe,
            stall_detector=self.stall_detector,
            budget_seconds=budget_seconds,
        )
        self.events = RunEventLog(run_dir, clock=clock)
        self.text_log = UnbufferedTextLog(run_dir)
        self.resume = ResumeManager(run_dir)
        self._completed_experiments: list[str] = []
        self.heartbeat_writer.update(run_id=run_id, task=task)

    @property
    def state(self) -> HeartbeatState:
        return self.heartbeat_writer.state

    def start(self, stage: str) -> None:
        """Begin a stage: update heartbeat, checkpoint resume state, log, render."""
        self.heartbeat_writer.update(status="running", stage=stage)
        self.heartbeat_writer.mark_progress()
        if self.resume.should_skip(stage):
            self.event("stage_skipped", {"stage": stage})
        else:
            self.resume.save_checkpoint(stage, status="running")
            self.event("stage_start", {"stage": stage})
        self.write_all()

    def heartbeat(self, **fields: Any) -> None:
        """Update heartbeat fields and persist (throttled by interval)."""
        progress_keys = {"latest_metric", "current_experiment", "substage", "rounds_used"}
        self.heartbeat_writer.update(**fields)
        if progress_keys & set(fields):
            self.heartbeat_writer.mark_progress()
        self.heartbeat_writer.tick()

    def complete_experiment(self, experiment_id: str) -> None:
        """Record a finished experiment for the dashboard listing."""
        experiment_id = str(experiment_id)
        if experiment_id not in self._completed_experiments:
            self._completed_experiments.append(experiment_id)

    def event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append a structured event and mirror a line to the text log."""
        self.events.append(event_type, payload)
        self.text_log.write(f"{event_type}: {payload or {}}")

    def complete_stage(
        self,
        stage_id: str,
        outputs: Iterable[str] = (),
        **fields: Any,
    ) -> None:
        """Mark a stage completed; it will be skipped after any resume."""
        if fields:
            self.resume.save_checkpoint(stage_id, **fields)
        self.resume.mark_completed(stage_id, list(outputs))
        self.heartbeat_writer.update(
            stage=str(stage_id),
            safe_resume_point=self.resume.safe_resume_point(),
        )
        self.heartbeat_writer.mark_progress()
        self.event("stage_complete", {"stage": stage_id, "outputs": [str(o) for o in outputs]})
        self.write_all()

    def write_all(self) -> None:
        """Write heartbeat.json, STATUS.md, and dashboard.html."""
        self.heartbeat_writer.tick(force=True)
        write_status(self.state, self.run_dir)
        write_dashboard(self.state, self.run_dir, completed_experiments=self._completed_experiments)
