# -*- coding: utf-8 -*-
"""Resume state management for AFAC v2.0 runs.

``ResumeManager`` persists ``resume_state.json`` in the run directory so a
crashed or interrupted run can resume without repeating completed work.

Guarantee: stages marked completed (and their recorded scientific rounds) are
never repeated or double counted after a reload. ``scientific_rounds_used``
is derived as the sum of per-stage ``rounds_used`` values stored in
``stage_state``, so re-checkpointing the same stage replaces rather than
accumulates its contribution.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..research.event_store import json_dumps, load_json

RESUME_FILENAME = "resume_state.json"

_TOP_LEVEL_FIELDS = ("remaining_budget", "best_candidate", "memory_event_id", "safe_resume_point")


class ResumeManager:
    """Persist and restore per-run resume state."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self._state: dict[str, Any] = self._read()

    @property
    def path(self) -> Path:
        return self.run_dir / RESUME_FILENAME

    def _default_state(self) -> dict[str, Any]:
        return {
            "stage_state": {},
            "completed_outputs": {},
            "remaining_budget": None,
            "scientific_rounds_used": 0,
            "best_candidate": None,
            "memory_event_id": "",
            "safe_resume_point": "",
        }

    def _read(self) -> dict[str, Any]:
        state = self._default_state()
        if self.path.exists():
            stored = load_json(self.path)
            state.update({k: stored[k] for k in state if k in stored})
        return state

    def _write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.run_dir / (RESUME_FILENAME + ".tmp")
        tmp_path.write_text(json_dumps(self._state) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.path)

    def _recompute_rounds(self) -> None:
        total = 0
        for entry in self._state["stage_state"].values():
            if isinstance(entry, dict):
                total += int(entry.get("rounds_used", 0) or 0)
        self._state["scientific_rounds_used"] = total

    def save_checkpoint(self, stage: str, **fields: Any) -> dict[str, Any]:
        """Checkpoint a stage's intermediate state.

        Known top-level fields (``remaining_budget``, ``best_candidate``,
        ``memory_event_id``, ``safe_resume_point``) are hoisted out of
        ``fields``; everything else is stored under ``stage_state[stage]``.
        """
        entry = self._state["stage_state"].setdefault(str(stage), {})
        for key, value in fields.items():
            if key in _TOP_LEVEL_FIELDS:
                self._state[key] = value
            else:
                entry[key] = value
        self._recompute_rounds()
        self._write()
        return self._state

    def mark_completed(self, stage_id: str, outputs: list[str] | tuple[str, ...] = ()) -> None:
        """Mark a stage as completed with its output artifacts."""
        stage_id = str(stage_id)
        entry = self._state["stage_state"].setdefault(stage_id, {})
        entry["status"] = "completed"
        self._state["completed_outputs"][stage_id] = [str(o) for o in outputs]
        self._state["safe_resume_point"] = f"after:{stage_id}"
        self._recompute_rounds()
        self._write()

    def should_skip(self, stage_id: str) -> bool:
        """True if the stage already completed and must not be repeated."""
        entry = self._state["stage_state"].get(str(stage_id), {})
        return isinstance(entry, dict) and entry.get("status") == "completed"

    def safe_resume_point(self) -> str:
        """Return the persisted safe resume point (empty string if none)."""
        return str(self._state.get("safe_resume_point", ""))

    def load(self) -> dict[str, Any]:
        """Reload state from disk and return it."""
        self._state = self._read()
        return self._state
