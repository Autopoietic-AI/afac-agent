# -*- coding: utf-8 -*-
"""Render a human-readable ``STATUS.md`` from a ``HeartbeatState``."""
from __future__ import annotations

import json
from pathlib import Path

from .heartbeat import HeartbeatState

STATUS_FILENAME = "STATUS.md"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s"


def _fmt_metric(metric: dict | None) -> str:
    if not metric:
        return "n/a"
    return json.dumps(metric, ensure_ascii=False, sort_keys=True)


def render_status(state: HeartbeatState) -> str:
    """Render the heartbeat state as a Markdown status document."""
    lines = [
        f"# Run Status: {state.run_id}",
        "",
        f"- **Task**: {state.task}",
        f"- **Status**: {state.status}",
        f"- **Stage**: {state.stage}",
        f"- **Substage**: {state.substage}",
        f"- **Current experiment**: {state.current_experiment}",
        f"- **Current model**: {state.current_model}",
        f"- **Rounds used**: {state.rounds_used}",
        f"- **Elapsed**: {_fmt_seconds(state.elapsed_seconds)}",
        f"- **Remaining**: {_fmt_seconds(state.remaining_seconds)}",
        f"- **Estimated completion**: {state.estimated_completion or 'n/a'}",
        f"- **CPU active**: {state.cpu_active:.1f}%",
        f"- **GPU active**: {'yes' if state.gpu_active else 'no'}",
        f"- **Latest metric**: {_fmt_metric(state.latest_metric)}",
        f"- **Next checkpoint**: {state.next_checkpoint or 'n/a'}",
        f"- **Stall status**: {state.stall_status}",
        f"- **Safe resume point**: {state.safe_resume_point or 'n/a'}",
        "",
    ]
    return "\n".join(lines)


def write_status(state: HeartbeatState, run_dir: str | Path) -> Path:
    """Render and write ``STATUS.md`` into the run directory."""
    path = Path(run_dir) / STATUS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_status(state), encoding="utf-8")
    return path
