# -*- coding: utf-8 -*-
"""Render a self-contained ``dashboard.html`` from a ``HeartbeatState``.

The page uses inline CSS only, refreshes itself every 30 seconds via a meta
tag, and references no external assets so it works from a plain file:// open.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

from .heartbeat import HeartbeatState

DASHBOARD_FILENAME = "dashboard.html"
REFRESH_SECONDS = 30

_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 2em; background: #f6f7f9; color: #222; }
h1 { font-size: 1.4em; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 8px;
        padding: 1em 1.2em; margin-bottom: 1em; }
.card h2 { font-size: 1em; margin: 0 0 0.6em 0; color: #555; text-transform: uppercase; }
table { border-collapse: collapse; }
td { padding: 0.2em 1em 0.2em 0; vertical-align: top; }
td.k { color: #666; white-space: nowrap; }
.stall-ok { color: #1a7f37; font-weight: bold; }
.stall-suspected_stall { color: #b35900; font-weight: bold; }
.stall-stalled { color: #c00; font-weight: bold; }
.v2-badge { background: #1a7f37; color: #fff; padding: 0.3em 0.8em; border-radius: 4px; font-weight: bold; }
.legacy-banner { background: #c00; color: #fff; padding: 0.6em 1em; border-radius: 4px; font-weight: bold; font-size: 1.1em; }
ul { margin: 0.3em 0; padding-left: 1.4em; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s"


def render_dashboard(
    state: HeartbeatState,
    completed_experiments: Iterable[str] | None = None,
) -> str:
    """Render the heartbeat state as a self-contained HTML dashboard."""
    experiments = list(completed_experiments or [])
    if experiments:
        exp_items = "".join(f"<li>{_esc(exp)}</li>" for exp in experiments)
        exp_html = f"<ul>{exp_items}</ul>"
    else:
        exp_html = "<p>No completed experiments yet.</p>"
    metric = json.dumps(state.latest_metric, ensure_ascii=False, sort_keys=True) if state.latest_metric else "n/a"
    if state.legacy_mode:
        identity_html = '<div class="legacy-banner">LEGACY EXECUTION — NOT A V2 RUN</div>'
    elif state.orchestrator_version:
        identity_html = (
            f'<div class="v2-badge">V2 RUN — orchestrator {_esc(state.orchestrator_version)}'
            f" | planner {_esc(state.planner_mode or 'n/a')}"
            f" | llm calls {state.llm_calls_count}</div>"
        )
    else:
        identity_html = '<div class="legacy-banner">LEGACY EXECUTION — NOT A V2 RUN</div>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<title>AFAC Run Dashboard - {_esc(state.run_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>AFAC Run Dashboard</h1>
{identity_html}
<div class="card"><h2>Run</h2>
<table>
<tr><td class="k">Run ID</td><td>{_esc(state.run_id)}</td></tr>
<tr><td class="k">Execution ID</td><td>{_esc(state.execution_id or 'n/a')}</td></tr>
<tr><td class="k">Input fingerprint</td><td>{_esc(state.input_fingerprint or 'n/a')}</td></tr>
<tr><td class="k">Orchestrator</td><td>{_esc(state.orchestrator_version or 'legacy')}</td></tr>
<tr><td class="k">Planner mode</td><td>{_esc(state.planner_mode or 'n/a')}</td></tr>
<tr><td class="k">LLM calls</td><td>{state.llm_calls_count}</td></tr>
<tr><td class="k">Cache status</td><td>{_esc(state.cache_status or 'n/a')}</td></tr>
<tr><td class="k">Task</td><td>{_esc(state.task)}</td></tr>
<tr><td class="k">Status</td><td>{_esc(state.status)}</td></tr>
<tr><td class="k">Current stage</td><td>{_esc(state.stage)}</td></tr>
<tr><td class="k">Substage</td><td>{_esc(state.substage)}</td></tr>
<tr><td class="k">Current problem</td><td>{_esc(state.current_problem_id or 'n/a')}</td></tr>
<tr><td class="k">Current proposal</td><td>{_esc(state.current_proposal_id or 'n/a')}</td></tr>
<tr><td class="k">Critic status</td><td>{_esc(state.current_critic_status or 'n/a')}</td></tr>
<tr><td class="k">Current experiment</td><td>{_esc(state.current_experiment)}</td></tr>
<tr><td class="k">Current model</td><td>{_esc(state.current_model)}</td></tr>
</table></div>
<div class="card"><h2>Progress</h2>
<table>
<tr><td class="k">Rounds used</td><td>{state.rounds_used}</td></tr>
<tr><td class="k">Latest metric</td><td>{_esc(metric)}</td></tr>
<tr><td class="k">Elapsed</td><td>{_fmt_seconds(state.elapsed_seconds)}</td></tr>
<tr><td class="k">Remaining</td><td>{_fmt_seconds(state.remaining_seconds)}</td></tr>
<tr><td class="k">Estimated completion</td><td>{_esc(state.estimated_completion or 'n/a')}</td></tr>
<tr><td class="k">Next checkpoint</td><td>{_esc(state.next_checkpoint or 'n/a')}</td></tr>
</table></div>
<div class="card"><h2>Completed experiments</h2>
{exp_html}</div>
<div class="card"><h2>Health</h2>
<table>
<tr><td class="k">CPU active</td><td>{state.cpu_active:.1f}%</td></tr>
<tr><td class="k">GPU active</td><td>{'yes' if state.gpu_active else 'no'}</td></tr>
<tr><td class="k">Stall status</td><td class="stall-{_esc(state.stall_status)}">{_esc(state.stall_status)}</td></tr>
<tr><td class="k">Safe resume point</td><td>{_esc(state.safe_resume_point or 'n/a')}</td></tr>
</table></div>
</body>
</html>
"""


def write_dashboard(
    state: HeartbeatState,
    run_dir: str | Path,
    completed_experiments: Iterable[str] | None = None,
) -> Path:
    """Render and write ``dashboard.html`` into the run directory."""
    path = Path(run_dir) / DASHBOARD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(state, completed_experiments), encoding="utf-8")
    return path
