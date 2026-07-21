# -*- coding: utf-8 -*-
"""Markdown rendering for deterministic plan decisions."""

from __future__ import annotations

from typing import Any


def render_plan_report(plan: dict[str, Any]) -> str:
    lines = [
        "# M5A Deterministic Plan",
        "",
        f"- plan_id: `{plan['plan_id']}`",
        f"- status: `{plan['status']}`",
        f"- primary_problem: `{plan['primary_problem']}`",
        f"- selected_action: `{plan['selected_action']}`",
        f"- selected_tool: `{plan['selected_tool']}`",
        f"- requires_human_approval: `{plan['requires_human_approval']}`",
        f"- auto_execution_allowed: `{plan['auto_execution_allowed']}`",
        "",
        "## Missing inputs",
        "",
    ]
    for item in plan.get("missing_inputs", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Reason codes", ""])
    for item in plan.get("reason_codes", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Ranked actions", ""])
    for action in plan.get("ranked_actions", []):
        lines.append(f"- `{action.get('action')}`: {', '.join(action.get('reason_codes', []))}")
    lines.extend(["", "## Blocked actions", ""])
    for action in plan.get("blocked_actions", []):
        lines.append(f"- `{action.get('action')}`: {', '.join(action.get('reason_codes', []))}")
    lines.extend(["", "## Deferred actions", ""])
    for action in plan.get("deferred_actions", []):
        lines.append(f"- `{action.get('action')}`: {', '.join(action.get('reason_codes', []))}")
    return "\n".join(lines) + "\n"
