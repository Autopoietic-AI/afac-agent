"""Markdown reporting for M6A shadow runs."""

from __future__ import annotations

from typing import Any


def render_shadow_report(proposal: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = [
        "# M6A LLM Shadow Planner Report",
        "",
        "M5A deterministic plan remains the only authoritative plan.",
        "",
        f"- Proposal status: `{proposal.get('status')}`",
        f"- Proposed action: `{proposal.get('proposed_action')}`",
        f"- Proposed tool: `{proposal.get('proposed_tool')}`",
        f"- Research needed: `{proposal.get('research_needed')}`",
        f"- Agreement level: `{comparison.get('agreement_level')}`",
        f"- LLM novelty: `{comparison.get('llm_novelty')}`",
        f"- LLM safety: `{comparison.get('llm_safety_status')}`",
        "",
        "## Summary",
        "",
        str(comparison.get("comparison_summary", "")),
    ]
    if proposal.get("research_query"):
        lines.extend(["", "## Advisory research query", "", str(proposal.get("research_query"))])
    return "\n".join(lines) + "\n"
