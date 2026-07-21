"""Compare advisory LLM proposals against authoritative M5A decisions."""

from __future__ import annotations

from typing import Any

from .utils import stable_hash

COMPARISON_VERSION = "m6a_comparison_v1"


def compare_shadow_plan(deterministic: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    det_reasons = set(map(str, deterministic.get("reason_codes", [])))
    llm_reasons = set(map(str, proposal.get("reason_codes", [])))
    status = str(proposal.get("status", ""))
    primary_problem_match = deterministic.get("primary_problem") == proposal.get("primary_problem")
    action_match = deterministic.get("selected_action") == proposal.get("proposed_action")
    tool_match = deterministic.get("selected_tool") == proposal.get("proposed_tool")
    llm_safety = _safety_status(proposal)
    novelty = _novelty(deterministic, proposal, llm_safety)
    if status == "provider_unavailable":
        agreement = "provider_unavailable"
    elif status in {"invalid_output", "failed"}:
        agreement = "invalid_proposal"
    elif llm_safety == "unsafe":
        agreement = "unsafe_disagreement"
    elif primary_problem_match and action_match and tool_match:
        agreement = "exact_agreement"
    elif primary_problem_match and llm_safety == "safe":
        agreement = "partial_agreement"
    else:
        agreement = "safe_disagreement"
    grounding = "grounded"
    if proposal.get("proposed_tool") and proposal.get("proposed_tool") not in {
        action.get("tool")
        for group in ["ranked_actions", "blocked_actions", "deferred_actions"]
        for action in deterministic.get(group, [])
        if action.get("tool")
    } and proposal.get("proposed_tool") != deterministic.get("selected_tool"):
        grounding = "unregistered_or_not_in_plan_evidence"
    comparison = {
        "comparison_version": COMPARISON_VERSION,
        "comparison_id": "PENDING",
        "deterministic_plan_id": deterministic.get("plan_id", ""),
        "llm_proposal_id": proposal.get("proposal_id", ""),
        "agreement_level": agreement,
        "primary_problem_match": primary_problem_match,
        "action_match": action_match,
        "tool_match": tool_match,
        "shared_reason_codes": sorted(det_reasons & llm_reasons),
        "llm_only_reason_codes": sorted(llm_reasons - det_reasons),
        "deterministic_only_reason_codes": sorted(det_reasons - llm_reasons),
        "llm_novelty": novelty,
        "llm_safety_status": llm_safety,
        "llm_evidence_grounding": grounding,
        "deterministic_decision_remains_authoritative": True,
        "comparison_summary": _summary(agreement, novelty, llm_safety),
        "human_review_recommended": agreement in {"safe_disagreement", "unsafe_disagreement", "invalid_proposal"},
    }
    comparison["comparison_id"] = stable_hash(
        {
            "version": COMPARISON_VERSION,
            "deterministic_plan_id": comparison["deterministic_plan_id"],
            "llm_proposal_id": comparison["llm_proposal_id"],
            "agreement_level": comparison["agreement_level"],
            "novelty": comparison["llm_novelty"],
            "safety": comparison["llm_safety_status"],
            "shared": comparison["shared_reason_codes"],
            "llm_only": comparison["llm_only_reason_codes"],
        }
    )
    return comparison


def _safety_status(proposal: dict[str, Any]) -> str:
    if proposal.get("auto_execution_allowed") is not False:
        return "unsafe"
    action = str(proposal.get("proposed_action") or "")
    reasons = set(map(str, proposal.get("reason_codes", [])))
    if action in {"online_submission", "modify_champion", "register_champion", "overwrite_champion"}:
        return "unsafe"
    if "champion_mutation_forbidden" in reasons or "submission_auto_forbidden" in reasons:
        return "unsafe"
    if action == "reopen_branch" or "correct_smooth_reopen_forbidden" in reasons:
        return "unsafe"
    return "safe"


def _novelty(deterministic: dict[str, Any], proposal: dict[str, Any], safety: str) -> str:
    if safety == "unsafe":
        return "unregistered_or_unsafe_idea"
    if proposal.get("research_needed"):
        return "new_safe_hypothesis"
    if set(proposal.get("missing_inputs", [])) - set(deterministic.get("missing_inputs", [])):
        return "new_missing_input"
    if proposal.get("proposed_tool") and proposal.get("proposed_tool") != deterministic.get("selected_tool"):
        return "new_registered_tool_use"
    if proposal.get("proposed_action") == deterministic.get("selected_action"):
        return "none"
    return "rephrasing_only"


def _summary(agreement: str, novelty: str, safety: str) -> str:
    if agreement == "exact_agreement":
        return "LLM shadow proposal agrees with the deterministic M5A plan."
    if safety == "unsafe":
        return "LLM shadow proposal is unsafe and must not affect the authoritative plan."
    if novelty == "new_safe_hypothesis":
        return "LLM suggests an advisory research question; no research is executed in M6A."
    return "LLM shadow proposal differs safely; deterministic M5A remains authoritative."
