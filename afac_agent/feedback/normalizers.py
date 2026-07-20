# -*- coding: utf-8 -*-
"""Whitelisted conversion from Adapter execution results to M4A feedback."""

from __future__ import annotations

from typing import Any, Callable

from .base import base_feedback, unavailable


Normalizer = Callable[[dict[str, Any], str, str], dict[str, Any]]


def normalize_v46a1_isolated(
    execution: dict[str, Any],
    execution_result_hash: str,
    execution_result_ref: str,
) -> dict[str, Any]:
    metrics = execution.get("metrics", {})
    feedback = base_feedback(
        execution=execution,
        execution_result_hash=execution_result_hash,
        execution_result_ref=execution_result_ref,
        feedback_kind="expert_scope_audit",
        evaluation_tier="expert_scope",
    )
    oof_status = str(metrics.get("oof_status", "") or "")
    if oof_status == "unavailable" or not oof_status:
        oof_evaluation = unavailable(metrics.get("oof_unavailable_reason") or "missing_final_oof")
        recommendation = "informational_only"
    else:
        oof_evaluation = {"status": oof_status}
        recommendation = "proceed_to_oof_evaluation"
    feedback["evidence"] = {
        "isolated_only_pass": bool(metrics.get("isolated_only_pass", False)),
        "diff_count": int(metrics.get("diff_count", 0)),
        "isolated_diff_count": int(metrics.get("isolated_diff_count", 0)),
        "graph_visible_diff_count": int(metrics.get("graph_visible_diff_count", 0)),
        "oof_status": oof_evaluation,
        "parent_identity_status": metrics.get("parent_identity_status", "unavailable"),
    }
    feedback["metrics"] = {
        "scope": {
            "isolated_only_pass": feedback["evidence"]["isolated_only_pass"],
            "diff_count": feedback["evidence"]["diff_count"],
            "isolated_diff_count": feedback["evidence"]["isolated_diff_count"],
            "graph_visible_diff_count": feedback["evidence"]["graph_visible_diff_count"],
        },
        "oof_evaluation": oof_evaluation,
    }
    feedback["limitations"].append(
        {
            "status": "unavailable",
            "reason": "missing_final_oof",
            "metric_family": "oof_gain",
        }
    )
    feedback["recommendation"] = recommendation
    return feedback


def normalize_v49a_edge_utility(
    execution: dict[str, Any],
    execution_result_hash: str,
    execution_result_ref: str,
) -> dict[str, Any]:
    metrics = execution.get("metrics", {})
    feedback = base_feedback(
        execution=execution,
        execution_result_hash=execution_result_hash,
        execution_result_ref=execution_result_ref,
        feedback_kind="signal_audit",
        evaluation_tier="signal_evidence",
    )
    coverage = {
        "covered_by_test_meta": int(metrics.get("champion_patches_covered_by_test_meta", 0)),
        "selected": int(metrics.get("champion_patches_selected", 0)),
        "gate_allowed": int(metrics.get("champion_patches_gate_allowed", 0)),
    }
    feedback["evidence"] = {
        "oof_selected_count": int(metrics.get("oof_selected_count", 0)),
        "test_selected_count": int(metrics.get("test_selected_count", 0)),
        "gate_config": metrics.get("gate_config", {}),
        "gate_recomputed_pass": bool(metrics.get("gate_recomputed_pass", False)),
        "champion_patch_coverage": coverage,
        "test_truth_usage_pass": bool(metrics.get("test_truth_usage_pass", False)),
    }
    feedback["metrics"] = {
        "selection_counts": {
            "oof_selected_count": feedback["evidence"]["oof_selected_count"],
            "test_selected_count": feedback["evidence"]["test_selected_count"],
        },
        "gate": {
            "gate_config": feedback["evidence"]["gate_config"],
            "gate_recomputed_pass": feedback["evidence"]["gate_recomputed_pass"],
        },
        "champion_patch_coverage": coverage,
        "model_gain_assessment": unavailable("signal_audit_does_not_estimate_model_gain"),
    }
    feedback["recommendation"] = (
        "proceed_to_candidate_generation"
        if feedback["evidence"]["gate_recomputed_pass"]
        and feedback["evidence"]["test_truth_usage_pass"]
        else "informational_only"
    )
    return feedback


def normalize_v53q1_patch_replay(
    execution: dict[str, Any],
    execution_result_hash: str,
    execution_result_ref: str,
) -> dict[str, Any]:
    metrics = execution.get("metrics", {})
    feedback = base_feedback(
        execution=execution,
        execution_result_hash=execution_result_hash,
        execution_result_ref=execution_result_ref,
        feedback_kind="candidate_replay",
        evaluation_tier="candidate_replay",
    )
    semantic_pass = bool(metrics.get("semantic_replay_pass", False))
    byte_pass = bool(metrics.get("byte_replay_pass", False))
    replay_diff = int(metrics.get("replay_to_champion_differing_row_count", 0))
    change_count = int(metrics.get("base_to_replay_diff_count", 0))
    feedback["evidence"] = {
        "semantic_replay_pass": semantic_pass,
        "byte_replay_pass": byte_pass,
        "base_to_replay_diff_count": change_count,
        "replay_to_champion_diff_count": replay_diff,
        "registered_as_champion": bool(metrics.get("registered_as_champion", False)),
        "submission_ready": bool(metrics.get("submission_ready", False)),
    }
    feedback["candidate_changes"] = {
        "status": "observed",
        "count": change_count,
        "rows": metrics.get("base_to_replay_diff", []),
    }
    feedback["metrics"] = {
        "semantic_replay": {
            "status": "observed",
            "passed": semantic_pass,
        },
        "byte_replay": {
            "status": "observed",
            "passed": byte_pass,
        },
        "replay_to_champion_diff": {
            "status": "observed",
            "differing_row_count": replay_diff,
        },
        "oof_gain": unavailable("candidate_replay_does_not_compute_oof_gain"),
    }
    feedback["safety"]["mutates_predictions"] = True
    feedback["safety"]["registered_as_champion"] = feedback["evidence"]["registered_as_champion"]
    feedback["safety"]["submission_ready"] = feedback["evidence"]["submission_ready"]
    feedback["recommendation"] = (
        "accept_as_reference"
        if semantic_pass
        and byte_pass
        and replay_diff == 0
        and not feedback["evidence"]["registered_as_champion"]
        and not feedback["evidence"]["submission_ready"]
        else "reject"
    )
    return feedback


NORMALIZER_REGISTRY: dict[str, Normalizer] = {
    "A1_V46A1_ISOLATED_AUDIT": normalize_v46a1_isolated,
    "A1_V49A_EDGE_UTILITY_AUDIT": normalize_v49a_edge_utility,
    "A1_V53Q1_PATCH_REPLAY_SAFE": normalize_v53q1_patch_replay,
}
