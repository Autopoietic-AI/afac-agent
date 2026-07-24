# -*- coding: utf-8 -*-
"""AFAC v2.0 run manifest spec and strict completion contract.

A run may only report ``status = completed`` when every contract condition
holds.  Smoke runs use ``completed_smoke`` with a reduced but still strict
condition set.  Missing any single condition forbids the completed status.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

MANIFEST_VERSION = "afac_v2_run_manifest_v1"
TRAJECTORY_VERSION = "afac_v2_trajectory_v1"
ORCHESTRATOR_VERSION = "2.0.0"

RUN_STATUSES = (
    "preparing",
    "running",
    "waiting_for_llm",
    "training",
    "evaluating",
    "saving",
    "completed",
    "completed_smoke",
    "completed_with_anchor_fallback",
    "incomplete_no_scientific_experiment",
    "incomplete_no_deployable_candidate",
    "blocked_data_contract_mismatch",
    "failed_budget_contract",
    "failed_deployment_permission",
    "failed_operator_compilation",
    "blocked_missing_llm",
    "blocked_llm_error",
    "degraded_deterministic_fallback",
    "incomplete",
    "invalid_replay",
    "interrupted",
    "failed",
)

# Artifact keys the completion contract inspects inside run manifests.
FORMAL_REQUIRED_ARTIFACTS = (
    "problem_node",
    "competition_research_state",
    "m6b_proposal",
    "m6c_critic",
    "m5_decision",
    "experiment_genome",
    "budget_decision",
    "experiment_result",
    "no_op_audit",
    "portfolio_update",
    "postmortem",
    "deployment_audit",
)

SMOKE_REQUIRED_ARTIFACTS = (
    "problem_node",
    "m6b_proposal",
    "m6c_critic",
    "m5_decision",
    "experiment_genome",
    "budget_decision",
    "experiment_result",
    "no_op_audit",
    "postmortem",
)


def build_manifest(
    *,
    execution_id: str,
    input_fingerprint: str,
    parent_execution_id: str,
    task: str,
    data_root_hash: str,
    code_commit: str,
    branch: str,
    planner_version: str,
    planner_policy_hash: str,
    planner_mode: str,
    llm_required: bool,
    llm_available: bool,
    llm_calls_count: int,
    capability_registry_hash: str,
    metric_contract_hash: str,
    validation_contract_hash: str,
    started_at: float,
    ended_at: float,
    max_wall_clock_seconds: float,
    cache_status: str,
    resume_status: str,
    status: str,
    current_stage: str,
    scientific_rounds_used: float,
    cheap_diagnostics_used: int,
    no_op_rounds_refunded: int,
    uses_test_truth: bool,
    mutates_frozen_assets: bool,
    deployment_generated: bool,
    smoke_mode: bool,
    artifacts: dict[str, str] | None = None,
    scientific_attempts_used: int = 0,
    effective_scientific_rounds: float = 0.0,
    diagnostics_used: int = 0,
    implementation_failures_used: int = 0,
    data_contract_status: str = "unknown",
    budget_contract_status: str = "unknown",
    deployment_permission_status: str = "unknown",
    incumbent_can_deploy: bool = False,
    no_op_in_portfolio: bool = False,
    all_rounds_diagnostic: bool = True,
    anchor_fallback: bool = False,
) -> dict[str, Any]:
    """Assemble a v2 run manifest (section 12 field set)."""
    if status not in RUN_STATUSES:
        raise ValueError(f"unknown run status: {status}")
    return {
        "manifest_version": MANIFEST_VERSION,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "execution_id": execution_id,
        "run_id": execution_id,
        "input_fingerprint": input_fingerprint,
        "parent_execution_id": parent_execution_id,
        "task": task,
        "data_root_hash": data_root_hash,
        "code_commit": code_commit,
        "branch": branch,
        "planner_version": planner_version,
        "planner_policy_hash": planner_policy_hash,
        "planner_mode": planner_mode,
        "llm_required": llm_required,
        "llm_available": llm_available,
        "llm_calls_count": llm_calls_count,
        "capability_registry_hash": capability_registry_hash,
        "metric_contract_hash": metric_contract_hash,
        "validation_contract_hash": validation_contract_hash,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_seconds": max(0.0, ended_at - started_at),
        "max_wall_clock_seconds": max_wall_clock_seconds,
        "cache_status": cache_status,
        "resume_status": resume_status,
        "status": status,
        "current_stage": current_stage,
        "scientific_rounds_used": scientific_rounds_used,
        "scientific_attempts_used": scientific_attempts_used,
        "effective_scientific_rounds": effective_scientific_rounds,
        "diagnostics_used": diagnostics_used,
        "cheap_diagnostics_used": cheap_diagnostics_used,
        "no_op_rounds_refunded": no_op_rounds_refunded,
        "implementation_failures_used": implementation_failures_used,
        "uses_test_truth": uses_test_truth,
        "mutates_frozen_assets": mutates_frozen_assets,
        "deployment_generated": deployment_generated,
        "smoke_mode": smoke_mode,
        "data_contract_status": data_contract_status,
        "budget_contract_status": budget_contract_status,
        "deployment_permission_status": deployment_permission_status,
        "incumbent_can_deploy": incumbent_can_deploy,
        "no_op_in_portfolio": no_op_in_portfolio,
        "all_rounds_diagnostic": all_rounds_diagnostic,
        "anchor_fallback": anchor_fallback,
        "artifacts": artifacts or {},
    }


def check_completion_contract(manifest: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Strict v2.1 completion contract (section 13 + 19).

    A formal run may only claim ``completed`` when it has executed at least one
    real scientific experiment, produced a deployable candidate, respected the
    hard wall-clock budget, and passed data/metric/deployment contracts.
    """
    missing: list[str] = []

    if not str(manifest.get("orchestrator_version", "")).startswith("2"):
        missing.append("orchestrator_version must start with 2")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        missing.append("manifest_version != afac_v2_run_manifest_v1")
    if manifest.get("trajectory_version") != TRAJECTORY_VERSION:
        missing.append("trajectory_version != afac_v2_trajectory_v1")
    if not manifest.get("execution_id"):
        missing.append("execution_id missing")
    if not manifest.get("code_commit"):
        missing.append("code_commit missing")
    if not manifest.get("planner_mode"):
        missing.append("planner_mode missing")

    artifacts = manifest.get("artifacts", {}) or {}
    required = SMOKE_REQUIRED_ARTIFACTS if smoke else FORMAL_REQUIRED_ARTIFACTS
    for key in required:
        if key not in artifacts:
            missing.append(f"artifact missing: {key}")

    gates = manifest.get("gates", {}) or {}
    for gate in ("metric_semantics_gate", "data_intelligence", "validation_reality", "test_truth_guard", "frozen_asset_hash"):
        if gates.get(gate) is not True:
            missing.append(f"gate not passed: {gate}")

    if manifest.get("llm_calls_count", 0) < (2 if manifest.get("llm_required") else 0):
        missing.append("llm_calls_count below contract")
    if manifest.get("cache_status") == "legacy_final_result_reuse":
        missing.append("v1 final trajectory/stop decision reused")
    if manifest.get("uses_test_truth"):
        missing.append("test truth used")
    if smoke and manifest.get("deployment_generated"):
        missing.append("smoke must not generate deployment")

    # Data / budget / deployment permission contracts
    if manifest.get("data_contract_status") != "passed":
        missing.append("data_contract_status not passed")
    if manifest.get("budget_contract_status") != "passed":
        missing.append("budget_contract_status not passed")
    if not smoke and manifest.get("deployment_permission_status") != "passed":
        missing.append("deployment_permission_status not passed")

    # Scientific execution reality
    if not smoke:
        anchor_fallback = bool(manifest.get("anchor_fallback"))
        if manifest.get("scientific_attempts_used", 0) < 1 and not anchor_fallback:
            missing.append("no scientific attempt executed and no anchor fallback")
        if manifest.get("effective_scientific_rounds", 0) < 1 and not anchor_fallback:
            missing.append("effective_scientific_rounds zero and no anchor fallback")
        if manifest.get("all_rounds_diagnostic", True) and not anchor_fallback:
            missing.append("all rounds were diagnostic-only")
        if manifest.get("no_op_in_portfolio"):
            missing.append("no-op candidate entered portfolio")
        if not manifest.get("incumbent_can_deploy", False) and not anchor_fallback:
            missing.append("incumbent lacks deployment permission")
        if not manifest.get("deployment_generated", False) and not anchor_fallback:
            missing.append("deployment artifact not generated")
        wall = manifest.get("wall_clock_seconds", 0.0)
        max_wall = manifest.get("max_wall_clock_seconds", 0.0)
        if wall > max_wall + 5.0:
            missing.append(f"wall_clock_seconds {wall:.1f} exceeds max {max_wall:.1f} + 5s")

    return {"status": "passed" if not missing else "failed", "missing": missing}
