# -*- coding: utf-8 -*-
"""Shared helpers for deterministic experiment feedback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FEEDBACK_VERSION = "m4a_v1"
NORMALIZER_VERSION = "m4a_normalizers_v1"
DEFAULT_FEEDBACK_ROOT = "artifacts/feedback_runs"

RECOMMENDATIONS = {
    "accept_as_reference",
    "proceed_to_oof_evaluation",
    "proceed_to_candidate_generation",
    "proceed_to_training",
    "informational_only",
    "reject",
    "blocked",
}

STATUSES = {"completed", "partial", "unavailable", "failed", "blocked"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def validate_adapter_execution_result(payload: dict[str, Any]) -> list[str]:
    required = [
        "tool_name",
        "adapter_id",
        "adapter_version",
        "status",
        "target_problem",
        "execution_mode",
        "read_only",
        "counts_as_experiment_round",
        "mutates_predictions",
        "mutates_project_state",
        "requires_gpu",
        "identity_hash",
        "started_at",
        "finished_at",
        "duration_seconds",
        "command",
        "returncode",
        "input_manifest",
        "input_hashes",
        "metrics",
        "bucket_metrics",
        "class_metrics",
        "artifacts",
        "stdout_log",
        "stderr_log",
        "warnings",
        "failure_reason",
        "missing_inputs",
    ]
    errors: list[str] = []
    for key in required:
        if key not in payload:
            errors.append(f"execution_result.{key}: missing")
    if payload.get("status") not in {
        "completed",
        "waiting_for_input",
        "failed",
        "blocked",
        "duplicate",
        "dry_run",
    }:
        errors.append("execution_result.status: invalid")
    if not isinstance(payload.get("metrics", {}), dict):
        errors.append("execution_result.metrics: must be object")
    return errors


def validate_experiment_feedback(payload: dict[str, Any]) -> list[str]:
    required = [
        "feedback_version",
        "feedback_id",
        "feedback_kind",
        "evaluation_tier",
        "task",
        "tool_name",
        "adapter_id",
        "adapter_version",
        "execution_identity_hash",
        "execution_result_ref",
        "execution_result_hash",
        "status",
        "validity",
        "evidence",
        "metrics",
        "class_metrics",
        "bucket_metrics",
        "candidate_changes",
        "overlap_conflict",
        "runtime",
        "resource_usage",
        "safety",
        "limitations",
        "warnings",
        "recommendation",
    ]
    errors: list[str] = []
    for key in required:
        if key not in payload:
            errors.append(f"feedback.{key}: missing")
    if payload.get("status") not in STATUSES:
        errors.append("feedback.status: invalid")
    if payload.get("recommendation") not in RECOMMENDATIONS:
        errors.append("feedback.recommendation: invalid")
    if not isinstance(payload.get("class_metrics"), dict):
        errors.append("feedback.class_metrics: must be object")
    if not isinstance(payload.get("bucket_metrics"), dict):
        errors.append("feedback.bucket_metrics: must be object")
    return errors


def base_feedback(
    *,
    execution: dict[str, Any],
    execution_result_hash: str,
    execution_result_ref: str,
    feedback_kind: str,
    evaluation_tier: str,
) -> dict[str, Any]:
    status = "completed" if execution.get("status") in {"completed", "duplicate"} else "partial"
    return {
        "feedback_version": FEEDBACK_VERSION,
        "feedback_id": "",
        "feedback_kind": feedback_kind,
        "evaluation_tier": evaluation_tier,
        "task": "A1",
        "tool_name": str(execution.get("tool_name", "")),
        "adapter_id": str(execution.get("adapter_id", "")),
        "adapter_version": str(execution.get("adapter_version", "")),
        "execution_identity_hash": str(execution.get("identity_hash", "")),
        "execution_result_ref": execution_result_ref,
        "execution_result_hash": execution_result_hash,
        "status": status,
        "validity": {
            "status": "valid" if status == "completed" else "limited",
            "audit_success_is_model_gain": False,
            "replay_success_is_new_experiment": False,
        },
        "evidence": {},
        "metrics": {},
        "class_metrics": unavailable("not_computed_by_feedback_contract"),
        "bucket_metrics": unavailable("not_computed_by_feedback_contract"),
        "candidate_changes": unavailable("not_applicable"),
        "overlap_conflict": unavailable("not_computed_by_feedback_contract"),
        "runtime": {
            "duration_seconds": execution.get("duration_seconds", 0.0),
            "returncode": execution.get("returncode"),
        },
        "resource_usage": {
            "requires_gpu": bool(execution.get("requires_gpu", False)),
            "counts_as_experiment_round": bool(
                execution.get("counts_as_experiment_round", True)
            ),
        },
        "safety": {
            "read_only": bool(execution.get("read_only", False)),
            "mutates_predictions": bool(execution.get("mutates_predictions", False)),
            "mutates_project_state": bool(execution.get("mutates_project_state", False)),
            "submission_creating": False,
            "test_truth_metrics_emitted": False,
        },
        "limitations": [
            {
                "status": "unavailable",
                "reason": "test_truth_unavailable",
                "metric_family": "test_truth_dependent_metrics",
            }
        ],
        "warnings": list(execution.get("warnings", [])),
        "recommendation": "informational_only",
    }
