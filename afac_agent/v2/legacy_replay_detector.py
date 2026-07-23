# -*- coding: utf-8 -*-
"""AFAC v2.0 legacy replay detector.

Detects v1 deterministic-loop artifacts masquerading as v2 runs.  A detected
replay must be marked ``invalid_replay``: it may never return ``completed``,
never generate a v2 deployment, never overwrite a v2 run directory, and never
enter a v2 portfolio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..research.event_store import load_json

LEGACY_MANIFEST_PREFIXES = ("b1_", "b2_", "a1_", "a2_")
V2_MANIFEST_VERSION = "afac_v2_run_manifest_v1"
V2_TRAJECTORY_VERSION = "afac_v2_trajectory_v1"


def detect_legacy_replay(
    manifest: dict[str, Any] | None = None,
    *,
    manifest_path: str | Path | None = None,
    report_name: str | None = None,
    execution_id: str | None = None,
    known_v1_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Classify a run manifest as ``ok`` (genuine v2) or ``invalid_replay``.

    Detection conditions (any one is sufficient):
    - manifest_version starts with a legacy prefix (b1_/b2_/a1_/a2_);
    - trajectory_version starts with a legacy prefix;
    - report name contains ``CLOSED_LOOP_V1``;
    - orchestrator_version / planner_mode / llm_calls_count missing;
    - problem node / M6B / M6C artifacts missing;
    - execution_id equals a known historical v1 run_id.
    """
    if manifest is None:
        if manifest_path is None:
            raise ValueError("manifest or manifest_path required")
        manifest = load_json(manifest_path)
    reasons: list[str] = []

    manifest_version = str(manifest.get("manifest_version") or "")
    trajectory_version = str(manifest.get("trajectory_version") or "")
    if manifest_version and manifest_version != V2_MANIFEST_VERSION:
        if manifest_version.startswith(LEGACY_MANIFEST_PREFIXES):
            reasons.append(f"legacy manifest_version: {manifest_version}")
        elif not manifest_version.startswith("afac_v2"):
            reasons.append(f"non-v2 manifest_version: {manifest_version}")
    if trajectory_version and trajectory_version != V2_TRAJECTORY_VERSION:
        if trajectory_version.startswith(LEGACY_MANIFEST_PREFIXES):
            reasons.append(f"legacy trajectory_version: {trajectory_version}")
    if report_name and "CLOSED_LOOP_V1" in report_name:
        reasons.append(f"legacy report name: {report_name}")
    if "orchestrator_version" not in manifest:
        reasons.append("orchestrator_version missing")
    else:
        version = str(manifest.get("orchestrator_version") or "")
        if not version.startswith("2"):
            reasons.append(f"orchestrator_version not v2: {version}")
    if "planner_mode" not in manifest:
        reasons.append("planner_mode missing")
    if "llm_calls_count" not in manifest:
        reasons.append("llm_calls_count missing")
    artifacts = manifest.get("artifacts", {})
    for key, label in (("problem_node", "problem node"), ("m6b_proposal", "M6B"), ("m6c_critic", "M6C")):
        if isinstance(artifacts, dict) and key not in artifacts and not manifest.get(f"has_{key}", False):
            reasons.append(f"{label} artifact missing")
    if execution_id and known_v1_run_ids and execution_id in known_v1_run_ids:
        reasons.append(f"execution_id equals historical v1 run_id: {execution_id}")

    is_replay = bool(reasons)
    return {
        "status": "invalid_replay" if is_replay else "ok",
        "is_legacy_replay": is_replay,
        "reasons": reasons,
        "forbidden_actions": [
            "return_completed",
            "generate_v2_deployment",
            "overwrite_v2_run_directory",
            "enter_v2_portfolio",
        ]
        if is_replay
        else [],
    }
