# -*- coding: utf-8 -*-
"""Planner policy loading and hard-safety validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_EVIDENCE_LEVELS = {
    "full_anchor_oof_comparison",
    "verified_self_contained_oof",
    "unverified_self_contained_oof",
    "signal_evidence",
    "expert_scope",
    "structure_only",
    "artifact_integrity",
}
REQUIRED_POLICY_KEYS = {
    "policy_version",
    "evidence_rank",
    "action_priority",
    "risk_policy",
    "budget_policy",
    "duplicate_policy",
    "missing_input_policy",
    "human_approval_policy",
    "branch_reopen_policy",
    "promotion_policy_refs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_policy(path: str | Path) -> tuple[dict[str, Any], str]:
    policy_path = Path(path)
    payload = load_json(policy_path)
    missing = sorted(REQUIRED_POLICY_KEYS - set(payload))
    if missing:
        raise ValueError(f"planner policy missing required keys: {missing}")
    evidence_rank = payload.get("evidence_rank", [])
    if not isinstance(evidence_rank, list):
        raise ValueError("planner policy evidence_rank must be a list")
    unknown_levels = sorted(set(evidence_rank) - REQUIRED_EVIDENCE_LEVELS)
    missing_levels = sorted(REQUIRED_EVIDENCE_LEVELS - set(evidence_rank))
    if unknown_levels:
        raise ValueError(f"planner policy references unknown evidence levels: {unknown_levels}")
    if missing_levels:
        raise ValueError(f"planner policy missing evidence levels: {missing_levels}")
    human = payload.get("human_approval_policy", {})
    if human.get("online_submission_requires_human_approval") is not True:
        raise ValueError("online_submission_requires_human_approval must be true")
    if human.get("training_requires_human_approval") is not True:
        raise ValueError("training_requires_human_approval must be true")
    if human.get("prediction_artifact_requires_human_approval") is not True:
        raise ValueError("prediction_artifact_requires_human_approval must be true")
    branch = payload.get("branch_reopen_policy", {})
    if branch.get("auto_reopen_closed_branch") is not False:
        raise ValueError("closed branch auto reopen must be false")
    if branch.get("requires_human_approval") is not True:
        raise ValueError("closed branch reopen requires_human_approval must be true")
    return payload, sha256_file(policy_path)
