# -*- coding: utf-8 -*-
"""Explicit-input evidence loading for M5A deterministic planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FEEDBACK_KEYS = {
    "feedback_id",
    "feedback_kind",
    "evaluation_tier",
    "task",
    "tool_name",
    "adapter_id",
    "adapter_version",
    "execution_identity_hash",
    "execution_result_hash",
    "status",
    "validity",
    "evidence",
    "metrics",
    "resource_usage",
    "safety",
    "recommendation",
}


@dataclass(frozen=True)
class EvidenceBundle:
    problem_map: dict[str, Any]
    feedbacks: list[dict[str, Any]]
    tool_registry: dict[str, Any]
    project_state: dict[str, Any]
    history: dict[str, Any]
    input_hashes: dict[str, str]
    evidence_refs: list[dict[str, Any]]
    feedback_refs: list[dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON root must be object")
    return payload


def _validate_problem_map(payload: dict[str, Any]) -> None:
    if "analysis_tier" not in payload or "problems" not in payload:
        raise ValueError("problem_map: missing analysis_tier/problems")
    if not isinstance(payload.get("problems"), list):
        raise ValueError("problem_map: problems must be list")


def _validate_feedback(payload: dict[str, Any], *, index: int) -> None:
    missing = sorted(REQUIRED_FEEDBACK_KEYS - set(payload))
    if missing:
        raise ValueError(f"feedback[{index}]: missing required keys {missing}")


def _logical_ref(path: Path, digest: str) -> dict[str, Any]:
    return {
        "name": path.name,
        "sha256": digest,
    }


def missing_paths(
    *,
    problem_map_path: Path,
    feedback_paths: list[Path],
    tool_registry_path: Path,
    project_state_path: Path,
    history_path: Path,
    policy_path: Path,
) -> list[str]:
    missing: list[str] = []
    if not problem_map_path.exists():
        missing.append("problem_map")
    if not feedback_paths:
        missing.append("feedback")
    for index, path in enumerate(feedback_paths):
        if not path.exists():
            missing.append(f"feedback[{index}]")
    for label, path in [
        ("tool_registry", tool_registry_path),
        ("project_state", project_state_path),
        ("history", history_path),
        ("policy", policy_path),
    ]:
        if not path.exists():
            missing.append(label)
    return missing


def load_evidence(
    *,
    problem_map_path: Path,
    feedback_paths: list[Path],
    tool_registry_path: Path,
    project_state_path: Path,
    history_path: Path,
) -> EvidenceBundle:
    problem_hash = sha256_file(problem_map_path)
    registry_hash = sha256_file(tool_registry_path)
    state_hash = sha256_file(project_state_path)
    history_hash = sha256_file(history_path)
    problem_map = _read_json(problem_map_path, label="problem_map")
    _validate_problem_map(problem_map)
    tool_registry = _read_json(tool_registry_path, label="tool_registry")
    project_state = _read_json(project_state_path, label="project_state")
    history = _read_json(history_path, label="history")
    feedbacks: list[dict[str, Any]] = []
    feedback_refs: list[dict[str, Any]] = []
    input_hashes = {
        "problem_map": problem_hash,
        "tool_registry": registry_hash,
        "project_state": state_hash,
        "history": history_hash,
    }
    for index, path in enumerate(feedback_paths):
        digest = sha256_file(path)
        payload = _read_json(path, label=f"feedback[{index}]")
        _validate_feedback(payload, index=index)
        feedbacks.append(payload)
        input_hashes[f"feedback[{index}]"] = digest
        feedback_refs.append(
            {
                "feedback_id": payload.get("feedback_id", ""),
                "tool_name": payload.get("tool_name", ""),
                "evaluation_tier": payload.get("evaluation_tier", ""),
                "sha256": digest,
            }
        )
    evidence_refs = [
        {"kind": "problem_map", **_logical_ref(problem_map_path, problem_hash)},
        {"kind": "tool_registry", **_logical_ref(tool_registry_path, registry_hash)},
        {"kind": "project_state", **_logical_ref(project_state_path, state_hash)},
        {"kind": "history", **_logical_ref(history_path, history_hash)},
    ]
    return EvidenceBundle(
        problem_map=problem_map,
        feedbacks=feedbacks,
        tool_registry=tool_registry,
        project_state=project_state,
        history=history,
        input_hashes=input_hashes,
        evidence_refs=evidence_refs,
        feedback_refs=feedback_refs,
    )
