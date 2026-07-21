# -*- coding: utf-8 -*-
"""Append-only research event store for M6R-A."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

EVENT_VERSION = "m6r_a_v2"


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): stable_payload(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [stable_payload(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        stable_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def rel_ref(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def make_event(
    *,
    event_type: str,
    task: str = "A1",
    experiment_id: str = "",
    execution_identity: str = "",
    branch_id: str = "",
    parent_branch_id: str = "",
    candidate_id: str = "",
    parent_candidate_id: str = "",
    scope_level: str = "global",
    scope_refs: dict[str, Any] | None = None,
    problem_ids: list[str] | None = None,
    method_ids: list[str] | None = None,
    hypothesis_ids: list[str] | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
    feedback_refs: list[dict[str, Any]] | None = None,
    artifact_refs: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    metrics_summary: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    outcome_status: str = "inconclusive",
    evidence_level: str = "structure_only",
    confidence: str = "low",
    failure_mode: str = "",
    success_mode: str = "",
    branch_effect: str = "none",
    research_priority_effect: str = "none",
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    core = {
        "event_version": EVENT_VERSION,
        "event_type": event_type,
        "task": task,
        "experiment_id": experiment_id,
        "execution_identity": execution_identity,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_candidate_id,
        "scope_level": scope_level,
        "scope_refs": scope_refs or {},
        "problem_ids": problem_ids or [],
        "method_ids": method_ids or [],
        "hypothesis_ids": hypothesis_ids or [],
        "evidence_refs": evidence_refs or [],
        "feedback_refs": feedback_refs or [],
        "artifact_refs": artifact_refs or [],
        "observations": observations or {},
        "metrics_summary": metrics_summary or {},
        "decision": decision or {},
        "outcome_status": outcome_status,
        "evidence_level": evidence_level,
        "confidence": confidence,
        "failure_mode": failure_mode,
        "success_mode": success_mode,
        "branch_effect": branch_effect,
        "research_priority_effect": research_priority_effect,
        "input_hashes": input_hashes or {},
    }
    return {**core, "event_id": stable_hash(core), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


class ResearchEventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{self.path}:{line_no}: event is not object")
            events.append(payload)
        return events

    def append_unique(self, events: list[dict[str, Any]]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = {event.get("event_id") for event in self.load()}
        new_events = [event for event in events if event.get("event_id") not in existing]
        if not new_events:
            return 0
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            for event in new_events:
                file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        return len(new_events)
