# -*- coding: utf-8 -*-
"""Deterministic duplicate/materialization index from explicit inputs."""

from __future__ import annotations

from collections import Counter
from typing import Any


class DuplicateIndex:
    def __init__(self, *, feedbacks: list[dict[str, Any]], history: dict[str, Any], project_state: dict[str, Any]):
        self.feedbacks = feedbacks
        self.history = history
        self.project_state = project_state
        self.execution_counts = Counter(
            str(item.get("execution_identity_hash", ""))
            for item in feedbacks
            if item.get("execution_identity_hash")
        )
        self.feedback_counts = Counter(
            str(item.get("feedback_id", ""))
            for item in feedbacks
            if item.get("feedback_id")
        )
        self.closed_branches = self._closed_branches()

    def _closed_branches(self) -> set[str]:
        result = {str(item) for item in self.project_state.get("closed_branches", [])}
        for experiment in self.history.get("experiments", []):
            decision = str(experiment.get("decision", "")).upper()
            if "CLOSE" in decision:
                result.add(str(experiment.get("version", "")))
                result.add(str(experiment.get("layer", "")))
        return {item for item in result if item}

    def duplicate_reason_codes(self) -> list[str]:
        reasons: list[str] = []
        if any(count > 1 for count in self.execution_counts.values()):
            reasons.append("duplicate_execution")
        if any(count > 1 for count in self.feedback_counts.values()):
            reasons.append("duplicate_feedback")
        return reasons

    def reference_verified(self) -> bool:
        for item in self.feedbacks:
            if item.get("feedback_kind") != "candidate_replay":
                continue
            evidence = item.get("evidence", {})
            if (
                evidence.get("semantic_replay_pass") is True
                and evidence.get("byte_replay_pass") is True
                and int(evidence.get("replay_to_champion_diff_count", 1)) == 0
            ):
                return True
        return False

    def candidate_materialized(self) -> bool:
        return self.reference_verified()

    def branch_closed(self, branch_id: str) -> bool:
        return branch_id in self.closed_branches
