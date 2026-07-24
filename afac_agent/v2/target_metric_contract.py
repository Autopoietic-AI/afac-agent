# -*- coding: utf-8 -*-
"""AFAC v2.1 target panel metric contract.

Every Problem Node declares which panel and metric it targets.  Promotion is
only valid when candidate and parent are compared on the same fold, same panel,
same metric, same K, and same evaluator version.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..research.event_store import stable_hash


@dataclass
class TargetMetricContract:
    target_panel_id: str
    target_metric_name: str
    target_k: int
    target_direction: str
    parent_target_metric: float | None
    candidate_target_metric: float | None
    target_metric_delta: float | None
    evaluator_version: str
    fold_hash: str
    status: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_panel_id": self.target_panel_id,
            "target_metric_name": self.target_metric_name,
            "target_k": self.target_k,
            "target_direction": self.target_direction,
            "parent_target_metric": self.parent_target_metric,
            "candidate_target_metric": self.candidate_target_metric,
            "target_metric_delta": self.target_metric_delta,
            "evaluator_version": self.evaluator_version,
            "fold_hash": self.fold_hash,
            "status": self.status,
            "reasons": self.reasons,
        }

    def contract_hash(self) -> str:
        return stable_hash(self.to_dict())


def evaluate_target_metric_contract(
    *,
    target_panel_id: str,
    target_metric_name: str,
    target_k: int,
    parent_target_metric: float | None,
    candidate_target_metric: float | None,
    evaluator_version: str,
    fold_hash: str,
    parent_fold_hash: str,
    target_direction: str = "maximize",
) -> TargetMetricContract:
    reasons: list[str] = []
    if parent_target_metric is None:
        reasons.append("parent_target_metric missing")
    if candidate_target_metric is None:
        reasons.append("candidate_target_metric missing")
    if fold_hash != parent_fold_hash:
        reasons.append(f"fold hash mismatch: {fold_hash} != {parent_fold_hash}")
    if not target_panel_id:
        reasons.append("target_panel_id missing")
    if not target_metric_name:
        reasons.append("target_metric_name missing")

    delta: float | None = None
    if parent_target_metric is not None and candidate_target_metric is not None:
        delta = float(candidate_target_metric) - float(parent_target_metric)

    status = "blocked_missing_parent_target_metric" if parent_target_metric is None else ("passed" if not reasons else "blocked")
    return TargetMetricContract(
        target_panel_id=target_panel_id,
        target_metric_name=target_metric_name,
        target_k=target_k,
        target_direction=target_direction,
        parent_target_metric=parent_target_metric,
        candidate_target_metric=candidate_target_metric,
        target_metric_delta=delta,
        evaluator_version=evaluator_version,
        fold_hash=fold_hash,
        status=status,
        reasons=reasons,
    )
