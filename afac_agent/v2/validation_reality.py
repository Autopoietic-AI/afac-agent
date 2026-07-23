# -*- coding: utf-8 -*-
"""AFAC v2.0 validation reality manager.

Tracks offline/online anchors, submission identities, offline-online gap math,
panel calibration, and deployment confidence.  Validates that a task's required
panel set is complete and free of duplicate membership before any panel-based
conclusion is trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from afac_agent.research.event_store import stable_hash

B1_STANDARD_PANEL = "B1_STANDARD_PANEL"
B1_DEGREE_MATCHED_PANEL = "B1_DEGREE_MATCHED_PANEL"
B1_PROPENSITY_MATCHED_PANEL = "B1_PROPENSITY_MATCHED_PANEL"
B1_LOW_DEGREE_PANEL = "B1_LOW_DEGREE_PANEL"
B1_TEST_LIKE_PANEL = "B1_TEST_LIKE_PANEL"
B1_WORST_ENVIRONMENT_PANEL = "B1_WORST_ENVIRONMENT_PANEL"

B2_STANDARD_PANEL = "B2_STANDARD_PANEL"
B2_SHORT_HISTORY_PANEL = "B2_SHORT_HISTORY_PANEL"
B2_LONG_HISTORY_PANEL = "B2_LONG_HISTORY_PANEL"
B2_HISTORY_TARGET_PANEL = "B2_HISTORY_TARGET_PANEL"
B2_NOVEL_TARGET_PANEL = "B2_NOVEL_TARGET_PANEL"
B2_LONG_TAIL_PANEL = "B2_LONG_TAIL_PANEL"
B2_COLD_USER_PANEL = "B2_COLD_USER_PANEL"
B2_TEST_LIKE_PANEL = "B2_TEST_LIKE_PANEL"
B2_TOP10_BOUNDARY_PANEL = "B2_TOP10_BOUNDARY_PANEL"

B1_REQUIRED_PANELS: list[str] = [
    B1_STANDARD_PANEL,
    B1_DEGREE_MATCHED_PANEL,
    B1_PROPENSITY_MATCHED_PANEL,
    B1_LOW_DEGREE_PANEL,
    B1_TEST_LIKE_PANEL,
    B1_WORST_ENVIRONMENT_PANEL,
]

B2_REQUIRED_PANELS: list[str] = [
    B2_STANDARD_PANEL,
    B2_SHORT_HISTORY_PANEL,
    B2_LONG_HISTORY_PANEL,
    B2_HISTORY_TARGET_PANEL,
    B2_NOVEL_TARGET_PANEL,
    B2_LONG_TAIL_PANEL,
    B2_COLD_USER_PANEL,
    B2_TEST_LIKE_PANEL,
    B2_TOP10_BOUNDARY_PANEL,
]

REQUIRED_PANELS: dict[str, list[str]] = {"B1": B1_REQUIRED_PANELS, "B2": B2_REQUIRED_PANELS}


@dataclass
class OfflineAnchor:
    """Offline evaluation anchor: a metric summary tied to a fold identity."""

    anchor_id: str
    fold_identity: str
    metric_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class OnlineAnchor:
    """Online evaluation anchor: a live-traffic score with its type."""

    anchor_id: str
    online_score: float
    score_type: str


@dataclass
class SubmissionIdentity:
    """Identity of a submission artifact for reproducibility checks."""

    file_path: str
    sha256: str
    model_id: str
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)


def offline_online_gap(offline_score: float, online_score: float) -> dict[str, float]:
    """Absolute and relative (to offline) gap between offline and online scores."""
    absolute_gap = float(online_score) - float(offline_score)
    relative_gap = absolute_gap / abs(float(offline_score)) if offline_score != 0 else 0.0
    return {"absolute_gap": absolute_gap, "relative_gap": relative_gap}


def panel_calibration(
    panel_id: str,
    offline_panel_scores: dict[str, float],
    online_signal: float | None = None,
    environment_weight: float = 1.0,
) -> dict[str, Any]:
    """Build a calibrated panel record combining offline scores and online signal."""
    scores = {str(k): float(v) for k, v in offline_panel_scores.items()}
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0
    calibrated_score = float(environment_weight) * mean_score
    if online_signal is not None:
        calibrated_score = (calibrated_score + float(online_signal)) / 2.0
    return {
        "panel_id": panel_id,
        "offline_panel_scores": scores,
        "online_signal": online_signal,
        "environment_weight": float(environment_weight),
        "calibrated_score": calibrated_score,
    }


def deployment_confidence(
    panel_scores: dict[str, float],
    gap: float,
    fold_stability: float,
) -> float:
    """Deployment confidence in [0, 1].

    Formula:
        confidence = agreement * gap_factor * stability_factor
    where
        agreement        = mean of clamped panel scores in [0, 1]
                           (cross-panel agreement on model quality);
        gap_factor       = exp(-|gap|) — confidence decays exponentially with
                           the offline-online gap magnitude;
        stability_factor = clamp(fold_stability, 0, 1).
    All three factors lie in [0, 1], so the product does too, and the score is
    monotonically decreasing in |gap|.
    """
    scores = [min(1.0, max(0.0, float(v))) for v in panel_scores.values()]
    agreement = sum(scores) / len(scores) if scores else 0.0
    gap_factor = math.exp(-abs(float(gap)))
    stability_factor = min(1.0, max(0.0, float(fold_stability)))
    confidence = agreement * gap_factor * stability_factor
    return min(1.0, max(0.0, confidence))


def _membership_hash(member_ids: list[Any]) -> str:
    """Local membership hash (stable hash of sorted member ids)."""
    return stable_hash(sorted(str(m) for m in member_ids))


class ValidationRealityManager:
    """Validates that a task's required panel set is complete and duplicate-free."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.required_panels = list(REQUIRED_PANELS.get(task, []))

    def validate_panel_set(self, panel_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Check panel records against the required panel set.

        Returns missing required panels, duplicate panels (identical membership
        hash), per-panel membership/distribution audit fields, and an overall
        status: ``verified`` | ``panels_missing`` | ``duplicate_panels``.
        Duplicate panels never count toward satisfying the required set.
        """
        panels: list[dict[str, Any]] = []
        duplicate_pairs: list[dict[str, str]] = []
        seen_hashes: dict[str, str] = {}  # membership_hash -> first panel_id
        present_independent: dict[str, str] = {}  # panel_id -> membership_hash

        for record in panel_records:
            panel = dict(record)
            panel_id = str(panel.get("panel_id", ""))
            member_ids = panel.get("member_ids")
            if member_ids is not None:
                members = [str(m) for m in member_ids]
                mhash = _membership_hash(members)
                panel["membership_hash"] = mhash
                panel["size"] = len(members)
            else:
                mhash = str(panel.get("membership_hash", ""))
            if mhash and mhash in seen_hashes:
                panel["status"] = "duplicate_panel"
                panel["duplicate_of"] = seen_hashes[mhash]
                duplicate_pairs.append({"panel_id": panel_id, "duplicate_of": seen_hashes[mhash]})
            else:
                panel.setdefault("status", "independent_panel")
                if mhash:
                    seen_hashes[mhash] = panel_id
                if panel_id:
                    present_independent[panel_id] = mhash
            panels.append(panel)

        missing = [pid for pid in self.required_panels if pid not in present_independent]
        if missing:
            status = "panels_missing"
        elif duplicate_pairs:
            status = "duplicate_panels"
        else:
            status = "verified"
        return {
            "task": self.task,
            "status": status,
            "missing_panels": missing,
            "duplicate_pairs": duplicate_pairs,
            "independent_panel_count": len(present_independent),
            "required_panels": self.required_panels,
            "panels": panels,
        }
