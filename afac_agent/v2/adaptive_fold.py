# -*- coding: utf-8 -*-
"""AFAC v2 adaptive fold policy and budget-aware validation.

Replaces fixed five-fold validation with a budget-aware fidelity ladder:

    F0_DETERMINISTIC (0 folds: static audit / cached replay / candidate recall
                     / metric audit / no-op audit)
    F1_SCREEN       (B2: 2 folds; B1: 1-2 folds — screening only, never a
                     final deployment conclusion)
    F2_CONFIRM      (3 folds — main confirmation level; candidates may be
                     promoted to incumbent)
    F3_FULL_CV      (5 folds — rare, only for final candidates when budget
                     and uncertainty justify it; never the default)
    F4_DEPLOYMENT   (full training data -> test inference -> deployment audit)

Canonical folds are FIXED: a hash-stable 5-fold master assignment, and every
fidelity selects a fixed prefix subset ([0,1] / [0,1,2] / [0,1,2,3,4]).
Nothing is ever re-randomized per round, so fold results stay comparable
across rounds and across candidates.  Parent comparisons are always paired
on identical folds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from ..research.event_store import stable_hash

FOLD_POLICY_VERSION = "afac_v2_adaptive_fold_v1"
MASTER_FOLD_COUNT = 5
CANONICAL_SEED = 2026


class FoldFidelity(IntEnum):
    F0_DETERMINISTIC = 0
    F1_SCREEN = 1
    F2_CONFIRM = 2
    F3_FULL_CV = 3
    F4_DEPLOYMENT = 4


FIDELITY_FOLD_IDS = {
    FoldFidelity.F0_DETERMINISTIC: [],
    FoldFidelity.F1_SCREEN: [0, 1],
    FoldFidelity.F2_CONFIRM: [0, 1, 2],
    FoldFidelity.F3_FULL_CV: [0, 1, 2, 3, 4],
}

# B1 may screen with a single fold (cheap triage only).
B1_SCREEN_FOLD_IDS = [0]


# --------------------------------------------------------------------------- canonical folds


@dataclass
class CanonicalFolds:
    """Hash-stable master fold assignment with fixed prefix subsets."""

    uids: list[str]
    folds: np.ndarray  # values 0..MASTER_FOLD_COUNT-1, aligned to uids
    fold_hash: str

    @classmethod
    def build(cls, uids: list[str], *, seed: int = CANONICAL_SEED, stratify_bins: np.ndarray | None = None) -> "CanonicalFolds":
        """Deterministic assignment: sorted-uid hash order, round-robin per bin."""
        uids = [str(u) for u in uids]
        order = sorted(range(len(uids)), key=lambda i: stable_hash({"uid": uids[i], "seed": seed}))
        folds = np.zeros(len(uids), dtype=np.int64)
        if stratify_bins is not None and len(stratify_bins) == len(uids):
            for b in np.unique(stratify_bins):
                idx = [i for i in order if stratify_bins[i] == b]
                for pos, i in enumerate(idx):
                    folds[i] = pos % MASTER_FOLD_COUNT
        else:
            for pos, i in enumerate(order):
                folds[i] = pos % MASTER_FOLD_COUNT
        fold_hash = stable_hash({"fold_policy": FOLD_POLICY_VERSION, "uids": uids, "folds": folds.tolist()})
        return cls(uids=uids, folds=folds, fold_hash=fold_hash)

    def fold_ids_for(self, fidelity: FoldFidelity, *, task: str = "B2") -> list[int]:
        if fidelity == FoldFidelity.F1_SCREEN and task == "B1":
            return list(B1_SCREEN_FOLD_IDS)
        return list(FIDELITY_FOLD_IDS.get(fidelity, []))

    def masks(self, fold_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (fit_mask, eval_mask) for one canonical fold."""
        eval_mask = self.folds == fold_id
        return ~eval_mask, eval_mask


@dataclass
class FoldPlan:
    """One experiment's fold plan, recorded in manifests."""

    task: str
    fidelity: int
    fold_count: int
    selected_fold_ids: list[int]
    canonical_fold_hash: str
    fold_policy_version: str = FOLD_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_fold_plan(canonical: CanonicalFolds, fidelity: FoldFidelity, *, task: str = "B2") -> FoldPlan:
    ids = canonical.fold_ids_for(fidelity, task=task)
    return FoldPlan(
        task=task,
        fidelity=int(fidelity),
        fold_count=len(ids),
        selected_fold_ids=ids,
        canonical_fold_hash=canonical.fold_hash,
    )


# --------------------------------------------------------------------------- paired comparison


def paired_fold_comparison(
    *,
    candidate_id: str,
    parent_id: str,
    selected_fold_ids: list[int],
    candidate_fold_metrics: dict[int, float],
    parent_fold_metrics: dict[int, float],
    recomputed_parent_folds: list[int] | None = None,
) -> dict[str, Any]:
    """Pair candidate and parent on IDENTICAL folds only.

    Comparing a candidate's 2-fold mean against a parent's historical 5-fold
    mean is forbidden: when the parent lacks a fold the comparison is marked
    not comparable unless that fold was recomputed (or loaded from a safe
    cache) and passed via ``recomputed_parent_folds``.
    """
    recomputed = set(recomputed_parent_folds or [])
    missing = [f for f in selected_fold_ids if f not in parent_fold_metrics]
    comparable = not missing or all(f in recomputed for f in missing)
    deltas: dict[str, float] = {}
    for f in selected_fold_ids:
        if f in candidate_fold_metrics and f in parent_fold_metrics:
            deltas[str(f)] = float(candidate_fold_metrics[f]) - float(parent_fold_metrics[f])
    values = list(deltas.values())
    return {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "selected_fold_ids": list(selected_fold_ids),
        "candidate_fold_metrics": {str(k): float(v) for k, v in candidate_fold_metrics.items()},
        "parent_fold_metrics": {str(k): float(v) for k, v in parent_fold_metrics.items()},
        "fold_deltas": deltas,
        "mean_delta": float(np.mean(values)) if values else 0.0,
        "worst_fold_delta": float(min(values)) if values else 0.0,
        "positive_fold_count": sum(1 for v in values if v > 0),
        "missing_parent_folds": missing,
        "recomputed_parent_folds": sorted(recomputed),
        "comparable": comparable and not missing,
    }


# --------------------------------------------------------------------------- promotion rules


@dataclass
class PromotionRules:
    """Configurable promotion thresholds (never hard-coded constants)."""

    min_mean_gain: float = 0.0
    min_positive_folds: int = 1
    severe_negative_fold_delta: float = -0.05
    require_target_bucket_improvement: bool = True
    forbid_no_op: bool = True
    b1_min_explainable_bucket_gain: float = 0.0


DEFAULT_B2_RULES = PromotionRules()
DEFAULT_B1_RULES = PromotionRules(severe_negative_fold_delta=-0.03, min_positive_folds=2)


def evaluate_promotion(
    comparison: dict[str, Any],
    *,
    rules: PromotionRules,
    target_bucket_gain: float,
    rescue: int,
    damage: int,
    prediction_changed: bool,
    no_op: bool,
    expected_information_gain: float,
    remaining_seconds: float,
    required_seconds: float,
) -> dict[str, Any]:
    """Screen(2-fold) -> confirm(3-fold) promotion decision."""
    reasons: list[str] = []
    if not comparison["comparable"]:
        reasons.append("fold comparison not comparable")
    if comparison["mean_delta"] <= rules.min_mean_gain:
        reasons.append(f"mean_delta {comparison['mean_delta']:.5f} <= {rules.min_mean_gain}")
    if comparison["positive_fold_count"] < rules.min_positive_folds:
        reasons.append(f"positive_fold_count {comparison['positive_fold_count']} < {rules.min_positive_folds}")
    if comparison["worst_fold_delta"] < rules.severe_negative_fold_delta:
        reasons.append(f"worst_fold_delta {comparison['worst_fold_delta']:.5f} severely negative")
    if rules.require_target_bucket_improvement and target_bucket_gain <= 0:
        reasons.append("no target-bucket improvement")
    if rules.forbid_no_op and no_op:
        reasons.append("no-op experiment")
    if not prediction_changed:
        reasons.append("predictions unchanged")
    if remaining_seconds < required_seconds:
        reasons.append("insufficient remaining budget for confirm + deployment reserve")
    promote = not reasons
    return {
        "decision": "promote_to_confirm" if promote else "hold",
        "reasons": reasons,
        "net_rescue_damage": rescue - damage,
        "expected_information_gain": expected_information_gain,
    }


# --------------------------------------------------------------------------- full-CV trigger


def should_trigger_full_cv(
    *,
    fold_metrics_3fold: dict[int, float],
    incumbent_delta: float,
    candidate_is_best: bool,
    could_replace_anchor: bool,
    candidates_indistinguishable: bool,
    remaining_seconds: float,
    estimated_5fold_runtime: float,
    deployment_reserve_seconds: float,
    safety_margin_seconds: float,
    variance_threshold: float = 0.02,
    boundary_threshold: float = 0.005,
) -> dict[str, Any]:
    """5-fold is never the default; it needs budget AND an uncertainty reason."""
    reasons: list[str] = []
    values = list(fold_metrics_3fold.values())
    variance = float(np.var(values)) if values else 0.0
    uncertainty = variance > variance_threshold
    near_boundary = abs(incumbent_delta) < boundary_threshold
    if not (uncertainty or near_boundary or could_replace_anchor or (candidates_indistinguishable and candidate_is_best)):
        reasons.append("no uncertainty justification (variance/boundary/anchor/indistinguishable)")
    if not candidate_is_best:
        reasons.append("candidate is not the current best")
    required = estimated_5fold_runtime + deployment_reserve_seconds + safety_margin_seconds
    if remaining_seconds <= required:
        reasons.append(f"budget insufficient: remaining {remaining_seconds:.0f}s <= required {required:.0f}s")
    return {"trigger_full_cv": not reasons, "reasons": reasons, "fold_variance": variance, "incumbent_delta": incumbent_delta}


# --------------------------------------------------------------------------- runtime estimator


class FoldRuntimeEstimator:
    """Tracks per-fold runtimes and predicts multi-fold costs."""

    def __init__(self, *, default_per_fold_seconds: float = 30.0, deployment_reserve_seconds: float = 300.0, safety_margin_seconds: float = 120.0) -> None:
        self.default_per_fold_seconds = default_per_fold_seconds
        self.deployment_reserve_seconds = deployment_reserve_seconds
        self.safety_margin_seconds = safety_margin_seconds
        self.per_fold_runtime_history: list[float] = []

    def record_fold_runtime(self, seconds: float) -> None:
        self.per_fold_runtime_history.append(float(seconds))

    @property
    def estimated_next_fold_runtime(self) -> float:
        if not self.per_fold_runtime_history:
            return self.default_per_fold_seconds
        return float(np.mean(self.per_fold_runtime_history))

    def estimated_runtime(self, fold_count: int) -> float:
        return self.estimated_next_fold_runtime * max(0, fold_count)

    def can_afford(self, fold_count: int, remaining_seconds: float, *, include_deployment_reserve: bool = True) -> bool:
        required = self.estimated_runtime(fold_count) + self.safety_margin_seconds
        if include_deployment_reserve:
            required += self.deployment_reserve_seconds
        return remaining_seconds > required

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_fold_runtime_history": self.per_fold_runtime_history,
            "estimated_next_fold_runtime": self.estimated_next_fold_runtime,
            "estimated_2fold_runtime": self.estimated_runtime(2),
            "estimated_3fold_runtime": self.estimated_runtime(3),
            "estimated_5fold_runtime": self.estimated_runtime(5),
            "deployment_reserve_seconds": self.deployment_reserve_seconds,
            "safety_margin_seconds": self.safety_margin_seconds,
        }


# --------------------------------------------------------------------------- stagnation semantics


def no_improvement_action(no_improvement_rounds: int) -> str:
    """No-improvement rounds never directly stop the agent."""
    if no_improvement_rounds <= 0:
        return "continue"
    if no_improvement_rounds == 1:
        return "revise_proposal_or_switch_operator_family"
    return "switch_problem_or_global_explore"


def stop_decision(
    *,
    no_improvement_rounds: int,
    operator_families_tried: list[str],
    problem_nodes_tried: list[str],
    global_explore_attempted: bool,
    high_roi_routes_remaining: bool,
    m5_admissible_routes_remaining: bool,
    remaining_seconds: float,
    deployment_reserve_entered: bool,
) -> dict[str, Any]:
    """Stopping requires global explore attempted + no viable routes, or the
    deployment reserve being entered."""
    allow_stop = (global_explore_attempted and not high_roi_routes_remaining and not m5_admissible_routes_remaining) or deployment_reserve_entered
    return {
        "allow_stop": allow_stop,
        "no_improvement_rounds": no_improvement_rounds,
        "operator_families_tried": list(operator_families_tried),
        "problem_nodes_tried": list(problem_nodes_tried),
        "global_explore_attempted": global_explore_attempted,
        "high_roi_routes_remaining": high_roi_routes_remaining,
        "m5_admissible_routes_remaining": m5_admissible_routes_remaining,
        "remaining_seconds": remaining_seconds,
        "deployment_possible": not deployment_reserve_entered,
        "required_action": "stop" if allow_stop else no_improvement_action(no_improvement_rounds),
    }


# --------------------------------------------------------------------------- policy facade


@dataclass
class AdaptiveFoldPolicy:
    """Per-task adaptive fold policy facade."""

    task: str = "B2"
    rules: PromotionRules = field(default_factory=PromotionRules)
    estimator: FoldRuntimeEstimator = field(default_factory=FoldRuntimeEstimator)

    @classmethod
    def for_task(cls, task: str, **kwargs: Any) -> "AdaptiveFoldPolicy":
        rules = DEFAULT_B1_RULES if task == "B1" else DEFAULT_B2_RULES
        return cls(task=task, rules=rules, **kwargs)

    def initial_fidelity(self) -> FoldFidelity:
        return FoldFidelity.F1_SCREEN

    def default_screen_folds(self) -> int:
        return 1 if self.task == "B1" else 2

    def confirm_folds(self) -> int:
        return 3

    def full_cv_folds(self) -> int:
        return 5

    def screen_is_final_evidence(self) -> bool:
        """Screen results are never the sole final evidence (B1 explicitly)."""
        return False

    def plan(self, canonical: CanonicalFolds, fidelity: FoldFidelity) -> FoldPlan:
        return build_fold_plan(canonical, fidelity, task=self.task)

    def promote(self, comparison: dict[str, Any], *, target_bucket_gain: float, rescue: int, damage: int, prediction_changed: bool, no_op: bool, expected_information_gain: float, remaining_seconds: float) -> dict[str, Any]:
        required = self.estimator.estimated_runtime(self.confirm_folds()) + self.estimator.deployment_reserve_seconds + self.estimator.safety_margin_seconds
        return evaluate_promotion(
            comparison,
            rules=self.rules,
            target_bucket_gain=target_bucket_gain,
            rescue=rescue,
            damage=damage,
            prediction_changed=prediction_changed,
            no_op=no_op,
            expected_information_gain=expected_information_gain,
            remaining_seconds=remaining_seconds,
            required_seconds=required,
        )
