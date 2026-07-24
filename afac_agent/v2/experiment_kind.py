# -*- coding: utf-8 -*-
"""AFAC v2.1 experiment kind and permission model.

Defines the kind of each experiment and what it is allowed to do.  The
permission model prevents the v2.0 failure where a `candidate_recall_diagnostic`
was deployed as if it were a confirmed scientific candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExperimentKind(str, Enum):
    DETERMINISTIC_DIAGNOSTIC = "DETERMINISTIC_DIAGNOSTIC"
    CACHED_REPLAY = "CACHED_REPLAY"
    SCREEN_EXPERIMENT = "SCREEN_EXPERIMENT"
    CONFIRM_EXPERIMENT = "CONFIRM_EXPERIMENT"
    FULL_CV_EXPERIMENT = "FULL_CV_EXPERIMENT"
    DEPLOYMENT_MODEL = "DEPLOYMENT_MODEL"


@dataclass(frozen=True)
class ExperimentPermission:
    """Static permission contract for an experiment kind."""

    kind: ExperimentKind
    consumes_scientific_round: bool
    can_enter_portfolio: bool
    can_be_incumbent: bool
    can_promote_fidelity: bool
    can_deploy: bool
    requires_training: bool
    requires_oof: bool
    requires_checkpoint: bool
    requires_same_fold_parent: bool
    min_fold_count: int
    max_wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "consumes_scientific_round": self.consumes_scientific_round,
            "can_enter_portfolio": self.can_enter_portfolio,
            "can_be_incumbent": self.can_be_incumbent,
            "can_promote_fidelity": self.can_promote_fidelity,
            "can_deploy": self.can_deploy,
            "requires_training": self.requires_training,
            "requires_oof": self.requires_oof,
            "requires_checkpoint": self.requires_checkpoint,
            "requires_same_fold_parent": self.requires_same_fold_parent,
            "min_fold_count": self.min_fold_count,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
        }


PERMISSIONS: dict[ExperimentKind, ExperimentPermission] = {
    ExperimentKind.DETERMINISTIC_DIAGNOSTIC: ExperimentPermission(
        kind=ExperimentKind.DETERMINISTIC_DIAGNOSTIC,
        consumes_scientific_round=False,
        can_enter_portfolio=False,
        can_be_incumbent=False,
        can_promote_fidelity=False,
        can_deploy=False,
        requires_training=False,
        requires_oof=False,
        requires_checkpoint=False,
        requires_same_fold_parent=False,
        min_fold_count=0,
        max_wall_clock_seconds=120.0,
    ),
    ExperimentKind.CACHED_REPLAY: ExperimentPermission(
        kind=ExperimentKind.CACHED_REPLAY,
        consumes_scientific_round=False,
        can_enter_portfolio=False,
        can_be_incumbent=False,
        can_promote_fidelity=False,
        can_deploy=False,
        requires_training=False,
        requires_oof=False,
        requires_checkpoint=False,
        requires_same_fold_parent=False,
        min_fold_count=0,
        max_wall_clock_seconds=30.0,
    ),
    ExperimentKind.SCREEN_EXPERIMENT: ExperimentPermission(
        kind=ExperimentKind.SCREEN_EXPERIMENT,
        consumes_scientific_round=True,
        can_enter_portfolio=True,
        can_be_incumbent=False,
        can_promote_fidelity=True,
        can_deploy=False,
        requires_training=True,
        requires_oof=True,
        requires_checkpoint=True,
        requires_same_fold_parent=True,
        min_fold_count=1,
        max_wall_clock_seconds=1800.0,
    ),
    ExperimentKind.CONFIRM_EXPERIMENT: ExperimentPermission(
        kind=ExperimentKind.CONFIRM_EXPERIMENT,
        consumes_scientific_round=True,
        can_enter_portfolio=True,
        can_be_incumbent=True,
        can_promote_fidelity=True,
        can_deploy=True,
        requires_training=True,
        requires_oof=True,
        requires_checkpoint=True,
        requires_same_fold_parent=True,
        min_fold_count=3,
        max_wall_clock_seconds=3600.0,
    ),
    ExperimentKind.FULL_CV_EXPERIMENT: ExperimentPermission(
        kind=ExperimentKind.FULL_CV_EXPERIMENT,
        consumes_scientific_round=True,
        can_enter_portfolio=True,
        can_be_incumbent=True,
        can_promote_fidelity=True,
        can_deploy=True,
        requires_training=True,
        requires_oof=True,
        requires_checkpoint=True,
        requires_same_fold_parent=True,
        min_fold_count=5,
        max_wall_clock_seconds=3600.0,
    ),
    ExperimentKind.DEPLOYMENT_MODEL: ExperimentPermission(
        kind=ExperimentKind.DEPLOYMENT_MODEL,
        consumes_scientific_round=False,
        can_enter_portfolio=False,
        can_be_incumbent=False,
        can_promote_fidelity=False,
        can_deploy=True,
        requires_training=False,
        requires_oof=False,
        requires_checkpoint=True,
        requires_same_fold_parent=False,
        min_fold_count=0,
        max_wall_clock_seconds=600.0,
    ),
}


def permission_for_kind(kind: str | ExperimentKind) -> ExperimentPermission:
    if isinstance(kind, ExperimentKind):
        return PERMISSIONS[kind]
    key = str(kind).upper()
    for ek in ExperimentKind:
        if ek.value == key:
            return PERMISSIONS[ek]
    # Any unknown / legacy diagnostic name is treated as deterministic diagnostic.
    return PERMISSIONS[ExperimentKind.DETERMINISTIC_DIAGNOSTIC]


def kind_from_operator_and_folds(operator_id: str, fold_count: int, *, smoke: bool = False) -> ExperimentKind:
    """Classify an executed operator into a permission kind.

    - 0-fold deterministic diagnostics are DETERMINISTIC_DIAGNOSTIC.
    - cached_replay is CACHED_REPLAY.
    - 1-2 fold model experiments are SCREEN_EXPERIMENT.
    - 3-fold are CONFIRM_EXPERIMENT.
    - 5-fold are FULL_CV_EXPERIMENT.
    """
    op = str(operator_id).lower()
    if op == "cached_replay":
        return ExperimentKind.CACHED_REPLAY
    if fold_count == 0:
        return ExperimentKind.DETERMINISTIC_DIAGNOSTIC
    if fold_count <= 2:
        return ExperimentKind.SCREEN_EXPERIMENT
    if fold_count == 3:
        return ExperimentKind.CONFIRM_EXPERIMENT
    return ExperimentKind.FULL_CV_EXPERIMENT
