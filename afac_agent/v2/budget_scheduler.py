# -*- coding: utf-8 -*-
"""AFAC v2.0 dynamic budget scheduler.

The scheduler treats ``max_scientific_rounds`` as a *safety cap*, not as a
fixed-length plan.  It must keep scheduling experiments while wall-clock and
round budget remain and at least one candidate has positive ROI.  This fixes
the v1.6 defect where the loop stopped after 3 rounds with only 75s of the
7200s budget consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

EPSILON = 1e-9

# Cost of one experiment in "scientific rounds" per fidelity level.
FIDELITY_ROUND_COST: dict[str, float] = {
    "static_audit": 0.0,
    "cached_replay": 0.0,
    "cheap_diagnostic": 0.0,
    "single_fold": 0.5,
    "full_oof": 1.0,
    "deployment": 0.0,
}

# ROI weight defaults; callers may override any subset.
DEFAULT_ROI_WEIGHTS: dict[str, float] = {
    "gain": 1.0,
    "information": 0.5,
    "novelty": 0.3,
    "coverage": 0.3,
    "deployment": 0.2,
    "risk": 0.5,
}

# Reference cost used to normalize compute cost into a unitless term.
REFERENCE_COST_SECONDS = 600.0

# Fraction of budget below which a stop is considered premature.
PREMATURE_STOP_FRACTION = 0.4


class Fidelity(str, Enum):
    STATIC_AUDIT = "static_audit"
    CACHED_REPLAY = "cached_replay"
    CHEAP_DIAGNOSTIC = "cheap_diagnostic"
    SINGLE_FOLD = "single_fold"
    FULL_OOF = "full_oof"
    DEPLOYMENT = "deployment"

    @property
    def round_cost(self) -> float:
        return FIDELITY_ROUND_COST[self.value]


@dataclass
class BudgetState:
    """Remaining-resource view of the research loop."""

    max_wall_clock_seconds: float = 7200.0
    elapsed: float = 0.0
    rounds_used: float = 0.0
    max_scientific_rounds: float = 12.0  # safety cap, NOT a fixed plan length

    @property
    def remaining_wall_clock_seconds(self) -> float:
        return max(0.0, self.max_wall_clock_seconds - self.elapsed)

    @property
    def remaining_rounds(self) -> float:
        return max(0.0, self.max_scientific_rounds - self.rounds_used)

    @property
    def wall_clock_fraction_consumed(self) -> float:
        if self.max_wall_clock_seconds <= 0:
            return 1.0
        return min(1.0, self.elapsed / self.max_wall_clock_seconds)

    @property
    def rounds_fraction_consumed(self) -> float:
        if self.max_scientific_rounds <= 0:
            return 1.0
        return min(1.0, self.rounds_used / self.max_scientific_rounds)


@dataclass
class ExperimentCandidate:
    candidate_id: str
    fidelity: Fidelity = Fidelity.SINGLE_FOLD
    expected_gain: float = 0.0
    expected_information_gain: float = 0.0
    compute_cost_seconds: float = 60.0
    validation_risk: float = 0.0
    novelty: float = 0.0
    coverage_gap: float = 0.0
    deployment_value: float = 0.0


def roi_score(candidate: ExperimentCandidate, weights: dict[str, float] | None = None) -> float:
    """Return the normalized return-on-investment score of a candidate.

    Formula::

        benefit = w_gain * expected_gain
                + w_info * expected_information_gain
                + w_novelty * novelty
                + w_coverage * coverage_gap
                + w_deployment * deployment_value
        risk_penalty = w_risk * validation_risk
        cost_term = compute_cost_seconds / REFERENCE_COST_SECONDS
        roi = (benefit - risk_penalty) / (1.0 + cost_term)

    The score rewards expected metric gain, information gain, novelty,
    coverage of open gaps and deployment value; it is penalized by
    validation risk and normalized down by relative compute cost.
    """
    w = {**DEFAULT_ROI_WEIGHTS, **(weights or {})}
    benefit = (
        w["gain"] * candidate.expected_gain
        + w["information"] * candidate.expected_information_gain
        + w["novelty"] * candidate.novelty
        + w["coverage"] * candidate.coverage_gap
        + w["deployment"] * candidate.deployment_value
    )
    risk_penalty = w["risk"] * candidate.validation_risk
    cost_term = candidate.compute_cost_seconds / REFERENCE_COST_SECONDS
    return (benefit - risk_penalty) / (1.0 + cost_term)


def _fits_budget(candidate: ExperimentCandidate, state: BudgetState) -> bool:
    fidelity = candidate.fidelity if isinstance(candidate.fidelity, Fidelity) else Fidelity(candidate.fidelity)
    return (
        candidate.compute_cost_seconds <= state.remaining_wall_clock_seconds + EPSILON
        and fidelity.round_cost <= state.remaining_rounds + EPSILON
    )


def schedule(candidates: list[ExperimentCandidate], state: BudgetState) -> list[ExperimentCandidate]:
    """Return a greedily ordered plan of candidates that fit the budget.

    Candidates are ranked by descending ROI (ties broken by candidate_id for
    determinism) and added while wall-clock and scientific-round budget last.
    """
    ranked = sorted(candidates, key=lambda c: (-roi_score(c), c.candidate_id))
    plan: list[ExperimentCandidate] = []
    elapsed = state.elapsed
    rounds_used = state.rounds_used
    for candidate in ranked:
        fidelity = candidate.fidelity if isinstance(candidate.fidelity, Fidelity) else Fidelity(candidate.fidelity)
        if (
            candidate.compute_cost_seconds <= state.max_wall_clock_seconds - elapsed + EPSILON
            and fidelity.round_cost <= state.max_scientific_rounds - rounds_used + EPSILON
        ):
            plan.append(candidate)
            elapsed += candidate.compute_cost_seconds
            rounds_used += fidelity.round_cost
    return plan


def should_continue(
    state: BudgetState,
    candidates: list[ExperimentCandidate],
    roi_threshold: float = 0.0,
) -> bool:
    """True when at least one affordable candidate clears the ROI threshold.

    Crucially this never stops merely because a fixed number of rounds (e.g.
    3) has completed; it stops only on budget exhaustion, the safety cap, or
    the absence of positive-ROI affordable candidates.
    """
    if state.remaining_wall_clock_seconds <= EPSILON or state.remaining_rounds <= EPSILON:
        return False
    return any(_fits_budget(c, state) and roi_score(c) > roi_threshold for c in candidates)


def detect_premature_stop(
    rounds_used: float,
    elapsed: float,
    state: BudgetState,
    candidates: list[ExperimentCandidate] | None = None,
) -> bool:
    """True when the loop stopped with <40% of budget consumed while
    positive-ROI candidates remained (or, if unknown, may have remained).
    """
    stopped_early = (
        elapsed / state.max_wall_clock_seconds < PREMATURE_STOP_FRACTION
        if state.max_wall_clock_seconds > 0
        else False
    ) and (
        rounds_used / state.max_scientific_rounds < PREMATURE_STOP_FRACTION
        if state.max_scientific_rounds > 0
        else False
    )
    if not stopped_early:
        return False
    if candidates is None:
        return True
    return any(roi_score(c) > 0.0 for c in candidates)
