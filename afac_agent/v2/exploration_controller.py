# -*- coding: utf-8 -*-
"""AFAC v2.0 global exploration controller.

Watches run history and the current portfolio for failure patterns
(stagnation, local optima, availability / anchor / scope / validation
bias, premature stop) and escalates the exploration mode:

    exploit -> adjacent_explore -> global_explore

Escalation is one step per decision round while stagnation persists, so a
run that keeps failing locally is pushed progressively further from its
current basin.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

STAGNATION_WINDOW = 2  # rounds without positive gain before stagnation fires
LOCAL_OPTIMUM_WINDOW = 4  # longer window of negligible gains
LOCAL_OPTIMUM_EPSILON = 1e-3
PREMATURE_STOP_FRACTION = 0.4


class Mode(str, Enum):
    EXPLOIT = "exploit"
    ADJACENT_EXPLORE = "adjacent_explore"
    GLOBAL_EXPLORE = "global_explore"


class Action(str, Enum):
    MOVE_UP_PROBLEM_HIERARCHY = "move_up_problem_hierarchy"
    SWITCH_PIPELINE_STAGE = "switch_pipeline_stage"
    SWITCH_INFORMATION_SOURCE = "switch_information_source"
    SWITCH_MODEL_LAYER = "switch_model_layer"
    SELECT_ORTHOGONAL_PARENT = "select_orthogonal_parent"
    RESET_TO_ANCHOR = "reset_to_anchor"
    FORCE_NEW_OPERATOR_FAMILY = "force_new_operator_family"


def _gains(history: list[dict[str, Any]]) -> list[float]:
    return [float(record.get("gain", 0.0)) for record in history]


def detect_stagnation(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when none of the last N=2 rounds produced a positive gain."""
    recent = _gains(history)[-STAGNATION_WINDOW:]
    flag = len(recent) >= STAGNATION_WINDOW and all(gain <= 0.0 for gain in recent)
    return flag, {"recent_gains": recent, "window": STAGNATION_WINDOW}


def detect_local_optimum(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when a longer window shows only negligible improvements."""
    recent = _gains(history)[-LOCAL_OPTIMUM_WINDOW:]
    flag = len(recent) >= LOCAL_OPTIMUM_WINDOW and all(
        gain < LOCAL_OPTIMUM_EPSILON for gain in recent
    )
    return flag, {"recent_gains": recent, "window": LOCAL_OPTIMUM_WINDOW}


def detect_availability_bias(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when available-but-not-best options are repeatedly selected."""
    biased = [
        record.get("candidate_id", record.get("experiment_id", ""))
        for record in history
        if record.get("availability_bias")
        or (
            record.get("best_scientific_option")
            and record.get("selected_option")
            and record.get("best_scientific_option") != record.get("selected_option")
        )
    ]
    return len(biased) >= 2, {"biased_selections": biased}


def detect_anchor_bias(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when all experiments derive from a single parent."""
    parents = {record.get("parent_id") for record in history if record.get("parent_id")}
    flag = len(history) >= 2 and len(parents) == 1
    return flag, {"distinct_parents": sorted(parents)}


def detect_scope_tunnel(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when all experiments sit in one pipeline stage."""
    stages = {record.get("pipeline_stage") for record in history if record.get("pipeline_stage")}
    flag = len(history) >= 2 and len(stages) == 1
    return flag, {"distinct_stages": sorted(stages)}


def detect_validation_bias(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when measured gains come from a single validation panel only."""
    panels = {
        record.get("panel")
        for record in history
        if record.get("panel") and float(record.get("gain", 0.0)) > 0.0
    }
    any_gain = any(float(record.get("gain", 0.0)) > 0.0 for record in history)
    flag = any_gain and len(panels) == 1
    return flag, {"panels_with_gain": sorted(panels)}


def detect_premature_stop(
    history: list[dict[str, Any]], portfolio_summary: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """True when the loop stopped with the budget mostly unconsumed."""
    fraction = float(portfolio_summary.get("budget_elapsed_fraction", 0.0))
    stopped = bool(portfolio_summary.get("stopped", False))
    flag = stopped and fraction < PREMATURE_STOP_FRACTION
    return flag, {"budget_elapsed_fraction": fraction, "stopped": stopped}


DETECTORS: dict[str, Callable[[list[dict[str, Any]], dict[str, Any]], tuple[bool, dict[str, Any]]]] = {
    "stagnation": detect_stagnation,
    "local_optimum": detect_local_optimum,
    "availability_bias": detect_availability_bias,
    "anchor_bias": detect_anchor_bias,
    "scope_tunnel": detect_scope_tunnel,
    "validation_bias": detect_validation_bias,
    "premature_stop": detect_premature_stop,
}

_DETECTOR_ACTIONS: dict[str, list[Action]] = {
    "stagnation": [Action.SWITCH_INFORMATION_SOURCE, Action.SWITCH_MODEL_LAYER],
    "local_optimum": [Action.SELECT_ORTHOGONAL_PARENT, Action.MOVE_UP_PROBLEM_HIERARCHY],
    "availability_bias": [Action.FORCE_NEW_OPERATOR_FAMILY],
    "anchor_bias": [Action.SELECT_ORTHOGONAL_PARENT],
    "scope_tunnel": [Action.SWITCH_PIPELINE_STAGE],
    "validation_bias": [Action.MOVE_UP_PROBLEM_HIERARCHY],
    "premature_stop": [],
}

_MODE_ORDER = [Mode.EXPLOIT, Mode.ADJACENT_EXPLORE, Mode.GLOBAL_EXPLORE]


def _escalate(mode: Mode) -> Mode:
    index = _MODE_ORDER.index(mode)
    return _MODE_ORDER[min(index + 1, len(_MODE_ORDER) - 1)]


def decide(
    history: list[dict[str, Any]],
    portfolio_summary: dict[str, Any],
    budget_state: Any = None,
) -> dict[str, Any]:
    """Decide the next exploration mode and corrective actions.

    Stagnation or a detected local optimum escalates the mode one step
    (exploit -> adjacent_explore -> global_explore); otherwise the mode
    returns to exploit.  Returns {"mode", "actions", "reasons", "evidence"}.
    """
    evidence: dict[str, Any] = {}
    reasons: list[str] = []
    for name, detector in DETECTORS.items():
        flag, detail = detector(history, portfolio_summary)
        evidence[name] = detail
        if flag:
            reasons.append(name)

    current_mode = Mode(portfolio_summary.get("mode", Mode.EXPLOIT.value))
    if "stagnation" in reasons or "local_optimum" in reasons:
        mode = _escalate(current_mode)
    else:
        mode = Mode.EXPLOIT

    actions: list[Action] = []
    for reason in reasons:
        for action in _DETECTOR_ACTIONS.get(reason, []):
            if action not in actions:
                actions.append(action)
    if mode == Mode.GLOBAL_EXPLORE:
        for action in (
            Action.RESET_TO_ANCHOR,
            Action.FORCE_NEW_OPERATOR_FAMILY,
            Action.MOVE_UP_PROBLEM_HIERARCHY,
        ):
            if action not in actions:
                actions.append(action)

    return {
        "mode": mode.value,
        "actions": [action.value for action in actions],
        "reasons": reasons,
        "evidence": evidence,
    }
