# -*- coding: utf-8 -*-
"""AFAC v2.0 no-op detector.

Compares a candidate's predictions against its parent and flags experiments
that changed nothing (identical proba arrays, identical ordered top-k lists,
or score-equal per-user score dicts).  No-op experiments are refunded their
round and excluded from the portfolio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from afac_agent.research.event_store import stable_hash

NO_OP_HINT = "implementation or configuration problem"


@dataclass
class NoOpReport:
    """Result of comparing candidate predictions against parent predictions."""

    prediction_hash: str = ""
    argmax_hash: str = ""
    changed_users: list[str] = field(default_factory=list)
    changed_items: list[str] = field(default_factory=list)
    changed_fraction: float = 0.0
    mean_abs_score_diff: float = 0.0
    max_abs_score_diff: float = 0.0
    route_fraction: float = 0.0
    effective_source_contribution: dict[str, float] = field(default_factory=dict)
    status: str = "ok"  # "ok" | "no_op"
    reasons: list[str] = field(default_factory=list)


def _source_contribution(source_scores: dict[str, Any] | None, eps: float) -> dict[str, float]:
    """Mean |score| share per source, normalized over all sources."""
    if not source_scores:
        return {}
    means: dict[str, float] = {}
    for source, scores in source_scores.items():
        arr = np.asarray(scores, dtype=np.float64).ravel()
        means[str(source)] = float(np.mean(np.abs(arr))) if arr.size else 0.0
    total = sum(means.values())
    if total <= eps:
        return {source: 0.0 for source in means}
    return {source: value / total for source, value in means.items()}


def _compare_proba(parent: np.ndarray, candidate: np.ndarray, eps: float) -> NoOpReport:
    parent_argmax = np.argmax(parent, axis=1)
    candidate_argmax = np.argmax(candidate, axis=1)
    argmax_changed = parent_argmax != candidate_argmax
    diffs = np.abs(parent - candidate)
    per_row_max = diffs.max(axis=1) if diffs.size else np.zeros(parent.shape[0])
    changed = argmax_changed | (per_row_max > eps)
    changed_users = [str(i) for i in np.nonzero(changed)[0]]
    changed_fraction = float(np.mean(changed)) if parent.shape[0] else 0.0
    report = NoOpReport(
        prediction_hash=stable_hash(np.asarray(candidate, dtype=np.float64).round(12).tolist()),
        argmax_hash=stable_hash(candidate_argmax.tolist()),
        changed_users=changed_users,
        changed_items=[],
        changed_fraction=changed_fraction,
        mean_abs_score_diff=float(diffs.mean()) if diffs.size else 0.0,
        max_abs_score_diff=float(diffs.max()) if diffs.size else 0.0,
        route_fraction=float(np.mean(argmax_changed)) if parent.shape[0] else 0.0,
    )
    if changed_fraction == 0.0 and report.max_abs_score_diff <= eps:
        report.status = "no_op"
        report.reasons.append("identical argmax and score differences within eps")
    else:
        report.status = "ok"
        report.reasons.append(f"changed_fraction={changed_fraction:.6f}")
    return report


def _compare_topk_lists(parent: list[list[str]], candidate: list[list[str]]) -> NoOpReport:
    changed_users: list[str] = []
    changed_items: set[str] = set()
    route_changed = 0
    for i, (parent_list, candidate_list) in enumerate(zip(parent, candidate)):
        parent_list = [str(x) for x in parent_list]
        candidate_list = [str(x) for x in candidate_list]
        if parent_list != candidate_list:
            changed_users.append(str(i))
            # Items with changed membership or changed position.
            changed_items.update(set(parent_list) ^ set(candidate_list))
            for old_item, new_item in zip(parent_list, candidate_list):
                if old_item != new_item:
                    changed_items.update((old_item, new_item))
            if (parent_list[:1] or [None]) != (candidate_list[:1] or [None]):
                route_changed += 1
    n = len(candidate)
    changed_fraction = len(changed_users) / n if n else 0.0
    report = NoOpReport(
        prediction_hash=stable_hash([[str(x) for x in row] for row in candidate]),
        argmax_hash=stable_hash([str(row[0]) if row else "" for row in candidate]),
        changed_users=changed_users,
        changed_items=sorted(changed_items),
        changed_fraction=changed_fraction,
        mean_abs_score_diff=0.0,
        max_abs_score_diff=0.0,
        route_fraction=route_changed / n if n else 0.0,
    )
    if changed_fraction == 0.0:
        report.status = "no_op"
        report.reasons.append("identical ordered top-k lists")
    else:
        report.status = "ok"
        report.reasons.append(f"changed_fraction={changed_fraction:.6f}")
    return report


def _compare_score_dicts(
    parent: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    eps: float,
) -> NoOpReport:
    changed_users: list[str] = []
    changed_items: set[str] = set()
    diffs: list[float] = []
    route_changed = 0
    uids = sorted(set(parent) | set(candidate))
    for uid in uids:
        parent_scores = parent.get(uid, {})
        candidate_scores = candidate.get(uid, {})
        user_changed = False
        for iid in sorted(set(parent_scores) | set(candidate_scores)):
            diff = abs(float(candidate_scores.get(iid, 0.0)) - float(parent_scores.get(iid, 0.0)))
            diffs.append(diff)
            if diff > eps:
                user_changed = True
                changed_items.add(str(iid))
        if user_changed:
            changed_users.append(str(uid))
        parent_top = max(parent_scores, key=parent_scores.get) if parent_scores else None
        candidate_top = max(candidate_scores, key=candidate_scores.get) if candidate_scores else None
        if parent_top != candidate_top:
            route_changed += 1
    n = len(uids)
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    changed_fraction = len(changed_users) / n if n else 0.0
    report = NoOpReport(
        prediction_hash=stable_hash({uid: {iid: round(float(s), 12) for iid, s in candidate[uid].items()} for uid in sorted(candidate)}),
        argmax_hash=stable_hash(
            {
                uid: (max(candidate[uid], key=candidate[uid].get) if candidate[uid] else "")
                for uid in sorted(candidate)
            }
        ),
        changed_users=changed_users,
        changed_items=sorted(changed_items),
        changed_fraction=changed_fraction,
        mean_abs_score_diff=float(diffs_arr.mean()) if diffs_arr.size else 0.0,
        max_abs_score_diff=float(diffs_arr.max()) if diffs_arr.size else 0.0,
        route_fraction=route_changed / n if n else 0.0,
    )
    if changed_fraction == 0.0 and report.max_abs_score_diff <= eps:
        report.status = "no_op"
        report.reasons.append("no score difference above eps for any user/item")
    else:
        report.status = "ok"
        report.reasons.append(f"changed_fraction={changed_fraction:.6f}")
    return report


def compare_predictions(
    parent: Any,
    candidate: Any,
    eps: float = 1e-12,
    source_scores: dict[str, Any] | None = None,
) -> NoOpReport:
    """Compare candidate predictions against parent predictions.

    Supported inputs:
    (a) classification proba arrays of shape (n, c) — argmax and score diffs;
    (b) recommendation top-k lists — ordered list membership/order;
    (c) per-user score dicts ``{uid: {iid: score}}``.

    Verdict: ``no_op`` when changed_fraction == 0 AND max_abs_score_diff <= eps
    (for top-k lists: identical ordered lists).
    """
    if isinstance(parent, dict) and isinstance(candidate, dict):
        report = _compare_score_dicts(parent, candidate, eps)
    else:
        parent_arr = np.asarray(parent, dtype=object)
        candidate_arr = np.asarray(candidate, dtype=object)
        if parent_arr.ndim == 2 and parent_arr.shape[1] > 0 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in parent_arr[0]
        ):
            report = _compare_proba(
                np.asarray(parent, dtype=np.float64),
                np.asarray(candidate, dtype=np.float64),
                eps,
            )
        else:
            report = _compare_topk_lists(list(parent), list(candidate))
    report.effective_source_contribution = _source_contribution(source_scores, eps)
    return report


def apply_noop_policy(experiment_record: dict[str, Any], report: NoOpReport) -> dict[str, Any]:
    """Apply the no-op accounting policy to an experiment record.

    A no-op experiment is refunded its round (``consumes_round=False``),
    excluded from the portfolio, and annotated with an issue note.
    """
    record = dict(experiment_record)
    record["noop_report"] = asdict(report)
    issues = list(record.get("issues", []))
    if report.status == "no_op":
        record["consumes_round"] = False
        record["portfolio_eligible"] = False
        issues.append({"type": "no_op", "hint": NO_OP_HINT})
    else:
        record["consumes_round"] = True
    record["issues"] = issues
    return record
