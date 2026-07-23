# -*- coding: utf-8 -*-
"""AFAC v2.0 metric semantics gate.

Separates candidate-pool recall (pre-ranking retrieval coverage) from final
top-K ranking metrics — fixing the v1.6 bug where ``candidate_recall@10`` was
computed identically to ``hit_rate@10`` — and audits classification panels for
membership duplicates so duplicate panels never count as independent evidence.
"""
from __future__ import annotations

from typing import Any

from afac_agent.research.event_store import stable_hash

TOP10_SUCCESS = "top10_success"
IN_POOL_OUTSIDE_TOP10 = "in_pool_outside_top10"
MISSING_FROM_CANDIDATE_POOL = "missing_from_candidate_pool"


def candidate_pool_recall(
    pool: list[list[str]],
    targets: list[str],
    ks: tuple[int, ...] = (20, 50, 100, 200),
) -> dict[str, float]:
    """Fraction of users whose target appears in the first K of the candidate pool.

    ``pool`` is the pre-ranking candidate union per user; this metric is
    independent of the final top-10 ranking.
    """
    if len(pool) != len(targets):
        raise ValueError(f"pool rows {len(pool)} != targets {len(targets)}")
    n = len(targets)
    result: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for cand, target in zip(pool, targets) if target in cand[:k])
        result[f"candidate_pool_recall@{k}"] = hits / n if n else 0.0
    return result


def ranking_metrics(
    topk: list[list[str]],
    targets: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Ranking metrics over the final top-K lists (single relevant item per user)."""
    if len(topk) != len(targets):
        raise ValueError(f"topk rows {len(topk)} != targets {len(targets)}")
    n = len(targets)
    hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    import math

    for pred, target in zip(topk, targets):
        pred_list = list(pred[:k])
        if target in pred_list:
            hits += 1
            rank = pred_list.index(target) + 1
            rr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)  # IDCG@K = 1 for single relevance
    return {
        f"hit_rate@{k}": hits / n if n else 0.0,
        f"ndcg@{k}": ndcg_sum / n if n else 0.0,
        f"mrr@{k}": rr_sum / n if n else 0.0,
    }


def error_decomposition(
    topk: list[list[str]],
    pool: list[list[str]],
    targets: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Mutually exclusive per-user error decomposition.

    Each user is assigned exactly one class: the target is in the final top-K
    (``top10_success``), is in the candidate pool but outside the top-K
    (``in_pool_outside_top10``), or is absent from the pool entirely
    (``missing_from_candidate_pool``).  Returned rates always sum to 1.
    """
    if not (len(topk) == len(pool) == len(targets)):
        raise ValueError("topk, pool and targets must have equal length")
    n = len(targets)
    counts = {TOP10_SUCCESS: 0, IN_POOL_OUTSIDE_TOP10: 0, MISSING_FROM_CANDIDATE_POOL: 0}
    for pred, cand, target in zip(topk, pool, targets):
        if target in list(pred[:k]):
            counts[TOP10_SUCCESS] += 1
        elif target in cand:
            counts[IN_POOL_OUTSIDE_TOP10] += 1
        else:
            counts[MISSING_FROM_CANDIDATE_POOL] += 1
    return {name: count / n if n else 0.0 for name, count in counts.items()}


def validate_error_decomposition(rates: dict[str, float], tol: float = 1e-9) -> dict[str, Any]:
    """Check that a decomposition covers every user exactly once (rates sum to 1)."""
    keys = [TOP10_SUCCESS, IN_POOL_OUTSIDE_TOP10, MISSING_FROM_CANDIDATE_POOL]
    total = float(sum(rates.get(name, 0.0) for name in keys))
    valid = abs(total - 1.0) <= tol
    return {"status": "ok" if valid else "invalid_metric_semantics", "sum": total}


def promotion_gate(status: str) -> dict[str, Any]:
    """Gate promotion on the decomposition validation status."""
    if status == "ok":
        return {"allowed": True, "reason": "metric semantics validated"}
    return {
        "allowed": False,
        "reason": f"metric semantics invalid ({status}): decomposition rates must sum to 1",
    }


def panel_record(
    panel_id: str,
    member_ids: list[Any],
    labels: list[Any] | None = None,
    degrees: list[float] | None = None,
    propensities: list[float] | None = None,
    distance_to_test: float | None = None,
) -> dict[str, Any]:
    """Build an auditable panel record with a membership hash and distributions.

    ``membership_hash`` is the stable hash of the sorted member ids, so two
    panels with identical membership share a hash regardless of ordering.
    """
    members = [str(m) for m in member_ids]
    membership_hash = stable_hash(sorted(members))

    def _distribution(values: list[Any] | None) -> dict[str, float] | None:
        if values is None:
            return None
        counts: dict[str, float] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0.0) + 1.0
        total = float(len(values))
        return {key: count / total for key, count in sorted(counts.items())}

    return {
        "panel_id": panel_id,
        "membership_hash": membership_hash,
        "size": len(members),
        "class_distribution": _distribution(list(labels) if labels is not None else None),
        "degree_distribution": _distribution(list(degrees) if degrees is not None else None),
        "propensity_distribution": _distribution(list(propensities) if propensities is not None else None),
        "distance_to_test": distance_to_test,
    }


def audit_panels(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag panels with identical membership hashes as duplicates.

    Every panel after the first occurrence of a membership hash is marked
    ``duplicate_panel`` with a ``duplicate_of`` field; duplicate panels must
    never count as independent evidence.
    """
    panels: list[dict[str, Any]] = []
    duplicate_pairs: list[dict[str, str]] = []
    seen: dict[str, str] = {}  # membership_hash -> first panel_id
    for record in records:
        panel = dict(record)
        panel_id = str(panel.get("panel_id", ""))
        mhash = str(panel.get("membership_hash", ""))
        if mhash in seen:
            panel["status"] = "duplicate_panel"
            panel["duplicate_of"] = seen[mhash]
            duplicate_pairs.append({"panel_id": panel_id, "duplicate_of": seen[mhash]})
        else:
            panel.setdefault("status", "independent_panel")
            seen[mhash] = panel_id
        panels.append(panel)
    return {
        "panels": panels,
        "duplicate_pairs": duplicate_pairs,
        "independent_panel_count": len(seen),
    }
