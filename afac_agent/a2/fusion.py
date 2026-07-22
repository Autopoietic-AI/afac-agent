# -*- coding: utf-8 -*-
"""A2 recommendation fusion operators.

All operators consume verified OOF score matrices aligned to the same user
order and item catalog.  Operators that learn a parameter (alpha, routing)
must be cross-fit: parameters are selected on the non-held-out folds of
AFAC_A2_FOLD_V1 and applied to the held-out fold.  Oracle/diagnostic variants
(union recall) are never executable candidates.

Operators:
- score_blend: per-user z-normalized convex score blend;
- rank_fusion: reciprocal-rank fusion (fixed, no learned parameter);
- candidate_union: diagnostic recall of the union of two top-K sets
  (diagnostic only) plus an executable re-scored union composition;
- bucket_route: route sequence-length buckets to different experts;
- topk_protected_rerank: Top10 set membership fixed; only ranks 1..7 are
  re-ordered by the expert; ranks 8..10 keep base order;
- slot_protected_rerank: history items keep their base relative order and
  slots; only novel items are re-ordered inside the Top10;
- retriever_ranker_composition: retriever proposes a candidate pool, the
  ranker re-scores inside the pool only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .evaluator import DEFAULT_K, topk_lists
from .task_adapter import A2Dataset

OPERATORS = (
    "score_blend",
    "rank_fusion",
    "candidate_union",
    "bucket_route",
    "topk_protected_rerank",
    "slot_protected_rerank",
    "retriever_ranker_composition",
)

ALPHA_GRID = (0.25, 0.50, 0.75)


def zscore_per_user(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (scores - mean) / std


def op_score_blend(a: np.ndarray, b: np.ndarray, *, alpha: float) -> np.ndarray:
    za = zscore_per_user(a)
    zb = zscore_per_user(b)
    return (1.0 - alpha) * za + alpha * zb


def op_rank_fusion(a: np.ndarray, b: np.ndarray, *, rrf_k: int = 60) -> np.ndarray:
    def rrf_scores(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable", axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(order.shape[0])[:, None]
        ranks[rows, order] = np.arange(order.shape[1])[None, :]
        return 1.0 / (rrf_k + ranks + 1.0)

    return rrf_scores(a) + rrf_scores(b)


def op_bucket_route(a: np.ndarray, b: np.ndarray, *, route_mask: np.ndarray) -> np.ndarray:
    za = zscore_per_user(a)
    zb = zscore_per_user(b)
    out = za.copy()
    out[route_mask] = zb[route_mask]
    return out


def union_candidate_recall(
    a: np.ndarray,
    b: np.ndarray,
    target_col: np.ndarray,
    *,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Diagnostic-only union recall of two top-K candidate sets."""
    top_a = topk_lists(a, k)
    top_b = topk_lists(b, k)
    rows = np.arange(a.shape[0])
    in_a = (top_a == target_col[:, None]).any(axis=1)
    in_b = (top_b == target_col[:, None]).any(axis=1)
    n = a.shape[0]
    return {
        "k": k,
        "user_count": n,
        "recall_a": float(in_a.mean()) if n else None,
        "recall_b": float(in_b.mean()) if n else None,
        "a_only": int((in_a & ~in_b).sum()),
        "b_only": int((~in_a & in_b).sum()),
        "both": int((in_a & in_b).sum()),
        "neither": int((~in_a & ~in_b).sum()),
        "oracle_union_recall": float((in_a | in_b).mean()) if n else None,
        "diagnostic_only": True,
        "executable": False,
    }


def op_retriever_ranker_composition(
    retriever: np.ndarray,
    ranker: np.ndarray,
    *,
    pool_size: int = 20,
) -> np.ndarray:
    """Candidates from the retriever's top-`pool_size`; ranker re-scores only those."""
    pool = topk_lists(retriever, pool_size)
    out = np.full(ranker.shape, -np.inf, dtype=np.float64)
    rows = np.arange(ranker.shape[0])[:, None]
    out[rows, pool] = np.take_along_axis(zscore_per_user(ranker), pool, axis=1)
    return out


def op_topk_protected_rerank(
    base: np.ndarray,
    expert: np.ndarray,
    *,
    set_size: int = DEFAULT_K,
    rerank_positions: int = 7,
) -> np.ndarray:
    """Top10 set membership fixed; rerank only the first `rerank_positions`.

    The first `rerank_positions` base items are re-ordered by expert scores;
    the remaining base items keep their relative base order in the tail
    slots.  Output is a score matrix whose top-`set_size` reproduces the
    protected list (lower ranks encode as decreasing scores).
    """
    base_top = topk_lists(base, set_size)
    n, m = base.shape
    out = np.full((n, m), -np.inf, dtype=np.float64)
    expert_z = zscore_per_user(expert)
    total = float(m + set_size)
    for i in range(n):
        head = base_top[i, :rerank_positions]
        tail = base_top[i, rerank_positions:set_size]
        head_order = head[np.argsort(-expert_z[i, head], kind="stable")]
        ordered = np.concatenate([head_order, tail])
        out[i, ordered] = total - np.arange(set_size)
    return out


def op_slot_protected_rerank(
    base: np.ndarray,
    expert: np.ndarray,
    history_mask: np.ndarray,
    *,
    set_size: int = DEFAULT_K,
) -> np.ndarray:
    """History slots protected; novel items reranked inside the Top10.

    `history_mask` is a boolean (n_users, n_items) mask marking each user's
    history items.  History items inside the base Top10 keep their base
    relative order and their slots; novel items are re-ordered by expert
    scores inside the remaining slots.  Set membership is unchanged.
    """
    base_top = topk_lists(base, set_size)
    n, m = base.shape
    out = np.full((n, m), -np.inf, dtype=np.float64)
    expert_z = zscore_per_user(expert)
    total = float(m + set_size)
    for i in range(n):
        top = base_top[i]
        is_hist = history_mask[i, top]
        hist_items = top[is_hist]  # base relative order preserved
        novel_items = top[~is_hist]
        novel_order = novel_items[np.argsort(-expert_z[i, novel_items], kind="stable")]
        ordered = np.empty(set_size, dtype=top.dtype)
        hist_slots = np.nonzero(is_hist)[0]
        novel_slots = np.nonzero(~is_hist)[0]
        ordered[hist_slots] = hist_items
        ordered[novel_slots] = novel_order
        out[i, ordered] = total - np.arange(set_size)
    return out


def build_history_mask(dataset: A2Dataset, uids: list[str]) -> np.ndarray:
    """Boolean (users x items) mask of each user's deduplicated history items."""
    mask = np.zeros((len(uids), len(dataset.item_ids)), dtype=bool)
    index = {iid: pos for pos, iid in enumerate(dataset.item_ids)}
    for row, uid in enumerate(uids):
        for iid in set(dataset.train_seq_dedup.get(uid, dataset.test_seq_dedup.get(uid, []))):
            col = index.get(iid)
            if col is not None:
                mask[row, col] = True
    return mask


def cross_fit_scores(
    *,
    operator: str,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    folds: np.ndarray,
    route_mask: np.ndarray,
    history_mask: np.ndarray | None,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    metric_ranks_fn: Any = None,
    selection_targets: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an operator with strict outer-fold cross-fit when it learns a parameter.

    - score_blend: alpha selected per held-out fold by NDCG@10 on the fit folds;
    - bucket_route: per-bucket routing is a fixed policy when `route_mask` is
      given by the declared Len3 rule; otherwise the route decision is
      selected per held-out fold (route vs no-route) on the fit folds;
    - fixed operators are applied once without learning.
    """
    from .evaluator import DEFAULT_K as K, ndcg_at_k, target_ranks

    n = scores_a.shape[0]
    out = np.zeros((n, scores_a.shape[1]), dtype=np.float64)
    assignments: list[dict[str, Any]] = []

    def apply(alpha: float | None, route: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
        if operator == "score_blend":
            return op_score_blend(scores_a[mask], scores_b[mask], alpha=float(alpha))
        if operator == "rank_fusion":
            return op_rank_fusion(scores_a[mask], scores_b[mask])
        if operator == "bucket_route":
            local_route = route[mask] if route is not None else np.zeros(int(mask.sum()), dtype=bool)
            return op_bucket_route(scores_a[mask], scores_b[mask], route_mask=local_route)
        if operator == "topk_protected_rerank":
            return op_topk_protected_rerank(scores_a[mask], scores_b[mask])
        if operator == "slot_protected_rerank":
            return op_slot_protected_rerank(scores_a[mask], scores_b[mask], history_mask[mask])
        if operator == "retriever_ranker_composition":
            return op_retriever_ranker_composition(scores_a[mask], scores_b[mask])
        raise ValueError(f"unsupported operator: {operator}")

    def ndcg(scores: np.ndarray, mask: np.ndarray) -> float:
        cols = selection_targets[mask]
        safe = np.where(cols >= 0, cols, 0)
        ranks = target_ranks(scores, safe)
        ranks = np.where(cols >= 0, ranks, scores.shape[1] + 1)
        return float(ndcg_at_k(ranks, K).mean())

    learned = operator in {"score_blend", "bucket_route"}
    if not learned:
        full = apply(None, route_mask, np.ones(n, dtype=bool))
        return full, {"mode": "fixed_no_learned_parameter", "fold_assignments": []}

    for held in sorted(set(folds.tolist())):
        fit = folds != held
        valid = folds == held
        if operator == "score_blend":
            best_alpha, best_score = None, -1.0
            for alpha in alpha_grid:
                score = ndcg(apply(alpha, None, fit), fit)
                if score > best_score + 1e-15:
                    best_alpha, best_score = alpha, score
            out[valid] = apply(best_alpha, None, valid)
            assignments.append({
                "held_out_fold": int(held),
                "selection_rows": int(fit.sum()),
                "application_rows": int(valid.sum()),
                "selected_params": {"alpha": best_alpha},
                "held_out_targets_used_for_selection": False,
            })
        else:  # bucket_route: select route-vs-base per held-out fold
            routed = apply(None, route_mask, fit)
            plain = apply(None, np.zeros(n, dtype=bool), fit)
            use_route = ndcg(routed, fit) >= ndcg(plain, fit)
            applied_route = route_mask if use_route else np.zeros(n, dtype=bool)
            out[valid] = apply(None, applied_route, valid)
            assignments.append({
                "held_out_fold": int(held),
                "selection_rows": int(fit.sum()),
                "application_rows": int(valid.sum()),
                "selected_params": {"route_exact_len3_to_expert": bool(use_route)},
                "held_out_targets_used_for_selection": False,
            })
    return out, {"mode": "strict_outer_fold_cross_fit", "fold_assignments": assignments}
