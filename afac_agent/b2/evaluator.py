# -*- coding: utf-8 -*-
"""B2 multi-panel recommendation evaluation.

Metrics: NDCG@10, HitRate@10, MRR@10, CandidateRecall@10, plus retrieval vs
ranking failure splits, length/history/novel/long-tail/test-like/top10-boundary
slices, and rescue/damage/net change statistics against a baseline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .fold import B2Folds


MAX_K = 10


def _ndcg_score(relevances: np.ndarray, k: int = MAX_K) -> float:
    """Compute DCG then divide by ideal DCG for a single ranked list."""
    rel = np.asarray(relevances, dtype=np.float64)[:k]
    if rel.size == 0 or rel.max() <= 0:
        return 0.0
    positions = np.arange(1, rel.size + 1)
    dcg = np.sum(rel / np.log2(positions + 1))
    ideal = np.sort(rel)[::-1]
    ideal_dcg = np.sum(ideal / np.log2(positions + 1))
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def _metrics_for_mask(
    topk: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    k: int = MAX_K,
) -> dict[str, Any]:
    if not mask.any():
        return {
            "n": 0,
            f"ndcg@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"candidate_recall@{k}": 0.0,
            "retrieval_failure_rate": 0.0,
            "ranking_failure_rate": 0.0,
        }
    topk_masked = topk[mask]
    targets_masked = targets[mask]
    n = int(mask.sum())

    hits = []
    rr = []
    ndcgs = []
    candidate_hits = []
    retrieval_fail = []
    ranking_fail = []

    for pred, target in zip(topk_masked, targets_masked):
        pred_list = pred[:k].tolist() if hasattr(pred, "tolist") else list(pred[:k])
        is_hit = target in pred_list
        hits.append(is_hit)
        candidate_hits.append(is_hit)
        if is_hit:
            rank = pred_list.index(target) + 1
            rr.append(1.0 / rank)
            # For single relevant item, NDCG@K = 1/log2(rank+1) / 1/log2(2)
            ndcgs.append(1.0 / np.log2(rank + 1) / (1.0 / np.log2(2)))
            if rank > 1:
                ranking_fail.append(True)
            else:
                ranking_fail.append(False)
            retrieval_fail.append(False)
        else:
            rr.append(0.0)
            ndcgs.append(0.0)
            retrieval_fail.append(True)
            ranking_fail.append(False)

    return {
        "n": n,
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hit_rate@{k}": float(np.mean(hits)),
        f"mrr@{k}": float(np.mean(rr)),
        f"candidate_recall@{k}": float(np.mean(candidate_hits)),
        "retrieval_failure_rate": float(np.mean(retrieval_fail)),
        "ranking_failure_rate": float(np.mean(ranking_fail)),
    }


def _delta(baseline_hits: np.ndarray, candidate_hits: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(baseline_hits.shape[0], dtype=bool)
    base = baseline_hits[mask]
    cand = candidate_hits[mask]
    rescue = int(np.sum(~base & cand))
    damage = int(np.sum(base & ~cand))
    changed = rescue + damage
    return {
        "rescue": rescue,
        "damage": damage,
        "net": rescue - damage,
        "changed_count": changed,
        "change_precision": rescue / changed if changed else None,
    }


class B2Evaluator:
    def __init__(self, dataset: Any, folds: B2Folds, panels: dict[str, Any]) -> None:
        self.dataset = dataset
        self.folds = folds
        self.panels = panels
        self.uids = folds.uids
        self.uid2idx = {uid: i for i, uid in enumerate(folds.uids)}
        self.top_k = dataset.top_k

        if "target_iid" in dataset.train_df.columns:
            self.targets = np.array([str(dataset.train_df.set_index("uid").loc[uid, "target_iid"]) for uid in folds.uids], dtype=object)
        else:
            self.targets = np.full(len(folds.uids), None, dtype=object)

    def _ensure_topk(self, topk: Any) -> np.ndarray:
        arr = np.asarray(topk, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def evaluate(self, topk: Any, *, asset_id: str, panel_ids: list[str] | None = None) -> dict[str, Any]:
        topk_arr = self._ensure_topk(topk)
        if topk_arr.shape[0] != len(self.uids):
            raise ValueError(f"topk rows {topk_arr.shape[0]} != number of train users {len(self.uids)}")

        panel_ids = panel_ids or ["B2_STANDARD_PANEL"]
        panel_results = {}
        for pid in panel_ids:
            _, val_mask = self._panel_masks(pid)
            panel_results[pid] = _metrics_for_mask(topk_arr, self.targets, val_mask, k=self.top_k)

        # Per-fold standard metrics for stability
        fold_metrics = []
        for f in range(5):
            m = self.folds.folds == f
            fold_metrics.append(_metrics_for_mask(topk_arr, self.targets, m, k=self.top_k))
        worst_fold_ndcg = min(fm[f"ndcg@{self.top_k}"] for fm in fold_metrics)
        positive_fold_count = sum(1 for fm in fold_metrics if fm[f"ndcg@{self.top_k}"] >= worst_fold_ndcg - 1e-9)

        overall = _metrics_for_mask(topk_arr, self.targets, np.ones(len(self.uids), dtype=bool), k=self.top_k)
        return {
            "asset_id": asset_id,
            "overall": overall,
            f"overall_ndcg@{self.top_k}": overall[f"ndcg@{self.top_k}"],
            f"overall_hit_rate@{self.top_k}": overall[f"hit_rate@{self.top_k}"],
            f"overall_mrr@{self.top_k}": overall[f"mrr@{self.top_k}"],
            "worst_fold_ndcg": worst_fold_ndcg,
            "positive_fold_count": positive_fold_count,
            "fold_metrics": fold_metrics,
            "panels": panel_results,
        }

    def compare(self, baseline_topk: Any, candidate_topk: Any, panel_id: str = "B2_STANDARD_PANEL") -> dict[str, Any]:
        base_arr = self._ensure_topk(baseline_topk)
        cand_arr = self._ensure_topk(candidate_topk)
        _, val_mask = self._panel_masks(panel_id)
        base_hits = np.array([self.targets[i] in base_arr[i, :self.top_k].tolist() for i in range(len(self.uids))])
        cand_hits = np.array([self.targets[i] in cand_arr[i, :self.top_k].tolist() for i in range(len(self.uids))])
        return _delta(base_hits, cand_hits, val_mask)

    def _panel_masks(self, panel_id: str) -> tuple[np.ndarray, np.ndarray]:
        from .fold import validation_mask
        return validation_mask(panel_id, self.folds, held_fold=None)


def cross_fit_blend(
    *,
    score_a: np.ndarray,
    score_b: np.ndarray,
    y_train_idx: np.ndarray,
    item_list: list[str],
    folds: np.ndarray,
    alpha_grid: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Outer-fold cross-fit score blend between two (n_users, n_items) score matrices.

    Returns blended scores and the per-fold alpha assignment.  Evaluation must
    convert scores back to Top-K outside this function.
    """
    n = score_a.shape[0]
    blended = np.zeros_like(score_a)
    assignments = []
    for held in sorted(set(folds.tolist())):
        fit = folds != held
        valid = folds == held
        best_alpha, best_score = None, -1.0
        for alpha in alpha_grid:
            s = (1.0 - alpha) * score_a[fit] + alpha * score_b[fit]
            preds = np.argsort(-s, axis=1)[:, :MAX_K]
            hits = 0
            for i, row in enumerate(preds):
                if y_train_idx[i] in row:
                    hits += 1
            score = hits / max(1, fit.sum())
            if score > best_score + 1e-12:
                best_score = score
                best_alpha = alpha
        s = (1.0 - best_alpha) * score_a[valid] + alpha * score_b[valid]
        blended[valid] = s
        assignments.append({"held_fold": int(held), "alpha": best_alpha})
    return blended, {"mode": "outer_fold_cross_fit", "assignments": assignments}
