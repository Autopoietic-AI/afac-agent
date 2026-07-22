# -*- coding: utf-8 -*-
"""A2 ranking evaluator.

Implements deterministic per-user ranking metrics over explicit candidate
score matrices:

- NDCG@10 / HitRate@10 / MRR@10 (single relevant item per user);
- Candidate Recall (target present in the candidate set at all);
- Target Rank (1-based, deterministic tie-break by item order);
- retrieval failure vs ranking failure separation;
- Sequence Length and History/Novel bucket breakdowns;
- Rescue/Damage/Net, Changed User Count and Change Precision against a
  baseline ranking.

The evaluator is offline-only: it never touches Test truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .task_adapter import A2Dataset

DEFAULT_K = 10
RETRIEVAL_FAILURE = "target_missing_from_candidates"
RANKING_FAILURE = "target_below_top10"
RANKING_SUCCESS = "target_in_top10"


def target_ranks(scores: np.ndarray, target_col: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
    """1-based rank of each row's target column with deterministic tie-break.

    rank = 1 + #(scores > target_score) + #(scores == target_score and col < target_col)
    Chunked over rows to bound memory on large candidate matrices.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n, m = scores.shape
    out = np.empty(n, dtype=np.int64)
    cols = np.arange(m)[None, :]
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        block = scores[start:end]
        target_score = block[np.arange(end - start), target_col[start:end]][:, None]
        greater = (block > target_score).sum(axis=1)
        equal_earlier = ((block == target_score) & (cols < target_col[start:end, None])).sum(axis=1)
        out[start:end] = 1 + greater + equal_earlier
    return out


def topk_lists(scores: np.ndarray, k: int = DEFAULT_K, *, chunk_rows: int = 4096) -> np.ndarray:
    """Per-row indices of the top-k columns, deterministic tie-break by column index."""
    scores = np.asarray(scores, dtype=np.float64)
    k = min(k, scores.shape[1])
    n = scores.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        block = scores[start:end]
        part = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-part_scores, kind="stable", axis=1)
        out[start:end] = np.take_along_axis(part, order, axis=1)
    return out


def ndcg_at_k(ranks: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    hit = ranks <= k
    gains = np.zeros(ranks.shape[0], dtype=np.float64)
    gains[hit] = 1.0 / np.log2(ranks[hit] + 1.0)
    return gains


def mrr_at_k(ranks: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    hit = ranks <= k
    out = np.zeros(ranks.shape[0], dtype=np.float64)
    out[hit] = 1.0 / ranks[hit]
    return out


def hit_at_k(ranks: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    return (ranks <= k).astype(np.float64)


@dataclass
class A2ScoreEvaluation:
    asset_id: str
    user_count: int
    candidate_size: int
    ranks: np.ndarray
    in_candidates: np.ndarray
    top10: np.ndarray
    ndcg10: float
    hit10: float
    mrr10: float
    candidate_recall: float
    mean_target_rank: float
    len_bucket_metrics: list[dict[str, Any]] = field(default_factory=list)
    type_bucket_metrics: list[dict[str, Any]] = field(default_factory=list)
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    failure_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "user_count": self.user_count,
            "candidate_size": self.candidate_size,
            "ndcg_at_10": self.ndcg10,
            "hit_rate_at_10": self.hit10,
            "mrr_at_10": self.mrr10,
            "candidate_recall": self.candidate_recall,
            "mean_target_rank": self.mean_target_rank,
            "failure_counts": self.failure_counts,
        }


class A2Evaluator:
    """Evaluate candidate rankings against A2 train targets (offline OOF only)."""

    def __init__(self, dataset: A2Dataset, fold_map: dict[str, int] | None = None) -> None:
        self.dataset = dataset
        self.fold_map = fold_map or {}
        self.item_index = {iid: pos for pos, iid in enumerate(dataset.item_ids)}
        self._len_bucket = {uid: dataset.len_bucket(uid) for uid in dataset.train_uids}
        self._target_type = {uid: dataset.target_type(uid) for uid in dataset.train_uids}

    def evaluate_scores(
        self,
        *,
        asset_id: str,
        uids: list[str],
        item_ids: list[str],
        scores: np.ndarray,
        targets: list[str] | None = None,
    ) -> A2ScoreEvaluation:
        """Evaluate a full-candidate score matrix aligned to `uids`/`item_ids`.

        `targets` defaults to the dataset train targets keyed by uid.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(uids), len(item_ids)):
            raise ValueError(f"scores shape {scores.shape} != ({len(uids)}, {len(item_ids)})")
        col_of = {iid: pos for pos, iid in enumerate(item_ids)}
        target_list = list(targets) if targets is not None else [self.dataset.train_targets.get(uid, "") for uid in uids]
        target_col = np.array([col_of.get(t, -1) for t in target_list], dtype=np.int64)
        in_candidates = target_col >= 0
        safe_col = np.where(in_candidates, target_col, 0)
        ranks = target_ranks(scores, safe_col)
        # Users whose target is absent from the candidate set get rank = +inf semantics:
        ranks = np.where(in_candidates, ranks, len(item_ids) + 1)
        top10 = topk_lists(scores, DEFAULT_K)

        ndcg = ndcg_at_k(ranks, DEFAULT_K)
        hit = hit_at_k(ranks, DEFAULT_K)
        mrr = mrr_at_k(ranks, DEFAULT_K)
        finite_ranks = ranks[in_candidates]
        failure_counts = {
            "target_in_candidates": int(in_candidates.sum()),
            "target_missing_from_candidates": int((~in_candidates).sum()),
            "target_in_top10": int((ranks <= DEFAULT_K).sum()),
            "target_below_top10": int((in_candidates & (ranks > DEFAULT_K)).sum()),
            "retrieval_failure": int((~in_candidates).sum()),
            "ranking_failure": int((in_candidates & (ranks > DEFAULT_K)).sum()),
        }
        evaluation = A2ScoreEvaluation(
            asset_id=asset_id,
            user_count=len(uids),
            candidate_size=len(item_ids),
            ranks=ranks,
            in_candidates=in_candidates,
            top10=top10,
            ndcg10=float(ndcg.mean()) if len(uids) else 0.0,
            hit10=float(hit.mean()) if len(uids) else 0.0,
            mrr10=float(mrr.mean()) if len(uids) else 0.0,
            candidate_recall=float(in_candidates.mean()) if len(uids) else 0.0,
            mean_target_rank=float(finite_ranks.mean()) if finite_ranks.size else math.inf,
            failure_counts=failure_counts,
        )
        evaluation.len_bucket_metrics = self._bucket_metrics(uids, ranks, in_candidates, axis="sequence_length", keys=self._len_bucket)
        evaluation.type_bucket_metrics = self._bucket_metrics(uids, ranks, in_candidates, axis="history_novel", keys=self._target_type)
        if self.fold_map:
            fold_keys = {uid: str(self.fold_map.get(uid, "unknown")) for uid in uids}
            evaluation.fold_metrics = self._bucket_metrics(uids, ranks, in_candidates, axis="fold", keys=fold_keys)
        return evaluation

    def _bucket_metrics(
        self,
        uids: list[str],
        ranks: np.ndarray,
        in_candidates: np.ndarray,
        *,
        axis: str,
        keys: dict[str, str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        key_array = np.array([keys.get(uid, "unknown") for uid in uids])
        for value in sorted(set(key_array.tolist())):
            mask = key_array == value
            count = int(mask.sum())
            bucket_ranks = ranks[mask]
            rows.append({
                "axis": axis,
                "bucket": value,
                "user_count": count,
                "ndcg_at_10": float(ndcg_at_k(bucket_ranks).mean()) if count else None,
                "hit_rate_at_10": float(hit_at_k(bucket_ranks).mean()) if count else None,
                "mrr_at_10": float(mrr_at_k(bucket_ranks).mean()) if count else None,
                "candidate_recall": float(in_candidates[mask].mean()) if count else None,
                "retrieval_failure": int((~in_candidates[mask]).sum()),
                "ranking_failure": int((in_candidates[mask] & (bucket_ranks > DEFAULT_K)).sum()),
            })
        return rows

    def compare(
        self,
        *,
        baseline: A2ScoreEvaluation,
        candidate: A2ScoreEvaluation,
        uids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Rescue/Damage/Net and change statistics of candidate vs baseline @10."""
        if baseline.ranks.shape != candidate.ranks.shape:
            raise ValueError("baseline/candidate user sets differ")
        base_hit = baseline.ranks <= DEFAULT_K
        cand_hit = candidate.ranks <= DEFAULT_K
        rescue = int((~base_hit & cand_hit).sum())
        damage = int((base_hit & ~cand_hit).sum())
        changed = int((baseline.top10 != candidate.top10).any(axis=1).sum())
        return {
            "baseline_asset_id": baseline.asset_id,
            "candidate_asset_id": candidate.asset_id,
            "rescue": rescue,
            "damage": damage,
            "net": rescue - damage,
            "changed_user_count": changed,
            "change_precision": (rescue / changed) if changed else None,
            "ndcg_at_10_gain": candidate.ndcg10 - baseline.ndcg10,
            "hit_rate_at_10_gain": candidate.hit10 - baseline.hit10,
            "mrr_at_10_gain": candidate.mrr10 - baseline.mrr10,
            "retrieval_failure_delta": candidate.failure_counts["retrieval_failure"] - baseline.failure_counts["retrieval_failure"],
            "ranking_failure_delta": candidate.failure_counts["ranking_failure"] - baseline.failure_counts["ranking_failure"],
        }

    def compare_by_bucket(
        self,
        *,
        baseline: A2ScoreEvaluation,
        candidate: A2ScoreEvaluation,
        uids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Rescue/damage broken down by sequence-length and history/novel buckets."""
        base_hit = baseline.ranks <= DEFAULT_K
        cand_hit = candidate.ranks <= DEFAULT_K
        changed = (baseline.top10 != candidate.top10).any(axis=1)
        out: dict[str, list[dict[str, Any]]] = {}
        for axis, keys in (("sequence_length", self._len_bucket), ("history_novel", self._target_type)):
            key_array = np.array([keys.get(uid, "unknown") for uid in uids])
            rows: list[dict[str, Any]] = []
            for value in sorted(set(key_array.tolist())):
                mask = key_array == value
                rescue = int((~base_hit[mask] & cand_hit[mask]).sum())
                damage = int((base_hit[mask] & ~cand_hit[mask]).sum())
                ch = int(changed[mask].sum())
                rows.append({
                    "axis": axis,
                    "bucket": value,
                    "user_count": int(mask.sum()),
                    "rescue": rescue,
                    "damage": damage,
                    "net": rescue - damage,
                    "changed_user_count": ch,
                    "change_precision": (rescue / ch) if ch else None,
                })
            out[axis] = rows
        return out
