# -*- coding: utf-8 -*-
"""AFAC v2.1 B2 anchor registry.

Popularity and history baselines are re-materialized on the current canonical
fold and current metric contract.  They can be used as a safe fallback when no
confirmed scientific candidate exists, but they are never confused with
scientific champions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..research.event_store import stable_hash
from .operators.recommendation import HistoryRetriever, PopularityRetriever


@dataclass
class AnchorRecord:
    candidate_id: str
    kind: str
    data_hash: str
    fold_hash: str
    evaluator_version: str
    oof_predictions: list[list[str]] = field(default_factory=list)
    oof_targets: list[str] = field(default_factory=list)
    oof_uids: list[str] = field(default_factory=list)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    deployment_ready: bool = False
    source_execution_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "data_hash": self.data_hash,
            "fold_hash": self.fold_hash,
            "evaluator_version": self.evaluator_version,
            "validation_metrics": self.validation_metrics,
            "deployment_ready": self.deployment_ready,
            "source_execution_id": self.source_execution_id,
            "oof_hash": stable_hash({"uids": self.oof_uids, "preds": self.oof_predictions}),
        }


def _ranking_metrics(top10: list[list[str]], targets: list[str], k: int = 10) -> dict[str, float]:
    hits = [t in lst[:k] for lst, t in zip(top10, targets)]
    rr = []
    for lst, t in zip(top10, targets):
        try:
            rr.append(1.0 / (lst.index(t) + 1))
        except ValueError:
            rr.append(0.0)
    return {
        "hit_rate@10": float(np.mean(hits)) if hits else 0.0,
        "mrr@10": float(np.mean(rr)) if rr else 0.0,
    }


class B2AnchorRegistry:
    """Registry of validated local anchors for B2."""

    EVALUATOR_VERSION = "afac_b2_ranking_v1"

    def __init__(self) -> None:
        self.anchors: dict[str, AnchorRecord] = {}

    def add(self, record: AnchorRecord) -> None:
        self.anchors[record.candidate_id] = record

    def best_anchor(self) -> AnchorRecord | None:
        candidates = [a for a in self.anchors.values() if a.deployment_ready]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.validation_metrics.get("hit_rate@10", 0.0))

    def build_anchors(
        self,
        *,
        canonical,
        dataset: Any,
        train_targets: dict[str, str],
        data_hash: str,
        source_execution_id: str,
    ) -> "B2AnchorRegistry":
        """Compute popularity and history baselines on the canonical folds."""
        seqs = {u: dataset.train_seq.get(u, []) for u in canonical.uids}
        all_eval_uids: list[str] = []
        all_targets: list[str] = []
        pop_top10: list[list[str]] = []
        hist_top10: list[list[str]] = []

        for f in range(5):
            fit_mask, eval_mask = canonical.masks(f)
            fit_uids = [u for u, m in zip(canonical.uids, fit_mask) if m]
            eval_uids = [u for u, m in zip(canonical.uids, eval_mask) if m]
            fit_seq = {u: seqs[u] for u in fit_uids}
            fit_targets = {u: train_targets[u] for u in fit_uids}

            popularity = PopularityRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            history = HistoryRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)

            pop_top10.extend(popularity.retrieve(eval_uids, max_per_user=10).to_topk_lists(10))
            hist_top10.extend(history.retrieve(eval_uids, max_per_user=10).to_topk_lists(10))
            all_eval_uids.extend(eval_uids)
            all_targets.extend([train_targets[u] for u in eval_uids])

        pop_metrics = _ranking_metrics(pop_top10, all_targets)
        hist_metrics = _ranking_metrics(hist_top10, all_targets)

        pop = AnchorRecord(
            candidate_id="anchor_popularity_b2",
            kind="ValidatedAnchor",
            data_hash=data_hash,
            fold_hash=canonical.fold_hash,
            evaluator_version=self.EVALUATOR_VERSION,
            oof_predictions=pop_top10,
            oof_targets=all_targets,
            oof_uids=all_eval_uids,
            validation_metrics=pop_metrics,
            deployment_ready=True,
            source_execution_id=source_execution_id,
        )
        hist = AnchorRecord(
            candidate_id="anchor_history_b2",
            kind="ValidatedAnchor",
            data_hash=data_hash,
            fold_hash=canonical.fold_hash,
            evaluator_version=self.EVALUATOR_VERSION,
            oof_predictions=hist_top10,
            oof_targets=all_targets,
            oof_uids=all_eval_uids,
            validation_metrics=hist_metrics,
            deployment_ready=True,
            source_execution_id=source_execution_id,
        )
        self.add(pop)
        self.add(hist)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_version": self.EVALUATOR_VERSION,
            "anchors": {k: v.to_dict() for k, v in self.anchors.items()},
            "best_anchor_id": self.best_anchor().candidate_id if self.best_anchor() else None,
        }
