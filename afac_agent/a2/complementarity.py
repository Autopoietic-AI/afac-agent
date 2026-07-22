# -*- coding: utf-8 -*-
"""A2 complementarity audit.

Analyzes retriever candidate-coverage complementarity and ranker ranking
complementarity between two verified OOF assets:
- candidate coverage at several K (A-only / B-only / both / neither);
- rank rescue/damage/net at @10 overall and by sequence-length and
  history/novel buckets;
- topK-range breakdown;
- oracle union recall (diagnostic only, never executable);
- explicit separation between changing the candidate set (retrieval) and
  reranking inside a fixed candidate set (ranking).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..research.event_store import stable_hash
from .assets import ScoreAsset
from .evaluator import A2Evaluator, DEFAULT_K, topk_lists
from .fusion import union_candidate_recall
from .task_adapter import A2Dataset

TOPK_RANGE = (1, 3, 5, 10, 20)


def build_complementarity_report(
    *,
    dataset: A2Dataset,
    evaluator: A2Evaluator,
    asset_a: ScoreAsset,
    asset_b: ScoreAsset,
) -> dict[str, Any]:
    a = asset_a.aligned_to(dataset.train_uids)
    b = asset_b.aligned_to(dataset.train_uids)
    uids = dataset.train_uids
    col_of = {iid: pos for pos, iid in enumerate(a.item_ids)}
    target_col = np.array([col_of.get(dataset.train_targets.get(uid, ""), -1) for uid in uids], dtype=np.int64)

    ev_a = evaluator.evaluate_scores(asset_id=a.asset_id, uids=uids, item_ids=a.item_ids, scores=a.scores, targets=a.targets)
    ev_b = evaluator.evaluate_scores(asset_id=b.asset_id, uids=uids, item_ids=b.item_ids, scores=b.scores, targets=b.targets)

    coverage_by_k = []
    for k in TOPK_RANGE:
        record = union_candidate_recall(a.scores, b.scores, target_col, k=k)
        record["interpretation"] = "retriever candidate coverage complementarity (diagnostic)"
        coverage_by_k.append(record)

    hit_a = ev_a.ranks <= DEFAULT_K
    hit_b = ev_b.ranks <= DEFAULT_K
    rank_complementarity = {
        "metric": f"hit_rate_at_{DEFAULT_K}",
        "a_only_hit": int((hit_a & ~hit_b).sum()),
        "b_only_hit": int((~hit_a & hit_b).sum()),
        "both_hit": int((hit_a & hit_b).sum()),
        "neither_hit": int((~hit_a & ~hit_b).sum()),
        "rescue_b_over_a": int((~hit_a & hit_b).sum()),
        "damage_b_over_a": int((hit_a & ~hit_b).sum()),
        "net_b_over_a": int((~hit_a & hit_b).sum()) - int((hit_a & ~hit_b).sum()),
        "oracle_union_hit_rate": float((hit_a | hit_b).mean()),
        "oracle_is_diagnostic_only": True,
    }

    bucket_rows = evaluator.compare_by_bucket(baseline=ev_a, candidate=ev_b, uids=uids)

    report = {
        "report_version": "a2_complementarity_v1",
        "task": "A2",
        "asset_a": a.asset_id,
        "asset_b": b.asset_id,
        "retriever_coverage_by_k": coverage_by_k,
        "ranker_complementarity_at_10": rank_complementarity,
        "rescue_damage_by_bucket": bucket_rows,
        "asset_a_summary": ev_a.summary(),
        "asset_b_summary": ev_b.summary(),
        "candidate_set_change_vs_fixed_set_rerank": {
            "candidate_set_change": (
                "modifies set membership (union/expansion); can fix retrieval failures "
                "(target_missing_from_candidates) but carries Top10-set protection risk"
            ),
            "fixed_set_rerank": (
                "keeps the Top10 candidate set fixed; can only fix ranking failures "
                "(target_below_top10); safer under the online champion strategy"
            ),
        },
        "oracle_union_diagnostic_only_not_executable": True,
        "view_hash": stable_hash({
            "a": a.asset_id,
            "b": b.asset_id,
            "coverage": coverage_by_k,
            "rank": rank_complementarity,
        }),
    }
    return report
