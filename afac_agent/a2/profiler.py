# -*- coding: utf-8 -*-
"""A2 data profiler.

Builds the A2 data profile and bucket registry from an A2Dataset:
- user/item counts;
- sequence length distribution (raw and dedup) with len0/1/2/exact_len3/4+ buckets;
- history/novel target ratio;
- item popularity, cold-start items, missing features;
- candidate coverage of the default item catalog;
- ranking-stage buckets when an evaluated asset is supplied.

The profiler is read-only, CPU-only, deterministic and idempotent.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..research.event_store import stable_hash
from .evaluator import A2ScoreEvaluation
from .task_adapter import A2Dataset, LEN_BUCKETS

BUCKET_AXES = {
    "sequence_length": list(LEN_BUCKETS),
    "candidate_item_type": ["history", "novel"],
    "ranking_stage": [
        "target_in_candidates",
        "target_missing_from_candidates",
        "target_in_top10",
        "target_below_top10",
    ],
}


def _distribution(values: list[str], order: list[str] | None = None) -> dict[str, int]:
    counts = Counter(values)
    keys = order if order is not None else sorted(counts)
    out = {key: int(counts.get(key, 0)) for key in keys}
    for key in sorted(counts):
        if key not in out:
            out[key] = int(counts[key])
    return out


def build_bucket_registry() -> dict[str, Any]:
    return {
        "registry_version": "a2_bucket_registry_v1",
        "task": "A2",
        "axes": [
            {
                "axis": "sequence_length",
                "buckets": BUCKET_AXES["sequence_length"],
                "definition": "deduplicated history length: len0/len1/len2/exact_len3/len4_plus",
            },
            {
                "axis": "candidate_item_type",
                "buckets": BUCKET_AXES["candidate_item_type"],
                "definition": "history: target iid appears in the user's deduplicated history; novel otherwise",
            },
            {
                "axis": "ranking_stage",
                "buckets": BUCKET_AXES["ranking_stage"],
                "definition": (
                    "target_missing_from_candidates = retrieval failure; "
                    "target_below_top10 = ranking failure; target_in_top10 = success"
                ),
            },
        ],
        "retrieval_vs_ranking_separation": "retrieval failure is measured on candidate recall; ranking failure requires the target inside the candidate set",
        "view_hash": stable_hash(BUCKET_AXES),
    }


def build_profile(
    dataset: A2Dataset,
    *,
    evaluations: list[A2ScoreEvaluation] | None = None,
) -> dict[str, Any]:
    train_uids = dataset.train_uids
    test_uids = dataset.test_uids

    len_buckets = [dataset.len_bucket(uid) for uid in train_uids]
    raw_lengths = [len(dataset.train_seq_raw.get(uid, [])) for uid in train_uids]
    dedup_lengths = [len(dataset.train_seq_dedup.get(uid, [])) for uid in train_uids]
    target_types = [dataset.target_type(uid) for uid in train_uids]

    # Item popularity over train histories (dedup presence) and targets.
    item_popularity: Counter[str] = Counter()
    for uid in train_uids:
        item_popularity.update(set(dataset.train_seq_dedup.get(uid, [])))
    target_popularity: Counter[str] = Counter(dataset.train_targets.values())
    catalog = set(dataset.item_ids)
    seen_items = set(item_popularity) | set(target_popularity)
    cold_start_items = sorted(catalog - seen_items)
    popularity_values = sorted(item_popularity.values(), reverse=True)

    # Test-side sequence profile for shift audit.
    test_len_buckets = [dataset.len_bucket(uid) for uid in test_uids]
    test_dedup_lengths = [len(dataset.test_seq_dedup.get(uid, [])) for uid in test_uids]

    profile: dict[str, Any] = {
        "profile_version": "a2_data_profiler_v1",
        "task": "A2",
        "counts": {
            "train_users": len(train_uids),
            "test_users": len(test_uids),
            "items": len(dataset.item_ids),
            "user_feature_rows": dataset.user_feature_uids,
            "item_feature_columns": dataset.item_feature_columns,
        },
        "sequence_length": {
            "bucket_axis": "sequence_length",
            "train_bucket_distribution": _distribution(len_buckets, LEN_BUCKETS),
            "test_bucket_distribution": _distribution(test_len_buckets, LEN_BUCKETS),
            "train_raw_length": _length_stats(raw_lengths),
            "train_dedup_length": _length_stats(dedup_lengths),
            "test_dedup_length": _length_stats(test_dedup_lengths),
            "dedup_reduction_ratio": (
                1.0 - (sum(dedup_lengths) / sum(raw_lengths)) if sum(raw_lengths) else None
            ),
        },
        "target_type": {
            "bucket_axis": "candidate_item_type",
            "train_distribution": _distribution(target_types, ["history", "novel"]),
        },
        "item_popularity": {
            "items_observed_in_history": len(item_popularity),
            "items_observed_as_target": len(target_popularity),
            "cold_start_items_in_catalog": len(cold_start_items),
            "cold_start_item_examples": cold_start_items[:10],
            "top10_popular_items": [
                {"iid": iid, "user_count": count} for iid, count in item_popularity.most_common(10)
            ],
            "popularity_percentiles": _percentiles(popularity_values),
        },
        "missing_features": {
            "train_empty_history_users": int(sum(1 for n in dedup_lengths if n == 0)),
            "test_empty_history_users": int(sum(1 for n in test_dedup_lengths if n == 0)),
            "train_missing_target": int(sum(1 for uid in train_uids if not dataset.train_targets.get(uid))),
        },
        "candidate_coverage": {
            "default_candidate_catalog": "item.csv",
            "catalog_size": len(dataset.item_ids),
            "train_targets_inside_catalog": int(
                sum(1 for uid in train_uids if dataset.train_targets.get(uid) in catalog)
            ),
            "train_targets_outside_catalog": int(
                sum(1 for uid in train_uids if dataset.train_targets.get(uid) not in catalog)
            ),
            "coverage_ratio": (
                sum(1 for uid in train_uids if dataset.train_targets.get(uid) in catalog) / len(train_uids)
                if train_uids else None
            ),
        },
        "validation": dataset.validation,
    }

    if evaluations:
        stages: dict[str, Any] = {}
        for ev in evaluations:
            stages[ev.asset_id] = {
                "bucket_axis": "ranking_stage",
                "failure_counts": ev.failure_counts,
                "retrieval_failure": ev.failure_counts["retrieval_failure"],
                "ranking_failure": ev.failure_counts["ranking_failure"],
                "retrieval_vs_ranking": "separated",
            }
        profile["ranking_stage"] = stages
    profile["view_hash"] = stable_hash({
        "counts": profile["counts"],
        "sequence_length": profile["sequence_length"]["train_bucket_distribution"],
        "target_type": profile["target_type"]["train_distribution"],
    })
    return profile


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None, "percentiles": {}}
    ordered = sorted(lengths)
    return {
        "count": len(lengths),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(lengths) / len(lengths),
        "percentiles": _percentiles(ordered),
    }


def _percentiles(ordered_values: list[int]) -> dict[str, int | None]:
    if not ordered_values:
        return {"p50": None, "p90": None, "p99": None}

    def pick(q: float) -> int:
        idx = min(len(ordered_values) - 1, max(0, int(round(q * (len(ordered_values) - 1)))))
        return int(ordered_values[idx])

    return {"p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99)}
