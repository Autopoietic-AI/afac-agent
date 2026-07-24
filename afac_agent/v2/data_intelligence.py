# -*- coding: utf-8 -*-
"""AFAC v2.0 Unified Data Intelligence.

Deterministic, LLM-free statistical profiling for the classification (B1)
and recommendation (B2) tasks.  All numbers are computed by code; the LLM
may only *explain* a report via ``llm_explain``, which snapshots the report
hash before and after the call and raises if the report object was mutated.

Regime thresholds (documented here, used below):

- SHIFT_SEPARABILITY_MODERATE = 0.65, SHIFT_SEPARABILITY_HIGH = 0.80:
  train/test propensity separability_auc = max(raw_auc, 1 - raw_auc);
  >=0.80 means a model can almost perfectly tell train from test apart.
- HOMOPHILY_HIGH = 0.60, HOMOPHILY_LOW = 0.40: adjusted train-train edge
  homophily; below 0.40 graph-smoothness assumptions are suspect.
- COLD_START_FRAC = 0.30: fraction of empty histories that marks a
  cold-start-dominated recommendation regime.
- POPULARITY_HEAD_SHARE = 0.50: top-1%-item interaction share that marks a
  strong popularity-bias regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..research.event_store import stable_hash

SEED = 2026

# Regime thresholds (see module docstring).
SHIFT_SEPARABILITY_MODERATE = 0.65
SHIFT_SEPARABILITY_HIGH = 0.80
HOMOPHILY_HIGH = 0.60
HOMOPHILY_LOW = 0.40
COLD_START_FRAC = 0.30
POPULARITY_HEAD_SHARE = 0.50

REPORT_VERSION = "afac_v2_data_intelligence_v1"


class ReportMutationError(RuntimeError):
    """Raised when an explain_fn mutates the DataIntelligenceReport."""


def _f(x: Any) -> float:
    return float(x)


def _separability_auc(raw_auc: float) -> float:
    """Direction-free separability: an inverted classifier is equally strong."""
    return float(max(raw_auc, 1.0 - raw_auc))


def _conclusion(
    statement: str,
    supporting_metrics: dict[str, Any],
    confidence: str,
    alternative_explanation: str,
) -> dict[str, Any]:
    return {
        "conclusion": statement,
        "evidence_id": stable_hash({"statement": statement, "metrics": supporting_metrics}),
        "supporting_metrics": supporting_metrics,
        "confidence": confidence,
        "alternative_explanation": alternative_explanation,
    }


def _propensity_auc(features: np.ndarray, is_test: np.ndarray) -> dict[str, float]:
    """In-sample LogisticRegression propensity AUC for train/test separation."""
    Xs = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED, n_jobs=1)
    clf.fit(Xs, is_test)
    scores = clf.predict_proba(Xs)[:, 1]
    raw = float(roc_auc_score(is_test, scores))
    return {"raw_auc": raw, "separability_auc": _separability_auc(raw)}


@dataclass
class DataIntelligenceReport:
    """Unified data-intelligence report.  All sections are plain dicts."""

    task: str
    status: str = "verified"  # "verified" | "error"
    errors: list[str] = field(default_factory=list)
    dataset_fingerprint: dict[str, Any] = field(default_factory=dict)
    regime_profile: dict[str, Any] = field(default_factory=dict)
    signal_reliability_map: dict[str, Any] = field(default_factory=dict)
    coverage_map: dict[str, Any] = field(default_factory=dict)
    shift_map: dict[str, Any] = field(default_factory=dict)
    validation_protocol: dict[str, Any] = field(default_factory=dict)
    primary_bottleneck: dict[str, Any] = field(default_factory=dict)
    secondary_problems: list[dict[str, Any]] = field(default_factory=list)
    headroom_map: dict[str, Any] = field(default_factory=dict)
    avoid_list: list[str] = field(default_factory=list)
    model_search_prior: list[dict[str, Any]] = field(default_factory=list)
    initial_hypotheses: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Shallow composition: section dicts are shared by reference.

        This is deliberate — it lets ``llm_explain`` detect an explain_fn
        that mutates the section content it was handed.
        """
        return {
            "report_version": REPORT_VERSION,
            "task": self.task,
            "status": self.status,
            "errors": self.errors,
            "dataset_fingerprint": self.dataset_fingerprint,
            "regime_profile": self.regime_profile,
            "signal_reliability_map": self.signal_reliability_map,
            "coverage_map": self.coverage_map,
            "shift_map": self.shift_map,
            "validation_protocol": self.validation_protocol,
            "primary_bottleneck": self.primary_bottleneck,
            "secondary_problems": self.secondary_problems,
            "headroom_map": self.headroom_map,
            "avoid_list": self.avoid_list,
            "model_search_prior": self.model_search_prior,
            "initial_hypotheses": self.initial_hypotheses,
        }

    def report_hash(self) -> str:
        return stable_hash(self.to_dict())


# ---------------------------------------------------------------------------
# Classification (B1)
# ---------------------------------------------------------------------------


def _dense(X: Any) -> np.ndarray:
    if issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _feature_stats(X: np.ndarray) -> dict[str, Any]:
    zero_frac = _f(np.mean(X == 0.0))
    l1 = np.abs(X).sum(axis=1)
    l2 = np.sqrt((X**2).sum(axis=1))
    Xs = StandardScaler().fit_transform(X)
    cap = int(min(Xs.shape[0], Xs.shape[1]))
    pca = PCA(n_components=cap, random_state=SEED)
    pca.fit(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    dims = {f"dims_at_{t}": int(np.searchsorted(cum, t / 100.0) + 1) for t in (50, 80, 90, 95, 99)}
    ev = pca.explained_variance_
    participation_ratio = _f(ev.sum() ** 2 / max(float((ev**2).sum()), 1e-12))
    numeric_rank = int(np.linalg.matrix_rank(Xs))
    return {
        "density": _f(1.0 - zero_frac),
        "zero_fraction": zero_frac,
        "l1_norm": {"mean": _f(l1.mean()), "std": _f(l1.std()), "min": _f(l1.min()), "max": _f(l1.max())},
        "l2_norm": {"mean": _f(l2.mean()), "std": _f(l2.std()), "min": _f(l2.min()), "max": _f(l2.max())},
        "pca_cumulative_variance_dims": dims,
        "participation_ratio": participation_ratio,
        "numeric_rank": numeric_rank,
    }


def _graph_stats(adj: csr_matrix) -> dict[str, Any]:
    adj = adj.tocsr().astype(np.float64)
    n = adj.shape[0]
    in_deg = np.asarray(adj.sum(axis=0)).ravel()
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    und = adj.maximum(adj.T)
    und_deg = np.asarray(und.sum(axis=1)).ravel()
    inter = adj.multiply(adj.T)
    recip_edges = float(inter.nnz) - float(inter.diagonal().astype(bool).sum())
    directed_edges = float(adj.nnz) - float((adj.diagonal() != 0).sum())
    self_loops = int((adj.diagonal() != 0).sum())
    n_comp, labels = connected_components(und, directed=False)
    comp_sizes = np.bincount(labels, minlength=n_comp)

    def _summ(v: np.ndarray) -> dict[str, float]:
        return {"mean": _f(v.mean()), "std": _f(v.std()), "min": _f(v.min()), "max": _f(v.max())}

    return {
        "n_nodes": int(n),
        "n_edges": int(adj.nnz),
        "in_degree": _summ(in_deg),
        "out_degree": _summ(out_deg),
        "undirected_degree": _summ(und_deg),
        "reciprocity": _f(recip_edges / max(directed_edges, 1.0)),
        "self_loops": self_loops,
        "component_count": int(n_comp),
        "largest_component_size": int(comp_sizes.max()),
        "isolated_nodes": int((und_deg == 0).sum()),
        "degrees": {"in": in_deg, "out": out_deg, "undirected": und_deg},  # internal, stripped later
    }


def _homophily(adj: csr_matrix, labels: np.ndarray, train_idx: np.ndarray) -> dict[str, Any]:
    """Raw + adjusted homophily over train-train edges ONLY (no test labels)."""
    und = adj.maximum(adj.T).tocsr()
    und.data[:] = 1.0
    train_mask = np.zeros(adj.shape[0], dtype=bool)
    train_mask[train_idx] = True
    sub = und[train_mask][:, train_mask].tocsr()
    rows, cols = sub.nonzero()
    keep = rows < cols  # count each undirected edge once
    rows, cols = rows[keep], cols[keep]
    y_train = labels[train_idx]
    n_edges = int(len(rows))
    if n_edges == 0:
        return {"raw_homophily": 0.0, "adjusted_homophily": 0.0, "n_train_train_edges": 0}
    same = (y_train[rows] == y_train[cols]).mean()
    deg = np.asarray(sub.sum(axis=1)).ravel()
    classes, counts = np.unique(y_train, return_counts=True)
    deg_per_class = {int(c): float(deg[y_train == c].sum()) for c in classes}
    total_deg = float(deg.sum())
    expected = sum((d / max(total_deg, 1e-12)) ** 2 for d in deg_per_class.values())
    adjusted = (same - expected) / max(1.0 - expected, 1e-12)
    return {
        "raw_homophily": _f(same),
        "adjusted_homophily": _f(adjusted),
        "n_train_train_edges": n_edges,
        "chance_level_degree_weighted": _f(expected),
        "train_class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }


def analyze_classification(
    X: Any,
    adj: csr_matrix,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> DataIntelligenceReport:
    """Profile a node-classification dataset (features + graph + labels)."""
    report = DataIntelligenceReport(task="classification")
    try:
        Xd = _dense(X)
        adj = adj.tocsr().astype(np.float64)
        labels = np.asarray(labels)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        feat = _feature_stats(Xd)
        graph = _graph_stats(adj)
        degrees = graph.pop("degrees")
        hom = _homophily(adj, labels, train_idx)

        classes, counts = np.unique(labels[train_idx], return_counts=True)
        report.dataset_fingerprint = {
            "n_nodes": int(Xd.shape[0]),
            "n_features": int(Xd.shape[1]),
            "n_classes": int(len(classes)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_edges": graph["n_edges"],
            "train_label_distribution": {str(int(c)): int(n) for c, n in zip(classes, counts)},
            "feature_hash": stable_hash({"shape": list(Xd.shape), "mean": _f(Xd.mean()), "std": _f(Xd.std())}),
        }

        # Train/test propensity on structural + feature-norm signals.
        feat_norm = np.sqrt((Xd**2).sum(axis=1))
        shift_feats = np.column_stack([degrees["undirected"], degrees["in"], degrees["out"], feat_norm])
        sel = np.concatenate([train_idx, test_idx])
        is_test = np.concatenate([np.zeros(len(train_idx)), np.ones(len(test_idx))]).astype(int)
        prop = _propensity_auc(shift_feats[sel], is_test)
        report.shift_map = {
            "propensity_features": ["degree", "in_degree", "out_degree", "feature_norm"],
            "raw_auc": prop["raw_auc"],
            "separability_auc": prop["separability_auc"],
            "train_mean": {name: _f(shift_feats[train_idx, i].mean()) for i, name in enumerate(report_shift_names())},
            "test_mean": {name: _f(shift_feats[test_idx, i].mean()) for i, name in enumerate(report_shift_names())},
        }

        sep = prop["separability_auc"]
        h_adj = hom["adjusted_homophily"]
        if sep >= SHIFT_SEPARABILITY_HIGH:
            shift_regime, shift_conf = "severe_shift", "high"
        elif sep >= SHIFT_SEPARABILITY_MODERATE:
            shift_regime, shift_conf = "moderate_shift", "medium"
        else:
            shift_regime, shift_conf = "stable", "medium"
        if h_adj >= HOMOPHILY_HIGH:
            graph_regime = "homophilic"
        elif h_adj <= HOMOPHILY_LOW:
            graph_regime = "heterophilic"
        else:
            graph_regime = "mixed"

        report.regime_profile = {
            "shift_regime": shift_regime,
            "graph_regime": graph_regime,
            "thresholds": {
                "shift_separability_moderate": SHIFT_SEPARABILITY_MODERATE,
                "shift_separability_high": SHIFT_SEPARABILITY_HIGH,
                "homophily_high": HOMOPHILY_HIGH,
                "homophily_low": HOMOPHILY_LOW,
            },
            "conclusions": [
                _conclusion(
                    f"train/test shift regime is {shift_regime}",
                    {"separability_auc": sep, "raw_auc": prop["raw_auc"]},
                    shift_conf,
                    "small sample or weak propensity features can fake/mask shift",
                ),
                _conclusion(
                    f"graph regime is {graph_regime}",
                    {"adjusted_homophily": h_adj, "raw_homophily": hom["raw_homophily"]},
                    "medium" if hom["n_train_train_edges"] > 50 else "low",
                    "label noise or degree imbalance can bias homophily estimates",
                ),
            ],
        }

        pca90 = feat["pca_cumulative_variance_dims"]["dims_at_90"]
        report.signal_reliability_map = {
            "features": _conclusion(
                "feature signal lives in a low-dimensional subspace"
                if pca90 <= max(1, Xd.shape[1] // 2)
                else "feature signal is high-dimensional / diffuse",
                {"pca_dims_at_90": pca90, "participation_ratio": feat["participation_ratio"], "numeric_rank": feat["numeric_rank"]},
                "medium",
                "PCA variance concentration does not guarantee class separability",
            ),
            "graph": _conclusion(
                "graph smoothness is exploitable" if graph_regime == "homophilic" else "graph smoothness is weak or harmful",
                {"adjusted_homophily": h_adj, "reciprocity": graph["reciprocity"], "component_count": graph["component_count"]},
                "medium",
                "train-train edges may not represent train-test edge behavior",
            ),
        }

        in_largest = graph["largest_component_size"] / max(graph["n_nodes"], 1)
        report.coverage_map = {
            "labeled_fraction": _f(len(train_idx) / max(graph["n_nodes"], 1)),
            "largest_component_fraction": _f(in_largest),
            "isolated_nodes": graph["isolated_nodes"],
            "self_loops": graph["self_loops"],
            "feature_stats": feat,
            "graph_stats": graph,
        }

        if shift_regime == "severe_shift":
            bottleneck_statement = "primary bottleneck is train/test distribution shift"
            bottleneck_metrics = {"separability_auc": sep}
            protocol = "grouped_or_shift_aware_split"
        elif graph_regime == "heterophilic":
            bottleneck_statement = "primary bottleneck is graph/feature signal mismatch (heterophily)"
            bottleneck_metrics = {"adjusted_homophily": h_adj}
            protocol = "stratified_kfold"
        elif report.coverage_map["labeled_fraction"] < 0.1:
            bottleneck_statement = "primary bottleneck is the labeled-node budget"
            bottleneck_metrics = {"labeled_fraction": report.coverage_map["labeled_fraction"]}
            protocol = "stratified_kfold"
        else:
            bottleneck_statement = "primary bottleneck is model capacity / fusion quality"
            bottleneck_metrics = {"separability_auc": sep, "adjusted_homophily": h_adj}
            protocol = "stratified_kfold"
        report.primary_bottleneck = _conclusion(
            bottleneck_statement,
            bottleneck_metrics,
            shift_conf,
            "single-run heuristics; bottleneck ranking can flip with better features",
        )

        report.secondary_problems = []
        if graph["isolated_nodes"] > 0:
            report.secondary_problems.append(
                _conclusion(
                    "isolated nodes cannot receive graph signal",
                    {"isolated_nodes": graph["isolated_nodes"]},
                    "high",
                    "self-loops in model operators may partially compensate",
                )
            )
        if graph_regime == "mixed":
            report.secondary_problems.append(
                _conclusion(
                    "mixed homophily: global propagation may hurt some classes",
                    {"adjusted_homophily": h_adj},
                    "low",
                    "homophily may be class-conditional rather than globally mixed",
                )
            )

        report.headroom_map = {
            "graph_propagation": "high" if graph_regime == "homophilic" else ("low" if graph_regime == "heterophilic" else "medium"),
            "feature_low_rank": "high" if pca90 <= max(1, Xd.shape[1] // 2) else "low",
            "fusion": "medium" if shift_regime == "stable" else "low",
        }
        report.avoid_list = []
        if graph_regime == "heterophilic":
            report.avoid_list.append("deep smooth GNN stacks (over-smoothing under heterophily)")
        if shift_regime == "severe_shift":
            report.avoid_list.append("trusting random-split offline gains for deployment")
        if graph["component_count"] > 1:
            report.avoid_list.append("propagation across component boundaries (none exist; expect component-locked behavior)")

        priors = [
            {"family": "feature_linear", "prior": 0.8},
            {"family": "feature_lowrank", "prior": 0.7 if pca90 <= max(1, Xd.shape[1] // 2) else 0.4},
            {"family": "graph_propagation", "prior": 0.8 if graph_regime == "homophilic" else 0.3},
            {"family": "fusion", "prior": 0.6},
        ]
        report.model_search_prior = sorted(priors, key=lambda d: -d["prior"])
        report.validation_protocol = {
            "protocol": protocol,
            "n_folds": 5,
            "seed": SEED,
            "notes": "propensity separability_auc >= 0.8 would require shift-aware evaluation",
        }
        report.initial_hypotheses = [
            {
                "hypothesis": "graph propagation adds over feature-only models"
                if graph_regime == "homophilic"
                else "feature models dominate; graph adds little",
                "evidence_id": report.regime_profile["conclusions"][1]["evidence_id"],
            },
            {
                "hypothesis": f"offline gains are deployment-fragile under {shift_regime}"
                if shift_regime != "stable"
                else "random-split offline evaluation is trustworthy",
                "evidence_id": report.regime_profile["conclusions"][0]["evidence_id"],
            },
        ]
        report.status = "verified"
    except Exception as exc:  # pragma: no cover - defensive
        report.status = "error"
        report.errors.append(f"{type(exc).__name__}: {exc}")
    return report


def report_shift_names() -> list[str]:
    return ["degree", "in_degree", "out_degree", "feature_norm"]


# ---------------------------------------------------------------------------
# Recommendation (B2)
# ---------------------------------------------------------------------------


def _length_buckets(lengths: np.ndarray) -> dict[str, Any]:
    n = max(len(lengths), 1)
    counts = {
        "len0": int((lengths == 0).sum()),
        "len1": int((lengths == 1).sum()),
        "len2": int((lengths == 2).sum()),
        "exact_len3": int((lengths == 3).sum()),
        "len4_plus": int((lengths >= 4).sum()),
    }
    return {"counts": counts, "fractions": {k: _f(v / n) for k, v in counts.items()}}


def _bucket_record(
    bucket_name: str,
    lengths: np.ndarray,
    *,
    length_definition: str,
    source_sequence: str,
    full_or_sampled: str,
) -> dict[str, Any]:
    return {
        "bucket_name": bucket_name,
        "length_definition": length_definition,
        "source_sequence": source_sequence,
        "full_or_sampled": full_or_sampled,
        "membership_hash": stable_hash({"lengths": lengths.tolist()}),
        "size": int(len(lengths)),
        **_length_buckets(lengths),
    }


def _normalize_seq(seqs: Any) -> list[list[Any]]:
    """Accept dict[str, list] or list[list]."""
    if isinstance(seqs, dict):
        return list(seqs.values())
    return list(seqs)


def _normalize_targets(targets: Any) -> list[Any]:
    if isinstance(targets, dict):
        return list(targets.values())
    return list(targets)


def analyze_recommendation(
    train_seq: Any,
    train_targets: Any,
    test_seq: Any,
    item_popularity: dict[Any, float] | None = None,
    *,
    n_train_total: int | None = None,
    n_test_total: int | None = None,
    profile_scope: str = "full",
    sampling_seed: int | None = None,
) -> DataIntelligenceReport:
    """Profile a sequential-recommendation dataset (histories + targets).

    ``train_seq`` and ``test_seq`` may be dicts mapping uid -> sequence or
    plain lists of sequences.  ``train_targets`` may be a dict uid -> target or
    a list of targets aligned to ``train_seq``.

    The report now separates full-dataset scale from profiler-sample scale and
    reports raw, dedup, unique-item, and repeat-intensity buckets.
    """
    report = DataIntelligenceReport(task="recommendation")
    try:
        train_seq_list = _normalize_seq(train_seq)
        test_seq_list = _normalize_seq(test_seq)
        train_targets_list = _normalize_targets(train_targets)

        train_lens = np.array([len(s) for s in train_seq_list], dtype=np.int64)
        test_lens = np.array([len(s) for s in test_seq_list], dtype=np.int64)
        train_dedup = np.array([len(set(s)) for s in train_seq_list], dtype=np.int64)
        test_dedup = np.array([len(set(s)) for s in test_seq_list], dtype=np.int64)
        train_unique = np.array([len({*s}) for s in train_seq_list], dtype=np.int64)
        test_unique = np.array([len({*s}) for s in test_seq_list], dtype=np.int64)
        train_repeat = train_lens - train_dedup
        test_repeat = test_lens - test_dedup

        if item_popularity is None:
            pop: dict[Any, float] = {}
            for seq, tgt in zip(train_seq_list, train_targets_list):
                for item in seq:
                    pop[item] = pop.get(item, 0.0) + 1.0
                if tgt is not None:
                    pop[tgt] = pop.get(tgt, 0.0) + 1.0
            item_popularity = pop

        in_history = [t in set(s) for s, t in zip(train_seq_list, train_targets_list)]
        history_ratio = _f(np.mean(in_history)) if in_history else 0.0

        pop_values = np.array(sorted(item_popularity.values(), reverse=True), dtype=np.float64)
        total_pop = float(pop_values.sum()) if pop_values.size else 0.0
        n_items = int(pop_values.size)
        top1_n = max(1, int(np.ceil(0.01 * n_items))) if n_items else 0
        top10_n = max(1, int(np.ceil(0.10 * n_items))) if n_items else 0
        popularity_stats = {
            "n_items": n_items,
            "top_1pct_share": _f(pop_values[:top1_n].sum() / max(total_pop, 1e-12)),
            "top_10pct_share": _f(pop_values[:top10_n].sum() / max(total_pop, 1e-12)),
            "median_count": _f(np.median(pop_values)) if n_items else 0.0,
            "max_count": _f(pop_values.max()) if n_items else 0.0,
        }

        n_train_profiled = n_train_total if n_train_total is not None else len(train_seq_list)
        n_test_profiled = n_test_total if n_test_total is not None else len(test_seq_list)

        report.dataset_fingerprint = {
            "n_train_total": int(n_train_total if n_train_total is not None else len(train_seq_list)),
            "n_train_profiled": int(n_train_profiled),
            "n_test_total": int(n_test_total if n_test_total is not None else len(test_seq_list)),
            "n_test_profiled": int(n_test_profiled),
            "n_items": n_items,
            "profile_scope": profile_scope,
            "sampling_seed": sampling_seed,
            "popularity_hash": stable_hash({str(k): _f(v) for k, v in item_popularity.items()}),
        }
        report.coverage_map = {
            "history_recall_coverage": history_ratio,
            "novel_target_ratio": _f(1.0 - history_ratio),
            "item_popularity": popularity_stats,
            "raw_vs_dedup": {
                "train_mean_raw_len": _f(train_lens.mean()) if len(train_lens) else 0.0,
                "train_mean_dedup_len": _f(train_dedup.mean()) if len(train_dedup) else 0.0,
                "test_mean_raw_len": _f(test_lens.mean()) if len(test_lens) else 0.0,
                "test_mean_dedup_len": _f(test_dedup.mean()) if len(test_dedup) else 0.0,
            },
            "raw_length_buckets": {
                "train": _bucket_record("train_raw", train_lens, length_definition="raw_sequence_length", source_sequence="item_seq_raw", full_or_sampled=profile_scope),
                "test": _bucket_record("test_raw", test_lens, length_definition="raw_sequence_length", source_sequence="item_seq_raw", full_or_sampled=profile_scope),
            },
            "dedup_length_buckets": {
                "train": _bucket_record("train_dedup", train_dedup, length_definition="deduplicated_sequence_length", source_sequence="item_seq_dedup", full_or_sampled=profile_scope),
                "test": _bucket_record("test_dedup", test_dedup, length_definition="deduplicated_sequence_length", source_sequence="item_seq_dedup", full_or_sampled=profile_scope),
            },
            "unique_item_count_buckets": {
                "train": _bucket_record("train_unique", train_unique, length_definition="unique_item_count", source_sequence="item_seq_dedup", full_or_sampled=profile_scope),
                "test": _bucket_record("test_unique", test_unique, length_definition="unique_item_count", source_sequence="item_seq_dedup", full_or_sampled=profile_scope),
            },
            "repeat_intensity_buckets": {
                "train": _bucket_record("train_repeat", train_repeat, length_definition="raw_minus_dedup_count", source_sequence="item_seq_raw - item_seq_dedup", full_or_sampled=profile_scope),
                "test": _bucket_record("test_repeat", test_repeat, length_definition="raw_minus_dedup_count", source_sequence="item_seq_raw - item_seq_dedup", full_or_sampled=profile_scope),
            },
            "test_length_buckets": _length_buckets(test_lens),
            "train_length_buckets": _length_buckets(train_lens),
        }

        # Train/test propensity on [seq_len, mean_history_popularity].
        def _pop_feats(seqs: list[list[Any]]) -> np.ndarray:
            rows = []
            for s in seqs:
                mean_pop = float(np.mean([item_popularity.get(i, 0.0) for i in s])) if s else 0.0
                rows.append([float(len(s)), mean_pop])
            return np.asarray(rows, dtype=np.float64)

        prop_feats = np.vstack([_pop_feats(train_seq_list), _pop_feats(test_seq_list)])
        is_test = np.concatenate([np.zeros(len(train_seq_list)), np.ones(len(test_seq_list))]).astype(int)
        prop = _propensity_auc(prop_feats, is_test)
        report.shift_map = {
            "propensity_features": ["seq_len", "mean_history_popularity"],
            "raw_auc": prop["raw_auc"],
            "separability_auc": prop["separability_auc"],
        }

        sep = prop["separability_auc"]
        cold_frac = report.coverage_map["train_length_buckets"]["fractions"]["len0"]
        head_share = popularity_stats["top_1pct_share"]
        regimes = []
        if cold_frac >= COLD_START_FRAC:
            regimes.append("cold_start_dominated")
        if head_share >= POPULARITY_HEAD_SHARE:
            regimes.append("popularity_biased")
        if sep >= SHIFT_SEPARABILITY_HIGH:
            regimes.append("severe_shift")
        elif sep >= SHIFT_SEPARABILITY_MODERATE:
            regimes.append("moderate_shift")
        if not regimes:
            regimes.append("stable_warm")
        report.regime_profile = {
            "regimes": regimes,
            "thresholds": {
                "cold_start_frac": COLD_START_FRAC,
                "popularity_head_share": POPULARITY_HEAD_SHARE,
                "shift_separability_moderate": SHIFT_SEPARABILITY_MODERATE,
                "shift_separability_high": SHIFT_SEPARABILITY_HIGH,
            },
            "conclusions": [
                _conclusion(
                    "recommendation regimes: " + ", ".join(regimes),
                    {"cold_start_frac": cold_frac, "top_1pct_share": head_share, "separability_auc": sep},
                    "medium",
                    "propensity features are coarse; hidden covariates may drive separation",
                )
            ],
        }

        report.signal_reliability_map = {
            "history_recall": _conclusion(
                "history-recall signal covers most targets" if history_ratio >= 0.5 else "most targets are novel w.r.t. history",
                {"history_recall_coverage": history_ratio},
                "high" if len(train_targets_list) > 100 else "medium",
                "history membership ignores item order and recency",
            ),
            "popularity": _conclusion(
                "popularity prior is strong" if head_share >= POPULARITY_HEAD_SHARE else "popularity prior is flat",
                {"top_1pct_share": head_share, "top_10pct_share": popularity_stats["top_10pct_share"]},
                "medium",
                "train popularity can drift toward test time",
            ),
        }

        if sep >= SHIFT_SEPARABILITY_HIGH:
            bottleneck = ("primary bottleneck is train/test behavior shift", {"separability_auc": sep})
        elif cold_frac >= COLD_START_FRAC:
            bottleneck = ("primary bottleneck is cold-start histories", {"len0_fraction": cold_frac})
        else:
            bottleneck = ("primary bottleneck is candidate retrieval quality", {"history_recall_coverage": history_ratio})
        report.primary_bottleneck = _conclusion(
            bottleneck[0], bottleneck[1], "medium", "heuristic ranking; may flip with richer features"
        )

        report.secondary_problems = []
        dedup_ratio = (
            report.coverage_map["raw_vs_dedup"]["train_mean_dedup_len"]
            / max(report.coverage_map["raw_vs_dedup"]["train_mean_raw_len"], 1e-12)
        )
        if dedup_ratio < 0.9:
            report.secondary_problems.append(
                _conclusion(
                    "histories contain many duplicate items",
                    {"dedup_ratio": _f(dedup_ratio)},
                    "high",
                    "duplicates may be meaningful re-consumption, not noise",
                )
            )
        if report.coverage_map["novel_target_ratio"] > 0.5:
            report.secondary_problems.append(
                _conclusion(
                    "majority of targets are novel w.r.t. user history",
                    {"novel_target_ratio": report.coverage_map["novel_target_ratio"]},
                    "medium",
                    "novelty may be an artifact of short histories",
                )
            )

        report.headroom_map = {
            "history_recall": "high" if history_ratio < 0.5 else "low",
            "popularity_prior": "high" if head_share >= POPULARITY_HEAD_SHARE else "medium",
            "sequence_modeling": "low" if cold_frac >= COLD_START_FRAC else "medium",
        }
        report.avoid_list = []
        if cold_frac >= COLD_START_FRAC:
            report.avoid_list.append("long-context sequence encoders (most histories are empty)")
        if sep >= SHIFT_SEPARABILITY_HIGH:
            report.avoid_list.append("trusting random-split offline gains for deployment")
        report.model_search_prior = sorted(
            [
                {"family": "popularity_baseline", "prior": 0.7},
                {"family": "history_recall", "prior": 0.8 if history_ratio >= 0.5 else 0.4},
                {"family": "sequence_model", "prior": 0.3 if cold_frac >= COLD_START_FRAC else 0.6},
                {"family": "fusion", "prior": 0.6},
            ],
            key=lambda d: -d["prior"],
        )
        report.validation_protocol = {
            "protocol": "time_aware_split" if sep >= SHIFT_SEPARABILITY_MODERATE else "random_split",
            "seed": SEED,
            "notes": "propensity separability_auc >= 0.65 suggests time-aware validation",
        }
        report.initial_hypotheses = [
            {
                "hypothesis": "history-recall features give strong target coverage"
                if history_ratio >= 0.5
                else "novel-target discovery is the main headroom",
                "evidence_id": report.signal_reliability_map["history_recall"]["evidence_id"],
            }
        ]
        report.status = "verified"
    except Exception as exc:  # pragma: no cover - defensive
        report.status = "error"
        report.errors.append(f"{type(exc).__name__}: {exc}")
    return report


# ---------------------------------------------------------------------------
# LLM explanation (read-only guard)
# ---------------------------------------------------------------------------


def llm_explain(report: DataIntelligenceReport, explain_fn: Callable[[dict[str, Any]], str]) -> str:
    """Run explain_fn on the report dict; the report itself must not change.

    The report hash is snapshotted before and after the call; any mutation
    of the report object (e.g. through the shared section dicts in
    ``report.to_dict()``) raises ReportMutationError.
    """
    before = report.report_hash()
    text = explain_fn(report.to_dict())
    after = report.report_hash()
    if after != before:
        raise ReportMutationError("explain_fn mutated the DataIntelligenceReport")
    return text
