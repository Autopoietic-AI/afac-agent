# -*- coding: utf-8 -*-
"""Synthetic tests for AFAC v2.0 classification operators + data intelligence."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from afac_agent.v2.data_intelligence import (
    DataIntelligenceReport,
    ReportMutationError,
    _separability_auc,
    analyze_classification,
    analyze_recommendation,
    llm_explain,
)
from afac_agent.v2.operators.classification import (
    APPNPOp,
    CLASSIFICATION_OPERATORS,
    GBDTOp,
    GraphViewRegistry,
    HeterophilyGNNOp,
    LabelPropagationOp,
    LinearOp,
    OperatorUnavailable,
    PCALowRankOp,
    ProbabilityBlendOp,
    PrototypeOp,
    ResidualMLPOp,
    SGCOp,
    cross_fit,
)

SEED = 2026
ALL_SECTIONS = [
    "dataset_fingerprint",
    "regime_profile",
    "signal_reliability_map",
    "coverage_map",
    "shift_map",
    "validation_protocol",
    "primary_bottleneck",
    "secondary_problems",
    "headroom_map",
    "avoid_list",
    "model_search_prior",
    "initial_hypotheses",
]


def _separable_features(n: int = 60, d: int = 20, n_classes: int = 3):
    rng = np.random.default_rng(SEED)
    y = np.tile(np.arange(n_classes), n // n_classes)
    centers = rng.normal(0.0, 1.0, size=(n_classes, d)) * 3.0
    X = centers[y] + rng.normal(0.0, 0.5, size=(n, d))
    return X, y


def _sbm_graph(n: int = 60, n_classes: int = 3, p_in: float = 0.4, p_out: float = 0.01):
    """Directed stochastic-block graph; labels = block id."""
    rng = np.random.default_rng(SEED)
    y = np.tile(np.arange(n_classes), n // n_classes)
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = p_in if y[i] == y[j] else p_out
            if rng.random() < p:
                rows.append(i)
                cols.append(j)
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return adj, y


def _weak_features(y: np.ndarray, d: int = 10, signal: float = 0.6):
    """Features with weak-but-present class signal (LR above chance)."""
    rng = np.random.default_rng(SEED + 1)
    n_classes = int(y.max() + 1)
    centers = rng.normal(0.0, 1.0, size=(n_classes, d)) * signal
    return centers[y] + rng.normal(0.0, 1.0, size=(len(y), d))


def _train_test_split(n: int, frac: float = 0.5):
    rng = np.random.default_rng(SEED + 2)
    perm = rng.permutation(n)
    n_train = int(n * frac)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def _acc(P: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    return float((P[idx].argmax(axis=1) == y[idx]).mean())


# ---------------------------------------------------------------------------
# Feature operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_cls", [LinearOp, GBDTOp, ResidualMLPOp, PCALowRankOp, PrototypeOp])
def test_feature_ops_beat_chance(op_cls):
    X, y = _separable_features()
    train_idx, test_idx = _train_test_split(len(y))
    op = op_cls().fit(X[train_idx], y[train_idx])
    P = op.predict_proba(X)
    assert P.shape == (len(y), 3)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert _acc(P, y, test_idx) > 1.0 / 3.0 + 0.2  # well above chance


# ---------------------------------------------------------------------------
# Graph operators on a homophilic SBM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_cls", [LabelPropagationOp, APPNPOp, SGCOp])
def test_graph_ops_beat_feature_only_chance(op_cls):
    adj, y = _sbm_graph()
    X = _weak_features(y)
    train_idx, test_idx = _train_test_split(len(y))
    linear = LinearOp().fit(X[train_idx], y[train_idx])
    linear_acc = _acc(linear.predict_proba(X), y, test_idx)
    op = op_cls().fit(adj, X, y, train_idx)
    P = op.predict_proba(X)
    assert P.shape == (len(y), 3)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    acc = _acc(P, y, test_idx)
    assert acc > 1.0 / 3.0 + 0.15  # clearly above chance
    assert acc >= linear_acc  # homophilic graph should not hurt vs weak features


def test_heterophily_gnn_stub_raises():
    adj, y = _sbm_graph()
    X = _weak_features(y)
    train_idx, _ = _train_test_split(len(y))
    with pytest.raises(OperatorUnavailable):
        HeterophilyGNNOp().fit(adj, X, y, train_idx)


# ---------------------------------------------------------------------------
# Graph view registry
# ---------------------------------------------------------------------------


def _nonsymmetric_adj():
    rows = [0, 1, 2, 3, 1]
    cols = [1, 2, 0, 1, 3]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(5, 5))


def test_directed_view_hashes_distinct():
    adj = _nonsymmetric_adj()
    _, h_out = GraphViewRegistry.build_view("directed_out", adj)
    _, h_in = GraphViewRegistry.build_view("directed_in", adj)
    _, h_union = GraphViewRegistry.build_view("undirected_union", adj)
    assert h_out != h_in
    assert h_union != h_out  # directed graph: symmetrized view must differ


def test_exact_two_hop_excludes_direct_edges():
    # Path 0-1-2 (undirected): exact two-hop should contain only (0, 2).
    rows = [0, 1, 1, 2]
    cols = [1, 0, 2, 1]
    adj = csr_matrix((np.ones(4), (rows, cols)), shape=(3, 3))
    two_hop, _ = GraphViewRegistry.build_view("exact_two_hop", adj)
    dense = two_hop.toarray()
    assert dense[0, 2] == 1.0 and dense[2, 0] == 1.0
    assert dense[0, 1] == 0.0 and dense[1, 0] == 0.0  # direct edges removed
    assert dense[1, 2] == 0.0 and dense[2, 1] == 0.0
    assert np.all(np.diag(dense) == 0.0)  # no self-loops


# ---------------------------------------------------------------------------
# Fusion: APPNP + Linear cross-fit blend
# ---------------------------------------------------------------------------


def test_appnp_cross_fit_fusion_valid_probabilities():
    adj, y = _sbm_graph()
    X = _weak_features(y)
    n = len(y)
    train_idx, test_idx = _train_test_split(n)
    # Two outer folds over the train indices; OOF probas from Linear + APPNP.
    folds = np.array_split(train_idx, 2)
    pa = np.zeros((len(train_idx), 3))
    pb = np.zeros((len(train_idx), 3))
    for i, val in enumerate(folds):
        fit = np.concatenate([folds[j] for j in range(2) if j != i])
        pa[[np.where(train_idx == v)[0][0] for v in val]] = LinearOp().fit(X[fit], y[fit]).predict_proba(X[val])
        appnp = APPNPOp().fit(adj, X, y, fit)
        pb[[np.where(train_idx == v)[0][0] for v in val]] = appnp.predict_proba(X)[val]
    pos = {v: k for k, v in enumerate(train_idx)}
    fold_rows = [np.array([pos[v] for v in f]) for f in folds]
    fused = cross_fit(ProbabilityBlendOp(), pa, pb, y[train_idx], fold_rows)
    assert fused.shape == (len(train_idx), 3)
    assert np.all(fused >= 0.0) and np.all(fused <= 1.0)
    np.testing.assert_allclose(fused.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_heterophily_gnn_unavailable():
    entry = CLASSIFICATION_OPERATORS["heterophily_gnn"]
    assert entry["implemented"] is False
    assert entry["available"] is False
    assert entry["missing_dependency"] == ["torch"]
    for op_id, e in CLASSIFICATION_OPERATORS.items():
        for key in (
            "family",
            "implemented",
            "available",
            "missing_dependency",
            "expected_runtime_seconds",
            "supports_oof",
            "supports_test",
            "deployment_ready",
            "scientific_priority",
        ):
            assert key in e, f"{op_id} missing catalog key {key}"


# ---------------------------------------------------------------------------
# Data intelligence: classification
# ---------------------------------------------------------------------------


def test_classification_di_report_sections_and_shift():
    adj, y = _sbm_graph()
    X = _weak_features(y)
    n = len(y)
    train_idx, test_idx = _train_test_split(n)
    # Inflate test-node feature norms to force a detectable shift signal.
    X_shift = X.copy()
    X_shift[test_idx] += 5.0
    report = analyze_classification(X_shift, adj, y, train_idx, test_idx)
    assert isinstance(report, DataIntelligenceReport)
    assert report.status == "verified"
    assert report.errors == []
    d = report.to_dict()
    for section in ALL_SECTIONS:
        assert section in d
    shift = report.shift_map
    assert 0.0 <= shift["raw_auc"] <= 1.0
    assert shift["separability_auc"] == pytest.approx(max(shift["raw_auc"], 1.0 - shift["raw_auc"]))
    assert shift["separability_auc"] > 0.8  # the injected +5 norm shift is detected
    assert report.report_hash() == report.report_hash()  # deterministic


def test_separability_auc_inverted_classifier():
    # Inverted propensity classifier (raw < 0.5) must map to > 0.5 separability.
    assert _separability_auc(0.1) == pytest.approx(0.9)
    assert _separability_auc(0.9) == pytest.approx(0.9)
    assert _separability_auc(0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Data intelligence: recommendation
# ---------------------------------------------------------------------------


def test_recommendation_di_buckets_and_novel_ratio():
    train_seq = [
        [],  # len0
        ["a"],  # len1
        ["a", "b"],  # len2
        ["a", "b", "c"],  # exact_len3
        ["a", "b", "c", "d"],  # len4_plus
        ["a", "b", "c", "d", "e"],
    ]
    # targets: 2 in history ("a" at idx 1 and 2), 4 novel
    train_targets = ["z0", "a", "a", "z2", "z3", "z4"]
    test_seq = [["a"], [], ["x", "y"], ["a", "b", "c", "d"]]
    report = analyze_recommendation(train_seq, train_targets, test_seq)
    assert report.status == "verified"
    d = report.to_dict()
    for section in ALL_SECTIONS:
        assert section in d
    counts = report.coverage_map["train_length_buckets"]["counts"]
    assert counts == {"len0": 1, "len1": 1, "len2": 1, "exact_len3": 1, "len4_plus": 2}
    history_ratio = report.coverage_map["history_recall_coverage"]
    assert history_ratio == pytest.approx(2 / 6)
    assert report.coverage_map["novel_target_ratio"] == pytest.approx(4 / 6)
    sep = report.shift_map
    assert sep["separability_auc"] == pytest.approx(max(sep["raw_auc"], 1.0 - sep["raw_auc"]))


# ---------------------------------------------------------------------------
# llm_explain
# ---------------------------------------------------------------------------


def test_llm_explain_returns_text_and_report_unchanged():
    adj, y = _sbm_graph()
    X = _weak_features(y)
    train_idx, test_idx = _train_test_split(len(y))
    report = analyze_classification(X, adj, y, train_idx, test_idx)
    before = report.report_hash()
    text = llm_explain(report, lambda d: f"task={d['task']} status={d['status']}")
    assert "task=classification" in text
    assert report.report_hash() == before


def test_llm_explain_mutating_fn_raises():
    adj, y = _sbm_graph()
    X = _weak_features(y)
    train_idx, test_idx = _train_test_split(len(y))
    report = analyze_classification(X, adj, y, train_idx, test_idx)

    def bad_explain(d):
        d["regime_profile"]["graph_regime"] = "tampered"
        return "oops"

    with pytest.raises(ReportMutationError):
        llm_explain(report, bad_explain)
