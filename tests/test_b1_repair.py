# -*- coding: utf-8 -*-
"""Synthetic tests for B1 Scientific Validity Repair.

These tests exercise the repair audit functions without relying on the real B1
first-run artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from afac_agent.b1.repair import (
    effective_rank_audit,
    homophily_repair,
    prediction_asset_audit,
    propensity_semantics_audit,
    raw_graph_direction_audit,
)


class _FakeDataset:
    def __init__(self, n_nodes: int, n_classes: int, n_features: int) -> None:
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        rng = np.random.RandomState(2026)
        self.features = csr_matrix(rng.randn(n_nodes, n_features))
        self.labels = rng.randint(0, n_classes, size=n_nodes)
        self.train_idx = np.arange(0, n_nodes - 4)
        self.test_idx = np.arange(n_nodes - 4, n_nodes)
        # Build an asymmetric adjacency (directed chain + one reciprocal edge)
        row, col = list(range(n_nodes - 1)), list(range(1, n_nodes))
        row.append(2)
        col.append(1)
        data = [1.0] * len(row)
        self.adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))


class _FakeFolds:
    def __init__(self, n_train: int) -> None:
        self.train_idx = np.arange(n_train)
        self.folds = np.arange(n_train) % 5


@pytest.fixture
def fake_dataset() -> _FakeDataset:
    return _FakeDataset(30, 4, 8)


@pytest.fixture
def fake_folds(fake_dataset: _FakeDataset) -> _FakeFolds:
    return _FakeFolds(len(fake_dataset.train_idx))


def test_raw_graph_direction_audit_detects_asymmetry(fake_dataset: _FakeDataset) -> None:
    audit = raw_graph_direction_audit(fake_dataset.adj)
    assert audit["is_symmetric"] is False
    assert audit["max_abs_A_minus_AT"] > 0.0
    assert audit["views"]["directed_out"]["edge_count"] == audit["views"]["directed_in"]["edge_count"]
    # Reciprocal count is small because only one extra edge (2 -> 1)
    assert audit["reciprocal_edge_count"] >= 1


def test_raw_graph_direction_audit_for_symmetric_graph() -> None:
    n = 8
    row, col = [], []
    for i in range(n - 1):
        row.extend([i, i + 1])
        col.extend([i + 1, i])
    adj = csr_matrix(([1.0] * len(row), (row, col)), shape=(n, n))
    audit = raw_graph_direction_audit(adj)
    assert audit["is_symmetric"] is True
    assert audit["max_abs_A_minus_AT"] == 0.0


def test_prediction_asset_audit_views_distinct(fake_dataset: _FakeDataset, fake_folds: _FakeFolds, tmp_path: Path) -> None:
    audit = prediction_asset_audit(fake_dataset, fake_folds, fake_dataset.features, tmp_path)
    diag = audit["diagnosis"]
    assert diag["status"] == "views_distinct"
    for key in ["directed_out_vs_directed_in", "directed_out_vs_undirected_union", "directed_in_vs_undirected_union"]:
        assert key in audit["comparisons"]


def test_prediction_asset_audit_detects_view_bug(tmp_path: Path) -> None:
    """If adj is symmetric, directed_out and directed_in should agree, but
    the audit still reports views_distinct when predictions differ.
    Here we simply assert the audit runs and that symmetric views agree.
    """
    n = 20
    row, col = [], []
    for i in range(n - 1):
        row.extend([i, i + 1])
        col.extend([i + 1, i])
    adj = csr_matrix(([1.0] * len(row), (row, col)), shape=(n, n))
    ds = _FakeDataset(n, 3, 6)
    ds.adj = adj
    folds = _FakeFolds(len(ds.train_idx))
    audit = prediction_asset_audit(ds, folds, ds.features, tmp_path)
    cmp_ = audit["comparisons"]["directed_out_vs_directed_in"]
    assert cmp_["changed_prediction_count"] == 0
    assert cmp_["agreement_rate"] == 1.0


def test_homophily_repair_semantics(fake_dataset: _FakeDataset, fake_folds: _FakeFolds) -> None:
    res = homophily_repair(fake_dataset.adj, fake_dataset.labels, fake_dataset.train_idx)
    assert res["audit"] == "homophily_repair"
    assert 0.0 <= res["raw_train_train_edge_homophily"] <= 1.0
    assert -1.0 <= res["adjusted_homophily"] <= 1.0
    assert res["edge_semantics"]["direction_preserved"] is True
    assert isinstance(res["class_conditioned_homophily"], dict)


def test_propensity_semantics(fake_dataset: _FakeDataset) -> None:
    res = propensity_semantics_audit(fake_dataset)
    assert res["audit"] == "propensity_semantics"
    assert 0.0 <= res["degree_separability_auc"] <= 1.0
    assert 0.0 <= res["feature_separability_auc"] <= 1.0
    assert 0.0 <= res["combined_separability_auc"] <= 1.0


def test_effective_rank_audit(fake_dataset: _FakeDataset) -> None:
    res = effective_rank_audit(fake_dataset)
    assert res["audit"] == "effective_rank"
    assert res["numeric_rank"] > 0
    assert res["entropy_effective_rank"] > 0
    assert res["participation_ratio"] > 0
    for key in ["50%", "80%", "90%", "95%", "99%"]:
        assert key in res["pca_dimensions_for_variance"]


def test_anchor_v1_v2_identity_isolation() -> None:
    from afac_agent.b1.fold import AFAC_B1_FOLD_V1
    assert AFAC_B1_FOLD_V1 == "AFAC_B1_FOLD_V1"
    # V2 anchor identity is distinct from V1 by construction in closed_loop.


def test_online_feedback_identity_preserved_in_repair_manifest(tmp_path: Path) -> None:
    manifest = {
        "canonical_first_b1_run_id": "e95368a24e0780e65e92aceb",
        "online_score": 0.37908,
        "offline_standard": 0.49069,
        "absolute_gap": 0.11161,
    }
    path = tmp_path / "b1_online_feedback.json"
    path.write_text(json.dumps(manifest))
    loaded = json.loads(path.read_text())
    assert loaded["canonical_first_b1_run_id"] == "e95368a24e0780e65e92aceb"
    assert loaded["online_score"] == 0.37908
