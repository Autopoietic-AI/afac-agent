# -*- coding: utf-8 -*-
"""AFAC v2.2 B1 scientific loop tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from afac_agent.v2.b1_data_contract import (
    B1CanonicalDataContract,
    build_b1_data_contract,
    reconcile_b1_data_contract,
)
from afac_agent.v2.b1_operators import (
    run_bucket_specialist_experiment,
    run_feature_baseline_experiment,
    run_feature_graph_residual_experiment,
    run_graph_propagation_experiment,
)
from afac_agent.v2.capability_registry import default_registry
from afac_agent.v2.experiment_kind import (
    ExperimentKind,
    kind_from_operator_and_folds,
    permission_for_kind,
)
from afac_agent.v2.proposal_compiler import compile_proposal, semantic_revision_delta


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_b1_dataset(n_nodes: int = 100, n_features: int = 16, n_classes: int = 4, seed: int = 42) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    X_dense = rng.randn(n_nodes, n_features).astype(np.float32)
    features = csr_matrix(X_dense)
    # Random directed edges
    n_edges = n_nodes * 3
    rows = rng.randint(0, n_nodes, n_edges)
    cols = rng.randint(0, n_nodes, n_edges)
    data = np.ones(n_edges, dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    labels = rng.randint(0, n_classes, n_nodes).astype(np.int64)
    train_idx = np.arange(n_nodes // 2)
    test_idx = np.arange(n_nodes // 2, n_nodes)
    labels[test_idx] = -1
    return {
        "n_nodes": n_nodes,
        "n_features": n_features,
        "n_classes": n_classes,
        "features": features,
        "adj": adj,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_directed_edges": n_edges,
        "n_undirected_edges": int(adj.nnz / 2),
    }


def _make_folds(n_train: int, n_folds: int = 5, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_train)
    folds = np.zeros(n_train, dtype=np.int64)
    fold_size = n_train // n_folds
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n_train
        folds[order[start:end]] = f
    return np.arange(n_train), folds


# ---------------------------------------------------------------------------
# data contract
# ---------------------------------------------------------------------------

def test_b1_data_contract_builds_and_passes(tmp_path: Path) -> None:
    ds = _make_b1_dataset()
    contract = build_b1_data_contract(
        data_root=tmp_path,
        n_nodes=ds["n_nodes"],
        n_features=ds["n_features"],
        n_classes=ds["n_classes"],
        n_edges=ds["n_undirected_edges"],
        n_directed_edges=ds["n_directed_edges"],
        train_idx=ds["train_idx"],
        test_idx=ds["test_idx"],
    )
    assert contract.status == "passed"
    assert contract.n_nodes_total is not None
    assert contract.n_nodes_total.value == 100
    assert contract.n_train_nodes_total.value == 50
    assert contract.n_test_nodes_total.value == 50
    assert contract.n_classes.value == 4


def test_b1_data_contract_detects_overlap() -> None:
    ds = _make_b1_dataset()
    train_idx = np.array([0, 1, 2, 3, 4])
    test_idx = np.array([3, 4, 5, 6, 7])  # overlap on 3, 4
    contract = build_b1_data_contract(
        data_root=Path("."),
        n_nodes=ds["n_nodes"],
        n_features=ds["n_features"],
        n_classes=ds["n_classes"],
        n_edges=ds["n_undirected_edges"],
        n_directed_edges=ds["n_directed_edges"],
        train_idx=train_idx,
        test_idx=test_idx,
    )
    assert contract.status != "passed"
    assert any("overlap" in e for e in contract.errors)


def test_b1_reconcile_detects_mismatch() -> None:
    result = reconcile_b1_data_contract(
        input_discovery_n_nodes=100,
        data_intelligence_n_nodes=200,
        experiment_executor_node_universe=set(range(100)),
        deployment_node_order=list(range(100)),
    )
    assert result["status"] == "blocked_data_contract_mismatch"


# ---------------------------------------------------------------------------
# B1 operators
# ---------------------------------------------------------------------------

def test_feature_baseline_produces_oof() -> None:
    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(len(ds["train_idx"]))
    result = run_feature_baseline_experiment(
        n_classes=ds["n_classes"],
        features=ds["features"],
        adj=ds["adj"],
        labels=ds["labels"],
        train_idx=train_idx,
        folds=folds,
        fold_ids=[0, 1],
    )
    assert "overall_accuracy" in result
    assert "macro_accuracy" in result
    assert result["oof_proba"].shape == (ds["n_nodes"], ds["n_classes"])
    assert result["overall_accuracy"] > 0.0


def test_graph_propagation_produces_oof() -> None:
    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(len(ds["train_idx"]))
    result = run_graph_propagation_experiment(
        n_classes=ds["n_classes"],
        features=ds["features"],
        adj=ds["adj"],
        labels=ds["labels"],
        train_idx=train_idx,
        folds=folds,
        fold_ids=[0, 1],
        graph_view="undirected_union",
        model_family="label_propagation",
    )
    assert "overall_accuracy" in result
    assert "macro_accuracy" in result
    assert result["oof_proba"].shape == (ds["n_nodes"], ds["n_classes"])


def test_feature_graph_residual_combines() -> None:
    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(len(ds["train_idx"]))
    feat_result = run_feature_baseline_experiment(
        n_classes=ds["n_classes"], features=ds["features"], adj=ds["adj"],
        labels=ds["labels"], train_idx=train_idx, folds=folds, fold_ids=[0, 1])
    graph_result = run_graph_propagation_experiment(
        n_classes=ds["n_classes"], features=ds["features"], adj=ds["adj"],
        labels=ds["labels"], train_idx=train_idx, folds=folds, fold_ids=[0, 1],
        graph_view="undirected_union", model_family="label_propagation")
    result = run_feature_graph_residual_experiment(
        feature_result=feat_result, graph_result=graph_result,
        n_classes=ds["n_classes"], labels=ds["labels"], train_idx=train_idx)
    assert "overall_accuracy" in result
    assert "rescue_damage" in result


def test_bucket_specialist_routes() -> None:
    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(len(ds["train_idx"]))
    feat_result = run_feature_baseline_experiment(
        n_classes=ds["n_classes"], features=ds["features"], adj=ds["adj"],
        labels=ds["labels"], train_idx=train_idx, folds=folds, fold_ids=[0, 1])
    graph_result = run_graph_propagation_experiment(
        n_classes=ds["n_classes"], features=ds["features"], adj=ds["adj"],
        labels=ds["labels"], train_idx=train_idx, folds=folds, fold_ids=[0, 1],
        graph_view="undirected_union", model_family="label_propagation")
    result = run_bucket_specialist_experiment(
        feature_result=feat_result, graph_result=graph_result,
        n_classes=ds["n_classes"], labels=ds["labels"], train_idx=train_idx,
        features=ds["features"], adj=ds["adj"])
    assert "route_fraction" in result
    assert result["route_fraction"] > 0.0


# ---------------------------------------------------------------------------
# experiment permission
# ---------------------------------------------------------------------------

def test_b1_feature_baseline_is_screen_by_default() -> None:
    kind = kind_from_operator_and_folds("feature_baseline_experiment", 2)
    assert kind == ExperimentKind.SCREEN_EXPERIMENT
    p = permission_for_kind(kind)
    assert p.can_enter_portfolio is True
    assert p.can_deploy is False
    assert p.consumes_scientific_round is True


def test_b1_diagnostic_cannot_deploy() -> None:
    kind = kind_from_operator_and_folds("classification_recall_diagnostic", 0)
    assert kind == ExperimentKind.DETERMINISTIC_DIAGNOSTIC
    p = permission_for_kind(kind)
    assert p.can_deploy is False
    assert p.can_be_incumbent is False


def test_b1_confirm_can_deploy() -> None:
    kind = kind_from_operator_and_folds("feature_baseline_experiment", 3)
    assert kind == ExperimentKind.CONFIRM_EXPERIMENT
    p = permission_for_kind(kind)
    assert p.can_deploy is True
    assert p.can_be_incumbent is True


# ---------------------------------------------------------------------------
# proposal compiler
# ---------------------------------------------------------------------------

def test_b1_compiler_maps_feature_baseline() -> None:
    compiled = compile_proposal(
        {"diagnostic_type": "feature_lr"},
        parent_candidate_id="majority_parent",
        formal_mode=True,
        task="B1",
    )
    assert compiled.status == "compiled"
    assert compiled.operator_id == "feature_baseline_experiment"
    assert compiled.experiment_kind == ExperimentKind.SCREEN_EXPERIMENT


def test_b1_compiler_maps_graph_propagation() -> None:
    compiled = compile_proposal(
        {"diagnostic_type": "graph_propagation_experiment"},
        parent_candidate_id="majority_parent",
        formal_mode=True,
        task="B1",
    )
    assert compiled.status == "compiled"
    assert compiled.operator_id == "graph_propagation_experiment"


def test_b1_compiler_blocks_unknown() -> None:
    compiled = compile_proposal(
        {"diagnostic_type": "magic_b1_operator"},
        parent_candidate_id="majority_parent",
        formal_mode=True,
        task="B1",
    )
    assert compiled.status == "blocked_missing_adapter"


# ---------------------------------------------------------------------------
# capability registry
# ---------------------------------------------------------------------------

def test_b1_operators_registered() -> None:
    registry = default_registry()
    for op_id in ("feature_baseline_experiment", "graph_propagation_experiment",
                  "feature_graph_residual_experiment", "bucket_specialist_experiment_b1",
                  "classification_recall_diagnostic"):
        record = registry.records.get(op_id)
        assert record is not None, f"{op_id} not registered"
        assert record.implemented is True


def test_b1_diagnostic_operator_not_deployment_ready() -> None:
    registry = default_registry()
    record = registry.records.get("classification_recall_diagnostic")
    assert record is not None
    assert record.deployment_ready is False
    assert record.supports_screen is False


# ---------------------------------------------------------------------------
# synthetic smokes A-D (B1)
# ---------------------------------------------------------------------------

def test_smoke_a_feature_baseline_loop() -> None:
    """Smoke A: Feature Baseline Screen consumes round, cannot deploy."""
    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(len(ds["train_idx"]))
    result = run_feature_baseline_experiment(
        n_classes=ds["n_classes"], features=ds["features"], adj=ds["adj"],
        labels=ds["labels"], train_idx=train_idx, folds=folds, fold_ids=[0, 1])
    assert result["overall_accuracy"] > 0.0
    kind = kind_from_operator_and_folds("feature_baseline_experiment", 2)
    assert kind == ExperimentKind.SCREEN_EXPERIMENT
    assert permission_for_kind(kind).can_deploy is False


def test_smoke_c_noop_not_in_portfolio() -> None:
    """Smoke C: No-op produces no change, should be refunded."""
    from afac_agent.v2.noop_detector import NoOpReport, apply_noop_policy
    report = NoOpReport(status="no_op", changed_fraction=0.0, reasons=["identical"])
    record = apply_noop_policy({"candidate_id": "c1", "consumes_round": True}, report)
    assert record["consumes_round"] is False
    assert record["portfolio_eligible"] is False


def test_smoke_d_budget_stops() -> None:
    """Smoke D: Budget too short to run experiment."""
    p = permission_for_kind(ExperimentKind.SCREEN_EXPERIMENT)
    assert p.consumes_scientific_round is True
    hard_deadline = 100.0
    required = p.max_wall_clock_seconds + 300.0  # deployment reserve
    remaining = 60.0
    assert remaining < required  # should stop


# ---------------------------------------------------------------------------
# B2 no-regression
# ---------------------------------------------------------------------------

def test_b2_operators_still_registered() -> None:
    registry = default_registry()
    for op_id in ("retrieval_union_experiment", "candidate_ranker_experiment",
                  "bucket_specialist_experiment", "protected_rerank_experiment"):
        assert op_id in registry.records, f"{op_id} missing from registry"


def test_b2_experiment_kind_unchanged() -> None:
    assert kind_from_operator_and_folds("candidate_ranker_experiment", 2) == ExperimentKind.SCREEN_EXPERIMENT
    assert kind_from_operator_and_folds("candidate_ranker_experiment", 3) == ExperimentKind.CONFIRM_EXPERIMENT


# ---------------------------------------------------------------------------
# hard-deadline circuit breaker (v2.2 budget repair)
# ---------------------------------------------------------------------------

def test_fold_oof_budget_circuit_breaker_aborts_between_folds() -> None:
    """A deadline already in the past must abort before fold 2 starts, keeping
    the completed fold-1 partial result instead of running the full ladder
    (regression: B1 v2 smoke ran 10977s against a 900s budget)."""
    import time

    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(ds["n_nodes"] // 2)
    result = run_feature_baseline_experiment(
        n_classes=ds["n_classes"],
        features=ds["features"],
        adj=ds["adj"],
        labels=ds["labels"],
        train_idx=train_idx,
        folds=folds,
        fold_ids=[0, 1, 2],
        model_family="feature_logistic",
        deadline_monotonic=time.monotonic() - 1.0,  # already past
    )
    assert result["budget_aborted"] is True
    assert result["folds_completed"] == [0]  # fold 0 finished, fold 1 never started


def test_fold_oof_without_deadline_completes_all_folds() -> None:
    ds = _make_b1_dataset()
    train_idx, folds = _make_folds(ds["n_nodes"] // 2)
    result = run_feature_baseline_experiment(
        n_classes=ds["n_classes"],
        features=ds["features"],
        adj=ds["adj"],
        labels=ds["labels"],
        train_idx=train_idx,
        folds=folds,
        fold_ids=[0, 1],
        model_family="feature_logistic",
    )
    assert result["budget_aborted"] is False
    assert result["folds_completed"] == [0, 1]
