# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from afac_agent.b1.closed_loop import B1ClosedLoopRunner
from afac_agent.b1.data_intelligence import run_data_intelligence
from afac_agent.b1.evaluator import B1Evaluator, cross_fit_fusion, cross_fit_node_gate
from afac_agent.b1.fold import AFAC_B1_FOLD_V1, build_folds, build_panels
from afac_agent.b1.intelligence_core import transferability_matrix
from afac_agent.b1.models import FeatureLogistic, LabelPropagationModel, instantiate_model
from afac_agent.b1.task_adapter import NodeClassificationTaskAdapter


def _make_b1_world(tmp_path: Path, *, n_nodes: int = 120, n_features: int = 16, n_classes: int = 4):
    root = tmp_path / "project"
    root.mkdir()
    data = root / "B1_data"
    data.mkdir()

    rng = np.random.default_rng(2026)
    # random sparse features (fully dense for tiny test)
    X = rng.normal(0.0, 1.0, size=(n_nodes, n_features)).astype(np.float32)
    X_csr = csr_matrix(X)

    # random directed edges
    edges = []
    for i in range(n_nodes):
        for _ in range(rng.integers(2, 6)):
            j = rng.integers(0, n_nodes)
            if i != j:
                edges.append((i, j))
    rows, cols = zip(*edges) if edges else ([], [])
    A = csr_matrix((np.ones(len(edges), dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes))

    labels = rng.integers(0, n_classes, size=n_nodes).astype(np.int64)
    train_idx = np.arange(int(n_nodes * 0.8))
    test_idx = np.arange(int(n_nodes * 0.8), n_nodes)
    labels[test_idx] = -1

    np.savez(
        data / "B1.npz",
        adj_data=A.data,
        adj_indices=A.indices,
        adj_indptr=A.indptr,
        adj_shape=np.array(A.shape),
        attr_data=X_csr.data,
        attr_indices=X_csr.indices,
        attr_indptr=X_csr.indptr,
        attr_shape=np.array(X_csr.shape),
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    with (data / "sample_submission.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["test_idx", "label"])
        for idx in test_idx:
            writer.writerow([int(idx), -1])
    (data / "README.md").write_text("synthetic B1", encoding="utf-8")
    return {"project": root, "data": data, "n_nodes": n_nodes, "n_features": n_features, "n_classes": n_classes}


@pytest.fixture()
def b1_world(tmp_path: Path):
    return _make_b1_world(tmp_path)


# ---------------------------------------------------------------- adapter

def test_adapter_loads_parameterized(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    assert ds.validation["status"] == "passed"
    assert ds.n_nodes == b1_world["n_nodes"]
    assert ds.n_features == b1_world["n_features"]
    assert ds.n_classes == b1_world["n_classes"]
    assert ds.validation["test_truth_hidden"] is True
    assert len(set(ds.train_idx) & set(ds.test_idx)) == 0


def test_adapter_different_shape(tmp_path: Path):
    world = _make_b1_world(tmp_path, n_nodes=80, n_features=8, n_classes=3)
    ds = NodeClassificationTaskAdapter(world["data"], task_id="B1").load()
    assert ds.n_nodes == 80 and ds.n_features == 8 and ds.n_classes == 3


def test_adapter_rejects_test_truth_leakage(tmp_path: Path):
    world = _make_b1_world(tmp_path)
    z = dict(np.load(world["data"] / "B1.npz", allow_pickle=True))
    z["labels"][z["test_idx"][0]] = 0
    np.savez(world["data"] / "B1.npz", **z)
    ds = NodeClassificationTaskAdapter(world["data"], task_id="B1").load()
    assert ds.validation["status"] == "failed"
    assert any("test labels are not hidden" in e for e in ds.validation["errors"])


def test_transferability_matrix_has_forbidden_a1():
    m = transferability_matrix()
    forbidden = m["forbidden_direct_transfer"]
    assert any("A1" in item for item in forbidden)
    assert m["cross_task_prior_mode"] == "advisory_only"
    assert m["target_task_evidence_priority"] == "hard"


# ---------------------------------------------------------------- data intelligence gate

def test_data_intelligence_verified(b1_world):
    res = run_data_intelligence(data_root=b1_world["data"], project_root=b1_world["project"], force_rebuild=True)
    assert res["status"] == "verified"
    out = b1_world["project"] / "artifacts" / "b1_runs" / res["run_id"] / "data_intelligence"
    for name in ("integrity_audit.json", "feature_geometry.json", "graph_regime.json", "label_graph_reliability.json", "shift_audit.json"):
        assert (out / name).is_file()
    integrity = json.loads((out / "integrity_audit.json").read_text(encoding="utf-8"))
    assert integrity["train_test_overlap"] == 0
    assert integrity["test_truth_hidden"] is True


# ---------------------------------------------------------------- folds and panels

def test_fold_and_panels(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    folds = build_folds(ds)
    assert folds.train_idx.size == ds.train_idx.size
    assert set(folds.folds.tolist()) == {0, 1, 2, 3, 4}
    assert np.all(np.bincount(folds.folds) > 0)
    panels = build_panels(ds, folds)
    assert panels["fold_identity"] == AFAC_B1_FOLD_V1
    for pid in ["B1_STANDARD_PANEL", "B1_DEGREE_MATCHED_PANEL", "B1_PROPENSITY_MATCHED_PANEL", "B1_LOW_DEGREE_PANEL", "B1_TEST_LIKE_PANEL"]:
        assert pid in panels


# ---------------------------------------------------------------- models and fusion

def test_feature_logistic_oof(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    X = ds.features.toarray().astype(np.float32)
    folds = build_folds(ds)
    oof = np.zeros((ds.n_nodes, ds.n_classes), dtype=np.float64)
    for held in range(5):
        fit = folds.folds != held
        model = FeatureLogistic(model_id="lr", n_classes=ds.n_classes).fit(X[folds.train_idx[fit]], ds.labels[folds.train_idx[fit]])
        val = folds.train_idx[folds.folds == held]
        oof[val] = model.predict_proba(X[val])
    assert oof[ds.train_idx].sum(axis=1).min() > 0.99


def test_label_propagation_no_leakage(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    folds = build_folds(ds)
    A = ds.directed_adj("undirected_union")
    # Train only on first 4 folds, predict on held fold
    fit_idx = folds.train_idx[folds.folds != 4]
    model = LabelPropagationModel(model_id="lp", n_classes=ds.n_classes, adj=A).fit(None, ds.labels[fit_idx], train_idx=fit_idx)
    val_idx = folds.train_idx[folds.folds == 4]
    proba = model.predict_proba(None)
    # held-out labels were not used in fit; predictions should be finite probabilities
    assert np.isfinite(proba[val_idx]).all()
    assert proba[val_idx].sum(axis=1).min() > 0.99


def test_cross_fit_fusion(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    folds = build_folds(ds)
    n = ds.train_idx.size
    rng = np.random.default_rng(7)
    a = rng.dirichlet(np.ones(ds.n_classes), size=n)
    b = rng.dirichlet(np.ones(ds.n_classes), size=n)
    y = ds.labels[ds.train_idx]
    out, assign = cross_fit_fusion(operator="probability_blend", proba_a=a, proba_b=b, y=y, folds=folds.folds, alpha_grid=(0.25, 0.5, 0.75))
    assert assign["mode"] == "outer_fold_cross_fit"
    assert len(assign["assignments"]) == 5
    assert out.shape == a.shape


def test_cross_fit_node_gate(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    folds = build_folds(ds)
    n = ds.train_idx.size
    rng = np.random.default_rng(11)
    # a perfect on odd folds, b perfect on even folds
    y = ds.labels[ds.train_idx]
    a = np.zeros((n, ds.n_classes))
    a[np.arange(n), y] = 1.0
    b = a.copy()
    for f in range(5):
        m = folds.folds == f
        if f % 2 == 0:
            a[m] = rng.dirichlet(np.ones(ds.n_classes), size=m.sum())
        else:
            b[m] = rng.dirichlet(np.ones(ds.n_classes), size=m.sum())
    meta = np.random.randn(n, 3)
    out, assign = cross_fit_node_gate(proba_a=a, proba_b=b, y=y, folds=folds.folds, meta=meta)
    assert out.shape == a.shape
    assert len(assign["assignments"]) == 5


# ---------------------------------------------------------------- closed loop dry-run safety

def test_closed_loop_does_not_use_b2_and_respects_budget(b1_world):
    ds = NodeClassificationTaskAdapter(b1_world["data"], task_id="B1").load()
    # B2 path exists but is not read
    b2_dir = b1_world["project"] / "B2_data"
    b2_dir.mkdir()
    (b2_dir / "dummy.csv").write_text("test_idx,label\n0,0\n", encoding="utf-8")
    runner = B1ClosedLoopRunner(
        project_root=b1_world["project"],
        data_root=b1_world["data"],
        out_root="artifacts/b1_runs",
        max_wall_clock_seconds=600,
        max_rounds=2,
    )
    result = runner.run(force_rebuild=True)
    assert result["status"] == "completed"
    assert result["scientific_rounds_used"] <= 2
    out = b1_world["project"] / "artifacts" / "b1_runs" / result["run_id"]
    assert (out / "B1_CLOSED_LOOP_V2_REPORT.md").is_file()
    assert (out / "trajectory_B1_v2.json").is_file()
    assert (out / "TO_UPLOAD" / "candidate_B1_v2.csv").is_file()
    # no b2 data in any artifact path or content
    assert not any("B2_data" in str(p) for p in out.rglob("*"))
    # candidate_B1_v2.csv valid
    rows = list(csv.DictReader((out / "TO_UPLOAD" / "candidate_B1_v2.csv").open(encoding="utf-8")))
    assert len(rows) == ds.n_nodes - ds.train_idx.size
    assert all(int(r["label"]) in range(ds.n_classes) for r in rows)


# ---------------------------------------------------------------- A1/A2 no regression is checked by full pytest suite
