# -*- coding: utf-8 -*-
"""AFAC_B1_FOLD_V1 and multi-panel validation protocol.

- Stratified 5-fold on train_idx/labels, fixed seed, hash-stable.
- Diagnostic panels: degree-matched, propensity-matched, low-degree, test-like.
- No Test node leakage; no held-out labels used for constructing folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..research.event_store import stable_hash
from .intelligence_core import _pagerank
from .task_adapter import NodeClassificationDataset

AFAC_B1_FOLD_V1 = "AFAC_B1_FOLD_V1"
RANDOM_STATE = 2026


@dataclass
class B1Folds:
    train_idx: np.ndarray
    folds: np.ndarray  # same length as train_idx, values 0..4
    fold_hash: str
    panel_configs: dict[str, Any]


def build_folds(dataset: NodeClassificationDataset) -> B1Folds:
    y = dataset.labels[dataset.train_idx]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = np.full(dataset.train_idx.size, -1, dtype=np.int64)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros_like(y), y)):
        folds[val_idx] = fold
    order_hash = stable_hash({"train_order": dataset.train_idx.tolist(), "folds": folds.tolist()})
    return B1Folds(
        train_idx=dataset.train_idx,
        folds=folds,
        fold_hash=order_hash,
        panel_configs={},
    )


def build_panels(dataset: NodeClassificationDataset, folds: B1Folds) -> dict[str, Any]:
    n = dataset.n_nodes
    train_mask = np.zeros(n, dtype=bool)
    train_mask[dataset.train_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[dataset.test_idx] = True

    deg_out = np.diff(dataset.adj.indptr).astype(np.float64)
    deg_in = np.diff(dataset.adj.T.tocsr().indptr).astype(np.float64)
    deg_und = np.diff(dataset.directed_adj("undirected_union").indptr).astype(np.float64)
    pr = _pagerank(dataset.directed_adj("undirected_union"))

    # Propensity score via cross-fit logistic on degree+PageRank+feature norm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold as SKF
    from sklearn.preprocessing import StandardScaler
    feat_norm = np.sqrt(np.asarray(dataset.features.multiply(dataset.features).sum(axis=1)).ravel())
    X_meta = np.column_stack([np.log1p(deg_out), np.log1p(deg_in), np.log1p(pr), feat_norm])
    y_meta = test_mask.astype(np.int64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_meta)
    propensity = np.zeros(n)
    skf = SKF(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(Xs, y_meta):
        model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=RANDOM_STATE)
        model.fit(Xs[tr], y_meta[tr])
        propensity[va] = model.predict_proba(Xs[va])[:, 1]

    panels: dict[str, Any] = {}
    k = max(1, int(0.2 * dataset.train_idx.size))
    prop_train = propensity[dataset.train_idx]
    deg_train = deg_und[dataset.train_idx]

    # Standard: each fold's validation set is the fold itself; weights uniform.
    panels["B1_STANDARD_PANEL"] = {"type": "uniform", "weights": np.ones(dataset.train_idx.size)}

    # Degree-matched: up-weight validation nodes whose degree is under-represented vs test
    test_deg_dist = np.histogram(deg_und[test_mask], bins=20, range=(0, deg_und.max()))[0].astype(np.float64)
    test_deg_dist /= test_deg_dist.sum() + 1e-12
    train_deg_hist = np.histogram(deg_train, bins=20, range=(0, deg_und.max()))[0].astype(np.float64)
    train_deg_hist /= train_deg_hist.sum() + 1e-12
    ratio = test_deg_dist / (train_deg_hist + 1e-12)
    deg_bin_idx = np.clip((deg_train / (deg_und.max() + 1e-12) * 20).astype(int), 0, 19)
    deg_weights = ratio[deg_bin_idx]
    panels["B1_DEGREE_MATCHED_PANEL"] = {"type": "degree_weighted", "weights": deg_weights}

    # Propensity-matched: focus validation on highest-propensity train nodes
    prop_mask = np.zeros(dataset.train_idx.size, dtype=bool)
    prop_mask[np.argsort(-prop_train)[:k]] = True
    panels["B1_PROPENSITY_MATCHED_PANEL"] = {"type": "subset", "mask": prop_mask}

    # Low-degree panel
    low_mask = np.zeros(dataset.train_idx.size, dtype=bool)
    low_mask[np.argsort(deg_train)[:k]] = True
    panels["B1_LOW_DEGREE_PANEL"] = {"type": "subset", "mask": low_mask}

    # Test-like panel
    test_like_mask = np.zeros(dataset.train_idx.size, dtype=bool)
    test_like_mask[np.argsort(-prop_train)[:k]] = True
    panels["B1_TEST_LIKE_PANEL"] = {"type": "subset", "mask": test_like_mask}

    # Convert arrays to compact JSON-friendly dict
    out = {"fold_identity": AFAC_B1_FOLD_V1, "fold_hash": folds.fold_hash}
    for pid, cfg in panels.items():
        out[pid] = {"type": cfg["type"]}
        if "weights" in cfg:
            out[pid]["weights"] = cfg["weights"].tolist()
        if "mask" in cfg:
            out[pid]["mask"] = cfg["mask"].tolist()
    folds.panel_configs = panels
    return out


def validation_mask(panel_id: str, folds: B1Folds, *, held_fold: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask) relative to train_idx for a panel.

    If held_fold is None, the whole train set is used (e.g. for final training).
    """
    n = folds.train_idx.size
    if panel_id == "B1_STANDARD_PANEL" or panel_id not in folds.panel_configs:
        if held_fold is None:
            return np.ones(n, dtype=bool), np.ones(n, dtype=bool)
        return folds.folds != held_fold, folds.folds == held_fold
    cfg = folds.panel_configs[panel_id]
    if cfg["type"] == "uniform":
        base_train = np.ones(n, dtype=bool) if held_fold is None else (folds.folds != held_fold)
        base_val = np.ones(n, dtype=bool) if held_fold is None else (folds.folds == held_fold)
        return base_train, base_val
    if cfg["type"] == "degree_weighted":
        # degree-matched weights only affect validation evaluation, not selection mask
        base_train = np.ones(n, dtype=bool) if held_fold is None else (folds.folds != held_fold)
        base_val = np.ones(n, dtype=bool) if held_fold is None else (folds.folds == held_fold)
        return base_train, base_val
    # subset panels: restrict validation to subset; train on rest of train nodes outside held fold
    mask = cfg["mask"]
    if held_fold is None:
        return np.ones(n, dtype=bool), mask
    return (folds.folds != held_fold), (folds.folds == held_fold) & mask
