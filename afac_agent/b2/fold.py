# -*- coding: utf-8 -*-
"""AFAC_B2_FOLD_V1 and multi-panel evaluation protocol.

- Looks for a trusted fold definition in the B2 data directory first.
- If none exists, builds a stratified user-group-aware 5-fold with seed 2026,
  hash-stable, with no history/target leakage.
- Defines diagnostic panels for cold-start, history/novel targets, long-tail,
  test-like, and Top10-boundary regimes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..research.event_store import stable_hash
from .task_adapter import B2Dataset

AFAC_B2_FOLD_V1 = "AFAC_B2_FOLD_V1"
RANDOM_STATE = 2026
N_SPLITS = 5


@dataclass
class B2Folds:
    uids: np.ndarray
    folds: np.ndarray  # values 0..4, aligned to uids
    fold_hash: str
    stratify_bins: np.ndarray
    panel_configs: dict[str, Any]


def _trusted_fold_path(data_root: Path) -> Path | None:
    """Search for an existing trusted fold artifact in the data directory."""
    candidates = sorted(data_root.glob("*AFAC_B2_FOLD_V1*.json"))
    if candidates:
        return candidates[0]
    return None


def _sequence_length_bucket(lengths: np.ndarray) -> np.ndarray:
    bins = np.zeros(len(lengths), dtype=np.int64)
    for i, length in enumerate(lengths):
        if length == 0:
            bins[i] = 0
        elif length <= 3:
            bins[i] = 1
        elif length <= 10:
            bins[i] = 2
        elif length <= 50:
            bins[i] = 3
        elif length <= 100:
            bins[i] = 4
        else:
            bins[i] = 5
    return bins


def _robust_stratified_folds(uids: np.ndarray, bins: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.full(len(uids), -1, dtype=np.int64)
    unique_bins = np.unique(bins)
    for b in unique_bins:
        mask = bins == b
        idx = np.where(mask)[0]
        n = len(idx)
        if n < N_SPLITS:
            # Too few samples for a full stratified split: assign round-robin by sorted uid hash
            h = np.array([stable_hash(u) for u in uids[idx]], dtype=object)
            order = np.argsort(h)
            folds[idx[order]] = np.arange(n) % N_SPLITS
            continue
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        y = bins[idx]
        for fold, (_, val_idx) in enumerate(skf.split(np.zeros(n), y)):
            folds[idx[val_idx]] = fold
    # Verify all assigned
    assert (folds >= 0).all(), "some users did not receive a fold assignment"
    return folds


def build_folds(dataset: B2Dataset, *, seed: int = RANDOM_STATE) -> B2Folds:
    data_root = dataset.data_root
    trusted = _trusted_fold_path(data_root)
    if trusted is not None:
        payload = json.loads(trusted.read_text(encoding="utf-8"))
        uids = np.asarray(payload["uids"], dtype=str)
        folds = np.asarray(payload["folds"], dtype=np.int64)
        fold_hash = payload.get("fold_hash", stable_hash({"uids": uids.tolist(), "folds": folds.tolist()}))
        return B2Folds(uids=uids, folds=folds, fold_hash=fold_hash, stratify_bins=np.zeros(len(uids)), panel_configs={})

    train_uids = dataset.train_df["uid"].astype(str).values
    seq_lengths = np.array([len(dataset.train_seq.get(uid, [])) for uid in train_uids], dtype=np.int64)
    bins = _sequence_length_bucket(seq_lengths)
    order = np.argsort(train_uids)
    uids = train_uids[order]
    bins = bins[order]
    folds = _robust_stratified_folds(uids, bins, seed=seed)
    fold_hash = stable_hash({"fold_identity": AFAC_B2_FOLD_V1, "uids": uids.tolist(), "folds": folds.tolist()})
    return B2Folds(uids=uids, folds=folds, fold_hash=fold_hash, stratify_bins=bins, panel_configs={})


def _propensity_scores(dataset: B2Dataset) -> pd.Series:
    """Train-vs-test propensity per user based on sequence length and item popularity."""
    from .data_intelligence import _item_popularity
    pop = _item_popularity(dataset)
    records = []
    for split, seqs in (("train", dataset.train_seq), ("test", dataset.test_seq)):
        for uid, seq in seqs.items():
            records.append({
                "uid": uid,
                "split": 1 if split == "test" else 0,
                "len": len(seq),
                "pop_mean": np.mean([pop.get(iid, 0) for iid in seq]) if seq else 0.0,
            })
    meta = pd.DataFrame(records)
    if meta.empty:
        return pd.Series(0.0, index=meta["uid"] if "uid" in meta.columns else [])
    X = meta[["len", "pop_mean"]].fillna(0).values
    y = meta["split"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=RANDOM_STATE)
    clf.fit(Xs, y)
    prob = clf.predict_proba(Xs)[:, 1]
    return pd.Series(prob, index=meta["uid"])


def build_panels(dataset: B2Dataset, folds: B2Folds) -> dict[str, Any]:
    train_uids = folds.uids
    uid2idx = {uid: i for i, uid in enumerate(train_uids)}
    n = len(train_uids)

    seq_lengths = np.array([len(dataset.train_seq.get(uid, [])) for uid in train_uids], dtype=np.int64)
    targets = {str(uid): str(tid) for uid, tid in zip(dataset.train_df["uid"], dataset.train_df["target_iid"])} if "target_iid" in dataset.train_df.columns else {}

    from .data_intelligence import _item_popularity
    pop = _item_popularity(dataset)
    item2rank = {iid: r for r, iid in enumerate(pop.index, start=1)}

    prop = _propensity_scores(dataset)
    prop_train = prop.reindex(train_uids).fillna(0).values

    panels: dict[str, Any] = {}

    # Standard: uniform
    panels["B2_STANDARD_PANEL"] = {"type": "uniform", "mask": np.ones(n, dtype=bool)}

    # Short history: length <= 3
    panels["B2_SHORT_HISTORY_PANEL"] = {"type": "subset", "mask": seq_lengths <= 3}

    # Exact length 3
    panels["B2_EXACT_LEN3_PANEL"] = {"type": "subset", "mask": seq_lengths == 3}

    # Length >= 4
    panels["B2_LEN4_PLUS_PANEL"] = {"type": "subset", "mask": seq_lengths >= 4}

    # History target: target appears in user's history
    hist_mask = np.zeros(n, dtype=bool)
    for i, uid in enumerate(train_uids):
        target = targets.get(uid)
        if target and target in dataset.train_seq.get(uid, []):
            hist_mask[i] = True
    panels["B2_HISTORY_TARGET_PANEL"] = {"type": "subset", "mask": hist_mask}

    # Novel target: target does not appear in user's history
    novel_mask = np.zeros(n, dtype=bool)
    for i, uid in enumerate(train_uids):
        target = targets.get(uid)
        if target and target not in dataset.train_seq.get(uid, []):
            novel_mask[i] = True
    panels["B2_NOVEL_TARGET_PANEL"] = {"type": "subset", "mask": novel_mask}

    # Long-tail target: target popularity rank in bottom 50%
    n_items = dataset.n_items
    lt_mask = np.zeros(n, dtype=bool)
    for i, uid in enumerate(train_uids):
        target = targets.get(uid)
        if target and item2rank.get(target, n_items) > n_items * 0.5:
            lt_mask[i] = True
    panels["B2_LONG_TAIL_PANEL"] = {"type": "subset", "mask": lt_mask}

    # Test-like panel: highest propensity train users
    k = max(1, int(0.2 * n))
    test_like_idx = np.argsort(-prop_train)[:k]
    test_like_mask = np.zeros(n, dtype=bool)
    test_like_mask[test_like_idx] = True
    panels["B2_TEST_LIKE_PANEL"] = {"type": "subset", "mask": test_like_mask}

    # Top10 boundary panel: target rank near 10 (approximate exploration of boundary cases)
    boundary_mask = np.zeros(n, dtype=bool)
    for i, uid in enumerate(train_uids):
        target = targets.get(uid)
        if target:
            rank = item2rank.get(target, n_items)
            if 5 <= rank <= 20:
                boundary_mask[i] = True
    panels["B2_TOP10_BOUNDARY_PANEL"] = {"type": "subset", "mask": boundary_mask}

    out = {"fold_identity": AFAC_B2_FOLD_V1, "fold_hash": folds.fold_hash}
    for pid, cfg in panels.items():
        out[pid] = {"type": cfg["type"], "mask": cfg["mask"].tolist()}
    folds.panel_configs = panels
    return out


def validation_mask(panel_id: str, folds: B2Folds, *, held_fold: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask) aligned to ``folds.uids``.

    If ``held_fold`` is None, the whole train set is used (final training).
    """
    n = folds.uids.size
    standard_train = np.ones(n, dtype=bool) if held_fold is None else (folds.folds != held_fold)
    standard_val = np.ones(n, dtype=bool) if held_fold is None else (folds.folds == held_fold)

    if panel_id == "B2_STANDARD_PANEL" or panel_id not in folds.panel_configs:
        return standard_train, standard_val

    cfg = folds.panel_configs[panel_id]
    if cfg["type"] == "uniform":
        return standard_train, standard_val
    if cfg["type"] == "subset":
        mask = cfg["mask"]
        if held_fold is None:
            return np.ones(n, dtype=bool), mask
        return standard_train, mask & standard_val

    raise ValueError(f"unknown panel config type: {cfg['type']}")
