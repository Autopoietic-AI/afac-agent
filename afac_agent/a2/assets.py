# -*- coding: utf-8 -*-
"""A2 prediction asset loading and alignment.

A ScoreAsset wraps an OOF/Test score matrix (uids x item_ids) with explicit
identity metadata.  Loading verifies:
- uid order against the dataset train/test order (re-aligned explicitly);
- item catalog identity against item.csv;
- score finiteness;
- OOF/Test scope separation (an asset is either offline_oof or test_scope,
  never mixed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..research.event_store import rel_ref, sha256_file
from .task_adapter import A2Dataset

SCOPE_OFFLINE_OOF = "offline_oof"
SCOPE_TEST = "test_scope"


@dataclass
class ScoreAsset:
    asset_id: str
    path: Path
    uids: list[str]
    item_ids: list[str]
    scores: np.ndarray  # float32, (len(uids), len(item_ids))
    targets: list[str] | None
    score_scope: str  # offline_oof | test_scope
    artifact_hash: str
    verification_status: str
    verification_errors: list[str]

    def aligned_to(self, uids: list[str]) -> "ScoreAsset":
        """Return a copy re-ordered to the given uid order (must be a permutation)."""
        if self.uids == uids:
            return self
        index = {uid: pos for pos, uid in enumerate(self.uids)}
        order = [index[uid] for uid in uids]
        scores = self.scores[order]
        targets = [self.targets[i] for i in order] if self.targets is not None else None
        return ScoreAsset(
            asset_id=self.asset_id,
            path=self.path,
            uids=list(uids),
            item_ids=self.item_ids,
            scores=scores,
            targets=targets,
            score_scope=self.score_scope,
            artifact_hash=self.artifact_hash,
            verification_status=self.verification_status,
            verification_errors=list(self.verification_errors),
        )


def load_score_asset(
    *,
    asset_id: str,
    path: str | Path,
    dataset: A2Dataset,
    expected_scope: str,
    project_root: Path | None = None,
) -> ScoreAsset:
    path = Path(path)
    errors: list[str] = []
    if not path.is_file():
        return ScoreAsset(asset_id, path, [], [], np.zeros((0, 0), dtype=np.float32), None,
                          expected_scope, "", "missing", ["artifact_missing"])
    artifact_hash = sha256_file(path)
    data = np.load(path, allow_pickle=True)  # uids are stored as object arrays in verified assets
    for key in ("uids", "scores", "item_ids"):
        if key not in data.files:
            errors.append(f"npz key {key} missing")
    if errors:
        return ScoreAsset(asset_id, path, [], [], np.zeros((0, 0), dtype=np.float32), None,
                          expected_scope, artifact_hash, "invalid", errors)
    uids = [str(u) for u in data["uids"].tolist()]
    item_ids = [str(i) for i in data["item_ids"].tolist()]
    scores = np.asarray(data["scores"], dtype=np.float32)
    targets = [str(t) for t in data["targets"].tolist()] if "targets" in data.files else None

    if scores.shape != (len(uids), len(item_ids)):
        errors.append(f"scores shape {scores.shape} != ({len(uids)}, {len(item_ids)})")
    if len(set(uids)) != len(uids):
        errors.append("uids not unique")
    if not np.isfinite(scores).all():
        errors.append("scores contain non-finite values")
    realigned_items = False
    if item_ids != dataset.item_ids:
        if set(item_ids) == set(dataset.item_ids):
            # Explicit column re-alignment to the canonical item.csv order.
            col_order = [item_ids.index(iid) for iid in dataset.item_ids]
            scores = scores[:, col_order]
            item_ids = list(dataset.item_ids)
            realigned_items = True
        else:
            errors.append("item_ids do not match item.csv catalog")

    if expected_scope == SCOPE_OFFLINE_OOF:
        expected_uids = set(dataset.train_uids)
        if set(uids) != expected_uids:
            errors.append("oof uids do not match train users")
        if targets is None:
            errors.append("oof asset must carry per-user targets")
        else:
            mismatched = sum(1 for uid, t in zip(uids, targets) if dataset.train_targets.get(uid) != t)
            if mismatched:
                errors.append(f"oof targets mismatch train.csv for {mismatched} users")
        if set(uids) & set(dataset.test_uids):
            errors.append("oof asset contains test users (OOF/Test mixing)")
    elif expected_scope == SCOPE_TEST:
        if set(uids) != set(dataset.test_uids):
            errors.append("test-scope uids do not match test users")
        if targets is not None:
            errors.append("test-scope asset must not carry targets (test truth isolation)")

    status = "verified_oof" if expected_scope == SCOPE_OFFLINE_OOF and not errors else (
        "verified_test_scope" if expected_scope == SCOPE_TEST and not errors else "invalid"
    )
    notes = ["item_columns_realigned_to_item_csv_order"] if realigned_items else []
    return ScoreAsset(
        asset_id=asset_id,
        path=path,
        uids=uids,
        item_ids=item_ids,
        scores=scores,
        targets=targets,
        score_scope=expected_scope,
        artifact_hash=artifact_hash,
        verification_status=status,
        verification_errors=errors + notes,
    )


def asset_ref(path: Path, project_root: Path | None) -> str:
    return rel_ref(path, project_root) if project_root else str(path)
