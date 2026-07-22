# -*- coding: utf-8 -*-
"""AFAC_A2_FOLD_V1 validation.

Validates the explicit A2 fold assignment file (uid,oof_fold):
- every train user appears exactly once;
- no Test user is present;
- fold values are legal (0..4) and all folds are non-empty;
- user order is explicit (file order recorded and hashed);
- file hash is recorded for stability audits;
- sequence-length and target-type distributions per fold are auditable.

The validator never re-assigns folds.  If the file fails validation the
caller must surface `validation_failed`, not silently re-split.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..research.event_store import sha256_file, stable_hash
from .task_adapter import A2Dataset

AFAC_A2_FOLD_V1 = "AFAC_A2_FOLD_V1"
EXPECTED_FOLDS = (0, 1, 2, 3, 4)


def validate_fold_csv(
    fold_csv: str | Path,
    dataset: A2Dataset,
    *,
    expected_folds: tuple[int, ...] = EXPECTED_FOLDS,
) -> dict[str, Any]:
    path = Path(fold_csv)
    manifest: dict[str, Any] = {
        "fold_protocol": AFAC_A2_FOLD_V1,
        "source_csv": str(path),
        "sha256": sha256_file(path) if path.is_file() else "",
        "status": "failed",
        "errors": [],
        "checks": {},
    }
    errors: list[str] = manifest["errors"]
    if not path.is_file():
        errors.append("fold_csv_missing")
        manifest["checks"]["file_exists"] = False
        return manifest
    manifest["checks"]["file_exists"] = True

    fold_map: dict[str, int] = {}
    file_order: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header[:2] != ["uid", "oof_fold"]:
            errors.append(f"unexpected header {header!r}; expected ['uid', 'oof_fold']")
        for row in reader:
            uid = (row.get("uid") or "").strip()
            raw_fold = (row.get("oof_fold") or "").strip()
            if not uid:
                errors.append("empty uid row")
                continue
            if uid in fold_map:
                errors.append(f"duplicate uid {uid}")
                continue
            try:
                fold_value = int(raw_fold)
            except ValueError:
                errors.append(f"uid {uid}: illegal fold value {raw_fold!r}")
                continue
            fold_map[uid] = fold_value
            file_order.append(uid)

    train_uids = set(dataset.train_uids)
    test_uids = set(dataset.test_uids)
    assigned = set(fold_map)

    missing_train = sorted(train_uids - assigned)
    extra_unknown = sorted(assigned - train_uids - test_uids)
    test_leak = sorted(assigned & test_uids)
    each_train_user_exactly_once = not missing_train and not test_leak and not extra_unknown and len(fold_map) == len(train_uids)
    if missing_train:
        errors.append(f"train users missing from fold file: {len(missing_train)}")
    if test_leak:
        errors.append(f"test users present in fold file: {len(test_leak)}")
    if extra_unknown:
        errors.append(f"unknown uids in fold file: {len(extra_unknown)}")

    fold_values = sorted(set(fold_map.values()))
    legal = fold_values == list(expected_folds)
    if not legal:
        errors.append(f"fold values {fold_values} != expected {list(expected_folds)}")

    fold_sizes = {str(f): sum(1 for v in fold_map.values() if v == f) for f in fold_values}
    manifest["checks"].update({
        "each_train_user_exactly_once": each_train_user_exactly_once,
        "contains_no_test_users": not test_leak,
        "fold_values_legal": legal,
        "fold_sizes": fold_sizes,
        "assigned_user_count": len(fold_map),
        "train_user_count": len(dataset.train_uids),
        "test_user_count": len(dataset.test_uids),
    })

    # Auditable per-fold distributions (sequence length buckets and target type).
    per_fold: dict[str, dict[str, Any]] = {}
    for f in fold_values:
        uids = [uid for uid in dataset.train_uids if fold_map.get(uid) == f]
        len_dist: dict[str, int] = {}
        type_dist: dict[str, int] = {}
        for uid in uids:
            lb = dataset.len_bucket(uid)
            len_dist[lb] = len_dist.get(lb, 0) + 1
            tt = dataset.target_type(uid)
            type_dist[tt] = type_dist.get(tt, 0) + 1
        per_fold[str(f)] = {
            "user_count": len(uids),
            "sequence_length_distribution": len_dist,
            "target_type_distribution": type_dist,
        }
    manifest["per_fold_audit"] = per_fold

    order_hash = stable_hash({"fold_file_order": file_order})
    manifest["user_order"] = {
        "explicit": True,
        "source": "fold_csv_file_order",
        "order_hash": order_hash,
        "first_uids": file_order[:5],
    }
    manifest["status"] = "validated" if not errors else "failed"
    manifest["view_hash"] = stable_hash({
        "fold_protocol": AFAC_A2_FOLD_V1,
        "sha256": manifest["sha256"],
        "fold_sizes": fold_sizes,
        "order_hash": order_hash,
    })
    manifest["fold_assignment_map_included"] = False  # map is large; consumers re-read the hashed CSV
    if manifest["status"] == "validated":
        manifest["fold_map"] = fold_map
    return manifest
