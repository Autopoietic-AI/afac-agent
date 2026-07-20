# -*- coding: utf-8 -*-
"""Leakage-safe, read-only A1 OOF candidate evaluation.

M4B deliberately evaluates train-side OOF probabilities only. It never reads
test labels, never writes prediction/submission artifacts, and never treats a
component parent as the current v53Q-1 anchor unless an explicit anchor manifest
proves that identity.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


N_CLASSES = 10
EVALUATION_VERSION = "m4b_v1"
CANDIDATE_PROBA_KEYS = ("proba", "final_proba", "candidate_proba")
PARENT_PROBA_KEYS = ("base_proba", "parent_proba")
TRAIN_INDEX_KEYS = ("train_idx", "global_idx")
LABEL_KEYS = ("labels", "y", "target")
OOF_BUCKET_KEYS = ("bucket", "buckets", "bucket_train")


@dataclass
class LoadedA1:
    adjacency: sparse.csr_matrix
    labels: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_degrees: np.ndarray
    train_buckets: dict[str, np.ndarray]


@dataclass
class LoadedOof:
    source: Path
    source_hash: str
    keys: list[str]
    proba_key: str
    proba: np.ndarray
    train_idx: np.ndarray
    labels: np.ndarray
    parent_proba_key: str
    embedded_parent_proba: np.ndarray | None
    bucket_key: str
    bucket_values: np.ndarray | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _stable_round(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_round(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_stable_round(v) for v in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _stable_round(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_present(files: set[str], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in files:
            return key
    return ""


def _load_a1_npz(path: Path) -> tuple[LoadedA1 | None, list[str]]:
    errors: list[str] = []
    required = {
        "adj_data",
        "adj_indices",
        "adj_indptr",
        "adj_shape",
        "labels",
        "train_idx",
        "test_idx",
    }
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        return None, [f"a1_npz: cannot read npz: {exc}"]
    with payload:
        missing = sorted(required - set(payload.files))
        if missing:
            return None, [f"a1_npz: missing keys {missing}"]
        try:
            adjacency = sparse.csr_matrix(
                (
                    payload["adj_data"],
                    payload["adj_indices"],
                    payload["adj_indptr"],
                ),
                shape=tuple(payload["adj_shape"]),
            ).astype(np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
            train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
            test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        except Exception as exc:
            return None, [f"a1_npz: invalid canonical arrays: {exc}"]
    if adjacency.shape[0] != adjacency.shape[1]:
        errors.append(f"a1_npz: adjacency must be square, got {adjacency.shape}")
    if labels.shape[0] != adjacency.shape[0]:
        errors.append("a1_npz: labels length must match adjacency node count")
    if len(set(map(int, train_idx))) != len(train_idx):
        errors.append("a1_npz: train_idx contains duplicates")
    if len(set(map(int, test_idx))) != len(test_idx):
        errors.append("a1_npz: test_idx contains duplicates")
    if errors:
        return None, errors
    degrees = _either_direction_degrees(adjacency)
    train_degrees = degrees[train_idx]
    train_buckets = {
        "Graph-visible": train_degrees > 0,
        "Isolated": train_degrees == 0,
        "degree_1": train_degrees == 1,
        "degree_2_5": (train_degrees >= 2) & (train_degrees <= 5),
        "degree_6p": train_degrees >= 6,
    }
    return LoadedA1(
        adjacency=adjacency,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        train_degrees=train_degrees,
        train_buckets=train_buckets,
    ), []


def _either_direction_degrees(adjacency: sparse.csr_matrix) -> np.ndarray:
    graph = adjacency.tocsr(copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    either = graph + graph.T
    either.data = np.ones_like(either.data, dtype=np.int8)
    either.eliminate_zeros()
    return np.asarray(either.getnnz(axis=1), dtype=np.int64)


def _manual_macro_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    scores: list[float] = []
    for klass in range(N_CLASSES):
        tp = int(((labels == klass) & (pred == klass)).sum())
        fp = int(((labels != klass) & (pred == klass)).sum())
        fn = int(((labels == klass) & (pred != klass)).sum())
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def _validate_probability_matrix(
    *,
    name: str,
    proba: np.ndarray,
    expected_rows: int,
) -> list[str]:
    errors: list[str] = []
    if proba.ndim != 2:
        return [f"{name}: proba must be 2D, got ndim={proba.ndim}"]
    if proba.shape != (expected_rows, N_CLASSES):
        errors.append(
            f"{name}: proba shape must be ({expected_rows}, {N_CLASSES}), got {proba.shape}"
        )
    if not np.isfinite(proba).all():
        errors.append(f"{name}: proba contains NaN/Inf")
    if (proba < -1e-8).any():
        errors.append(f"{name}: proba contains negative values")
    if proba.ndim == 2 and proba.shape[1] == N_CLASSES:
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            errors.append(f"{name}: probability rows must sum to 1")
    return errors


def _load_oof_npz(
    *,
    path: Path,
    a1: LoadedA1,
    name: str,
    allow_parent_keys_as_primary: bool = False,
) -> tuple[LoadedOof | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        return None, [f"{name}: cannot read npz: {exc}"], warnings
    with payload:
        files = set(payload.files)
        primary_keys = (
            (*CANDIDATE_PROBA_KEYS, *PARENT_PROBA_KEYS)
            if allow_parent_keys_as_primary
            else CANDIDATE_PROBA_KEYS
        )
        proba_key = _first_present(files, primary_keys)
        parent_key = _first_present(files, PARENT_PROBA_KEYS)
        index_key = _first_present(files, TRAIN_INDEX_KEYS)
        label_key = _first_present(files, LABEL_KEYS)
        bucket_key = _first_present(files, OOF_BUCKET_KEYS)
        if not proba_key:
            errors.append(f"{name}: missing whitelisted proba key {list(primary_keys)}")
        if not index_key:
            errors.append(f"{name}: missing whitelisted train index key {list(TRAIN_INDEX_KEYS)}")
        if not label_key:
            errors.append(f"{name}: missing whitelisted label key {list(LABEL_KEYS)}")
        if errors:
            return None, errors, warnings
        train_idx = np.asarray(payload[index_key], dtype=np.int64)
        proba = np.asarray(payload[proba_key], dtype=np.float64)
        labels = np.asarray(payload[label_key], dtype=np.int64)
        embedded_parent = (
            np.asarray(payload[parent_key], dtype=np.float64)
            if parent_key and parent_key != proba_key
            else None
        )
        bucket_values = (
            np.asarray(payload[bucket_key])
            if bucket_key
            else None
        )
        keys = sorted(payload.files)

    if train_idx.ndim != 1:
        errors.append(f"{name}: train_idx must be 1D")
    if labels.ndim != 1:
        errors.append(f"{name}: labels must be 1D")
    if len(train_idx) != len(a1.train_idx):
        errors.append(
            f"{name}: train_idx length must match A1 train count {len(a1.train_idx)}, got {len(train_idx)}"
        )
    if len(set(map(int, train_idx))) != len(train_idx):
        errors.append(f"{name}: train_idx contains duplicates")
    if set(map(int, train_idx)) != set(map(int, a1.train_idx)):
        errors.append(f"{name}: train_idx must exactly cover A1 train_idx")
    errors.extend(
        _validate_probability_matrix(
            name=name,
            proba=proba,
            expected_rows=len(train_idx),
        )
    )
    if embedded_parent is not None:
        errors.extend(
            _validate_probability_matrix(
                name=f"{name}:embedded_parent",
                proba=embedded_parent,
                expected_rows=len(train_idx),
            )
        )
    if labels.shape[0] != len(train_idx):
        errors.append(f"{name}: labels length must match train_idx length")
    if errors:
        return None, errors, warnings

    order = np.asarray([int(x) for x in train_idx], dtype=np.int64)
    position = {int(idx): i for i, idx in enumerate(order)}
    reorder = np.asarray([position[int(idx)] for idx in a1.train_idx], dtype=np.int64)
    reordered_labels = labels[reorder]
    expected_labels = a1.labels[a1.train_idx]
    if not np.array_equal(reordered_labels, expected_labels):
        return None, [f"{name}: labels must match A1 train labels after safe reorder"], warnings
    if not np.array_equal(order, a1.train_idx):
        warnings.append(f"{name}: train_idx order differed from A1; safely reordered by global_idx")
    reordered_bucket = bucket_values[reorder] if bucket_values is not None and len(bucket_values) == len(train_idx) else bucket_values
    if bucket_values is not None and len(bucket_values) != len(train_idx):
        warnings.append(f"{name}: OOF bucket field length does not match train_idx")
    loaded = LoadedOof(
        source=path,
        source_hash=sha256_file(path),
        keys=keys,
        proba_key=proba_key,
        proba=proba[reorder],
        train_idx=a1.train_idx.copy(),
        labels=expected_labels.copy(),
        parent_proba_key=parent_key,
        embedded_parent_proba=embedded_parent[reorder] if embedded_parent is not None else None,
        bucket_key=bucket_key,
        bucket_values=reordered_bucket,
    )
    warnings.extend(_bucket_consistency_warnings(name=name, loaded=loaded, a1=a1))
    return loaded, [], warnings


def _bucket_consistency_warnings(
    *,
    name: str,
    loaded: LoadedOof,
    a1: LoadedA1,
) -> list[str]:
    if loaded.bucket_values is None or len(loaded.bucket_values) != len(a1.train_idx):
        return []
    canonical = np.full(len(a1.train_idx), "degree_6p", dtype=object)
    canonical[a1.train_degrees == 0] = "Isolated"
    canonical[a1.train_degrees == 1] = "degree_1"
    canonical[(a1.train_degrees >= 2) & (a1.train_degrees <= 5)] = "degree_2_5"
    observed = loaded.bucket_values
    if np.issubdtype(observed.dtype, np.number):
        mapping = {
            0: "Isolated",
            1: "degree_1",
            2: "degree_2_5",
            3: "degree_6p",
            4: "Graph-visible",
        }
        mapped = np.asarray([mapping.get(int(x), f"unknown:{int(x)}") for x in observed], dtype=object)
    else:
        mapped = np.asarray([str(x) for x in observed], dtype=object)
    mismatch = int((mapped != canonical).sum())
    if mismatch:
        return [
            f"{name}: OOF bucket field '{loaded.bucket_key}' mismatches canonical A1.npz buckets for {mismatch} train rows; canonical buckets are used"
        ]
    return [f"{name}: OOF bucket field '{loaded.bucket_key}' verified against canonical A1.npz buckets"]


def _prediction_distribution(pred: np.ndarray) -> dict[str, int]:
    return {str(klass): int((pred == klass).sum()) for klass in range(N_CLASSES)}


def _class_metrics(
    *,
    labels: np.ndarray,
    candidate_pred: np.ndarray,
    parent_pred: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_correct = candidate_pred == labels
    parent_correct = parent_pred == labels if parent_pred is not None else None
    for klass in range(N_CLASSES):
        mask = labels == klass
        support = int(mask.sum())
        row: dict[str, Any] = {
            "class_id": int(klass),
            "support": support,
            "candidate_correct": int(candidate_correct[mask].sum()),
            "candidate_class_metric": (
                float(candidate_correct[mask].mean()) if support else None
            ),
        }
        if parent_correct is not None and parent_pred is not None:
            rescue_mask = (~parent_correct) & candidate_correct & mask
            damage_mask = parent_correct & (~candidate_correct) & mask
            parent_count = int(parent_correct[mask].sum())
            rescue = int(rescue_mask.sum())
            damage = int(damage_mask.sum())
            row.update(
                {
                    "parent_correct": parent_count,
                    "parent_class_metric": float(parent_correct[mask].mean()) if support else None,
                    "gain": (
                        float(candidate_correct[mask].mean() - parent_correct[mask].mean())
                        if support
                        else None
                    ),
                    "rescue": rescue,
                    "damage": damage,
                    "net": rescue - damage,
                }
            )
        rows.append(row)
    return rows


def _aggregate_bucket(
    *,
    name: str,
    mask: np.ndarray,
    labels: np.ndarray,
    candidate_pred: np.ndarray,
    parent_pred: np.ndarray | None,
) -> dict[str, Any]:
    candidate_correct = candidate_pred == labels
    row: dict[str, Any] = {
        "bucket": name,
        "node_count": int(mask.sum()),
        "candidate_accuracy": (
            float(candidate_correct[mask].mean()) if int(mask.sum()) else None
        ),
    }
    if parent_pred is None:
        return row
    parent_correct = parent_pred == labels
    changed = (parent_pred != candidate_pred) & mask
    rescue_mask = (~parent_correct) & candidate_correct & mask
    damage_mask = parent_correct & (~candidate_correct) & mask
    rescue = int(rescue_mask.sum())
    changed_count = int(changed.sum())
    row.update(
        {
            "parent_accuracy": (
                float(parent_correct[mask].mean()) if int(mask.sum()) else None
            ),
            "gain": (
                float(candidate_correct[mask].mean() - parent_correct[mask].mean())
                if int(mask.sum())
                else None
            ),
            "rescue": rescue,
            "damage": int(damage_mask.sum()),
            "net": rescue - int(damage_mask.sum()),
            "changed_count": changed_count,
            "change_precision": (
                float(rescue / changed_count) if changed_count else None
            ),
        }
    )
    return row


def _fold_metrics(
    *,
    path: Path | None,
    train_idx: np.ndarray,
    labels: np.ndarray,
    candidate_pred: np.ndarray,
    parent_pred: np.ndarray | None,
) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {"status": "unavailable", "reason": "missing_canonical_fold"}, []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as exc:
        return {}, [f"canonical_fold_csv: cannot read csv: {exc}"]
    idx_col = next((c for c in ("global_idx", "train_idx", "node_id") if c in fieldnames), "")
    fold_col = next((c for c in ("fold", "fold_id") if c in fieldnames), "")
    label_col = next((c for c in ("label", "labels", "y", "target") if c in fieldnames), "")
    if not idx_col or not fold_col:
        return {}, ["canonical_fold_csv: requires global_idx/train_idx/node_id and fold/fold_id columns"]
    assignments: dict[int, int] = {}
    for row_number, row in enumerate(rows):
        try:
            idx = int(str(row[idx_col]).strip())
            fold = int(str(row[fold_col]).strip())
        except ValueError:
            errors.append(f"canonical_fold_csv: row {row_number} has non-integer index/fold")
            continue
        if fold < 0:
            errors.append(f"canonical_fold_csv: row {row_number} has negative fold")
        if idx in assignments:
            errors.append(f"canonical_fold_csv: duplicate node {idx}")
        assignments[idx] = fold
        if label_col and str(row.get(label_col, "")).strip():
            try:
                observed_label = int(str(row[label_col]).strip())
            except ValueError:
                errors.append(f"canonical_fold_csv: row {row_number} has non-integer label")
                continue
            pos = np.where(train_idx == idx)[0]
            if len(pos) == 1 and observed_label != int(labels[pos[0]]):
                errors.append(f"canonical_fold_csv: label mismatch for node {idx}")
    if set(assignments) != set(map(int, train_idx)):
        errors.append("canonical_fold_csv: fold assignment must exactly cover train_idx")
    if errors:
        return {}, errors
    folds = np.asarray([assignments[int(idx)] for idx in train_idx], dtype=np.int64)
    unique_folds = sorted(set(map(int, folds)))
    if parent_pred is None:
        return {
            "status": "unavailable",
            "reason": "missing_parent_oof_for_fold_gain",
            "fold_count": len(unique_folds),
        }, []
    parent_correct = parent_pred == labels
    candidate_correct = candidate_pred == labels
    fold_rows: list[dict[str, Any]] = []
    gains: list[float] = []
    for fold in unique_folds:
        mask = folds == fold
        parent_acc = float(parent_correct[mask].mean())
        candidate_acc = float(candidate_correct[mask].mean())
        gain = candidate_acc - parent_acc
        gains.append(gain)
        fold_rows.append(
            {
                "fold": int(fold),
                "node_count": int(mask.sum()),
                "parent_accuracy": parent_acc,
                "candidate_accuracy": candidate_acc,
                "gain": gain,
            }
        )
    return {
        "status": "observed",
        "fold_count": len(unique_folds),
        "per_fold": fold_rows,
        "positive_fold_count": int(sum(gain > 0 for gain in gains)),
        "negative_fold_count": int(sum(gain < 0 for gain in gains)),
        "zero_fold_count": int(sum(gain == 0 for gain in gains)),
        "worst_fold_gain": float(min(gains)) if gains else None,
        "median_fold_gain": float(np.median(gains)) if gains else None,
        "mean_fold_gain": float(np.mean(gains)) if gains else None,
    }, []


def _anchor_identity(
    *,
    anchor_manifest_json: Path | None,
    parent_hash: str,
) -> tuple[bool, list[str], dict[str, Any]]:
    if anchor_manifest_json is None:
        return False, [], {}
    warnings: list[str] = []
    try:
        manifest = json.loads(anchor_manifest_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"anchor_manifest_json: cannot read JSON: {exc}"], {}
    text = json.dumps(manifest, ensure_ascii=False).lower()
    hash_values = {
        str(value).lower()
        for value in _walk_values(manifest)
        if isinstance(value, str) and len(value) == 64
    }
    version_ok = "v53q-1" in text or "v53q1" in text
    hash_ok = bool(parent_hash) and parent_hash.lower() in hash_values
    if not version_ok:
        warnings.append("anchor_manifest_json: v53Q-1 identity text not found")
    if not hash_ok:
        warnings.append("anchor_manifest_json: parent OOF hash not proven by manifest")
    return bool(version_ok and hash_ok), warnings, manifest


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_a1_oof_candidate(
    *,
    a1_npz: Path,
    candidate_oof_npz: Path,
    output_dir: Path | None = None,
    evaluation_id: str = "",
    parent_oof_npz: Path | None = None,
    canonical_fold_csv: Path | None = None,
    anchor_manifest_json: Path | None = None,
    node_bucket_csv: Path | None = None,
    m2_profile_json: Path | None = None,
    evaluation_policy_json: Path | None = None,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], list[str], list[str]]:
    del node_bucket_csv  # reserved for future cross-checks; A1.npz remains canonical.
    errors: list[str] = []
    warnings: list[str] = []
    a1, a1_errors = _load_a1_npz(a1_npz)
    if a1 is None:
        return {}, a1_errors, warnings
    candidate, candidate_errors, candidate_warnings = _load_oof_npz(
        path=candidate_oof_npz,
        a1=a1,
        name="candidate_oof_npz",
    )
    warnings.extend(candidate_warnings)
    if candidate is None:
        return {}, candidate_errors, warnings

    external_parent: LoadedOof | None = None
    parent_proba: np.ndarray | None = None
    parent_hash = ""
    parent_identity_status = "unavailable"
    comparison_scope = "candidate_integrity_only"
    analysis_tier = "integrity_only"
    parent_source = ""
    if candidate.embedded_parent_proba is not None:
        parent_proba = candidate.embedded_parent_proba
        parent_hash = candidate.source_hash
        parent_identity_status = "unverified"
        comparison_scope = "embedded_parent_unverified"
        analysis_tier = "self_contained_oof_comparison"
        parent_source = f"{candidate.source}:{candidate.parent_proba_key}"
    elif parent_oof_npz is not None:
        external_parent, parent_errors, parent_warnings = _load_oof_npz(
            path=parent_oof_npz,
            a1=a1,
            name="parent_oof_npz",
            allow_parent_keys_as_primary=True,
        )
        warnings.extend(parent_warnings)
        if parent_errors:
            return {}, parent_errors, warnings
        assert external_parent is not None
        parent_proba = external_parent.proba
        parent_hash = external_parent.source_hash
        parent_identity_status = "unverified"
        comparison_scope = "external_parent_unverified"
        analysis_tier = "self_contained_oof_comparison"
        parent_source = str(external_parent.source)

    anchor_manifest: dict[str, Any] = {}
    if parent_proba is not None and anchor_manifest_json is not None and canonical_fold_csv is not None:
        anchor_ok, anchor_warnings, anchor_manifest = _anchor_identity(
            anchor_manifest_json=anchor_manifest_json,
            parent_hash=parent_hash,
        )
        warnings.extend(anchor_warnings)
        if anchor_ok:
            parent_identity_status = "current_anchor_verified"
            comparison_scope = "current_champion_anchor"
            analysis_tier = "full_anchor_oof_comparison"
        else:
            warnings.append("full_anchor_oof_comparison downgraded because anchor identity was not verified")
    elif parent_proba is not None and (anchor_manifest_json is not None or canonical_fold_csv is not None):
        warnings.append("full_anchor_oof_comparison unavailable; missing one or more required anchor/fold inputs")

    policy: dict[str, Any] = {}
    policy_status = "unavailable"
    if evaluation_policy_json is not None:
        try:
            policy = json.loads(evaluation_policy_json.read_text(encoding="utf-8"))
            policy_status = "observed"
        except Exception as exc:
            return {}, [f"evaluation_policy_json: cannot read JSON: {exc}"], warnings

    labels = candidate.labels
    candidate_pred = candidate.proba.argmax(axis=1)
    parent_pred = parent_proba.argmax(axis=1) if parent_proba is not None else None
    candidate_correct = candidate_pred == labels

    metric_semantics = {
        "source_files": [
            "historical code audit: a1_label_anchored_public_graph_alignment_stack_v2.py",
            "M4B implementation: manual macro-F1 and train-side OOF only",
        ],
        "overall_accuracy": "mean(argmax(proba) == train_label) on A1 train_idx OOF rows",
        "macro_metric": "macro_f1 over 10 classes",
        "class_metric": "per-true-class recall/class accuracy: correct/support",
        "rescue": "parent wrong and candidate correct",
        "damage": "parent correct and candidate wrong",
        "net": "rescue - damage",
        "changed": "argmax(parent_proba) != argmax(candidate_proba)",
        "change_precision": "rescue / changed_count",
        "oracle_correct": "parent correct OR candidate correct",
        "bucket_source": "A1.npz canonical adjacency; unique either-direction degree after self-loop removal",
        "test_truth_used": False,
    }

    overall_metrics: dict[str, Any] = {
        "status": "observed",
        "candidate_accuracy": float(candidate_correct.mean()),
        "node_count": int(len(labels)),
    }
    macro_metrics: dict[str, Any] = {
        "status": "observed",
        "macro_metric_name": "macro_f1",
        "candidate_macro_metric": _manual_macro_f1(labels, candidate_pred),
    }
    rescue_damage: dict[str, Any] = {"status": "unavailable", "reason": "missing_parent_oof"}
    oracle_metrics: dict[str, Any] = {"status": "unavailable", "reason": "missing_parent_oof"}
    prediction_shift: dict[str, Any] = {
        "status": "observed",
        "candidate_prediction_distribution": _prediction_distribution(candidate_pred),
    }
    if parent_pred is not None:
        parent_correct = parent_pred == labels
        changed = parent_pred != candidate_pred
        rescue_mask = (~parent_correct) & candidate_correct
        damage_mask = parent_correct & (~candidate_correct)
        rescue = int(rescue_mask.sum())
        damage = int(damage_mask.sum())
        changed_count = int(changed.sum())
        overall_metrics.update(
            {
                "parent_accuracy": float(parent_correct.mean()),
                "gain": float(candidate_correct.mean() - parent_correct.mean()),
            }
        )
        macro_metrics.update(
            {
                "parent_macro_metric": _manual_macro_f1(labels, parent_pred),
                "gain": _manual_macro_f1(labels, candidate_pred) - _manual_macro_f1(labels, parent_pred),
            }
        )
        rescue_damage = {
            "status": "observed",
            "changed_count": changed_count,
            "rescue": rescue,
            "damage": damage,
            "net": rescue - damage,
            "change_precision": float(rescue / changed_count) if changed_count else None,
        }
        oracle_correct = parent_correct | candidate_correct
        oracle_metrics = {
            "status": "observed",
            "oracle_accuracy": float(oracle_correct.mean()),
            "oracle_gain": float(oracle_correct.mean() - parent_correct.mean()),
        }
        prediction_shift["parent_prediction_distribution"] = _prediction_distribution(parent_pred)
        prediction_shift["changed_count"] = changed_count

    class_rows = _class_metrics(
        labels=labels,
        candidate_pred=candidate_pred,
        parent_pred=parent_pred,
    )
    bucket_rows = [
        _aggregate_bucket(
            name=name,
            mask=mask,
            labels=labels,
            candidate_pred=candidate_pred,
            parent_pred=parent_pred,
        )
        for name, mask in a1.train_buckets.items()
    ]
    fold_metrics, fold_errors = _fold_metrics(
        path=canonical_fold_csv,
        train_idx=a1.train_idx,
        labels=labels,
        candidate_pred=candidate_pred,
        parent_pred=parent_pred,
    )
    if fold_errors:
        return {}, fold_errors, warnings

    input_hashes = {
        "a1_npz": sha256_file(a1_npz),
        "candidate_oof_npz": candidate.source_hash,
    }
    if parent_oof_npz is not None:
        input_hashes["parent_oof_npz"] = sha256_file(parent_oof_npz)
    if canonical_fold_csv is not None:
        input_hashes["canonical_fold_csv"] = sha256_file(canonical_fold_csv)
    if anchor_manifest_json is not None:
        input_hashes["anchor_manifest_json"] = sha256_file(anchor_manifest_json)
    if m2_profile_json is not None and m2_profile_json.exists():
        input_hashes["m2_profile_json"] = sha256_file(m2_profile_json)
    if evaluation_policy_json is not None:
        input_hashes["evaluation_policy_json"] = sha256_file(evaluation_policy_json)

    identity_payload = {
        "evaluation_version": EVALUATION_VERSION,
        "analysis_tier": analysis_tier,
        "comparison_scope": comparison_scope,
        "candidate_hash": candidate.source_hash,
        "parent_hash": parent_hash,
        "input_hashes": input_hashes,
        "policy": policy,
    }
    final_evaluation_id = evaluation_id or stable_hash(identity_payload)

    limitations = []
    if analysis_tier != "full_anchor_oof_comparison":
        limitations.append(
            {
                "status": "unavailable",
                "reason": "missing_verified_v53q1_anchor_oof_or_canonical_fold",
                "metric_family": "current_champion_anchor_gain",
            }
        )
    if fold_metrics.get("status") == "unavailable":
        limitations.append(
            {
                "status": "unavailable",
                "reason": fold_metrics.get("reason", "missing_canonical_fold"),
                "metric_family": "fold_gain",
            }
        )

    evaluation = {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_id": final_evaluation_id,
        "analysis_tier": analysis_tier,
        "comparison_scope": comparison_scope,
        "candidate_identity": {
            "source": str(candidate.source),
            "source_role": "historical_component_or_candidate",
            "keys": candidate.keys,
            "proba_key": candidate.proba_key,
            "eligible_for_research_reopen": False,
            "used_only_for_evaluator_validation": True,
        },
        "candidate_hash": candidate.source_hash,
        "parent_identity": {
            "source": parent_source,
            "anchor_manifest_source": str(anchor_manifest_json) if anchor_manifest_json else "",
            "manifest_observed": bool(anchor_manifest),
        },
        "parent_hash": parent_hash,
        "parent_identity_status": parent_identity_status,
        "input_validation": {
            "status": "passed",
            "train_idx_count": int(len(a1.train_idx)),
            "class_count": N_CLASSES,
            "candidate_rows": int(candidate.proba.shape[0]),
            "probability_rows_sum_to_one": True,
            "test_truth_used": False,
        },
        "metric_semantics": metric_semantics,
        "evaluation_policy": {
            "status": policy_status,
            "policy": policy,
        },
        "overall_metrics": overall_metrics,
        "macro_metrics": macro_metrics,
        "class_metrics": class_rows,
        "bucket_metrics": bucket_rows,
        "fold_metrics": fold_metrics,
        "rescue_damage": rescue_damage,
        "oracle_metrics": oracle_metrics,
        "prediction_shift": prediction_shift,
        "limitations": limitations,
        "warnings": warnings,
        "test_truth_used": False,
        "leakage_safety_pass": True,
        "evaluation_integrity_pass": True,
    }

    artifacts: dict[str, str] = {}
    if write_outputs and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = output_dir / "a1_oof_evaluation.json"
        manifest_path = output_dir / "evaluation_manifest.json"
        report_path = output_dir / "OOF_EVALUATION_REPORT.md"
        class_path = output_dir / "class_metrics.csv"
        bucket_path = output_dir / "bucket_metrics.csv"
        changed_path = output_dir / "changed_nodes_audit.csv"
        evaluation_path.write_text(_json_dumps(evaluation) + "\n", encoding="utf-8")
        manifest = {
            "evaluation_id": final_evaluation_id,
            "evaluation_version": EVALUATION_VERSION,
            "analysis_tier": analysis_tier,
            "comparison_scope": comparison_scope,
            "input_hashes": input_hashes,
            "test_truth_used": False,
            "artifacts": {
                "a1_oof_evaluation": str(evaluation_path),
                "class_metrics": str(class_path),
                "bucket_metrics": str(bucket_path),
                "changed_nodes_audit": str(changed_path),
                "report": str(report_path),
            },
        }
        manifest_path.write_text(_json_dumps(manifest) + "\n", encoding="utf-8")
        _write_csv(class_path, class_rows, sorted({k for row in class_rows for k in row}))
        _write_csv(bucket_path, bucket_rows, sorted({k for row in bucket_rows for k in row}))
        changed_rows: list[dict[str, Any]] = []
        if parent_pred is not None:
            changed_mask = parent_pred != candidate_pred
            parent_correct = parent_pred == labels
            for idx, truth, pp, cp, pc, cc in zip(
                a1.train_idx[changed_mask],
                labels[changed_mask],
                parent_pred[changed_mask],
                candidate_pred[changed_mask],
                parent_correct[changed_mask],
                candidate_correct[changed_mask],
            ):
                changed_rows.append(
                    {
                        "global_idx": int(idx),
                        "label": int(truth),
                        "parent_pred": int(pp),
                        "candidate_pred": int(cp),
                        "rescue": bool((not pc) and cc),
                        "damage": bool(pc and (not cc)),
                    }
                )
        _write_csv(
            changed_path,
            changed_rows,
            ["global_idx", "label", "parent_pred", "candidate_pred", "rescue", "damage"],
        )
        fold_path = output_dir / "fold_metrics.csv"
        if fold_metrics.get("status") == "observed":
            _write_csv(
                fold_path,
                list(fold_metrics.get("per_fold", [])),
                ["fold", "node_count", "parent_accuracy", "candidate_accuracy", "gain"],
            )
            manifest["artifacts"]["fold_metrics"] = str(fold_path)
            manifest_path.write_text(_json_dumps(manifest) + "\n", encoding="utf-8")
        report_path.write_text(_render_report(evaluation), encoding="utf-8")
        artifacts = dict(manifest["artifacts"])
        artifacts["evaluation_manifest"] = str(manifest_path)

    evaluation["artifacts"] = artifacts
    return evaluation, [], warnings


def _render_report(evaluation: dict[str, Any]) -> str:
    overall = evaluation["overall_metrics"]
    macro = evaluation["macro_metrics"]
    rd = evaluation["rescue_damage"]
    oracle = evaluation["oracle_metrics"]
    lines = [
        "# A1 OOF Candidate Evaluation",
        "",
        f"- evaluation_id: `{evaluation['evaluation_id']}`",
        f"- analysis_tier: `{evaluation['analysis_tier']}`",
        f"- comparison_scope: `{evaluation['comparison_scope']}`",
        f"- parent_identity_status: `{evaluation['parent_identity_status']}`",
        f"- test_truth_used: `{evaluation['test_truth_used']}`",
        f"- leakage_safety_pass: `{evaluation['leakage_safety_pass']}`",
        f"- evaluation_integrity_pass: `{evaluation['evaluation_integrity_pass']}`",
        "",
        "## Overall",
        "",
        f"- candidate_accuracy: `{overall.get('candidate_accuracy')}`",
        f"- candidate_macro_f1: `{macro.get('candidate_macro_metric')}`",
    ]
    if "parent_accuracy" in overall:
        lines.extend(
            [
                f"- parent_accuracy: `{overall.get('parent_accuracy')}`",
                f"- overall_gain: `{overall.get('gain')}`",
                f"- parent_macro_f1: `{macro.get('parent_macro_metric')}`",
                f"- macro_gain: `{macro.get('gain')}`",
                f"- changed_count: `{rd.get('changed_count')}`",
                f"- rescue: `{rd.get('rescue')}`",
                f"- damage: `{rd.get('damage')}`",
                f"- net: `{rd.get('net')}`",
                f"- change_precision: `{rd.get('change_precision')}`",
                f"- oracle_gain: `{oracle.get('oracle_gain')}`",
            ]
        )
    else:
        lines.append("- parent comparison: `unavailable`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for item in evaluation.get("limitations", []):
        lines.append(f"- {item.get('metric_family')}: {item.get('reason')}")
    for warning in evaluation.get("warnings", []):
        lines.append(f"- warning: {warning}")
    return "\n".join(lines) + "\n"
