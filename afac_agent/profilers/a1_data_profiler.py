# -*- coding: utf-8 -*-
"""M2 A1 Data Profiler.

This module is intentionally read-only with respect to competition assets.  It
loads the canonical A1 graph from the NPZ adjacency CSR, optionally cross-checks
an edge CSV, and emits deterministic profile artifacts.  It does not train,
predict, modify champion artifacts, write memory, or mutate project state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse


PROFILE_VERSION = "m2_a1_v1"
MAX_HOP = 4
CORE_OUTPUT_FILES = [
    "a1_data_profile.json",
    "a1_input_validation.json",
    "a1_signal_inventory.json",
    "a1_problem_map.json",
    "a1_node_buckets.csv",
    "a1_structure_profile.csv",
    "a1_train_neighbor_profile.csv",
    "a1_hop_profile.csv",
    "a1_class_profile.csv",
    "a1_confusion_transitions.csv",
    "a1_prediction_sink_source.csv",
    "a1_fold_profile.csv",
    "a1_shift_profile.csv",
    "A1_DATA_PROFILE_REPORT.md",
]


@dataclass(frozen=True)
class ProfileRequest:
    npz_path: Path
    out_dir: Path
    edges_csv: Path | None = None
    champion_csv: Path | None = None
    fold_file: Path | None = None
    anchor_oof_npz: Path | None = None
    reference_oof_npz: Path | None = None
    strict: bool = False
    require_fold: bool = False
    require_oof: bool = False
    profile_version: str = PROFILE_VERSION


@dataclass(frozen=True)
class ProfileRunResult:
    status: str
    reason: str
    analysis_tier: str
    out_dir: str
    core_result_hash: str | None = None
    generated_files: list[str] | None = None
    warnings: list[str] | None = None
    errors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(
        _json_safe(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write_text(path, text + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_path(path: Path) -> str:
    return path.resolve().as_posix()


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: counts[key] for key in sorted(counts)}


def _champion_prediction_distribution(
    path: Path | None,
    *,
    num_classes: int,
) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {
            "status": "unavailable",
            "reason": "missing_champion_csv",
        }, []
    if not path.exists():
        return {
            "status": "unavailable",
            "reason": "missing_champion_csv",
            "source": path.name,
        }, [f"optional champion_csv missing: {path}"]

    errors: list[str] = []
    counts = {str(class_id): 0 for class_id in range(num_classes)}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["test_idx", "label"]:
            errors.append(
                f"champion csv columns must be ['test_idx', 'label'], got {reader.fieldnames}"
            )
            return {
                "status": "unavailable",
                "reason": "invalid_champion_csv",
                "source": path.name,
            }, errors
        total = 0
        for row_index, row in enumerate(reader):
            try:
                label = int(row["label"])
            except Exception:
                errors.append(f"champion csv row {row_index}: label must be int")
                continue
            if not 0 <= label < num_classes:
                errors.append(
                    f"champion csv row {row_index}: label {label} out of range"
                )
                continue
            counts[str(label)] += 1
            total += 1

    ratios = {
        key: (value / total if total else 0.0)
        for key, value in counts.items()
    }
    return {
        "status": "observed",
        "source": path.name,
        "source_sha256": _sha256(path),
        "total_test_nodes": total,
        "predicted_class_counts": counts,
        "predicted_class_ratios": ratios,
        "interpretation": "predicted_labels_not_test_truth",
    }, errors


def _load_npz_bundle(path: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "adj_data",
        "adj_indices",
        "adj_indptr",
        "adj_shape",
        "attr_data",
        "attr_indices",
        "attr_indptr",
        "attr_shape",
        "labels",
        "train_idx",
        "test_idx",
    ]
    errors: list[str] = []
    data = np.load(path, allow_pickle=False)
    missing = [key for key in required if key not in data.files]
    if missing:
        return {}, [f"npz missing keys: {missing}"]

    adj_shape = tuple(int(x) for x in data["adj_shape"])
    attr_shape = tuple(int(x) for x in data["attr_shape"])
    labels = data["labels"].astype(np.int64)
    train_idx = data["train_idx"].astype(np.int64)
    test_idx = data["test_idx"].astype(np.int64)
    if len(adj_shape) != 2 or adj_shape[0] != adj_shape[1]:
        errors.append("adj_shape must be square")
    if len(attr_shape) != 2 or attr_shape[0] != adj_shape[0]:
        errors.append("attr_shape row count must match adj nodes")
    if labels.shape[0] != adj_shape[0]:
        errors.append("labels length must match node count")
    all_idx = np.concatenate([train_idx, test_idx])
    if np.any(all_idx < 0) or np.any(all_idx >= adj_shape[0]):
        errors.append("train_idx/test_idx out of bounds")
    if len(np.unique(all_idx)) != len(all_idx):
        errors.append("train_idx and test_idx must be unique and disjoint")
    if not np.all(labels[train_idx] >= 0):
        errors.append("train labels must be non-negative")
    if not np.all(labels[test_idx] == -1):
        errors.append("test labels must be -1")
    if errors:
        return {}, errors

    adjacency = sparse.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=adj_shape,
    )
    features = sparse.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=attr_shape,
    )
    return {
        "adjacency": adjacency,
        "features": features,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "raw_adj_indices": data["adj_indices"].astype(np.int64),
        "raw_adj_indptr": data["adj_indptr"].astype(np.int64),
        "raw_adj_shape": adj_shape,
    }, []


def _raw_directed_edges(indptr: np.ndarray, indices: np.ndarray) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for source in range(len(indptr) - 1):
        for pos in range(int(indptr[source]), int(indptr[source + 1])):
            edges.append((source, int(indices[pos])))
    return edges


def _canonical_graph(
    raw_edges: list[tuple[int, int]], n_nodes: int
) -> dict[str, Any]:
    self_loop_count = sum(1 for source, target in raw_edges if source == target)
    non_self = [(source, target) for source, target in raw_edges if source != target]
    directed_counts = Counter(non_self)
    duplicate_count = sum(count - 1 for count in directed_counts.values() if count > 1)
    unique_edges = sorted(directed_counts)

    out_neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
    in_neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
    either_neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
    for source, target in unique_edges:
        out_neighbors[source].add(target)
        in_neighbors[target].add(source)
        either_neighbors[source].add(target)
        either_neighbors[target].add(source)

    return {
        "self_loop_count": self_loop_count,
        "duplicate_directed_edge_count": duplicate_count,
        "unique_edges": unique_edges,
        "out_neighbors": out_neighbors,
        "in_neighbors": in_neighbors,
        "either_neighbors": either_neighbors,
    }


def _read_edges_csv(path: Path) -> tuple[list[tuple[int, int]], list[str]]:
    errors: list[str] = []
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["source", "target"]:
            return [], [f"edge csv columns must be ['source', 'target'], got {reader.fieldnames}"]
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append((int(row["source"]), int(row["target"])))
            except Exception:
                errors.append(f"edge csv line {line_no}: source/target must be int")
    return rows, errors


def _load_fold_file(
    path: Path, train_idx: np.ndarray
) -> tuple[dict[int, int], list[str]]:
    mapping: dict[int, int] = {}
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["global_idx", "fold"]:
            return {}, [f"fold columns must be ['global_idx', 'fold'], got {reader.fieldnames}"]
        for line_no, row in enumerate(reader, start=2):
            try:
                global_idx = int(row["global_idx"])
                fold = int(row["fold"])
            except Exception:
                errors.append(f"fold line {line_no}: global_idx/fold must be int")
                continue
            if global_idx in mapping:
                errors.append(f"fold line {line_no}: duplicate global_idx {global_idx}")
            mapping[global_idx] = fold

    train_set = set(int(x) for x in train_idx)
    if set(mapping) != train_set:
        missing = sorted(train_set - set(mapping))[:10]
        extra = sorted(set(mapping) - train_set)[:10]
        errors.append(f"fold must cover train_idx exactly; missing={missing}, extra={extra}")
    return mapping, errors


def _load_oof(
    path: Path,
    *,
    labels: np.ndarray,
    train_idx: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    data = np.load(path, allow_pickle=False)
    proba_key = next((key for key in ["proba", "oof_proba", "oof"] if key in data.files), "")
    if not proba_key:
        return {}, ["oof npz must contain proba/oof_proba/oof"]
    if "train_idx" not in data.files:
        return {}, ["oof npz must contain train_idx for global alignment"]

    proba = np.asarray(data[proba_key], dtype=np.float64)
    oof_train_idx = data["train_idx"].astype(np.int64)
    if proba.shape != (len(train_idx), num_classes):
        errors.append(f"oof proba shape must be {(len(train_idx), num_classes)}, got {proba.shape}")
    if set(int(x) for x in oof_train_idx) != set(int(x) for x in train_idx):
        errors.append("oof train_idx must cover dataset train_idx exactly")
    if len(np.unique(oof_train_idx)) != len(oof_train_idx):
        errors.append("oof train_idx contains duplicates")
    if not np.isfinite(proba).all() or np.any(proba < 0):
        errors.append("oof proba must be finite and non-negative")
    row_sums = proba.sum(axis=1) if proba.ndim == 2 else np.array([])
    if len(row_sums) and not np.allclose(row_sums, 1.0, atol=1e-4):
        errors.append("oof proba rows must sum to 1")
    if "labels" in data.files:
        oof_labels = data["labels"].astype(np.int64)
        for idx, label in zip(oof_train_idx, oof_labels):
            if labels[int(idx)] != int(label):
                errors.append("oof labels must match dataset train labels")
                break
    if errors:
        return {}, errors

    order = np.argsort(oof_train_idx)
    aligned_idx = oof_train_idx[order]
    aligned_proba = proba[order]
    true_labels = labels[aligned_idx]
    pred = aligned_proba.argmax(axis=1).astype(np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, predicted in zip(true_labels, pred):
        confusion[int(true), int(predicted)] += 1
    return {
        "train_idx": aligned_idx,
        "proba": aligned_proba,
        "true_labels": true_labels,
        "pred": pred,
        "confusion": confusion,
    }, []


def _bfs_shells(neighbors: list[set[int]], start: int, max_hop: int) -> dict[int, set[int]]:
    shells: dict[int, set[int]] = {}
    visited = {start}
    frontier = {start}
    for distance in range(1, max_hop + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(neighbors[node])
        next_frontier -= visited
        shells[distance] = set(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            for rest in range(distance + 1, max_hop + 1):
                shells[rest] = set()
            break
    return shells


def _visible_train_for_node(
    node: int,
    *,
    split: str,
    train_set: set[int],
    fold_map: dict[int, int],
) -> tuple[set[int], str | None]:
    if split == "test" or not fold_map or node not in fold_map:
        return set(train_set), str(fold_map[node]) if node in fold_map else None
    fold = fold_map[node]
    return {idx for idx in train_set if fold_map.get(idx) != fold}, str(fold)


def _bucket_degree(either_degree: int) -> str:
    if either_degree == 0:
        return "isolated"
    if either_degree == 1:
        return "degree_1"
    if 2 <= either_degree <= 5:
        return "degree_2_5"
    return "degree_6p"


def _primary_bucket(counts: dict[str, int]) -> str:
    if counts["exact1_visible_train_count"] > 0:
        return "one_hop_available"
    if counts["exact2_visible_train_count"] > 0:
        return "exact2_only"
    if counts["exact3_visible_train_count"] + counts["exact4_visible_train_count"] > 0:
        return "exact3_4_only"
    return "no_visible_train_within_4_hops"


def _node_rows(
    *,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    graph: dict[str, Any],
    fold_map: dict[int, int],
) -> list[dict[str, Any]]:
    train_set = set(int(x) for x in train_idx)
    test_set = set(int(x) for x in test_idx)
    rows: list[dict[str, Any]] = []
    for node in range(len(labels)):
        split = "train" if node in train_set else "test" if node in test_set else "unknown"
        visible_train, fold = _visible_train_for_node(
            node,
            split=split,
            train_set=train_set,
            fold_map=fold_map,
        )
        in_n = graph["in_neighbors"][node]
        out_n = graph["out_neighbors"][node]
        either_n = graph["either_neighbors"][node]
        shells = _bfs_shells(graph["either_neighbors"], node, MAX_HOP)
        hop_counts = {
            f"exact{distance}_visible_train_count": len(
                shells[distance] & visible_train
            )
            for distance in range(1, MAX_HOP + 1)
        }
        row = {
            "global_idx": node,
            "split": split,
            "true_label": int(labels[node]) if split == "train" else None,
            "fold": fold,
            "in_degree": len(in_n),
            "out_degree": len(out_n),
            "incident_edge_degree": len(in_n) + len(out_n),
            "either_neighbor_degree": len(either_n),
            "graph_visible": len(either_n) > 0,
            "isolated": len(either_n) == 0,
            "degree_bucket": _bucket_degree(len(either_n)),
            "incoming_visible_train_count": len(in_n & visible_train),
            "outgoing_visible_train_count": len(out_n & visible_train),
            "either_visible_train_count": len(either_n & visible_train),
            **hop_counts,
        }
        row["primary_supervision_bucket"] = _primary_bucket(row)
        rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row[key]) for row in rows)
    return [{"bucket": key_value, "node_count": counts[key_value]} for key_value in sorted(counts)]


def _class_profile(labels: np.ndarray, train_idx: np.ndarray) -> list[dict[str, Any]]:
    counts = Counter(int(labels[int(idx)]) for idx in train_idx)
    total = sum(counts.values())
    return [
        {
            "class_id": class_id,
            "train_count": counts[class_id],
            "train_share": counts[class_id] / total if total else 0.0,
        }
        for class_id in sorted(counts)
    ]


def _fold_profile(
    labels: np.ndarray, train_idx: np.ndarray, fold_map: dict[int, int]
) -> list[dict[str, Any]]:
    if not fold_map:
        return [{"status": "unavailable", "reason": "missing_fold_file"}]
    by_fold: dict[int, list[int]] = defaultdict(list)
    for idx in train_idx:
        by_fold[int(fold_map[int(idx)])].append(int(idx))
    return [
        {
            "fold": fold,
            "valid_node_count": len(nodes),
            "class_distribution": _distribution(labels[nodes].tolist()),
        }
        for fold, nodes in sorted(by_fold.items())
    ]


def _feature_shift(features: sparse.csr_matrix, train_idx: np.ndarray, test_idx: np.ndarray) -> list[dict[str, Any]]:
    nnz = np.diff(features.indptr).astype(np.float64)
    l1 = np.asarray(np.abs(features).sum(axis=1)).ravel().astype(np.float64)
    metrics = {"feature_nnz": nnz, "feature_l1": l1}
    rows: list[dict[str, Any]] = []
    for name, values in metrics.items():
        train_values = values[train_idx]
        test_values = values[test_idx]
        pooled = math.sqrt((float(np.var(train_values)) + float(np.var(test_values))) / 2.0)
        smd = 0.0 if pooled < 1e-12 else (float(np.mean(test_values)) - float(np.mean(train_values))) / pooled
        rows.append(
            {
                "status": "observed",
                "shift_family": "feature",
                "metric": name,
                "train_mean": float(np.mean(train_values)),
                "test_mean": float(np.mean(test_values)),
                "smd": float(smd),
            }
        )
    return rows


def _structure_shift_rows(node_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ["degree_bucket", "primary_supervision_bucket"]:
        train_values = [row[metric] for row in node_rows if row["split"] == "train"]
        test_values = [row[metric] for row in node_rows if row["split"] == "test"]
        train_total = len(train_values)
        test_total = len(test_values)
        keys = sorted(set(train_values) | set(test_values))
        train_counts = Counter(train_values)
        test_counts = Counter(test_values)
        for value in keys:
            rows.append(
                {
                    "status": "observed",
                    "shift_family": "structure",
                    "metric": metric,
                    "bucket": value,
                    "train_count": train_counts[value],
                    "test_count": test_counts[value],
                    "train_ratio": train_counts[value] / train_total
                    if train_total
                    else 0.0,
                    "test_ratio": test_counts[value] / test_total
                    if test_total
                    else 0.0,
                }
            )
    return rows


def _unavailable_shift_rows(oof_available: bool) -> list[dict[str, Any]]:
    if oof_available:
        return []
    return [
        {
            "status": "unavailable",
            "shift_family": "oof_prediction_proba",
            "metric": "prediction_probability_shift",
            "reason": "missing_anchor_oof",
        },
        {
            "status": "not_applicable",
            "shift_family": "test_truth",
            "metric": "test_truth_dependent_shift",
            "reason": "test_labels_unavailable",
        },
    ]


def _confusion_rows(confusion: np.ndarray | None) -> list[dict[str, Any]]:
    if confusion is None:
        return [{"status": "unavailable", "reason": "missing_anchor_oof"}]
    rows: list[dict[str, Any]] = []
    for true_label in range(confusion.shape[0]):
        for pred_label in range(confusion.shape[1]):
            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(confusion[true_label, pred_label]),
                }
            )
    return rows


def _prediction_sink_source(confusion: np.ndarray | None) -> list[dict[str, Any]]:
    if confusion is None:
        return [{"status": "unavailable", "reason": "missing_anchor_oof"}]
    total_errors = int(confusion.sum() - np.trace(confusion))
    rows: list[dict[str, Any]] = []
    for class_id in range(confusion.shape[0]):
        true_count = int(confusion[class_id, :].sum())
        predicted_count = int(confusion[:, class_id].sum())
        sink_inflow = int(confusion[:, class_id].sum() - confusion[class_id, class_id])
        source_outflow = int(confusion[class_id, :].sum() - confusion[class_id, class_id])
        rows.append(
            {
                "class_id": class_id,
                "sink_inflow": sink_inflow,
                "source_outflow": source_outflow,
                "sink_net": sink_inflow - source_outflow,
                "source_error_rate": source_outflow / true_count if true_count else 0.0,
                "sink_false_positive_share": sink_inflow / predicted_count if predicted_count else 0.0,
                "overall_error_share": (sink_inflow + source_outflow) / total_errors if total_errors else 0.0,
            }
        )
    return rows


def _problem_map(
    *,
    analysis_tier: str,
    anchor_identity: str,
    node_rows: list[dict[str, Any]],
    confusion: np.ndarray | None,
    signal_inventory: dict[str, Any],
    train_test_shift: dict[str, Any],
) -> dict[str, Any]:
    structure_counts = Counter(row["primary_supervision_bucket"] for row in node_rows)
    rankings: dict[str, Any] = {
        "coverage_gap": [
            {"bucket": bucket, "node_count": count}
            for bucket, count in sorted(structure_counts.items())
        ],
        "train_test_shift": train_test_shift,
    }
    problems: list[dict[str, Any]] = []
    for bucket, count in sorted(structure_counts.items()):
        problems.append(
            {
                "problem_id": f"structure::{bucket}",
                "scope": bucket,
                "node_count": count,
                "error_count": None if confusion is None else 0,
                "error_rate": None if confusion is None else 0.0,
                "gap_to_overall": None,
                "confidence_evidence": "unavailable_without_anchor_oof" if confusion is None else "observed_oof",
                "structure_evidence": "observed",
                "confusion_evidence": "missing" if confusion is None else "observed",
                "signal_status": signal_inventory["signals"],
                "evidence_files": ["a1_node_buckets.csv"],
                "eligible_tool_families": [],
                "blocked_tool_families": ["closed_branches", "model_training", "m3_adapter"],
            }
        )
    return {
        "analysis_tier": analysis_tier,
        "anchor_identity": anchor_identity,
        "rankings": rankings,
        "problems": problems,
        "notes": [
            "Generated by deterministic rules only.",
            "No LLM calls and no model actions were executed.",
        ],
    }


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys or ["status", "reason"]


def _write_unavailable_csv(path: Path, reason: str) -> None:
    _write_csv(path, [{"status": "unavailable", "reason": reason}], ["status", "reason"])


def core_result_hash(out_dir: Path) -> str:
    digest = hashlib.sha256()
    for filename in CORE_OUTPUT_FILES:
        path = out_dir / filename
        if path.exists():
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def run_profile(request: ProfileRequest) -> ProfileRunResult:
    warnings: list[str] = []
    errors: list[str] = []
    npz_path = Path(request.npz_path)
    out_dir = Path(request.out_dir)

    if not npz_path.exists():
        return ProfileRunResult(
            status="waiting_for_input",
            reason="missing_npz_path",
            analysis_tier="unavailable",
            out_dir=str(out_dir),
            warnings=[],
            errors=[str(npz_path)],
        )
    if request.require_fold and (not request.fold_file or not Path(request.fold_file).exists()):
        return ProfileRunResult(
            status="waiting_for_input",
            reason="missing_required_fold_file",
            analysis_tier="dataset_only",
            out_dir=str(out_dir),
            warnings=[],
            errors=[],
        )
    if request.require_oof and (not request.anchor_oof_npz or not Path(request.anchor_oof_npz).exists()):
        return ProfileRunResult(
            status="waiting_for_input",
            reason="missing_required_anchor_oof_npz",
            analysis_tier="dataset_only",
            out_dir=str(out_dir),
            warnings=[],
            errors=[],
        )

    bundle, npz_errors = _load_npz_bundle(npz_path)
    if npz_errors:
        return ProfileRunResult(
            status="failed",
            reason="invalid_npz_schema",
            analysis_tier="unavailable",
            out_dir=str(out_dir),
            errors=npz_errors,
            warnings=warnings,
        )

    labels = bundle["labels"]
    train_idx = bundle["train_idx"]
    test_idx = bundle["test_idx"]
    features: sparse.csr_matrix = bundle["features"]
    raw_edges = _raw_directed_edges(bundle["raw_adj_indptr"], bundle["raw_adj_indices"])
    graph = _canonical_graph(raw_edges, len(labels))
    train_labels = labels[train_idx]
    num_classes = int(train_labels.max()) + 1 if len(train_labels) else 0
    champion_distribution, champion_errors = _champion_prediction_distribution(
        Path(request.champion_csv) if request.champion_csv else None,
        num_classes=num_classes,
    )
    if champion_errors:
        warnings.extend(champion_errors)

    edge_csv_status = "not_provided"
    edge_csv_rows = None
    if request.edges_csv:
        edge_path = Path(request.edges_csv)
        if edge_path.exists():
            edge_rows, edge_errors = _read_edges_csv(edge_path)
            edge_csv_rows = len(edge_rows)
            if edge_errors:
                errors.extend(edge_errors)
            elif Counter(edge_rows) != Counter(raw_edges):
                message = "edge csv does not match canonical NPZ raw adjacency edge multiset"
                if request.strict:
                    errors.append(message)
                else:
                    warnings.append(message)
                edge_csv_status = "mismatch"
            else:
                edge_csv_status = "matched_npz_raw_edges"
        elif request.strict:
            errors.append(f"edges_csv missing: {edge_path}")
        else:
            warnings.append(f"optional edges_csv missing: {edge_path}")
    if errors:
        return ProfileRunResult(
            status="failed",
            reason="input_validation_failed",
            analysis_tier="unavailable",
            out_dir=str(out_dir),
            errors=errors,
            warnings=warnings,
        )

    fold_map: dict[int, int] = {}
    fold_status = "missing"
    if request.fold_file and Path(request.fold_file).exists():
        fold_map, fold_errors = _load_fold_file(Path(request.fold_file), train_idx)
        if fold_errors:
            return ProfileRunResult(
                status="failed",
                reason="invalid_fold_file",
                analysis_tier="dataset_only",
                out_dir=str(out_dir),
                errors=fold_errors,
                warnings=warnings,
            )
        fold_status = "available"
    else:
        warnings.append("canonical fold assignment not provided; fold-aware OOF tier unavailable")

    oof_payload: dict[str, Any] = {}
    oof_status = "missing"
    if request.anchor_oof_npz and Path(request.anchor_oof_npz).exists():
        oof_payload, oof_errors = _load_oof(
            Path(request.anchor_oof_npz),
            labels=labels,
            train_idx=train_idx,
            num_classes=num_classes,
        )
        if oof_errors:
            return ProfileRunResult(
                status="failed",
                reason="invalid_anchor_oof",
                analysis_tier="fold_aware_structure" if fold_map else "dataset_only",
                out_dir=str(out_dir),
                errors=oof_errors,
                warnings=warnings,
            )
        oof_status = "available"
    else:
        warnings.append("v53Q-1 anchor OOF proba not provided; full_anchor_oof tier unavailable")

    analysis_tier = "dataset_only"
    if fold_map:
        analysis_tier = "fold_aware_structure"
    if fold_map and oof_payload:
        analysis_tier = "full_anchor_oof"

    node_rows = _node_rows(
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        graph=graph,
        fold_map=fold_map,
    )
    node_rows.sort(key=lambda row: int(row["global_idx"]))
    structure_rows = _summarize_rows(node_rows, "degree_bucket")
    train_neighbor_rows = _summarize_rows(node_rows, "primary_supervision_bucket")
    hop_rows = [
        {
            "hop": hop,
            "nodes_with_visible_train": sum(
                1 for row in node_rows if int(row[f"exact{hop}_visible_train_count"]) > 0
            ),
        }
        for hop in range(1, MAX_HOP + 1)
    ]
    class_rows = _class_profile(labels, train_idx)
    fold_rows = _fold_profile(labels, train_idx, fold_map)
    feature_shift_rows = _feature_shift(features, train_idx, test_idx)
    structure_shift_rows = _structure_shift_rows(node_rows)
    confusion = oof_payload.get("confusion") if oof_payload else None
    unavailable_shift_rows = _unavailable_shift_rows(confusion is not None)
    shift_rows = [
        *feature_shift_rows,
        *structure_shift_rows,
        *unavailable_shift_rows,
    ]
    confusion_rows = _confusion_rows(confusion)
    sink_source_rows = _prediction_sink_source(confusion)
    signal_inventory = {
        "signals": {
            "champion_csv": "available" if request.champion_csv and Path(request.champion_csv).exists() else "missing",
            "canonical_fold": "available" if fold_map else "missing",
            "v53q1_anchor_oof": "available" if oof_payload else "missing",
            "edge_csv_cross_check": "observed" if edge_csv_status.startswith("matched") else edge_csv_status,
            "closed_branches": "closed",
            "unverified_experts": "unsupported",
        }
    }
    train_test_shift = {
        "status": "observed",
        "observed": [
            {
                "shift_family": "feature",
                "metric": row["metric"],
                "train_mean": row["train_mean"],
                "test_mean": row["test_mean"],
                "smd": row["smd"],
            }
            for row in feature_shift_rows
        ]
        + [
            {
                "shift_family": "structure",
                "metric": row["metric"],
                "bucket": row["bucket"],
                "train_ratio": row["train_ratio"],
                "test_ratio": row["test_ratio"],
            }
            for row in structure_shift_rows
        ],
        "unavailable": [
            row for row in unavailable_shift_rows if row["status"] == "unavailable"
        ],
        "not_applicable": [
            row for row in unavailable_shift_rows if row["status"] == "not_applicable"
        ],
    }
    input_validation = {
        "status": "passed",
        "npz_path": {"sha256": _sha256(npz_path), "schema": "a1_npz_csr"},
        "edge_csv": {
            "status": edge_csv_status,
            "rows": edge_csv_rows,
        },
        "fold": {"status": fold_status},
        "oof": {"status": oof_status},
        "graph_audit": {
            "self_loop_count": graph["self_loop_count"],
            "duplicate_directed_edge_count": graph["duplicate_directed_edge_count"],
            "raw_directed_edge_count": len(raw_edges),
            "unique_directed_edges_excluding_self": len(graph["unique_edges"]),
        },
        "warnings": warnings,
    }
    graph_direction_profile = {
        "in_degree_mean": float(np.mean([len(x) for x in graph["in_neighbors"]])),
        "out_degree_mean": float(np.mean([len(x) for x in graph["out_neighbors"]])),
        "either_neighbor_degree_mean": float(np.mean([len(x) for x in graph["either_neighbors"]])),
        "bidirectional_pair_count": sum(
            1
            for source, target in graph["unique_edges"]
            if source < target and source in graph["out_neighbors"][target]
        ),
    }
    profile = {
        "metadata": {
            "profile_version": request.profile_version,
            "deterministic": True,
            "read_only": True,
            "cpu_only": True,
            "max_hop": MAX_HOP,
        },
        "analysis_tier": analysis_tier,
        "anchor_identity": "v53Q-1" if oof_payload else None,
        "dataset": {
            "num_nodes": int(len(labels)),
            "num_features": int(features.shape[1]),
            "num_classes": int(num_classes),
            "train_nodes": int(len(train_idx)),
            "test_nodes": int(len(test_idx)),
            "feature_nnz": int(features.nnz),
        },
        "graph": {
            "edge_source": "npz_adj_csr",
            "edge_csv_role": "optional_cross_check_only",
            "directed": True,
            "self_loops_removed_for_structure": True,
            "duplicates_deduplicated_for_structure": True,
            "raw_directed_edge_count": len(raw_edges),
            "unique_directed_edges_excluding_self": len(graph["unique_edges"]),
        },
        "graph_direction_profile": graph_direction_profile,
        "structure_buckets": {
            "degree_bucket_counts": {
                bucket: sum(1 for row in node_rows if row["degree_bucket"] == bucket)
                for bucket in ["isolated", "degree_1", "degree_2_5", "degree_6p"]
            }
        },
        "exact_hop_policy": {
            "primary_exact_hop_view": "either_direction",
            "directed_exact_hop_breakdown": {
                "status": "not_generated",
                "reason": (
                    "not included in M2 v1; reserved for directed H2 signal audit"
                ),
            },
        },
        "folds": {
            "status": fold_status,
            "fold_count": len(set(fold_map.values())) if fold_map else 0,
        },
        "oof_profile": {
            "status": oof_status,
            "alignment": "global_train_idx" if oof_payload else None,
            "probability_shape": list(oof_payload["proba"].shape) if oof_payload else None,
        },
        "prediction_sink_source": {
            "status": "available" if oof_payload else "unavailable",
            "definition": "OOF confusion matrix C[true,pred]",
        },
        "test_policy": {
            "truth_available": False,
            "truth_dependent_metrics_emitted": False,
        },
        "test_profile": {
            "champion_predicted_label_distribution": champion_distribution,
            "test_label_distribution": {
                "status": "not_applicable",
                "reason": "test_truth_unavailable",
            },
        },
        "warnings": warnings,
    }
    problem_map = _problem_map(
        analysis_tier=analysis_tier,
        anchor_identity="v53Q-1" if oof_payload else "unavailable",
        node_rows=node_rows,
        confusion=confusion,
        signal_inventory=signal_inventory,
        train_test_shift=train_test_shift,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "a1_data_profile.json", profile)
    _write_json(out_dir / "a1_input_validation.json", input_validation)
    _write_json(out_dir / "a1_signal_inventory.json", signal_inventory)
    _write_json(out_dir / "a1_problem_map.json", problem_map)
    _write_csv(out_dir / "a1_node_buckets.csv", node_rows, _fieldnames(node_rows))
    _write_csv(out_dir / "a1_structure_profile.csv", structure_rows, _fieldnames(structure_rows))
    _write_csv(out_dir / "a1_train_neighbor_profile.csv", train_neighbor_rows, _fieldnames(train_neighbor_rows))
    _write_csv(out_dir / "a1_hop_profile.csv", hop_rows, _fieldnames(hop_rows))
    _write_csv(out_dir / "a1_class_profile.csv", class_rows, _fieldnames(class_rows))
    _write_csv(out_dir / "a1_confusion_transitions.csv", confusion_rows, _fieldnames(confusion_rows))
    _write_csv(out_dir / "a1_prediction_sink_source.csv", sink_source_rows, _fieldnames(sink_source_rows))
    _write_csv(out_dir / "a1_fold_profile.csv", fold_rows, _fieldnames(fold_rows))
    _write_csv(out_dir / "a1_shift_profile.csv", shift_rows, _fieldnames(shift_rows))
    _atomic_write_text(
        out_dir / "A1_DATA_PROFILE_REPORT.md",
        "\n".join(
            [
                "# A1 Data Profile Report",
                "",
                f"- analysis_tier: {analysis_tier}",
                f"- nodes: {len(labels)}",
                f"- train_nodes: {len(train_idx)}",
                f"- test_nodes: {len(test_idx)}",
                "- primary_exact_hop_view: either_direction",
                "- directed_exact_hop_breakdown: not_generated "
                "(reserved for directed H2 signal audit).",
                "- no model training, prediction generation, or submission was performed.",
                "",
            ]
        ),
    )
    core_hash = core_result_hash(out_dir)
    manifest = {
        "profile_version": request.profile_version,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "absolute_input_path": _stable_path(npz_path),
        "analysis_tier": analysis_tier,
        "core_result_hash": core_hash,
        "generated_files": CORE_OUTPUT_FILES,
        "read_only": True,
        "counts_as_experiment_round": False,
        "mutates_project_state": False,
        "mutates_predictions": False,
        "requires_gpu": False,
    }
    _write_json(out_dir / "a1_profile_manifest.json", manifest)
    return ProfileRunResult(
        status="completed",
        reason="ok",
        analysis_tier=analysis_tier,
        out_dir=str(out_dir),
        core_result_hash=core_hash,
        generated_files=["a1_profile_manifest.json", *CORE_OUTPUT_FILES],
        warnings=warnings,
        errors=[],
    )


def _parse_args() -> ProfileRequest:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--edges_csv", default="")
    parser.add_argument("--champion_csv", default="")
    parser.add_argument("--fold_file", default="")
    parser.add_argument("--anchor_oof_npz", default="")
    parser.add_argument("--reference_oof_npz", default="")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require_fold", action="store_true")
    parser.add_argument("--require_oof", action="store_true")
    parser.add_argument("--profile_version", default=PROFILE_VERSION)
    args = parser.parse_args()

    def optional_path(value: str) -> Path | None:
        return Path(value) if value else None

    return ProfileRequest(
        npz_path=Path(args.npz_path),
        out_dir=Path(args.out_dir),
        edges_csv=optional_path(args.edges_csv),
        champion_csv=optional_path(args.champion_csv),
        fold_file=optional_path(args.fold_file),
        anchor_oof_npz=optional_path(args.anchor_oof_npz),
        reference_oof_npz=optional_path(args.reference_oof_npz),
        strict=args.strict,
        require_fold=args.require_fold,
        require_oof=args.require_oof,
        profile_version=args.profile_version,
    )


def main() -> None:
    result = run_profile(_parse_args())
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if result.status == "failed":
        raise SystemExit(2)
    if result.status == "waiting_for_input":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
