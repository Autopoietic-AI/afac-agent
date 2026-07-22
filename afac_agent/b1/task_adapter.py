# -*- coding: utf-8 -*-
"""Generic node classification task adapter (parameterized over A1/B1/etc.).

Reads node features, adjacency (CSR), train/test indices, labels, and sample
submission.  Validates no train/test overlap, no label leakage, no NaN/Inf,
valid ranges, and preserves the official submission node order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class NodeClassificationDataset:
    task_id: str
    data_root: Path
    n_nodes: int
    n_features: int
    n_classes: int
    features: csr_matrix  # (n_nodes, n_features)
    adj: csr_matrix  # directed as stored
    labels: np.ndarray  # full length, -1 for test
    train_idx: np.ndarray
    test_idx: np.ndarray
    sample_submission: list[dict[str, Any]]  # official test order
    validation: dict[str, Any] = field(default_factory=dict)

    def y_train(self) -> np.ndarray:
        return self.labels[self.train_idx]

    def directed_adj(self, direction: str) -> csr_matrix:
        if direction == "directed_out":
            return self.adj
        if direction == "directed_in":
            return self.adj.T.tocsr()
        if direction == "undirected_union":
            sym = self.adj + self.adj.T
            sym.data = np.ones_like(sym.data)
            return sym.tocsr()
        raise ValueError(f"unknown direction: {direction}")


class NodeClassificationTaskAdapter:
    """Read-only loader for node-classification datasets stored as NPZ + CSV."""

    def __init__(self, data_root: str | Path, *, task_id: str = "B1") -> None:
        self.data_root = Path(data_root)
        self.task_id = task_id

    def missing_files(self) -> list[str]:
        missing = []
        for name in ("B1.npz", "sample_submission.csv"):
            if not (self.data_root / name).is_file():
                missing.append(name)
        return missing

    def load(self) -> NodeClassificationDataset:
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(f"{self.task_id} data files missing: {missing}")
        errors: list[str] = []
        warnings: list[str] = []

        npz_path = self.data_root / "B1.npz"
        data = np.load(npz_path, allow_pickle=True)
        for key in ("adj_data", "adj_indices", "adj_indptr", "adj_shape",
                    "attr_data", "attr_indices", "attr_indptr", "attr_shape",
                    "labels", "train_idx", "test_idx"):
            if key not in data.files:
                errors.append(f"B1.npz missing key: {key}")

        adj = csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=tuple(data["adj_shape"]),
        )
        features = csr_matrix(
            (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
            shape=tuple(data["attr_shape"]),
        )
        labels = np.asarray(data["labels"]).astype(np.int64)
        train_idx = np.asarray(data["train_idx"]).astype(np.int64)
        test_idx = np.asarray(data["test_idx"]).astype(np.int64)

        n_nodes = int(features.shape[0])
        n_features = int(features.shape[1])
        if adj.shape != (n_nodes, n_nodes):
            errors.append(f"adj shape {adj.shape} != ({n_nodes},{n_nodes})")

        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())
        overlap = sorted(train_set & test_set)
        if overlap:
            errors.append(f"train/test overlap: {len(overlap)} nodes")
        if train_idx.min() < 0 or train_idx.max() >= n_nodes or test_idx.min() < 0 or test_idx.max() >= n_nodes:
            errors.append("node index out of range")
        if len(train_set) != len(train_idx) or len(test_set) != len(test_idx):
            errors.append("duplicate indices in train/test")

        y_train = labels[train_idx]
        if np.any(y_train < 0):
            errors.append("train labels contain -1")
        n_classes = int(np.unique(y_train).max()) + 1
        if not np.array_equal(np.unique(y_train), np.arange(n_classes)):
            warnings.append("train labels not contiguous from 0")
        if np.any(labels[test_idx] != -1):
            errors.append("test labels are not hidden (-1)")

        if not np.isfinite(features.data).all():
            errors.append("features contain non-finite values")
        if not np.isfinite(adj.data).all():
            errors.append("adjacency weights contain non-finite values")

        sample_submission = self._load_submission(warnings)
        if sample_submission:
            sub_ids = [int(r["test_idx"]) for r in sample_submission]
            if set(sub_ids) != test_set:
                errors.append("sample submission test_idx does not match npz test_idx")
        else:
            errors.append("sample submission empty or unreadable")

        validation = {
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "warnings": warnings,
            "n_nodes": n_nodes,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_truth_hidden": bool(np.all(labels[test_idx] == -1)),
        }
        return NodeClassificationDataset(
            task_id=self.task_id,
            data_root=self.data_root,
            n_nodes=n_nodes,
            n_features=n_features,
            n_classes=n_classes,
            features=features,
            adj=adj,
            labels=labels,
            train_idx=train_idx,
            test_idx=test_idx,
            sample_submission=sample_submission,
            validation=validation,
        )

    def _load_submission(self, warnings: list[str]) -> list[dict[str, Any]]:
        path = self.data_root / "sample_submission.csv"
        if not path.is_file():
            return []
        import csv
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({"test_idx": int(row["test_idx"]), "label": int(row["label"])})
        return rows

    def submission_order(self, dataset: NodeClassificationDataset) -> np.ndarray:
        return np.array([r["test_idx"] for r in dataset.sample_submission], dtype=np.int64)
