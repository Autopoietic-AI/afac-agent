# -*- coding: utf-8 -*-
"""AFAC v2.2 canonical data contract for node classification (B1).

Reconciles node, edge, feature, and label counts across Input Discovery,
Data Intelligence, the Experiment Executor, and Deployment.  Every scale
value keeps a provenance record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..research.event_store import stable_hash


@dataclass
class ScaleValue:
    """One measured scale with full provenance."""

    value: int
    source_file: str
    source_column: str
    counting_rule: str
    deduplicated: bool
    sampled: bool
    membership_hash: str


@dataclass
class B1CanonicalDataContract:
    """Canonical B1 data contract for node classification."""

    n_nodes_total: ScaleValue | None = None
    n_train_nodes_total: ScaleValue | None = None
    n_test_nodes_total: ScaleValue | None = None
    n_edges_total: ScaleValue | None = None
    n_directed_edges: ScaleValue | None = None
    n_features: ScaleValue | None = None
    n_classes: ScaleValue | None = None
    # Sources
    train_node_source: str = ""
    test_node_source: str = ""
    feature_source: str = ""
    edge_source: str = ""
    label_source: str = ""
    submission_template_source: str = ""
    # Column names
    node_id_column: str = ""
    source_node_column: str = ""
    target_node_column: str = ""
    label_column: str = ""
    # Hashes
    feature_schema_hash: str = ""
    node_universe_hash: str = ""
    edge_hash: str = ""
    train_membership_hash: str = ""
    test_membership_hash: str = ""
    # Status
    status: str = "unknown"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def sv(v: ScaleValue | None) -> dict[str, Any] | None:
            if v is None:
                return None
            return {
                "value": v.value,
                "source_file": v.source_file,
                "source_column": v.source_column,
                "counting_rule": v.counting_rule,
                "deduplicated": v.deduplicated,
                "sampled": v.sampled,
                "membership_hash": v.membership_hash,
            }

        return {
            "n_nodes_total": sv(self.n_nodes_total),
            "n_train_nodes_total": sv(self.n_train_nodes_total),
            "n_test_nodes_total": sv(self.n_test_nodes_total),
            "n_edges_total": sv(self.n_edges_total),
            "n_directed_edges": sv(self.n_directed_edges),
            "n_features": sv(self.n_features),
            "n_classes": sv(self.n_classes),
            "train_node_source": self.train_node_source,
            "test_node_source": self.test_node_source,
            "feature_source": self.feature_source,
            "edge_source": self.edge_source,
            "label_source": self.label_source,
            "submission_template_source": self.submission_template_source,
            "node_id_column": self.node_id_column,
            "source_node_column": self.source_node_column,
            "target_node_column": self.target_node_column,
            "label_column": self.label_column,
            "feature_schema_hash": self.feature_schema_hash,
            "node_universe_hash": self.node_universe_hash,
            "edge_hash": self.edge_hash,
            "train_membership_hash": self.train_membership_hash,
            "test_membership_hash": self.test_membership_hash,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def data_hash(self) -> str:
        return stable_hash(self.to_dict())


def _hash_array(arr: Any) -> str:
    return stable_hash({"values": np.asarray(arr).tolist()})


def build_b1_data_contract(
    *,
    data_root: Path,
    n_nodes: int,
    n_features: int,
    n_classes: int,
    n_edges: int,
    n_directed_edges: int,
    train_idx: Any,
    test_idx: Any,
    node_ids: list[int] | None = None,
) -> B1CanonicalDataContract:
    """Build the canonical B1 contract from the loaded NPZ dataset.

    ``train_idx`` and ``test_idx`` are numpy arrays.
    ``node_ids`` if provided are the 0..n_nodes-1 implicit indices.
    """
    train_arr = np.asarray(train_idx, dtype=np.int64)
    test_arr = np.asarray(test_idx, dtype=np.int64)
    if node_ids is None:
        node_ids = list(range(n_nodes))

    errors: list[str] = []
    warnings: list[str] = []

    # Basic sanity
    train_set = set(train_arr.tolist())
    test_set = set(test_arr.tolist())
    overlap = sorted(train_set & test_set)
    if overlap:
        errors.append(f"train/test node overlap: {len(overlap)} nodes")
    if len(train_set) != len(train_arr):
        errors.append("duplicate nodes in train_idx")
    if len(test_set) != len(test_arr):
        errors.append("duplicate nodes in test_idx")
    if train_arr.min() < 0 or train_arr.max() >= n_nodes:
        errors.append("train_idx out of node range")
    if test_arr.min() < 0 or test_arr.max() >= n_nodes:
        errors.append("test_idx out of node range")
    if len(set(node_ids)) != n_nodes:
        errors.append("node_ids contain duplicates or gaps")

    npz_file = str(data_root / "B1.npz")
    submission_file = str(data_root / "sample_submission.csv")

    contract = B1CanonicalDataContract(
        n_nodes_total=ScaleValue(
            value=n_nodes,
            source_file=npz_file,
            source_column="features.shape[0]",
            counting_rule="node_count",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_array(node_ids),
        ),
        n_train_nodes_total=ScaleValue(
            value=len(train_arr),
            source_file=npz_file,
            source_column="train_idx",
            counting_rule="node_count",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_array(sorted(train_arr)),
        ),
        n_test_nodes_total=ScaleValue(
            value=len(test_arr),
            source_file=npz_file,
            source_column="test_idx",
            counting_rule="node_count",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_array(sorted(test_arr)),
        ),
        n_edges_total=ScaleValue(
            value=n_edges,
            source_file=npz_file,
            source_column="adj_data",
            counting_rule="edge_count_undirected",
            deduplicated=False,
            sampled=False,
            membership_hash=stable_hash({"n_edges": n_edges}),
        ),
        n_directed_edges=ScaleValue(
            value=n_directed_edges,
            source_file=npz_file,
            source_column="adj_data",
            counting_rule="edge_count_directed",
            deduplicated=True,
            sampled=False,
            membership_hash=stable_hash({"n_directed": n_directed_edges}),
        ),
        n_features=ScaleValue(
            value=n_features,
            source_file=npz_file,
            source_column="attr_data",
            counting_rule="feature_dimension",
            deduplicated=True,
            sampled=False,
            membership_hash=stable_hash({"n_features": n_features}),
        ),
        n_classes=ScaleValue(
            value=n_classes,
            source_file=npz_file,
            source_column="labels[train_idx]",
            counting_rule="unique_label_classes",
            deduplicated=True,
            sampled=False,
            membership_hash=stable_hash({"n_classes": n_classes}),
        ),
        train_node_source=f"{npz_file}::train_idx",
        test_node_source=f"{npz_file}::test_idx",
        feature_source=f"{npz_file}::attr_data",
        edge_source=f"{npz_file}::adj_data",
        label_source=f"{npz_file}::labels",
        submission_template_source=submission_file,
        node_id_column="node_index_implicit",
        source_node_column="adj_indices[row]",
        target_node_column="adj_indices[col]",
        label_column="labels[node_idx]",
        status="passed" if not errors else "blocked_data_contract_mismatch",
        errors=errors,
        warnings=warnings,
    )
    return contract


def reconcile_b1_data_contract(
    input_discovery_n_nodes: int,
    data_intelligence_n_nodes: int,
    experiment_executor_node_universe: set[int],
    deployment_node_order: list[int],
) -> dict[str, Any]:
    """Cross-source consistency check for B1."""
    errors: list[str] = []
    if input_discovery_n_nodes != data_intelligence_n_nodes:
        errors.append(
            f"input_discovery n_nodes={input_discovery_n_nodes} != "
            f"data_intelligence n_nodes={data_intelligence_n_nodes}"
        )
    if len(experiment_executor_node_universe) != input_discovery_n_nodes:
        errors.append(
            f"experiment_executor node universe size={len(experiment_executor_node_universe)} != "
            f"input_discovery n_nodes={input_discovery_n_nodes}"
        )
    if set(deployment_node_order) != experiment_executor_node_universe:
        errors.append("deployment node order differs from experiment executor node universe")
    return {
        "status": "passed" if not errors else "blocked_data_contract_mismatch",
        "errors": errors,
    }
