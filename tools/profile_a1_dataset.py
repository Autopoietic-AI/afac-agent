# -*- coding: utf-8 -*-
"""A1图节点分类数据画像工具。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def smd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt(
        (np.var(a) + np.var(b)) / 2.0
    )
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def load_npz(path: Path):
    data = np.load(path)
    adjacency = sp.csr_matrix(
        (
            data["adj_data"],
            data["adj_indices"],
            data["adj_indptr"],
        ),
        shape=tuple(data["adj_shape"]),
    )
    features = sp.csr_matrix(
        (
            data["attr_data"],
            data["attr_indices"],
            data["attr_indptr"],
        ),
        shape=tuple(data["attr_shape"]),
    )
    return {
        "raw": data,
        "adj": adjacency,
        "features": features,
        "labels": data["labels"],
        "train_idx": data["train_idx"].astype(np.int64),
        "test_idx": data["test_idx"].astype(np.int64),
    }


def feature_row_stats(x: sp.csr_matrix) -> pd.DataFrame:
    x = x.tocsr()
    nnz = np.diff(x.indptr).astype(np.float64)
    l1 = np.asarray(np.abs(x).sum(axis=1)).ravel()
    l2 = np.sqrt(
        np.asarray(x.multiply(x).sum(axis=1)).ravel()
    )
    row_sum = np.asarray(x.sum(axis=1)).ravel()
    mean_nonzero = np.divide(
        row_sum,
        nnz,
        out=np.zeros_like(row_sum, dtype=np.float64),
        where=nnz > 0,
    )
    return pd.DataFrame(
        {
            "feature_nnz": nnz,
            "feature_l1": l1,
            "feature_l2": l2,
            "feature_mean_nonzero": mean_nonzero,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_npz(Path(args.npz_path))
    adj = bundle["adj"].astype(bool).astype(np.int8)
    x = bundle["features"]
    labels = bundle["labels"]
    train_idx = bundle["train_idx"]
    test_idx = bundle["test_idx"]

    n_nodes, n_features = x.shape
    train_labels = labels[train_idx]
    valid_classes = np.unique(
        train_labels[train_labels >= 0]
    )
    num_classes = int(len(valid_classes))

    out_degree = np.diff(adj.indptr).astype(np.int64)
    in_degree = np.asarray(adj.sum(axis=0)).ravel().astype(np.int64)
    isolated = (out_degree == 0) & (in_degree == 0)
    graph_visible = ~isolated

    adj_no_diag = adj.copy().tolil()
    adj_no_diag.setdiag(0)
    adj_no_diag = adj_no_diag.tocsr()
    adj_no_diag.eliminate_zeros()

    reverse_overlap = adj_no_diag.multiply(adj_no_diag.T)
    directed_edge_count = int(adj_no_diag.nnz)
    reciprocal_directed_edges = int(reverse_overlap.nnz)
    reciprocity = (
        reciprocal_directed_edges / directed_edge_count
        if directed_edge_count else 0.0
    )

    rows, cols = adj_no_diag.nonzero()
    train_mask_global = np.zeros(n_nodes, dtype=bool)
    train_mask_global[train_idx] = True
    train_train_edge = (
        train_mask_global[rows]
        & train_mask_global[cols]
    )
    tt_rows = rows[train_train_edge]
    tt_cols = cols[train_train_edge]
    if len(tt_rows):
        edge_homophily = float(
            np.mean(labels[tt_rows] == labels[tt_cols])
        )
    else:
        edge_homophily = None

    class_counts = pd.Series(
        train_labels
    ).value_counts().sort_index()
    majority_prior = float(
        class_counts.max() / class_counts.sum()
    )

    row_stats = feature_row_stats(x)
    train_stats = row_stats.iloc[train_idx]
    test_stats = row_stats.iloc[test_idx]
    shift_rows = []
    for column in row_stats.columns:
        shift_rows.append(
            {
                "feature": column,
                "train_mean": float(
                    train_stats[column].mean()
                ),
                "test_mean": float(
                    test_stats[column].mean()
                ),
                "smd": smd(
                    train_stats[column].to_numpy(),
                    test_stats[column].to_numpy(),
                ),
            }
        )
    shift_frame = pd.DataFrame(shift_rows)

    degree_total = out_degree + in_degree
    bucket_labels = np.select(
        [
            isolated,
            degree_total == 1,
            (degree_total >= 2) & (degree_total <= 5),
            degree_total >= 6,
        ],
        [
            "isolated",
            "degree_1",
            "degree_2_5",
            "degree_6_plus",
        ],
        default="other",
    )
    bucket_frame = (
        pd.DataFrame(
            {
                "bucket": bucket_labels,
                "is_train": train_mask_global,
            }
        )
        .groupby(["bucket", "is_train"])
        .size()
        .reset_index(name="count")
    )

    feature_sparsity = float(
        1.0 - x.nnz / (x.shape[0] * x.shape[1])
    )
    avg_total_degree = float(np.mean(degree_total))
    isolated_ratio = float(np.mean(isolated))
    directed = reciprocity < 0.5
    sparse_graph = avg_total_degree < 10
    mixed_visibility = isolated_ratio >= 0.05
    homophily_signal = (
        edge_homophily is not None
        and edge_homophily >= majority_prior + 0.10
    )

    archetype_parts = []
    archetype_parts.append(
        "directed" if directed else "reciprocal"
    )
    archetype_parts.append(
        "sparse" if sparse_graph else "dense"
    )
    archetype_parts.append(
        "mixed_visibility"
        if mixed_visibility
        else "mostly_visible"
    )
    archetype_parts.append(
        "homophilous"
        if homophily_signal
        else "weak_or_heterophilous"
    )
    archetype = "_".join(archetype_parts)

    profile = {
        "dataset_path": str(Path(args.npz_path).resolve()),
        "num_nodes": int(n_nodes),
        "num_features": int(n_features),
        "num_classes": num_classes,
        "train_nodes": int(len(train_idx)),
        "test_nodes": int(len(test_idx)),
        "feature_nnz": int(x.nnz),
        "feature_sparsity": feature_sparsity,
        "directed_edges_excluding_self": directed_edge_count,
        "average_total_degree": avg_total_degree,
        "reciprocity": reciprocity,
        "isolated_all": int(isolated.sum()),
        "isolated_train": int(isolated[train_idx].sum()),
        "isolated_test": int(isolated[test_idx].sum()),
        "isolated_ratio": isolated_ratio,
        "graph_visible_train": int(graph_visible[train_idx].sum()),
        "graph_visible_test": int(graph_visible[test_idx].sum()),
        "train_train_edge_homophily": edge_homophily,
        "majority_class_prior": majority_prior,
        "dataset_archetype": archetype,
        "routing_recommendation": {
            "split_graph_visible_and_isolated":
                mixed_visibility,
            "preserve_direction":
                directed,
            "start_with_homophily_anchor":
                bool(homophily_signal),
            "require_isolated_tabular_expert":
                bool(isolated[train_idx].sum() > 0),
            "require_train_test_shift_audit":
                bool(
                    shift_frame["smd"].abs().max() > 0.25
                ),
        },
    }

    class_frame = pd.DataFrame(
        {
            "class": class_counts.index.astype(int),
            "train_count": class_counts.values.astype(int),
            "train_ratio": (
                class_counts.values / class_counts.values.sum()
            ),
        }
    )

    (out_dir / "a1_data_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    class_frame.to_csv(
        out_dir / "a1_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bucket_frame.to_csv(
        out_dir / "a1_structure_buckets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shift_frame.to_csv(
        out_dir / "a1_train_test_feature_shift.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
