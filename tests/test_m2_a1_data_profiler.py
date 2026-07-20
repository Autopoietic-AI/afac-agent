# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from afac_agent.profilers.a1_data_profiler import (
    ProfileRequest,
    core_result_hash,
    run_profile,
)


def _write_npz(path: Path) -> None:
    # Canonical directed edges include a self-loop and a duplicate.
    adj_data = np.ones(11, dtype=np.float32)
    adj_indices = np.array([0, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    adj_indptr = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
    attr = sparse.identity(9, dtype=np.float32, format="csr")
    labels = np.array([0, 1, 2, 0, 1, 2, 0, -1, -1], dtype=np.int64)
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    test_idx = np.array([7, 8], dtype=np.int64)
    np.savez(
        path,
        adj_data=adj_data,
        adj_indices=adj_indices,
        adj_indptr=adj_indptr,
        adj_shape=np.array([9, 9]),
        attr_data=attr.data,
        attr_indices=attr.indices,
        attr_indptr=attr.indptr,
        attr_shape=np.array(attr.shape),
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def _write_edges(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writeheader()
        # Strictly mirrors the raw NPZ edge stream before CSR deduplication.
        for source, target in [
            (0, 0),
            (0, 1),
            (0, 1),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
        ]:
            writer.writerow({"source": source, "target": target})


def _write_mismatched_edges(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writeheader()
        writer.writerow({"source": 0, "target": 2})


def _write_fold(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["global_idx", "fold"])
        writer.writeheader()
        for global_idx, fold in [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 0),
            (5, 1),
            (6, 0),
        ]:
            writer.writerow({"global_idx": global_idx, "fold": fold})


def _write_oof(path: Path) -> None:
    train_idx = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    labels = np.array([0, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    pred = np.array([0, 1, 1, 2, 2, 1, 1], dtype=np.int64)
    proba = np.full((7, 3), 0.05, dtype=np.float64)
    proba[np.arange(7), pred] = 0.9
    np.savez(path, proba=proba, train_idx=train_idx, labels=labels)


def _write_champion_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_idx", "label"])
        writer.writeheader()
        writer.writerow({"test_idx": 7, "label": 1})
        writer.writerow({"test_idx": 8, "label": 2})


def test_directed_degrees_loops_duplicates_and_buckets(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    edges = tmp_path / "edges.csv"
    out_dir = tmp_path / "out"
    _write_npz(npz)
    _write_edges(edges)

    result = run_profile(
        ProfileRequest(npz_path=npz, edges_csv=edges, out_dir=out_dir)
    )

    assert result.status == "completed"
    profile = json.loads((out_dir / "a1_data_profile.json").read_text())
    validation = json.loads((out_dir / "a1_input_validation.json").read_text())
    assert validation["graph_audit"]["self_loop_count"] == 1
    assert validation["graph_audit"]["duplicate_directed_edge_count"] == 1

    rows = {
        int(row["global_idx"]): row
        for row in csv.DictReader((out_dir / "a1_node_buckets.csv").open())
    }
    assert rows[0]["in_degree"] == "1"
    assert rows[0]["out_degree"] == "1"
    assert rows[0]["incident_edge_degree"] == "2"
    assert rows[0]["either_neighbor_degree"] == "1"
    assert rows[0]["degree_bucket"] == "degree_1"
    assert rows[0]["graph_visible"] == "true"
    assert rows[0]["isolated"] == "false"

    assert profile["graph"]["edge_source"] == "npz_adj_csr"
    assert profile["structure_buckets"]["degree_bucket_counts"]["degree_6p"] == 0


def test_champion_prediction_distribution_and_shift_status(
    tmp_path: Path,
) -> None:
    npz = tmp_path / "a1.npz"
    champion = tmp_path / "champion.csv"
    out_dir = tmp_path / "out"
    _write_npz(npz)
    _write_champion_csv(champion)

    result = run_profile(
        ProfileRequest(
            npz_path=npz,
            champion_csv=champion,
            out_dir=out_dir,
        )
    )

    assert result.status == "completed"
    profile = json.loads((out_dir / "a1_data_profile.json").read_text())
    test_profile = profile["test_profile"]
    distribution = test_profile["champion_predicted_label_distribution"]
    assert distribution["status"] == "observed"
    assert distribution["total_test_nodes"] == 2
    assert distribution["predicted_class_counts"]["1"] == 1
    assert distribution["predicted_class_counts"]["2"] == 1
    assert sum(distribution["predicted_class_counts"].values()) == 2
    assert abs(sum(distribution["predicted_class_ratios"].values()) - 1.0) < 1e-12
    assert distribution["interpretation"] == "predicted_labels_not_test_truth"
    assert "true" not in json.dumps(distribution).lower()

    problem_map = json.loads((out_dir / "a1_problem_map.json").read_text())
    shift = problem_map["rankings"]["train_test_shift"]
    assert isinstance(shift, dict)
    assert shift["status"] == "observed"
    assert shift["observed"]
    assert shift["unavailable"][0]["status"] == "unavailable"
    assert shift["unavailable"][0]["reason"] == "missing_anchor_oof"

    shift_rows = list(csv.DictReader((out_dir / "a1_shift_profile.csv").open()))
    statuses = {row["status"] for row in shift_rows}
    assert "observed" in statuses
    assert "unavailable" in statuses


def test_directed_exact_hop_breakdown_is_explicitly_not_generated(
    tmp_path: Path,
) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "out"
    _write_npz(npz)

    result = run_profile(ProfileRequest(npz_path=npz, out_dir=out_dir))

    assert result.status == "completed"
    profile = json.loads((out_dir / "a1_data_profile.json").read_text())
    exact_hop = profile["exact_hop_policy"]
    assert exact_hop["primary_exact_hop_view"] == "either_direction"
    assert exact_hop["directed_exact_hop_breakdown"]["status"] == "not_generated"
    assert (
        exact_hop["directed_exact_hop_breakdown"]["reason"]
        == "not included in M2 v1; reserved for directed H2 signal audit"
    )


def test_exact_hops_fold_visible_train_and_oof_alignment(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    fold = tmp_path / "fold.csv"
    oof = tmp_path / "oof.npz"
    out_dir = tmp_path / "out"
    _write_npz(npz)
    _write_fold(fold)
    _write_oof(oof)

    result = run_profile(
        ProfileRequest(
            npz_path=npz,
            fold_file=fold,
            anchor_oof_npz=oof,
            out_dir=out_dir,
            require_fold=True,
            require_oof=True,
        )
    )

    assert result.status == "completed"
    rows = {
        int(row["global_idx"]): row
        for row in csv.DictReader((out_dir / "a1_node_buckets.csv").open())
    }
    # Node 2 is in fold 1. Its 1-hop train node 1 is also in fold 0,
    # so it is outer-train visible; fold-valid labels from fold 1 are not used.
    assert rows[2]["fold"] == "1"
    assert rows[2]["exact1_visible_train_count"] == "1"
    assert rows[2]["primary_supervision_bucket"] == "one_hop_available"
    # Node 8 can reach train node 6 in exactly two either-directed hops via
    # non-train intermediate node 7.
    assert rows[8]["exact1_visible_train_count"] == "0"
    assert rows[8]["exact2_visible_train_count"] == "1"
    assert rows[8]["primary_supervision_bucket"] == "exact2_only"

    profile = json.loads((out_dir / "a1_data_profile.json").read_text())
    assert profile["oof_profile"]["status"] == "available"
    assert profile["oof_profile"]["alignment"] == "global_train_idx"
    assert profile["prediction_sink_source"]["status"] == "available"


def test_test_metrics_do_not_include_truth_dependent_fields(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "out"
    _write_npz(npz)

    result = run_profile(ProfileRequest(npz_path=npz, out_dir=out_dir))

    assert result.status == "completed"
    raw = (out_dir / "a1_data_profile.json").read_text()
    forbidden = ["test_accuracy", "rescue", "damage", "oracle", "correct", "wrong"]
    lowered = raw.lower()
    for token in forbidden:
        assert token not in lowered


def test_missing_required_inputs_and_strict_edge_failure(tmp_path: Path) -> None:
    missing = run_profile(
        ProfileRequest(npz_path=tmp_path / "missing.npz", out_dir=tmp_path / "out")
    )
    assert missing.status == "waiting_for_input"

    npz = tmp_path / "a1.npz"
    bad_edges = tmp_path / "bad_edges.csv"
    _write_npz(npz)
    _write_mismatched_edges(bad_edges)
    failed = run_profile(
        ProfileRequest(
            npz_path=npz,
            edges_csv=bad_edges,
            out_dir=tmp_path / "bad_out",
            strict=True,
        )
    )
    assert failed.status == "failed"


def test_require_fold_and_require_oof_waiting(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    _write_npz(npz)

    need_fold = run_profile(
        ProfileRequest(
            npz_path=npz,
            out_dir=tmp_path / "fold_out",
            require_fold=True,
        )
    )
    assert need_fold.status == "waiting_for_input"

    need_oof = run_profile(
        ProfileRequest(
            npz_path=npz,
            out_dir=tmp_path / "oof_out",
            require_oof=True,
        )
    )
    assert need_oof.status == "waiting_for_input"


def test_idempotent_core_hash_excludes_manifest_time(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "out"
    _write_npz(npz)

    first = run_profile(ProfileRequest(npz_path=npz, out_dir=out_dir))
    first_hash = core_result_hash(out_dir)
    second = run_profile(ProfileRequest(npz_path=npz, out_dir=out_dir))
    second_hash = core_result_hash(out_dir)

    assert first.status == "completed"
    assert second.status == "completed"
    assert first_hash == second_hash
    manifest = json.loads((out_dir / "a1_profile_manifest.json").read_text())
    assert "started_at" in manifest
    assert "finished_at" in manifest
    assert manifest["core_result_hash"] == first_hash


def test_chinese_space_path_smoke(tmp_path: Path) -> None:
    root = tmp_path / "中文 路径"
    root.mkdir()
    npz = root / "a1 数据.npz"
    out_dir = root / "输出 目录"
    _write_npz(npz)

    result = run_profile(ProfileRequest(npz_path=npz, out_dir=out_dir))

    assert result.status == "completed"
    assert (out_dir / "a1_data_profile.json").exists()
