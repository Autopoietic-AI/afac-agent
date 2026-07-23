# -*- coding: utf-8 -*-
"""Synthetic integration tests for the B2 sequence-recommendation module.

These tests use tiny in-memory datasets and do not execute full real training.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from afac_agent.b2.closed_loop import B2ClosedLoopRunner
from afac_agent.b2.data_intelligence import run_data_intelligence
from afac_agent.b2.evaluator import B2Evaluator
from afac_agent.b2.fold import AFAC_B2_FOLD_V1, build_folds, build_panels, validation_mask
from afac_agent.b2.models import (
    CandidateRankerModel,
    HistoryRecallModel,
    ItemItemCooccurrenceModel,
    LastItemTransitionModel,
    PopularityModel,
    ScoreBlendModel,
)
from afac_agent.b2.task_adapter import B2TaskAdapter


N_ITEMS = 20
TOP_K = 10


def _make_b2_dir(tmp_path: Path) -> Path:
    root = tmp_path / "b2_data"
    root.mkdir()

    items = [f"i{i:03d}" for i in range(1, N_ITEMS + 1)]
    users_train = [f"u{i:03d}" for i in range(1, 21)]
    users_test = [f"u{i:03d}" for i in range(21, 26)]

    # Train: deterministic sequences where target is usually in history
    with (root / "train.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
        for i, uid in enumerate(users_train):
            seq_len = [3, 5, 3, 5, 5, 1, 4, 5, 3, 5, 5, 3, 4, 5, 5, 3, 5, 5, 4, 5][i]
            seq = [items[(i + j) % N_ITEMS] for j in range(seq_len)]
            target = seq[-1] if i % 2 == 0 else items[(i + 7) % N_ITEMS]
            raw = ",".join(seq * 2)
            dedup = ",".join(seq)
            counts = ",".join(f"{iid}:{2}" for iid in seq)
            writer.writerow([uid, target, raw, dedup, counts])

    # Test: some empty sequences to exercise cold-start
    with (root / "test.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
        for i, uid in enumerate(users_test):
            if i == 0:
                writer.writerow([uid, "", "", ""])
            else:
                seq = [items[(i + j) % N_ITEMS] for j in range(3)]
                writer.writerow([uid, ",".join(seq), ",".join(seq), ",".join(f"{iid}:1" for iid in seq)])

    with (root / "user.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "u_cat_01", "u_cat_02"])
        for uid in users_train + users_test:
            writer.writerow([uid, "1", "2"])

    with (root / "item.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iid", "i_cat_01", "i_cat_02", "i_bucket_01"])
        for iid in items:
            writer.writerow([iid, "1", "2", "3"])

    with (root / "sample_submission.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "prediction"])
        for uid in users_test:
            writer.writerow([uid, ",".join(items[:TOP_K])])

    with (root / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "dataset_name": "b2_synthetic",
            "task_family": "sequence_recommendation",
            "submission_format": {"prediction_format": "comma-separated top-K iid list"},
        }, fh)

    return root


@pytest.fixture
def b2_dir(tmp_path: Path) -> Path:
    return _make_b2_dir(tmp_path)


@pytest.fixture
def b2_dataset(b2_dir: Path) -> Any:
    return B2TaskAdapter(b2_dir).load()


def test_adapter_parameterization_and_validation(b2_dir: Path) -> None:
    adapter = B2TaskAdapter(b2_dir, top_k=TOP_K)
    dataset = adapter.load()
    assert dataset.validation["status"] == "passed"
    assert dataset.top_k == TOP_K
    assert dataset.validation["test_truth_hidden"] is True
    assert dataset.validation["train_test_uid_overlap"] == 0


def test_uid_iid_legality_and_sequence_parsing(b2_dataset: Any) -> None:
    all_items = b2_dataset.all_item_ids()
    assert all(iid.startswith("i") for iid in all_items)
    for uid, seq in b2_dataset.train_seq.items():
        for iid in seq:
            assert iid in all_items
    assert len(b2_dataset.test_seq) == 5
    assert b2_dataset.test_seq[list(b2_dataset.test_seq.keys())[0]] == []


def test_illegal_iid_detected(tmp_path: Path) -> None:
    root = _make_b2_dir(tmp_path)
    # Inject illegal item into one train sequence
    train = list(csv.DictReader((root / "train.csv").read_text(encoding="utf-8").splitlines()))
    train[0]["item_seq_dedup"] = "i999999," + train[0]["item_seq_dedup"]
    with (root / "train.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=train[0].keys())
        writer.writeheader()
        writer.writerows(train)
    dataset = B2TaskAdapter(root).load()
    assert dataset.validation["status"] == "failed"
    assert any("illegal iids" in e for e in dataset.validation["errors"])


def test_fold_no_leakage(b2_dataset: Any) -> None:
    folds = build_folds(b2_dataset)
    assert folds.uids.size == len(b2_dataset.train_df)
    assert set(folds.folds.tolist()) == set(range(5))
    # Each user appears exactly once
    assert len(folds.uids) == len(np.unique(folds.uids))
    assert folds.fold_hash


def test_panels_have_required_masks(b2_dataset: Any) -> None:
    folds = build_folds(b2_dataset)
    panels = build_panels(b2_dataset, folds)
    required = [
        "B2_STANDARD_PANEL",
        "B2_SHORT_HISTORY_PANEL",
        "B2_EXACT_LEN3_PANEL",
        "B2_LEN4_PLUS_PANEL",
        "B2_HISTORY_TARGET_PANEL",
        "B2_NOVEL_TARGET_PANEL",
        "B2_LONG_TAIL_PANEL",
        "B2_TEST_LIKE_PANEL",
        "B2_TOP10_BOUNDARY_PANEL",
    ]
    for pid in required:
        assert pid in panels
        train_mask, val_mask = validation_mask(pid, folds, held_fold=0)
        assert train_mask.sum() > 0
        assert val_mask.sum() > 0
        assert not (train_mask & val_mask).any()


def test_evaluator_metrics_and_failure_split(b2_dataset: Any) -> None:
    folds = build_folds(b2_dataset)
    panels = build_panels(b2_dataset, folds)
    evaluator = B2Evaluator(b2_dataset, folds, panels)
    item_list = sorted(b2_dataset.all_item_ids())
    # Perfect oracle candidate set (target at rank 1)
    targets = evaluator.targets
    oracle = np.full((len(folds.uids), b2_dataset.top_k), "", dtype=object)
    for i, target in enumerate(targets):
        oracle[i, 0] = target
        rest = [iid for iid in item_list if iid != target][: b2_dataset.top_k - 1]
        oracle[i, 1 : 1 + len(rest)] = rest
    metrics = evaluator.evaluate(oracle, asset_id="ORACLE", panel_ids=["B2_STANDARD_PANEL"])
    assert metrics["overall"]["hit_rate@10"] == 1.0
    assert metrics["overall"]["ndcg@10"] == 1.0
    assert metrics["overall"]["retrieval_failure_rate"] == 0.0
    assert metrics["overall"]["ranking_failure_rate"] == 0.0

    # Retrieval failure only candidate (items that never appear as targets)
    target_set = set(targets.tolist())
    non_target = [iid for iid in item_list if iid not in target_set][: b2_dataset.top_k]
    wrong = np.tile(non_target, (len(folds.uids), 1))
    wrong_metrics = evaluator.evaluate(wrong, asset_id="WRONG", panel_ids=["B2_STANDARD_PANEL"])
    assert wrong_metrics["overall"]["retrieval_failure_rate"] == 1.0
    assert wrong_metrics["overall"]["ranking_failure_rate"] == 0.0


def test_evaluator_compare_rescue_damage(b2_dataset: Any) -> None:
    folds = build_folds(b2_dataset)
    panels = build_panels(b2_dataset, folds)
    evaluator = B2Evaluator(b2_dataset, folds, panels)
    item_list = sorted(b2_dataset.all_item_ids())
    targets = evaluator.targets

    target_set = set(targets.tolist())
    non_target = [iid for iid in item_list if iid not in target_set][: b2_dataset.top_k]
    base = np.tile(non_target, (len(folds.uids), 1))
    cand = np.full((len(folds.uids), b2_dataset.top_k), "", dtype=object)
    for i, target in enumerate(targets):
        cand[i, 0] = target
        rest = [iid for iid in item_list if iid != target][: b2_dataset.top_k - 1]
        cand[i, 1 : 1 + len(rest)] = rest

    delta = evaluator.compare(base, cand, panel_id="B2_STANDARD_PANEL")
    assert delta["rescue"] > 0
    assert delta["damage"] == 0
    assert delta["net"] > 0


def test_topk_legality_and_oof_test_isolation(b2_dataset: Any) -> None:
    folds = build_folds(b2_dataset)
    item_list = sorted(b2_dataset.all_item_ids())
    model = PopularityModel().fit(folds.uids.tolist(), item_list, b2_dataset.train_seq)
    topk = model.topk(folds.uids.tolist(), k=b2_dataset.top_k)
    assert topk.shape == (len(folds.uids), b2_dataset.top_k)
    for row in topk:
        assert len(set(row)) == len(row)
        assert all(iid in item_list for iid in row)

    # OOF: val predictions never trained on val users
    for held in range(5):
        fit_uids = folds.uids[folds.folds != held].tolist()
        val_uids = folds.uids[folds.folds == held].tolist()
        m = PopularityModel().fit(fit_uids, item_list, b2_dataset.train_seq)
        pred = m.topk(val_uids, k=b2_dataset.top_k)
        assert pred.shape[0] == len(val_uids)


def test_models_runnable(b2_dataset: Any) -> None:
    folds = build_folds(b2_dataset)
    item_list = sorted(b2_dataset.all_item_ids())
    train_targets = {uid: tid for uid, tid in zip(b2_dataset.train_df["uid"], b2_dataset.train_df["target_iid"])}
    uids = folds.uids.tolist()
    for ModelClass, kwargs in [
        (PopularityModel, {}),
        (HistoryRecallModel, {}),
        (ItemItemCooccurrenceModel, {"window": 3}),
        (LastItemTransitionModel, {}),
        (ScoreBlendModel, {"weights": {"history": 1.0, "popularity": 0.5}}),
        (CandidateRankerModel, {"n_candidates": 20, "n_negatives": 5, "base": "logistic"}),
    ]:
        model = ModelClass(model_id=f"test_{ModelClass.__name__}", **kwargs)
        model.fit(uids, item_list, b2_dataset.train_seq, train_targets=train_targets, user_df=b2_dataset.user_df, item_df=b2_dataset.item_df)
        topk = model.topk(uids, k=b2_dataset.top_k)
        assert topk.shape == (len(uids), b2_dataset.top_k)


def test_data_intelligence_runs(b2_dir: Path, tmp_path: Path) -> None:
    result = run_data_intelligence(
        data_root=b2_dir,
        out_root=tmp_path / "b2_runs",
        project_root=tmp_path,
        force_rebuild=True,
    )
    assert result["status"] == "verified"
    assert result["run_id"]


def test_closed_loop_runs_and_produces_candidate_b2(b2_dir: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "b2_runs"
    runner = B2ClosedLoopRunner(
        project_root=tmp_path,
        data_root=b2_dir,
        out_root=out_root,
        max_wall_clock_seconds=600,
        max_rounds=1,
    )
    result = runner.run(force_rebuild=True)
    assert result["status"] == "completed"
    assert result["run_id"]
    run_dir = out_root / result["run_id"]
    sub_path = run_dir / "TO_UPLOAD" / "candidate_B2.csv"
    assert sub_path.is_file()
    rows = list(csv.DictReader(sub_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 5
    for row in rows:
        preds = row["prediction"].split(",")
        assert len(preds) == TOP_K
        assert len(set(preds)) == len(preds)
        assert all(p.startswith("i") for p in preds)
    # No A1/A2/B1 assets touched
    assert result["artifacts"]["candidate_B2_csv"]


def test_namespace_isolation_from_a1_a2_b1() -> None:
    from afac_agent.b1.fold import AFAC_B1_FOLD_V1
    from afac_agent.b2 import B2_EVAL_ANCHOR_ID, B2_ONLINE_ANCHOR_ID
    assert AFAC_B1_FOLD_V1 != AFAC_B2_FOLD_V1
    assert B2_EVAL_ANCHOR_ID != "B1_EVAL_ANCHOR_V1"
    assert B2_ONLINE_ANCHOR_ID == "B2_ONLINE_ANCHOR"
