# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from afac_agent.a2.anchors import (
    A2_EVAL_ANCHOR_ID,
    A2_ONLINE_ANCHOR_ID,
    materialize_online_anchor,
)
from afac_agent.a2.assets import SCOPE_OFFLINE_OOF, SCOPE_TEST, load_score_asset
from afac_agent.a2.evaluator import (
    A2Evaluator,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    target_ranks,
    topk_lists,
)
from afac_agent.a2.fold import AFAC_A2_FOLD_V1, validate_fold_csv
from afac_agent.a2.fusion import (
    cross_fit_scores,
    op_rank_fusion,
    op_retriever_ranker_composition,
    op_score_blend,
    op_slot_protected_rerank,
    op_topk_protected_rerank,
    union_candidate_recall,
)
from afac_agent.a2.integration import run_a2_integration
from afac_agent.a2.task_adapter import A2TaskAdapter, sequence_length_bucket

N_TRAIN = 25
N_TEST = 6
N_ITEMS = 20
N_FOLDS = 5


def _iid(i: int) -> str:
    return f"i{i:06d}"


def _uid(i: int) -> str:
    return f"u{i:06d}"


def _write_dataset(root: Path) -> Path:
    data = root / "a2_data"
    data.mkdir(parents=True)
    items = [_iid(i) for i in range(N_ITEMS)]
    with (data / "item.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iid", "i_cat_01", "i_cat_02", "i_cat_03", "i_bucket_01"])
        for iid in items:
            writer.writerow([iid, 0, 0, 0, 1])
    with (data / "user.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid"] + [f"u_cat_{i:02d}" for i in range(1, 9)])
        for u in range(N_TRAIN + N_TEST):
            writer.writerow([_uid(u)] + [1] * 8)
    with (data / "train.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
        for u in range(N_TRAIN):
            # history length cycles through 0,1,2,3,4+ buckets
            bucket = u % 5
            hist_len = [0, 1, 2, 3, 5][bucket]
            hist = [_iid((u + j + 1) % N_ITEMS) for j in range(hist_len)]
            target = _iid(u % N_ITEMS)
            # even users: target inside history (history type); odd: novel
            if u % 2 == 0 and hist:
                hist[0] = target
            raw = hist + hist[:1]
            dedup = list(dict.fromkeys(hist))
            counts = ",".join(f"{iid}:{raw.count(iid)}" for iid in dict.fromkeys(raw))
            writer.writerow([_uid(u), target, ",".join(raw), ",".join(dedup), counts])
    with (data / "test.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
        for t in range(N_TEST):
            uid = _uid(N_TRAIN + t)
            hist = [_iid((t + 2) % N_ITEMS)]
            writer.writerow([uid, ",".join(hist), ",".join(hist), f"{hist[0]}:1"])
    with (data / "sample_submission.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "prediction"])
        for t in range(N_TEST):
            writer.writerow([_uid(N_TRAIN + t), ",".join(items[:10])])
    (data / "metadata.json").write_text(json.dumps({"dataset_name": "synthetic_a2"}), encoding="utf-8")
    return data


def _scores(seed: int, targets: list[str], items: list[str], rank_of_target: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1.0, size=(len(targets), len(items)))
    for row, target in enumerate(targets):
        col = items.index(target)
        order = np.argsort(-scores[row], kind="stable")
        position = int(np.nonzero(order == col)[0][0])
        if position != rank_of_target - 1:
            swap_col = order[rank_of_target - 1]
            scores[row, col], scores[row, swap_col] = scores[row, swap_col], scores[row, col]
    return scores.astype(np.float32)


def _write_runs(root: Path, train_uids: list[str], test_uids: list[str], targets: list[str], items: list[str]) -> Path:
    runs = root / "a2_runs" / "runs"
    (runs / "stageA").mkdir(parents=True)
    with (runs / "stageA" / "train_folds.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "oof_fold"])
        for pos, uid in enumerate(train_uids):
            writer.writerow([uid, pos % N_FOLDS])
    for name, seed, rank in (("C2_v42_merged", 7, 1), ("C2_v48_merged", 11, 2)):
        merged = runs / name
        merged.mkdir(parents=True)
        np.savez(
            merged / "oof_scores_full.npz",
            uids=np.array(train_uids, dtype=object),
            scores=_scores(seed, targets, items, rank),
            item_ids=np.array(items),
            targets=np.array(targets),
        )
        rng = np.random.default_rng(seed + 1)
        np.savez(
            merged / "test_scores_cvmean.npz",
            uids=np.array(test_uids, dtype=object),
            scores=rng.normal(0.0, 1.0, size=(len(test_uids), len(items))).astype(np.float32),
            item_ids=np.array(items),
        )
    return runs.parent


@pytest.fixture()
def a2_world(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    data_dir = _write_dataset(project)
    dataset = A2TaskAdapter(data_dir).load()
    targets = [dataset.train_targets[uid] for uid in dataset.train_uids]
    runs_root = _write_runs(project, dataset.train_uids, dataset.test_uids, targets, dataset.item_ids)
    return {"project": project, "data_dir": data_dir, "runs_root": runs_root, "dataset": dataset}


# ---------------------------------------------------------------- adapter

def test_adapter_loads_and_validates(a2_world):
    dataset = a2_world["dataset"]
    assert dataset.validation["status"] == "passed"
    assert dataset.validation["train_users"] == N_TRAIN
    assert dataset.validation["test_users"] == N_TEST
    assert len(dataset.item_ids) == N_ITEMS
    assert dataset.test_uids == [_uid(N_TRAIN + t) for t in range(N_TEST)]  # order preserved
    assert dataset.validation["test_truth_isolated"] is True


def test_adapter_rejects_invalid_uid(tmp_path: Path):
    data = _write_dataset(tmp_path)
    with (data / "train.csv").open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(["bad_uid", "i000001", "", "", ""])
    dataset = A2TaskAdapter(data).load()
    assert dataset.validation["status"] == "failed"
    assert any("invalid uid" in e for e in dataset.validation["errors"])


def test_adapter_rejects_target_column_in_test(tmp_path: Path):
    data = _write_dataset(tmp_path)
    test_path = data / "test.csv"
    lines = test_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0] + ",target_iid"
    lines[1:] = [line + ",i000001" for line in lines[1:]]
    test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dataset = A2TaskAdapter(data).load()
    assert dataset.validation["status"] == "failed"
    assert any("test truth isolation" in e for e in dataset.validation["errors"])


def test_top10_validation(a2_world):
    adapter = A2TaskAdapter(a2_world["data_dir"])
    items = a2_world["dataset"].item_ids
    assert adapter.validate_top10("u000001", items[:10], set(items)) == []
    assert "top10 contains duplicate items" in adapter.validate_top10("u000001", items[:9] + items[:1], set(items))
    assert any("illegal item" in e for e in adapter.validate_top10("u000001", ["xx"] * 10, set(items)))
    assert adapter.validate_top10("u000001", items[:9], set(items)) != []


def test_sequence_length_bucket():
    assert sequence_length_bucket(0) == "len0"
    assert sequence_length_bucket(1) == "len1"
    assert sequence_length_bucket(2) == "len2"
    assert sequence_length_bucket(3) == "exact_len3"
    assert sequence_length_bucket(4) == "len4_plus"
    assert sequence_length_bucket(50) == "len4_plus"


# ---------------------------------------------------------------- metrics

def test_ndcg_mrr_hit_at_10():
    ranks = np.array([1, 2, 10, 11, 100])
    ndcg = ndcg_at_k(ranks, 10)
    assert ndcg[0] == pytest.approx(1.0)
    assert ndcg[1] == pytest.approx(1.0 / np.log2(3))
    assert ndcg[2] == pytest.approx(1.0 / np.log2(11))
    assert ndcg[3] == 0.0 and ndcg[4] == 0.0
    assert hit_at_k(ranks, 10).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert mrr_at_k(ranks, 10)[2] == pytest.approx(0.1)
    assert mrr_at_k(ranks, 10)[3] == 0.0


def test_target_ranks_tie_break():
    scores = np.array([[1.0, 2.0, 2.0, 0.5]])
    # target at col 2: one greater (col 1), one equal earlier (col 1? no col<2 equal at col 1 has 2.0 > 2.0? no equal)
    ranks = target_ranks(scores, np.array([2]))
    assert ranks[0] == 2  # col1 strictly greater; col1 is greater not equal
    ranks = target_ranks(scores, np.array([1]))
    assert ranks[0] == 1  # tie with col2 but col1 comes first


def test_retrieval_vs_ranking_failure(a2_world):
    dataset = a2_world["dataset"]
    evaluator = A2Evaluator(dataset)
    items = dataset.item_ids
    uids = dataset.train_uids[:4]
    targets = [dataset.train_targets[u] for u in uids]
    scores = np.zeros((4, len(items)), dtype=np.float32)
    # user 0: target ranked 1 (success); user 1: target ranked 15 (ranking failure);
    # user 2: target removed from candidates later (retrieval failure); user 3: target ranked 5
    for row, target in enumerate(targets):
        col = items.index(target)
        scores[row, col] = 100.0
    t1 = items.index(targets[1])
    descending = [j for j in range(len(items)) if j != t1]
    for pos, j in enumerate(descending):
        scores[1, j] = float(len(items) - pos)  # 20..2 over non-target items
    scores[1, t1] = 6.5  # exactly 14 items strictly above -> target rank 15
    scores[3, :] = 0.0
    others = [j for j in range(len(items)) if j != items.index(targets[3])]
    for pos, j in enumerate(others[:4]):
        scores[3, j] = 10.0 - pos
    scores[3, items.index(targets[3])] = 5.0
    ev = evaluator.evaluate_scores(asset_id="synthetic", uids=uids, item_ids=items, scores=scores, targets=targets)
    assert ev.failure_counts["target_in_top10"] == 3
    assert ev.failure_counts["ranking_failure"] == 1
    # retrieval failure: drop target column by evaluating with a catalog missing the target
    reduced_items = [i for i in items if i != targets[2]]
    reduced_scores = np.delete(scores, items.index(targets[2]), axis=1)
    ev2 = evaluator.evaluate_scores(asset_id="synthetic", uids=uids, item_ids=reduced_items, scores=reduced_scores, targets=targets)
    assert ev2.failure_counts["retrieval_failure"] == 1
    assert ev2.failure_counts["target_missing_from_candidates"] == 1
    assert ev2.candidate_recall == pytest.approx(0.75)


def test_len_and_history_novel_buckets(a2_world):
    dataset = a2_world["dataset"]
    evaluator = A2Evaluator(dataset)
    items = dataset.item_ids
    uids = dataset.train_uids
    targets = [dataset.train_targets[u] for u in uids]
    scores = np.zeros((len(uids), len(items)), dtype=np.float32)
    for row, target in enumerate(targets):
        scores[row, items.index(target)] = 1.0
    ev = evaluator.evaluate_scores(asset_id="synthetic", uids=uids, item_ids=items, scores=scores, targets=targets)
    len_buckets = {row["bucket"] for row in ev.len_bucket_metrics}
    assert len_buckets == {"len0", "len1", "len2", "exact_len3", "len4_plus"}
    type_buckets = {row["bucket"]: row for row in ev.type_bucket_metrics}
    assert set(type_buckets) == {"history", "novel"}
    expected_history = sum(1 for u in uids if dataset.target_type(u) == "history")
    assert type_buckets["history"]["user_count"] == expected_history
    assert dataset.len_bucket("u000000") == "len0"
    assert dataset.len_bucket("u000003") == "exact_len3"


# ---------------------------------------------------------------- fold

def test_fold_validation_ok(a2_world):
    dataset = a2_world["dataset"]
    manifest = validate_fold_csv(a2_world["runs_root"] / "runs" / "stageA" / "train_folds.csv", dataset)
    assert manifest["status"] == "validated"
    assert manifest["fold_protocol"] == AFAC_A2_FOLD_V1
    assert manifest["checks"]["each_train_user_exactly_once"] is True
    assert manifest["checks"]["contains_no_test_users"] is True
    assert manifest["checks"]["fold_values_legal"] is True
    assert manifest["sha256"]
    assert manifest["user_order"]["explicit"] is True
    assert len(manifest["per_fold_audit"]) == N_FOLDS


def test_fold_validation_rejects_test_leak_and_duplicates(a2_world, tmp_path: Path):
    dataset = a2_world["dataset"]
    bad = tmp_path / "bad_folds.csv"
    with bad.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "oof_fold"])
        writer.writerow([dataset.train_uids[0], 0])
        writer.writerow([dataset.train_uids[0], 1])  # duplicate
        writer.writerow([dataset.test_uids[0], 0])   # test leak
    manifest = validate_fold_csv(bad, dataset)
    assert manifest["status"] == "failed"
    assert any("duplicate uid" in e for e in manifest["errors"])
    assert any("test users present" in e for e in manifest["errors"])
    assert manifest["checks"]["each_train_user_exactly_once"] is False


# ---------------------------------------------------------------- OOF/Test isolation

def test_oof_test_isolation(a2_world):
    dataset = a2_world["dataset"]
    runs = a2_world["runs_root"] / "runs"
    oof = load_score_asset(asset_id="v42", path=runs / "C2_v42_merged" / "oof_scores_full.npz", dataset=dataset, expected_scope=SCOPE_OFFLINE_OOF)
    assert oof.verification_status == "verified_oof"
    test_asset = load_score_asset(asset_id="v42t", path=runs / "C2_v42_merged" / "test_scores_cvmean.npz", dataset=dataset, expected_scope=SCOPE_TEST)
    assert test_asset.verification_status == "verified_test_scope"
    # using the test asset as OOF must fail
    mixed = load_score_asset(asset_id="bad", path=runs / "C2_v42_merged" / "test_scores_cvmean.npz", dataset=dataset, expected_scope=SCOPE_OFFLINE_OOF)
    assert mixed.verification_status == "invalid"
    assert any("oof uids do not match" in e for e in mixed.verification_errors)


# ---------------------------------------------------------------- fusion operators

def test_score_blend_and_rank_fusion(a2_world):
    dataset = a2_world["dataset"]
    evaluator = A2Evaluator(dataset)
    items = dataset.item_ids
    uids = dataset.train_uids
    targets = [dataset.train_targets[u] for u in uids]
    a = _scores(7, targets, items, 2).astype(np.float64)
    b = _scores(11, targets, items, 1).astype(np.float64)
    blended = op_score_blend(a, b, alpha=1.0)  # pure b -> rank 1
    cols = np.array([items.index(t) for t in targets])
    assert (target_ranks(blended, cols) == 1).all()
    rrf = op_rank_fusion(a, b)
    ev = evaluator.evaluate_scores(asset_id="rrf", uids=uids, item_ids=items, scores=rrf, targets=targets)
    assert ev.hit10 == 1.0


def test_candidate_union_diagnostic(a2_world):
    dataset = a2_world["dataset"]
    items = dataset.item_ids
    targets = [dataset.train_targets[u] for u in dataset.train_uids]
    cols = np.array([items.index(t) for t in targets])
    a = _scores(7, targets, items, 1).astype(np.float64)   # always inside top-k
    b = _scores(11, targets, items, 20).astype(np.float64)  # outside top-10
    record = union_candidate_recall(a, b, cols, k=10)
    assert record["a_only"] == len(targets)
    assert record["b_only"] == 0
    assert record["oracle_union_recall"] == 1.0
    assert record["diagnostic_only"] is True and record["executable"] is False


def test_topk_protected_rerank_preserves_set_and_tail(a2_world):
    dataset = a2_world["dataset"]
    items = dataset.item_ids
    targets = [dataset.train_targets[u] for u in dataset.train_uids]
    a = _scores(7, targets, items, 1).astype(np.float64)
    b = _scores(11, targets, items, 5).astype(np.float64)
    out = op_topk_protected_rerank(a, b, set_size=10, rerank_positions=7)
    base_top = topk_lists(a, 10)
    new_top = topk_lists(out, 10)
    for row in range(len(targets)):
        assert set(new_top[row].tolist()) == set(base_top[row].tolist())  # membership fixed
        assert new_top[row, 7:].tolist() == base_top[row, 7:].tolist()    # tail order preserved


def test_slot_protected_rerank_protects_history(a2_world):
    dataset = a2_world["dataset"]
    from afac_agent.a2.fusion import build_history_mask

    items = dataset.item_ids
    uids = dataset.train_uids
    targets = [dataset.train_targets[u] for u in uids]
    a = _scores(7, targets, items, 1).astype(np.float64)
    b = _scores(11, targets, items, 5).astype(np.float64)
    mask = build_history_mask(dataset, uids)
    out = op_slot_protected_rerank(a, b, mask, set_size=10)
    base_top = topk_lists(a, 10)
    new_top = topk_lists(out, 10)
    for row in range(len(uids)):
        assert set(new_top[row].tolist()) == set(base_top[row].tolist())
        hist_cols = [c for c in base_top[row].tolist() if mask[row, c]]
        assert [c for c in new_top[row].tolist() if mask[row, c]] == hist_cols  # history relative order preserved


def test_retriever_ranker_composition(a2_world):
    dataset = a2_world["dataset"]
    items = dataset.item_ids
    targets = [dataset.train_targets[u] for u in dataset.train_uids]
    a = _scores(7, targets, items, 15).astype(np.float64)  # retriever: target at 15 (in pool 20)
    b = _scores(11, targets, items, 1).astype(np.float64)  # ranker loves target
    out = op_retriever_ranker_composition(a, b, pool_size=20)
    cols = np.array([items.index(t) for t in targets])
    assert (target_ranks(out, cols) == 1).all()
    # outside the retriever pool everything is -inf
    pool = topk_lists(a, 20)
    outside = np.ones((len(targets), len(items)), dtype=bool)
    rows = np.arange(len(targets))[:, None]
    outside[rows, pool] = False
    assert np.isneginf(out[outside]).all()


def test_cross_fit_selects_alpha_on_fit_folds(a2_world):
    dataset = a2_world["dataset"]
    items = dataset.item_ids
    uids = dataset.train_uids
    targets = [dataset.train_targets[u] for u in uids]
    cols = np.array([items.index(t) for t in targets])
    folds = np.array([i % N_FOLDS for i in range(len(uids))])
    a = _scores(7, targets, items, 1).astype(np.float64)   # perfect and strongly dominant
    for row, target in enumerate(targets):
        a[row, items.index(target)] += 100.0
    b = _scores(11, targets, items, 20).astype(np.float64)  # bad
    out, assignment = cross_fit_scores(
        operator="score_blend",
        scores_a=a,
        scores_b=b,
        folds=folds,
        route_mask=np.zeros(len(uids), dtype=bool),
        history_mask=None,
        alpha_grid=(0.25, 0.5, 0.75),
        selection_targets=cols,
    )
    assert assignment["mode"] == "strict_outer_fold_cross_fit"
    assert len(assignment["fold_assignments"]) == N_FOLDS
    for record in assignment["fold_assignments"]:
        assert record["selected_params"]["alpha"] == 0.25  # closest to the good asset
        assert record["held_out_targets_used_for_selection"] is False
    assert (target_ranks(out, cols) == 1).all()


# ---------------------------------------------------------------- anchors

def test_anchor_identities(a2_world, tmp_path: Path):
    online = materialize_online_anchor(out_dir=tmp_path / "online", project_root=a2_world["project"])
    assert online["anchor_id"] == A2_ONLINE_ANCHOR_ID
    assert online["online_score"] == 0.5093
    assert online["usable_as_offline_oof"] is False
    assert online["offline_or_test"] == "online_test_score"


# ---------------------------------------------------------------- full dry-run

def test_full_dry_run(a2_world):
    result = run_a2_integration(
        project_root=a2_world["project"],
        a2_data_dir=a2_world["data_dir"],
        a2_runs_root=a2_world["runs_root"],
        out_root=a2_world["project"] / "artifacts" / "a2_integration",
    )
    assert result["status"] == "ready_for_experiment_design"
    assert result["scientific_rounds_used"] == 0
    out_dir = a2_world["project"] / "artifacts" / "a2_integration" / result["run_id"]
    expected = [
        "a2_profile.json", "a2_bucket_registry.json", "a2_fold_manifest.json",
        "online_anchor_manifest.json", "evaluation_anchor_manifest.json",
        "portfolio_snapshot.json", "asset_verification.json",
        "complementarity_report.json", "fusion_plan.json", "m6b_proposal.json",
        "m6c_review.json", "m5_admission.json", "execution_preview.json",
        "evaluation_preview.json", "trajectory_preview.json",
        "A2_INTEGRATION_REPORT.md", "integration_manifest.json", "REPORT_PACKAGE",
    ]
    for name in expected:
        assert (out_dir / name).exists(), name
    manifest = json.loads((out_dir / "integration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["trains_model"] is False
    assert manifest["generates_test_prediction"] is False
    assert manifest["creates_submission"] is False
    assert manifest["uses_test_truth"] is False
    assert manifest["mutates_a1_champion_or_anchor"] is False
    assert manifest["scientific_rounds_used"] == 0
    # no submission/prediction artifacts anywhere in the run dir
    assert not list(out_dir.rglob("*.zip"))
    assert not list(out_dir.rglob("prediction*"))
    anchor = json.loads((out_dir / "evaluation_anchor_manifest.json").read_text(encoding="utf-8"))
    assert anchor["anchor_id"] == A2_EVAL_ANCHOR_ID
    assert anchor["status"] == "materialized"
    assert anchor["deployment_equivalent"] is False
    assert anchor["offline_or_test"] == "offline_oof"
    assert anchor["test_scores_used_as_oof"] is False
    verification = json.loads((out_dir / "asset_verification.json").read_text(encoding="utf-8"))
    assert verification["oof_test_isolation"] == "pass"
    portfolio = json.loads((out_dir / "portfolio_snapshot.json").read_text(encoding="utf-8"))
    by_id = {a["asset_id"]: a for a in portfolio["assets"]}
    assert by_id["A2_V42C_DIN_OOF"]["verification_status"] == "verified_oof"
    assert by_id["A2_V48A_SASREC_OOF"]["verification_status"] == "verified_oof"
    assert by_id["A2_V23_TOP10_FIXED_CANDIDATES"]["verification_status"] == "declared_unmaterialized"
    # idempotent resume
    again = run_a2_integration(
        project_root=a2_world["project"],
        a2_data_dir=a2_world["data_dir"],
        a2_runs_root=a2_world["runs_root"],
        out_root=a2_world["project"] / "artifacts" / "a2_integration",
    )
    assert again["run_id"] == result["run_id"]
    assert again["status"] == "ready_for_experiment_design"


def test_missing_asset_reports_single_item(a2_world):
    result = run_a2_integration(
        project_root=a2_world["project"],
        a2_data_dir=a2_world["data_dir"],
        a2_runs_root=a2_world["project"] / "nonexistent",
        out_root=a2_world["project"] / "artifacts" / "a2_integration_missing",
    )
    assert result["status"] == "waiting_for_explicit_asset"
    assert result["missing_asset"] == "a2_asset:runs/stageA/train_folds.csv"
    assert result["scientific_rounds_used"] == 0
