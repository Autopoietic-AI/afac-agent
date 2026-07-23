# -*- coding: utf-8 -*-
"""Tests for the AFAC v2.0 recommendation operator space (memory-safe)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from afac_agent.v2.memory_safe import (
    FORBIDDEN_PATTERN,
    MemoryPreflight,
    SparseCandidateTable,
    chunked,
    topk_streaming,
)
from afac_agent.v2.operators.recommendation import (
    ALL_RETRIEVER_CLASSES,
    CandidateTableRanker,
    CandidateUnion,
    ColdStartRetriever,
    ExpertRouter,
    HistToTargetRetriever,
    Item2VecRetriever,
    ItemAttributeRetriever,
    ItemCF1HopRetriever,
    ItemCF2HopRetriever,
    LastTransitionRetriever,
    MultiExpertEnsemble,
    NovelRecallRetriever,
    PairTransitionRetriever,
    PopularityRetriever,
    RandomWalkRetriever,
    SequenceRecallRetriever,
    StableTop10Anchor,
    UserAttributeRetriever,
    fallback_keep_parent,
    history_slot_protection,
    novel_only_rerank,
    position10_admission,
    topk_protection,
)

GROUPS = [("i0", "i20"), ("i1", "i21"), ("i2", "i22")]


def _synth_data():
    """30 users / 60 items with planted a_g -> t_g pair transitions."""
    train_seq: dict[str, list[str]] = {}
    targets: dict[str, str] = {}
    for g, (anchor, target) in enumerate(GROUPS):
        for j in range(8):
            uid = f"u{g * 8 + j}"
            f1 = f"i{30 + (g * 8 + j) % 30}"
            f2 = f"i{30 + (g * 8 + j + 7) % 30}"
            if j < 4:
                seq = [f1, anchor, target, f2]  # bigram in history
            else:
                seq = [f1, f2, anchor]  # target only implied by transition
            train_seq[uid] = seq
            targets[uid] = target
    train_seq["u_rep"] = ["i5", "i6", "i5", "i6"]
    targets["u_rep"] = "i5"
    train_seq["u_len3"] = ["i7", "i8", "i9"]
    targets["u_len3"] = "i8"
    train_seq["u_len5"] = ["i10", "i11", "i12", "i13", "i14"]
    targets["u_len5"] = "i12"
    train_seq["u_cold"] = []

    user_rows = []
    for idx, uid in enumerate(train_seq):
        g = idx // 8 if idx < 24 else -1
        if 0 <= g <= 2:
            age, seg = str(20 + g), f"s{g}"
        elif uid == "u_cold":
            age, seg = "20", "s0"
        else:
            age, seg = str(40 + idx), f"sx{idx}"
        user_rows.append({"uid": uid, "age": age, "seg": seg})
    user_df = pd.DataFrame(user_rows)

    item_rows = []
    for n in range(60):
        item_rows.append(
            {
                "iid": f"i{n}",
                "price": str(n),
                "cat": f"c{n % 10}",
                "seg": "s0" if n % 2 == 0 else "s1",
            }
        )
    item_df = pd.DataFrame(item_rows)
    return train_seq, targets, user_df, item_df


def _planted_users(group: int = 0) -> list[str]:
    """Users of one group whose target is implied only by transition."""
    return [f"u{group * 8 + j}" for j in range(4, 8)]


# ---------------------------------------------------------------------------
# SparseCandidateTable / memory preflight
# ---------------------------------------------------------------------------


def test_sparse_table_union_rrf_and_source_count():
    t1 = SparseCandidateTable(source_id="s1")
    t1.add_scores("u1", {"a": 3.0, "b": 2.0}, "s1")
    t2 = SparseCandidateTable(source_id="s2")
    t2.add_scores("u1", {"b": 5.0, "c": 1.0}, "s2")

    merged = SparseCandidateTable.union(t1, t2, rrf_k=60)
    b = merged.get("u1", "b")
    a = merged.get("u1", "a")
    c = merged.get("u1", "c")
    assert b is not None and b.source_count == 2
    assert a is not None and a.source_count == 1
    # a: rank1 in s1 -> 1/61 ; b: rank2 in s1 + rank1 in s2 ; c: rank2 in s2
    assert a.rrf == pytest.approx(1.0 / 61)
    assert b.rrf == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert c.rrf == pytest.approx(1.0 / 62)
    assert merged.topk("u1", 3)[0] == "b"
    assert merged.to_topk_lists(2) == [["b", "a"]]
    assert merged.n_entries == 3

    cov = merged.coverage({"u1": "c", "u2": "zz"})
    assert cov["n_users"] == 2 and cov["n_hit"] == 1
    assert cov["hit_rate"] == pytest.approx(0.5)
    assert cov["mean_rank"] == pytest.approx(3.0)


def test_sparse_table_max_per_user_and_csr_bounded():
    t = SparseCandidateTable(source_id="s", max_per_user=5)
    t.add_scores("u", {f"i{n}": float(n) for n in range(20)}, "s")
    assert len(t.entries_for("u")) == 5
    assert t.topk("u", 3) == ["i19", "i18", "i17"]
    csr = t.to_csr()
    assert csr.shape == (1, 20)  # bounded vocab, not the full catalog
    assert csr.nnz == 5


def test_chunked_and_topk_streaming():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    stream = [("u", f"i{n}", float(n)) for n in range(50)]
    out = topk_streaming(stream, 5)
    assert out["u"] == ["i49", "i48", "i47", "i46", "i45"]


def test_memory_preflight_blocks_full_dense_float64():
    pre = MemoryPreflight(budget_mb=4096.0)
    decision = pre.check_dense(100000, 100000, np.float64)
    assert decision["allowed"] is False
    assert decision["suggested"] == "use_sparse"
    assert decision["estimated_mb"] > 4096.0
    assert "forbidden" in decision["reason"]
    assert "SparseCandidateTable" in FORBIDDEN_PATTERN

    ok = pre.check_dense(10, 10, np.float64)
    assert ok["allowed"] is True and ok["suggested"] == "ok"

    batch = pre.safe_batch_size(100000, 100000, np.float32)
    assert 1 <= batch <= 100000
    # a single row wider than the budget still yields the minimum of 1
    assert pre.safe_batch_size(10, 10**12, np.float64) == 1


# ---------------------------------------------------------------------------
# R1 retrievers / R2 union
# ---------------------------------------------------------------------------


def _fit_all(train_seq, targets, user_df, item_df, classes=ALL_RETRIEVER_CLASSES):
    tables = {}
    for cls in classes:
        retriever = cls().fit(train_seq, targets, user_df, item_df)
        tables[cls.source_id] = retriever.retrieve(list(train_seq.keys()), max_per_user=50)
    return tables


def test_retrievers_recover_planted_target():
    train_seq, targets, user_df, item_df = _synth_data()
    tables = _fit_all(train_seq, targets, user_df, item_df)
    users = _planted_users(0)
    recovering = [
        PopularityRetriever,
        LastTransitionRetriever,
        PairTransitionRetriever,
        HistToTargetRetriever,
        ItemCF1HopRetriever,
        ItemCF2HopRetriever,
        Item2VecRetriever,
        RandomWalkRetriever,
        UserAttributeRetriever,
        ItemAttributeRetriever,
        SequenceRecallRetriever,
        NovelRecallRetriever,
    ]
    for cls in recovering:
        table = tables[cls.source_id]
        for uid in users:
            assert table.get(uid, targets[uid]) is not None, (
                f"{cls.source_id} failed to recover target for {uid}"
            )
    # pair transition should rank the planted target near the top
    table = tables[PairTransitionRetriever.source_id]
    for uid in users:
        assert table.get(uid, targets[uid]).rank <= 3
    # item2vec is honestly marked degraded
    assert Item2VecRetriever.degraded is True


def test_cold_start_retriever_only_for_empty_history():
    train_seq, targets, user_df, item_df = _synth_data()
    table = ColdStartRetriever().fit(train_seq, targets, user_df, item_df).retrieve(
        ["u_cold", "u0"], max_per_user=10
    )
    assert table.entries_for("u_cold")  # fallback candidates exist
    assert table.entries_for("u0") == []  # not emitted for warm users


def test_candidate_union_keeps_target():
    train_seq, targets, user_df, item_df = _synth_data()
    tables = _fit_all(train_seq, targets, user_df, item_df)
    merged = CandidateUnion().merge(tables.values())
    planted = {uid: targets[uid] for uid in list(train_seq)[:24]}
    cov = merged.coverage(planted)
    assert cov["hit_rate"] == pytest.approx(1.0)
    for uid, tgt in planted.items():
        entry = merged.get(uid, tgt)
        assert entry is not None and entry.source_count >= 3


# ---------------------------------------------------------------------------
# R3 ranker
# ---------------------------------------------------------------------------


def test_ranker_ranks_target_above_hard_negatives():
    train_seq, targets, user_df, item_df = _synth_data()
    tables = _fit_all(train_seq, targets, user_df, item_df)
    merged = CandidateUnion().merge(tables.values())

    ranker = CandidateTableRanker(n_negatives=25, n_estimators=48)
    assert ranker.BACKEND_INFO["actual"] == "sklearn_gbdt_binary"
    assert ranker.BACKEND_INFO["requested"] == "lightgbm_lambdarank"
    ranker.set_context(
        train_seq=train_seq, train_targets=targets, user_df=user_df, item_df=item_df
    )
    rows, labels = ranker.build_training_rows(merged)
    assert set(labels) == {0, 1}
    n_users_with_target = len([u for u in train_seq if u in targets])
    assert labels.count(1) == n_users_with_target
    ranker.fit(rows, labels)
    assert ranker.clf is not None

    users = list(train_seq)[:24]
    topk = ranker.rerank(users, k=10)
    hits = 0
    for uid, lst in zip(users, topk):
        assert len(lst) <= 10
        if targets[uid] in lst:
            hits += 1
    assert hits / len(users) >= 0.75

    # target score above the median hard negative for most users
    better = 0
    for uid in users:
        user_rows = [r for r in ranker.score(ranker.build_scoring_rows(user_ids=[uid]))]
        tgt_row = next(r for r in user_rows if r["item_id"] == targets[uid])
        negs = [r for r in user_rows if r["item_id"] != targets[uid]]
        if negs and tgt_row["score"] > np.median([r["score"] for r in negs]):
            better += 1
    assert better / len(users) >= 0.75


def test_ranker_fallback_without_targets():
    train_seq, targets, user_df, item_df = _synth_data()
    ranker = CandidateTableRanker()
    ranker.set_context(train_seq=train_seq, train_targets={})
    rows, labels = ranker.build_training_rows(
        SparseCandidateTable(source_id="empty")
    )
    ranker.fit(rows, labels)
    assert ranker.fallback_used is True


# ---------------------------------------------------------------------------
# R4 anchor / R5 router
# ---------------------------------------------------------------------------


def test_stable_top10_anchor_snapshot():
    a1 = StableTop10Anchor.snapshot("u1", [f"i{n}" for n in range(12)], scores=[1.0 - 0.1 * n for n in range(12)])
    a2 = StableTop10Anchor.snapshot("u1", [f"i{n}" for n in range(10)], scores=[1.0 - 0.1 * n for n in range(10)])
    assert a1.anchor_hash == a2.anchor_hash  # snapshot truncates to top-10
    assert len(a1.top10) == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        a1.top10 = ("x",)  # type: ignore[misc]


def test_expert_router_buckets():
    train_seq, targets, _, _ = _synth_data()
    router = ExpertRouter().fit(train_seq)
    assert router.route("u_cold") == ["Len0", "ColdUser"]
    assert "ExactLen3" in router.route("u_len3")
    assert "Len4Plus" in router.route("u_len5")
    assert "Repeat" in router.route("u_rep")
    assert "History" in router.route("u_len3")

    rows = [{"features": {"rrf": 1.0, "history_count": 2.0, "repeat_flag": 1.0}}]
    out = router.reweight_rows(rows, ["Repeat"])
    assert out[0]["expert_score"] > 0.0


# ---------------------------------------------------------------------------
# R6 protection / R7 admission / R8 ensemble
# ---------------------------------------------------------------------------

_PARENT = [f"p{n}" for n in range(10)]


def test_topk_protection_keeps_first_k_slots():
    proposed = list(reversed(_PARENT))
    new, audit = topk_protection(_PARENT, proposed, k=3)
    assert new[:3] == _PARENT[:3]
    assert audit["kept_slots"] == [0, 1, 2]
    assert not audit["fallback_used"]
    assert len(set(new)) == len(_PARENT)


def test_history_and_novel_slot_protection():
    history = {"p1", "p4"}
    proposed = ["x0", "x1", "p0", "p2", "p3", "p5", "p6", "p7", "p8", "p9"]
    new, audit = history_slot_protection(_PARENT, proposed, history)
    assert new[1] == "p1" and new[4] == "p4"
    assert audit["kept_slots"] == [1, 4]

    new2, audit2 = novel_only_rerank(_PARENT, proposed, history)
    assert new2[1] == "p1" and new2[4] == "p4"
    # only novel items (not in history) may move, and no history item is
    # introduced into a free slot
    for i in range(10):
        if new2[i] != _PARENT[i]:
            assert _PARENT[i] not in history
            assert new2[i] not in history
    assert {"x0", "x1"} <= set(new2)


def test_position10_admission_rules():
    # admitted: strictly above incumbent + margin, only at the last slot
    new, audit = position10_admission(_PARENT, "ext", margin=0.1, incumbent_score=0.5, external_score=0.7)
    assert new[:-1] == _PARENT[:-1]
    assert new[-1] == "ext"
    assert len(audit["admissions"]) == 1
    assert audit["admissions"][0]["position"] == 10
    assert audit["changed_count"] == 1

    # rejected: margin not cleared -> parent unchanged, no admission
    new2, audit2 = position10_admission(_PARENT, "ext", margin=0.1, incumbent_score=0.5, external_score=0.55)
    assert new2 == _PARENT
    assert audit2["admissions"] == []
    assert audit2["fallback_used"] is False

    # violation: candidate already present -> fallback keeps parent unchanged
    new3, audit3 = position10_admission(_PARENT, "p3", margin=0.1, incumbent_score=0.5, external_score=9.9)
    assert new3 == _PARENT
    assert audit3["fallback_used"] is True
    assert audit3["violations"]


def test_fallback_keep_parent_returns_parent_unchanged():
    new, audit = fallback_keep_parent(_PARENT, "boom")
    assert new == _PARENT
    assert new is not _PARENT
    assert audit["fallback_used"] is True
    assert audit["violations"] == ["boom"]


def test_multi_expert_ensemble_respects_protection():
    experts = {
        "e1": ["x0", "x1", "p0", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
        "e2": ["x1", "x0", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
    }
    new, audit = MultiExpertEnsemble(protect_k=3).combine(_PARENT, experts)
    assert new[:3] == _PARENT[:3]
    assert len(new) == len(_PARENT)
    assert len(set(new)) == len(_PARENT)
    assert not audit["fallback_used"]
