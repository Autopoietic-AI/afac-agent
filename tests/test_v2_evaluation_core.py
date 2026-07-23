# -*- coding: utf-8 -*-
"""Synthetic tests for the AFAC v2.0 evaluation core modules."""
from __future__ import annotations

import numpy as np
import pytest

from afac_agent.v2.metric_semantics import (
    audit_panels,
    candidate_pool_recall,
    error_decomposition,
    panel_record,
    promotion_gate,
    ranking_metrics,
    validate_error_decomposition,
)
from afac_agent.v2.noop_detector import apply_noop_policy, compare_predictions
from afac_agent.v2.validation_reality import (
    B1_REQUIRED_PANELS,
    B2_REQUIRED_PANELS,
    ValidationRealityManager,
    deployment_confidence,
    offline_online_gap,
    panel_calibration,
)


def _pool_and_targets():
    # 4 users; targets sit at varying pool depths and ranking positions.
    targets = ["t0", "t1", "t2", "t3"]
    pool = [
        ["x", "t0"] + ["f"] * 198,  # target at pool rank 2
        ["f"] * 25 + ["t1"] + ["f"] * 174,  # target at pool rank 26
        ["f"] * 150 + ["t2"] + ["f"] * 49,  # target at pool rank 151
        ["f"] * 200,  # target missing from pool
    ]
    topk = [
        ["t0"] + ["x"] * 9,  # success at rank 1
        ["x"] * 10,  # in pool (rank 26) but outside top-10
        ["x"] * 10,  # in pool (rank 151) but outside top-10
        ["x"] * 10,  # missing from pool
    ]
    return pool, topk, targets


def test_candidate_pool_recall_differs_from_hit_rate():
    pool, topk, targets = _pool_and_targets()
    recall = candidate_pool_recall(pool, targets)
    assert recall["candidate_pool_recall@20"] == pytest.approx(1 / 4)
    assert recall["candidate_pool_recall@50"] == pytest.approx(2 / 4)
    assert recall["candidate_pool_recall@100"] == pytest.approx(2 / 4)
    assert recall["candidate_pool_recall@200"] == pytest.approx(3 / 4)
    rank = ranking_metrics(topk, targets)
    assert rank["hit_rate@10"] == pytest.approx(1 / 4)
    # The v1.6 bug made these identical; they must now be separable.
    assert recall["candidate_pool_recall@50"] != rank["hit_rate@10"]
    assert rank["ndcg@10"] == pytest.approx(1 / 4)  # single hit at rank 1
    assert rank["mrr@10"] == pytest.approx(1 / 4)


def test_error_decomposition_is_exclusive_and_sums_to_one():
    pool, topk, targets = _pool_and_targets()
    rates = error_decomposition(topk, pool, targets)
    assert rates["top10_success"] == pytest.approx(1 / 4)
    assert rates["in_pool_outside_top10"] == pytest.approx(2 / 4)
    assert rates["missing_from_candidate_pool"] == pytest.approx(1 / 4)
    check = validate_error_decomposition(rates)
    assert check["status"] == "ok"
    assert check["sum"] == pytest.approx(1.0)
    assert promotion_gate(check["status"])["allowed"] is True


def test_broken_decomposition_rejected_and_blocks_promotion():
    broken = {"top10_success": 0.5, "in_pool_outside_top10": 0.3, "missing_from_candidate_pool": 0.3}
    check = validate_error_decomposition(broken)
    assert check["status"] == "invalid_metric_semantics"
    gate = promotion_gate(check["status"])
    assert gate["allowed"] is False
    assert gate["reason"]


def test_duplicate_panels_detected_via_membership():
    members = ["u1", "u2", "u3"]
    record_a = panel_record("PANEL_A", members, labels=[0, 1, 1])
    record_b = panel_record("PANEL_B", list(reversed(members)))  # same membership, different order
    record_c = panel_record("PANEL_C", ["u1", "u2", "u4"])
    assert record_a["membership_hash"] == record_b["membership_hash"]
    assert record_a["class_distribution"] == {"0": 1 / 3, "1": 2 / 3}
    audit = audit_panels([record_a, record_b, record_c])
    assert audit["independent_panel_count"] == 2
    assert audit["duplicate_pairs"] == [{"panel_id": "PANEL_B", "duplicate_of": "PANEL_A"}]
    dup = [p for p in audit["panels"] if p["panel_id"] == "PANEL_B"][0]
    assert dup["status"] == "duplicate_panel"
    assert dup["duplicate_of"] == "PANEL_A"


def test_noop_detected_for_identical_proba():
    parent = np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5]])
    report = compare_predictions(parent, parent.copy())
    assert report.status == "no_op"
    assert report.changed_fraction == 0.0
    assert report.max_abs_score_diff <= 1e-12
    assert report.changed_users == []


def test_noop_detected_for_identical_topk_lists():
    topk = [["a", "b", "c"], ["d", "e", "f"]]
    report = compare_predictions(topk, [list(row) for row in topk])
    assert report.status == "no_op"
    assert report.changed_fraction == 0.0


def test_real_change_is_not_noop():
    parent = np.array([[0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
    candidate = np.array([[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]])
    report = compare_predictions(parent, candidate)
    assert report.status == "ok"
    assert report.changed_fraction == pytest.approx(1 / 3)
    assert report.max_abs_score_diff > 1e-12
    assert report.route_fraction == pytest.approx(1 / 3)

    parent_lists = [["a", "b", "c"], ["d", "e", "f"]]
    candidate_lists = [["a", "b", "c"], ["d", "f", "e"]]  # order changed
    report_lists = compare_predictions(parent_lists, candidate_lists)
    assert report_lists.status == "ok"
    assert report_lists.changed_fraction == pytest.approx(0.5)
    assert set(report_lists.changed_items) == {"e", "f"}

    parent_scores = {"u1": {"i1": 0.5, "i2": 0.4}}
    candidate_scores = {"u1": {"i1": 0.9, "i2": 0.4}}
    report_scores = compare_predictions(parent_scores, candidate_scores)
    assert report_scores.status == "ok"
    assert report_scores.changed_users == ["u1"]
    assert report_scores.changed_items == ["i1"]


def test_round_refund_applied_on_noop():
    topk = [["a", "b"], ["c", "d"]]
    report = compare_predictions(topk, [list(row) for row in topk])
    record = apply_noop_policy({"experiment_id": "e1"}, report)
    assert record["consumes_round"] is False
    assert record["portfolio_eligible"] is False
    assert {"type": "no_op", "hint": "implementation or configuration problem"} in record["issues"]

    changed = compare_predictions(topk, [["a", "x"], ["c", "d"]])
    record_ok = apply_noop_policy({"experiment_id": "e2"}, changed)
    assert record_ok["consumes_round"] is True
    assert record_ok.get("portfolio_eligible") is not False


def test_offline_online_gap_math():
    gap = offline_online_gap(0.5, 0.4)
    assert gap["absolute_gap"] == pytest.approx(-0.1)
    assert gap["relative_gap"] == pytest.approx(-0.2)
    zero = offline_online_gap(0.0, 0.3)
    assert zero["absolute_gap"] == pytest.approx(0.3)
    assert zero["relative_gap"] == 0.0


def test_missing_required_panel_flagged():
    manager = ValidationRealityManager("B1")
    records = [
        {"panel_id": pid, "member_ids": [f"u_{pid}_{i}" for i in range(5)]}
        for pid in B1_REQUIRED_PANELS[:-1]
    ]
    result = manager.validate_panel_set(records)
    assert result["status"] == "panels_missing"
    assert result["missing_panels"] == [B1_REQUIRED_PANELS[-1]]

    full = records + [{"panel_id": B1_REQUIRED_PANELS[-1], "member_ids": ["a", "b"]}]
    assert manager.validate_panel_set(full)["status"] == "verified"

    b2 = ValidationRealityManager("B2")
    assert b2.required_panels == B2_REQUIRED_PANELS
    empty = b2.validate_panel_set([])
    assert empty["status"] == "panels_missing"
    assert empty["missing_panels"] == B2_REQUIRED_PANELS


def test_duplicate_panels_flagged_by_manager():
    manager = ValidationRealityManager("B1")
    members = [f"u{i}" for i in range(10)]
    records = [
        {"panel_id": pid, "member_ids": [f"{pid}_{i}" for i in range(4)]}
        for pid in B1_REQUIRED_PANELS
    ]
    records.append({"panel_id": "B1_EXTRA_PANEL", "member_ids": list(members)})
    records.append({"panel_id": "B1_EXTRA_PANEL_COPY", "member_ids": list(reversed(members))})
    result = manager.validate_panel_set(records)
    assert result["status"] == "duplicate_panels"
    assert result["duplicate_pairs"] == [
        {"panel_id": "B1_EXTRA_PANEL_COPY", "duplicate_of": "B1_EXTRA_PANEL"}
    ]


def test_deployment_confidence_bounds_and_gap_monotonicity():
    panel_scores = {"P1": 0.8, "P2": 0.6}
    conf_small = deployment_confidence(panel_scores, gap=0.05, fold_stability=0.9)
    conf_large = deployment_confidence(panel_scores, gap=0.5, fold_stability=0.9)
    assert 0.0 <= conf_small <= 1.0
    assert 0.0 <= conf_large <= 1.0
    assert conf_large < conf_small
    assert deployment_confidence(panel_scores, gap=0.0, fold_stability=1.0) == pytest.approx(0.7)


def test_panel_calibration_weight():
    record = panel_calibration("P1", {"f1": 0.5, "f2": 0.7}, environment_weight=2.0)
    assert record["calibrated_score"] == pytest.approx(1.2)
    with_signal = panel_calibration("P1", {"f1": 0.5}, online_signal=0.9)
    assert with_signal["calibrated_score"] == pytest.approx(0.7)
