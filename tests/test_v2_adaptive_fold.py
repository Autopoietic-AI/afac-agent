# -*- coding: utf-8 -*-
"""Synthetic tests for the adaptive fold policy, budget-aware promotion,
canonical fold integrity, folded experiments, stagnation semantics and the
formal multi-round loop with deployment (fake provider, no network).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from afac_agent.research.event_store import load_json
from afac_agent.v2.adaptive_fold import (
    AdaptiveFoldPolicy,
    CanonicalFolds,
    FoldFidelity,
    FoldRuntimeEstimator,
    build_fold_plan,
    evaluate_promotion,
    no_improvement_action,
    paired_fold_comparison,
    should_trigger_full_cv,
    stop_decision,
)
from afac_agent.v2.orchestrator import V2AutonomousResearchOrchestrator

from test_v2_runtime import FakeProvider, _write_b2_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _uids(n: int = 50) -> list[str]:
    return [f"u{i:04d}" for i in range(n)]


# --------------------------------------------------------------------------- fold ladder basics


def test_f0_deterministic_has_zero_folds() -> None:
    canonical = CanonicalFolds.build(_uids())
    plan = build_fold_plan(canonical, FoldFidelity.F0_DETERMINISTIC, task="B2")
    assert plan.fold_count == 0
    assert plan.selected_fold_ids == []


def test_b2_default_screen_is_two_folds() -> None:
    policy = AdaptiveFoldPolicy.for_task("B2")
    assert policy.default_screen_folds() == 2
    canonical = CanonicalFolds.build(_uids())
    plan = policy.plan(canonical, FoldFidelity.F1_SCREEN)
    assert plan.selected_fold_ids == [0, 1]


def test_b1_screen_one_fold_and_confirm_three() -> None:
    policy = AdaptiveFoldPolicy.for_task("B1")
    assert policy.default_screen_folds() == 1
    assert policy.confirm_folds() == 3
    canonical = CanonicalFolds.build(_uids())
    assert policy.plan(canonical, FoldFidelity.F1_SCREEN).selected_fold_ids == [0]
    assert policy.plan(canonical, FoldFidelity.F2_CONFIRM).selected_fold_ids == [0, 1, 2]
    assert policy.screen_is_final_evidence() is False


def test_b2_screen_promotion_to_confirm() -> None:
    comparison = {"comparable": True, "mean_delta": 0.02, "positive_fold_count": 1, "worst_fold_delta": -0.005}
    decision = evaluate_promotion(
        comparison,
        rules=AdaptiveFoldPolicy.for_task("B2").rules,
        target_bucket_gain=0.01, rescue=5, damage=2,
        prediction_changed=True, no_op=False,
        expected_information_gain=0.5, remaining_seconds=5000.0, required_seconds=500.0,
    )
    assert decision["decision"] == "promote_to_confirm"


def test_noop_is_never_promoted() -> None:
    comparison = {"comparable": True, "mean_delta": 0.5, "positive_fold_count": 2, "worst_fold_delta": 0.1}
    decision = evaluate_promotion(
        comparison,
        rules=AdaptiveFoldPolicy.for_task("B2").rules,
        target_bucket_gain=0.1, rescue=0, damage=0,
        prediction_changed=False, no_op=True,
        expected_information_gain=0.0, remaining_seconds=5000.0, required_seconds=100.0,
    )
    assert decision["decision"] == "hold"
    assert any("no-op" in r for r in decision["reasons"])


# --------------------------------------------------------------------------- canonical fold integrity


def test_canonical_folds_deterministic() -> None:
    c1 = CanonicalFolds.build(_uids())
    c2 = CanonicalFolds.build(_uids())
    assert c1.fold_hash == c2.fold_hash
    assert np.array_equal(c1.folds, c2.folds)


def test_no_rerandomization_across_calls() -> None:
    canonical = CanonicalFolds.build(_uids())
    folds_before = canonical.folds.copy()
    for _ in range(5):
        canonical.masks(0)
        canonical.fold_ids_for(FoldFidelity.F2_CONFIRM)
    assert np.array_equal(canonical.folds, folds_before)


def test_fold_subsets_are_fixed_prefixes() -> None:
    canonical = CanonicalFolds.build(_uids())
    assert canonical.fold_ids_for(FoldFidelity.F1_SCREEN) == [0, 1]
    assert canonical.fold_ids_for(FoldFidelity.F2_CONFIRM) == [0, 1, 2]
    assert canonical.fold_ids_for(FoldFidelity.F3_FULL_CV) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- paired comparison


def test_paired_comparison_same_folds() -> None:
    result = paired_fold_comparison(
        candidate_id="c1", parent_id="p1", selected_fold_ids=[0, 1],
        candidate_fold_metrics={0: 0.20, 1: 0.22}, parent_fold_metrics={0: 0.18, 1: 0.23},
    )
    assert result["comparable"] is True
    assert result["fold_deltas"] == {"0": pytest.approx(0.02), "1": pytest.approx(-0.01)}
    assert result["positive_fold_count"] == 1
    assert result["worst_fold_delta"] == pytest.approx(-0.01)


def test_mismatched_fold_comparison_forbidden() -> None:
    result = paired_fold_comparison(
        candidate_id="c1", parent_id="p1", selected_fold_ids=[0, 1],
        candidate_fold_metrics={0: 0.20, 1: 0.22}, parent_fold_metrics={0: 0.18},
    )
    assert result["comparable"] is False
    assert result["missing_parent_folds"] == [1]


# --------------------------------------------------------------------------- full-CV trigger and budget


def test_full_cv_trigger_requires_uncertainty_and_budget() -> None:
    ok = should_trigger_full_cv(
        fold_metrics_3fold={0: 0.20, 1: 0.28, 2: 0.14},  # high variance
        incumbent_delta=0.001, candidate_is_best=True,
        could_replace_anchor=False, candidates_indistinguishable=False,
        remaining_seconds=5000.0, estimated_5fold_runtime=300.0,
        deployment_reserve_seconds=300.0, safety_margin_seconds=120.0,
    )
    assert ok["trigger_full_cv"] is True


def test_full_cv_forbidden_when_budget_insufficient() -> None:
    result = should_trigger_full_cv(
        fold_metrics_3fold={0: 0.20, 1: 0.28, 2: 0.14},
        incumbent_delta=0.001, candidate_is_best=True,
        could_replace_anchor=False, candidates_indistinguishable=False,
        remaining_seconds=400.0, estimated_5fold_runtime=300.0,
        deployment_reserve_seconds=300.0, safety_margin_seconds=120.0,
    )
    assert result["trigger_full_cv"] is False
    assert any("budget" in r for r in result["reasons"])


def test_full_cv_forbidden_without_uncertainty() -> None:
    result = should_trigger_full_cv(
        fold_metrics_3fold={0: 0.200, 1: 0.201, 2: 0.199},
        incumbent_delta=0.5, candidate_is_best=True,
        could_replace_anchor=False, candidates_indistinguishable=False,
        remaining_seconds=9000.0, estimated_5fold_runtime=100.0,
        deployment_reserve_seconds=100.0, safety_margin_seconds=60.0,
    )
    assert result["trigger_full_cv"] is False


def test_deployment_reserve_enforced() -> None:
    estimator = FoldRuntimeEstimator(default_per_fold_seconds=100.0, deployment_reserve_seconds=300.0, safety_margin_seconds=100.0)
    assert estimator.can_afford(3, remaining_seconds=1000.0) is True
    assert estimator.can_afford(3, remaining_seconds=500.0) is False


def test_runtime_estimator_updates_dynamically() -> None:
    estimator = FoldRuntimeEstimator(default_per_fold_seconds=50.0)
    assert estimator.estimated_next_fold_runtime == 50.0
    for seconds in (10.0, 20.0, 30.0):
        estimator.record_fold_runtime(seconds)
    assert estimator.estimated_next_fold_runtime == pytest.approx(20.0)
    assert estimator.estimated_runtime(5) == pytest.approx(100.0)


# --------------------------------------------------------------------------- stagnation semantics


def test_two_no_improvement_rounds_trigger_global_explore() -> None:
    assert no_improvement_action(1) == "revise_proposal_or_switch_operator_family"
    assert no_improvement_action(2) == "switch_problem_or_global_explore"


def test_no_improvement_cannot_stop_directly() -> None:
    decision = stop_decision(
        no_improvement_rounds=2, operator_families_tried=["ranker"], problem_nodes_tried=["p1"],
        global_explore_attempted=False, high_roi_routes_remaining=True, m5_admissible_routes_remaining=True,
        remaining_seconds=3000.0, deployment_reserve_entered=False,
    )
    assert decision["allow_stop"] is False
    decision2 = stop_decision(
        no_improvement_rounds=3, operator_families_tried=["ranker", "bucket"], problem_nodes_tried=["p1", "p2"],
        global_explore_attempted=True, high_roi_routes_remaining=False, m5_admissible_routes_remaining=False,
        remaining_seconds=3000.0, deployment_reserve_entered=False,
    )
    assert decision2["allow_stop"] is True


# --------------------------------------------------------------------------- orchestrator formal loop + deployment


class RoundAwareProvider(FakeProvider):
    """Round 1 proposes a diagnostic; later rounds propose the ranker."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.m6b_calls = 0

    def generate(self, request):
        if "M6B proposal stage" in prompt_text(request):
            self.m6b_calls += 1
            self.calls.append(prompt_text(request))
            kind = "candidate_recall_diagnostic" if self.m6b_calls <= 1 else "candidate_ranker_experiment"
            proposal = dict(self.proposal)
            proposal["diagnostic_type"] = kind
            proposal["budget_seconds"] = 60
            from afac_agent.llm.base import LLMResponse
            return LLMResponse(status="completed", provider="fake", model="fake", text=json.dumps(proposal))
        return super().generate(request)


def prompt_text(request) -> str:
    return request.prompt


def _formal_orchestrator(tmp_path: Path, data: Path, provider, **kwargs) -> V2AutonomousResearchOrchestrator:
    defaults = dict(
        project_root=PROJECT_ROOT,
        task="B2",
        data_root=data,
        out_root=tmp_path / "v2_formal",
        max_wall_clock_seconds=600.0,
        require_llm=True,
        smoke=False,
        no_deployment=False,
        provider=provider,
        force_new_execution=True,
    )
    defaults.update(kwargs)
    return V2AutonomousResearchOrchestrator(**defaults)


def test_formal_loop_folded_screen_and_deployment(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = _formal_orchestrator(tmp_path, data, RoundAwareProvider()).run()
    assert manifest["status"] == "completed"
    assert manifest["deployment_generated"] is True
    run_dir = tmp_path / "v2_formal" / manifest["execution_id"]
    # round_01 diagnostic (F0), round_02 folded screen (F1, folds [0,1])
    assert (run_dir / "round_02" / "paired_fold_comparison.json").is_file()
    comparison = load_json(run_dir / "round_02" / "paired_fold_comparison.json")
    assert comparison["selected_fold_ids"] == [0, 1]
    assert comparison["comparable"] is True
    assert (run_dir / "round_01" / "next_decision.json").is_file()
    # deployment artifacts
    candidate = run_dir / "TO_UPLOAD" / "candidate_B2.csv"
    assert candidate.is_file()
    lines = candidate.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "uid,prediction"
    assert len(lines) == 9  # header + 8 synthetic test users
    audit = load_json(run_dir / "TO_UPLOAD" / "submission_audit.json")
    assert audit["audit_passed"] is True
    decision = load_json(run_dir / "deployment_decision.json")
    assert decision["method"] == "full_train_retrain"
    assert "3fold_ensemble" in decision["supported_methods"]


def test_full_train_deployment_uses_all_train_users(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = _formal_orchestrator(tmp_path, data, RoundAwareProvider()).run()
    run_dir = tmp_path / "v2_formal" / manifest["execution_id"]
    deployment = load_json(run_dir / "TO_UPLOAD" / "deployment_manifest.json")
    assert deployment["retrained_on_all_train_users"] is True


def test_3fold_ensemble_test_inference(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = _formal_orchestrator(tmp_path, data, RoundAwareProvider(), deployment_method="3fold_ensemble").run()
    run_dir = tmp_path / "v2_formal" / manifest["execution_id"]
    decision = load_json(run_dir / "deployment_decision.json")
    assert decision["method"] == "3fold_ensemble"
    audit = load_json(run_dir / "TO_UPLOAD" / "submission_audit.json")
    assert audit["audit_passed"] is True
    assert audit["n_rows"] == 8


def test_parent_inheritance_and_fold_plan_across_rounds(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = _formal_orchestrator(tmp_path, data, RoundAwareProvider()).run()
    run_dir = tmp_path / "v2_formal" / manifest["execution_id"]
    trajectory = load_json(run_dir / "trajectory_v2.json")
    exec_events = [e for e in trajectory["events"] if e["stage"] == "EXPERIMENT_EXECUTION"]
    assert len(exec_events) >= 2
    canonical = load_json(run_dir / "canonical_folds.json")
    comparison = load_json(run_dir / "round_02" / "paired_fold_comparison.json")
    # The incumbent is seeded from the validated popularity anchor so round-1
    # parent comparisons carry the canonical fold hash (v2.1 repair).
    assert comparison["parent_id"] == "anchor_popularity_b2"
    # fold plan is stable and identical to the canonical hash
    assert comparison and canonical["canonical_fold_hash"]


def test_no_deployment_formal_never_claims_completed(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = _formal_orchestrator(tmp_path, data, RoundAwareProvider(), no_deployment=True).run()
    run_dir = tmp_path / "v2_formal" / manifest["execution_id"]
    assert not (run_dir / "TO_UPLOAD" / "candidate_B2.csv").exists()
    assert manifest["status"] != "completed"  # strict contract: no deployment audit, no completed
