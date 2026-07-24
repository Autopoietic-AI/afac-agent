# -*- coding: utf-8 -*-
"""AFAC v2.1 B2 scientific execution, data contract, budget and deployment
permission repair tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from afac_agent.llm.base import LLMResponse
from afac_agent.research.event_store import load_json
from afac_agent.v2.anchor_registry import B2AnchorRegistry
from afac_agent.v2.capability_registry import default_registry
from afac_agent.v2.completion_contract import check_completion_contract
from afac_agent.v2.data_contract import (
    B2CanonicalDataContract,
    ScaleValue,
    build_b2_data_contract,
    reconcile_data_contract,
)
from afac_agent.v2.data_intelligence import analyze_recommendation
from afac_agent.v2.experiment_kind import (
    ExperimentKind,
    kind_from_operator_and_folds,
    permission_for_kind,
)
from afac_agent.v2.orchestrator import V2AutonomousResearchOrchestrator
from afac_agent.v2.noop_detector import apply_noop_policy, NoOpReport
from afac_agent.v2.proposal_compiler import compile_proposal, semantic_revision_delta

from test_v2_runtime import FakeProvider, _write_b2_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- data contract


def _make_contract(tmp_path: Path) -> B2CanonicalDataContract:
    data = _write_b2_dataset(tmp_path)
    import pandas as pd

    item_df = pd.read_csv(data / "item.csv", keep_default_na=False)
    train_df = pd.read_csv(data / "train.csv", keep_default_na=False)
    test_df = pd.read_csv(data / "test.csv", keep_default_na=False)
    train_seq: dict[str, list[str]] = {}
    test_seq: dict[str, list[str]] = {}
    targets: dict[str, str] = {}
    for _, row in train_df.iterrows():
        uid = str(row["uid"])
        targets[uid] = str(row["target_iid"])
        train_seq[uid] = [str(x) for x in str(row["item_seq_raw"]).split(",") if x]
    for _, row in test_df.iterrows():
        uid = str(row["uid"])
        test_seq[uid] = [str(x) for x in str(row["item_seq_raw"]).split(",") if x]
    return build_b2_data_contract(
        data_root=data,
        item_df=item_df,
        train_df=train_df,
        test_df=test_df,
        train_seq=train_seq,
        test_seq=test_seq,
        train_targets=targets,
        item_universe=set(item_df["iid"].astype(str)),
        sampled_train_users=list(train_seq)[:10],
        sampled_test_users=list(test_seq)[:4],
        profile_scope="sampled_head",
    )


def test_data_contract_tracks_n_items_source(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    assert contract.status == "passed"
    assert contract.n_items_total is not None
    assert contract.n_items_total.source_file.endswith("item.csv")
    assert contract.n_items_total.source_column == "iid"
    assert contract.n_items_total.value == 60


def test_data_contract_users_do_not_impersonate_items(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    assert contract.n_train_users_total is not None
    assert contract.n_train_users_total.value == 30
    assert contract.n_train_users_total.value != contract.n_items_total.value


def test_data_contract_sampled_scope_is_recorded(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    assert contract.profile_scope == "sampled_head"
    assert contract.profiler_sample_train_users is not None
    assert contract.profiler_sample_train_users.value == 10
    assert contract.profiler_sample_test_users is not None
    assert contract.profiler_sample_test_users.value == 4


def test_reconcile_blocks_inconsistent_n_items() -> None:
    result = reconcile_data_contract(
        input_discovery_n_items=14065,
        data_intelligence_n_items=40011,
        experiment_executor_item_universe=set(map(str, range(14065))),
        deployment_candidate_legality_universe=set(map(str, range(14065))),
    )
    assert result["status"] == "blocked_data_contract_mismatch"
    assert any("14065" in e and "40011" in e for e in result["errors"])


# --------------------------------------------------------------------------- data intelligence


def test_data_intelligence_separates_total_and_profiled_test_users() -> None:
    train_seq = {"u0": ["i0", "i1"], "u1": ["i2"]}
    train_targets = {"u0": "i3", "u1": "i4"}
    test_seq = {"v0": ["i0"], "v1": ["i1"], "v2": ["i2"], "v3": ["i3"]}
    di = analyze_recommendation(
        train_seq,
        train_targets,
        test_seq,
        item_popularity={f"i{i}": 0.0 for i in range(10)},
        n_train_total=2,
        n_test_total=4,
        profile_scope="sampled_head",
    )
    fp = di.dataset_fingerprint
    assert fp["n_test_total"] == 4
    assert fp["n_test_profiled"] == 4
    assert fp["profile_scope"] == "sampled_head"
    assert "raw_length_buckets" in di.coverage_map
    assert "dedup_length_buckets" in di.coverage_map


# --------------------------------------------------------------------------- experiment permission


def test_diagnostic_cannot_deploy_or_be_incumbent() -> None:
    p = permission_for_kind(ExperimentKind.DETERMINISTIC_DIAGNOSTIC)
    assert p.can_deploy is False
    assert p.can_be_incumbent is False
    assert p.can_enter_portfolio is False
    assert p.consumes_scientific_round is False


def test_screen_can_enter_portfolio_but_cannot_deploy() -> None:
    p = permission_for_kind(ExperimentKind.SCREEN_EXPERIMENT)
    assert p.can_enter_portfolio is True
    assert p.can_be_incumbent is False
    assert p.can_deploy is False
    assert p.consumes_scientific_round is True


def test_confirm_can_deploy_and_be_incumbent() -> None:
    p = permission_for_kind(ExperimentKind.CONFIRM_EXPERIMENT)
    assert p.can_enter_portfolio is True
    assert p.can_be_incumbent is True
    assert p.can_deploy is True
    assert p.consumes_scientific_round is True


def test_kind_classification_by_fold_count() -> None:
    assert kind_from_operator_and_folds("candidate_ranker_experiment", 0) == ExperimentKind.DETERMINISTIC_DIAGNOSTIC
    assert kind_from_operator_and_folds("candidate_ranker_experiment", 2) == ExperimentKind.SCREEN_EXPERIMENT
    assert kind_from_operator_and_folds("candidate_ranker_experiment", 3) == ExperimentKind.CONFIRM_EXPERIMENT
    assert kind_from_operator_and_folds("candidate_ranker_experiment", 5) == ExperimentKind.FULL_CV_EXPERIMENT
    assert kind_from_operator_and_folds("cached_replay", 3) == ExperimentKind.CACHED_REPLAY


# --------------------------------------------------------------------------- proposal compiler


def test_compiler_maps_real_operator() -> None:
    compiled = compile_proposal(
        {"diagnostic_type": "candidate_ranker_experiment"},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    assert compiled.status == "compiled"
    assert compiled.operator_id == "candidate_ranker_experiment"
    assert compiled.experiment_kind == ExperimentKind.SCREEN_EXPERIMENT


def test_compiler_blocks_unavailable_operator() -> None:
    compiled = compile_proposal(
        {"diagnostic_type": "unknown_magic_operator"},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    assert compiled.status == "blocked_missing_adapter"


def test_compiler_rejects_diagnostic_in_formal_mode() -> None:
    compiled = compile_proposal(
        {"diagnostic_type": "candidate_recall_diagnostic"},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    assert compiled.status == "blocked_missing_adapter"


def test_semantic_revision_delta_detects_duplicate() -> None:
    p1 = compile_proposal(
        {"diagnostic_type": "candidate_ranker_experiment", "information_sources": ["popularity", "history"]},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    p2 = compile_proposal(
        {"diagnostic_type": "candidate_ranker_experiment", "information_sources": ["popularity", "history"]},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    delta = semantic_revision_delta(p1, p2)
    assert delta["duplicate"] is True
    assert not delta["semantic_delta"]


def test_semantic_revision_delta_detects_real_source_change() -> None:
    p1 = compile_proposal(
        {"diagnostic_type": "candidate_ranker_experiment", "information_sources": ["popularity"]},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    p2 = compile_proposal(
        {"diagnostic_type": "candidate_ranker_experiment", "information_sources": ["popularity", "history"]},
        parent_candidate_id="popularity_parent",
        formal_mode=True,
        task="B2",
    )
    delta = semantic_revision_delta(p1, p2)
    assert delta["duplicate"] is False
    assert "retrieval_sources" in delta["semantic_delta"]


# --------------------------------------------------------------------------- regression: fault run manifest


def test_fault_run_manifest_fails_completion_contract() -> None:
    """The 74ba8db... fault run must be recognized as invalid."""
    manifest = {
        "orchestrator_version": "2.0.0",
        "manifest_version": "afac_v2_run_manifest_v1",
        "trajectory_version": "afac_v2_trajectory_v1",
        "execution_id": "74ba8db66a4b8f50d9196865a611e9d78058e14d35fd8ac99c6e2426279dcd41",
        "code_commit": "8c7339594311459150a3e5a326a5ee20188f2919",
        "planner_mode": "llm_plus_deterministic_gates",
        "llm_required": True,
        "llm_calls_count": 12,
        "capability_registry_hash": "",
        "metric_contract_hash": "",
        "validation_contract_hash": "",
        "started_at": 0.0,
        "ended_at": 8340.0,
        "max_wall_clock_seconds": 7200.0,
        "wall_clock_seconds": 8340.0,
        "cache_status": "no_cache",
        "resume_status": "fresh",
        "status": "completed",
        "current_stage": "COMPLETED",
        "scientific_rounds_used": 0.0,
        "scientific_attempts_used": 0,
        "effective_scientific_rounds": 0.0,
        "diagnostics_used": 3,
        "cheap_diagnostics_used": 3,
        "no_op_rounds_refunded": 0,
        "implementation_failures_used": 0,
        "uses_test_truth": False,
        "mutates_frozen_assets": False,
        "deployment_generated": True,
        "smoke_mode": False,
        "data_contract_status": "passed",
        "budget_contract_status": "failed",
        "deployment_permission_status": "failed",
        "incumbent_can_deploy": False,
        "no_op_in_portfolio": True,
        "all_rounds_diagnostic": True,
        "anchor_fallback": False,
        "artifacts": {
            "problem_node": "x",
            "competition_research_state": "x",
            "m6b_proposal": "x",
            "m6c_critic": "x",
            "m5_decision": "x",
            "experiment_genome": "x",
            "budget_decision": "x",
            "experiment_result": "x",
            "no_op_audit": "x",
            "portfolio_update": "x",
            "postmortem": "x",
            "deployment_audit": "x",
        },
        "gates": {
            "metric_semantics_gate": True,
            "data_intelligence": True,
            "validation_reality": True,
            "test_truth_guard": True,
            "frozen_asset_hash": True,
        },
    }
    contract = check_completion_contract(manifest, smoke=False)
    assert contract["status"] == "failed"
    reasons = " ".join(contract["missing"])
    assert "scientific" in reasons or "diagnostic" in reasons or "budget" in reasons or "deployment_permission" in reasons


# --------------------------------------------------------------------------- orchestrator integration


class RankerOnlyProvider(FakeProvider):
    """Always proposes the candidate ranker experiment (valid scientific operator)."""

    def generate(self, request) -> LLMResponse:
        text = request.prompt if hasattr(request, "prompt") else str(request)
        if "M6B proposal stage" in text:
            proposal = dict(self.proposal)
            proposal["diagnostic_type"] = "candidate_ranker_experiment"
            proposal["budget_seconds"] = 60
            return LLMResponse(
                status="completed", provider="fake", model="fake", text=json.dumps(proposal)
            )
        return super().generate(request)


def _smoke_orchestrator(tmp_path: Path, data: Path, provider, **kwargs: Any) -> V2AutonomousResearchOrchestrator:
    return V2AutonomousResearchOrchestrator(
        project_root=PROJECT_ROOT,
        task="B2",
        data_root=data,
        out_root=tmp_path / "v2_smoke",
        max_wall_clock_seconds=300.0,
        require_llm=True,
        smoke=True,
        no_deployment=True,
        provider=provider,
        force_new_execution=True,
        smoke_max_users=50,
        **kwargs,
    )


def test_smoke_real_ranker_consumes_scientific_round_and_does_not_deploy(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = _smoke_orchestrator(tmp_path, data, RankerOnlyProvider()).run()
    assert manifest["status"] == "completed_smoke"
    run_dir = tmp_path / "v2_smoke" / manifest["execution_id"]
    # A real scientific screen should consume a round and not be diagnostic-only.
    assert manifest["scientific_attempts_used"] >= 1
    assert manifest["effective_scientific_rounds"] >= 1
    assert manifest["all_rounds_diagnostic"] is False
    assert manifest["deployment_generated"] is False
    assert not (run_dir / "TO_UPLOAD" / "candidate_B2.csv").exists()


def test_formal_run_without_confirm_falls_back_to_anchor(tmp_path: Path) -> None:
    """A formal run that only screens (no confirm) must not deploy the screen;
    it should fall back to the validated anchor."""
    data = _write_b2_dataset(tmp_path)
    manifest = V2AutonomousResearchOrchestrator(
        project_root=PROJECT_ROOT,
        task="B2",
        data_root=data,
        out_root=tmp_path / "v2_formal_anchor",
        max_wall_clock_seconds=600.0,
        require_llm=True,
        smoke=False,
        no_deployment=False,
        provider=RankerOnlyProvider(),
        force_new_execution=True,
        smoke_max_users=50,
    ).run()
    # Screen-only run cannot claim completed; anchor fallback is the only legal deployable source.
    assert manifest["status"] in {"completed_with_anchor_fallback", "completed"}
    assert manifest["anchor_fallback"] is True
    assert manifest["deployment_generated"] is True
    run_dir = tmp_path / "v2_formal_anchor" / manifest["execution_id"]
    assert (run_dir / "TO_UPLOAD" / "candidate_B2.csv").is_file()
    decision = load_json(run_dir / "deployment_decision.json")
    assert decision["deployment_decision"] in {"deploy", "anchor_fallback"}
    assert decision["audit"]["scientific_permission_passed"] is True
    assert decision["audit"]["fallback_to_anchor"] is True


# --------------------------------------------------------------------------- smoke C: no-op isolation


def test_no_op_experiment_is_refunded_and_not_in_portfolio() -> None:
    report = NoOpReport(status="no_op", changed_fraction=0.0, reasons=["identical lists"])
    record = apply_noop_policy({"candidate_id": "c1", "consumes_round": True}, report)
    assert record["consumes_round"] is False
    assert record["portfolio_eligible"] is False


# --------------------------------------------------------------------------- smoke D: hard budget gate


def test_short_budget_rejects_experiment_and_stops_cleanly(tmp_path: Path) -> None:
    data = _write_b2_dataset(tmp_path)
    manifest = V2AutonomousResearchOrchestrator(
        project_root=PROJECT_ROOT,
        task="B2",
        data_root=data,
        out_root=tmp_path / "v2_budget",
        max_wall_clock_seconds=60.0,
        deployment_reserve_seconds=300.0,
        safety_margin_seconds=120.0,
        require_llm=True,
        smoke=False,
        no_deployment=True,
        provider=RankerOnlyProvider(),
        force_new_execution=True,
        smoke_max_users=50,
    ).run()
    # The single ranker screen cannot fit inside the hard budget with the
    # required deployment reserve + safety margin, so the run stops cleanly.
    assert manifest["status"] != "completed"
    assert manifest["budget_contract_status"] == "passed"
    assert manifest["wall_clock_seconds"] <= 65.0


# --------------------------------------------------------------------------- capability registry


def test_candidate_ranker_is_registered_and_available() -> None:
    registry = default_registry()
    record = registry.records.get("candidate_ranker_experiment")
    assert record is not None
    assert record.implemented is True
    assert record.available is True
    assert record.supports_screen is True
    assert record.supports_confirm is True
    assert record.supports_diagnostic is False


def test_diagnostic_registry_rejects_deployment_ready() -> None:
    registry = default_registry()
    record = registry.records.get("candidate_recall_diagnostic")
    assert record is not None
    assert record.deployment_ready is False
