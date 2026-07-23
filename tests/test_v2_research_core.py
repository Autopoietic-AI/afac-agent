# -*- coding: utf-8 -*-
"""Synthetic fast tests for the AFAC v2.0 research core modules."""

from __future__ import annotations

import pytest

from afac_agent.v2.budget_scheduler import (
    BudgetState,
    ExperimentCandidate,
    Fidelity,
    detect_premature_stop,
    roi_score,
    schedule,
    should_continue,
)
from afac_agent.v2.capability_registry import default_registry
from afac_agent.v2.competition_intelligence import (
    CardStore,
    CompetitionSolutionCard,
    MLESTARLoop,
    MLESTAR_STAGES,
    MethodCard,
)
from afac_agent.v2.exploration_controller import decide
from afac_agent.v2.model_genome import ModelGenome, validate_genome_delta
from afac_agent.v2.problem_hierarchy import ProblemHierarchy, ProblemNode


def _candidate(candidate_id: str, gain: float = 0.5, cost: float = 60.0) -> ExperimentCandidate:
    return ExperimentCandidate(
        candidate_id=candidate_id,
        fidelity=Fidelity.SINGLE_FOLD,
        expected_gain=gain,
        expected_information_gain=0.2,
        compute_cost_seconds=cost,
    )


# ---------------------------------------------------------------- budget


def test_scheduler_continues_past_round_three_with_budget_remaining() -> None:
    state = BudgetState(elapsed=75.0, rounds_used=3.0)
    candidates = [_candidate("c1"), _candidate("c2")]
    assert should_continue(state, candidates) is True
    plan = schedule(candidates, state)
    assert [c.candidate_id for c in plan] == ["c1", "c2"]


def test_scheduler_stops_at_budget_exhaustion() -> None:
    candidates = [_candidate("c1")]
    exhausted_time = BudgetState(elapsed=7200.0, rounds_used=3.0)
    assert should_continue(exhausted_time, candidates) is False
    exhausted_rounds = BudgetState(elapsed=100.0, rounds_used=12.0)
    assert should_continue(exhausted_rounds, candidates) is False


def test_premature_stop_detection_fires_on_v16_pattern() -> None:
    state = BudgetState()  # 7200s wall clock, 12-round safety cap
    # v1.6 defect: stopped after 3 rounds with 75s of 7200s consumed.
    assert detect_premature_stop(rounds_used=3, elapsed=75.0, state=state) is True
    assert detect_premature_stop(rounds_used=3, elapsed=75.0, state=state, candidates=[_candidate("c1")]) is True
    # No positive-ROI candidate left -> not a premature stop.
    assert (
        detect_premature_stop(rounds_used=3, elapsed=75.0, state=state, candidates=[_candidate("c1", gain=-1.0)])
        is False
    )
    # Budget mostly consumed -> not premature.
    assert detect_premature_stop(rounds_used=10, elapsed=6000.0, state=state) is False


def test_roi_score_prefers_gain_and_penalizes_risk() -> None:
    good = _candidate("good", gain=1.0)
    risky = ExperimentCandidate(candidate_id="risky", expected_gain=1.0, validation_risk=2.0)
    assert roi_score(good) > roi_score(risky)


# ---------------------------------------------------------------- hierarchy


def _build_hierarchy() -> tuple[ProblemHierarchy, ProblemNode, ProblemNode]:
    hierarchy = ProblemHierarchy()
    task = ProblemNode(level=0, task="A1")
    hierarchy.add_node(task)
    stage = ProblemNode(level=1, task="A1", pipeline_stage="model")
    hierarchy.add_node(stage, parent_id=task.node_id)
    low = ProblemNode(
        level=2,
        task="A1",
        pipeline_stage="model",
        bucket="underfit",
        headroom=0.1,
        expected_information_gain=0.5,
        compute_cost=2.0,
        open_hypotheses=["h_low"],
    )
    high = ProblemNode(
        level=2,
        task="A1",
        pipeline_stage="model",
        bucket="overfit",
        headroom=0.5,
        expected_information_gain=1.0,
        compute_cost=1.0,
        open_hypotheses=["h_high"],
    )
    hierarchy.add_node(low, parent_id=stage.node_id)
    hierarchy.add_node(high, parent_id=stage.node_id)
    return hierarchy, low, high


def test_problem_hierarchy_ranking_and_round_trip() -> None:
    hierarchy, low, high = _build_hierarchy()
    targets = hierarchy.select_targets()
    assert [node.node_id for node in targets] == [high.node_id, low.node_id]

    hierarchy.close_route(high.node_id, "h_high")
    hierarchy.open_hypothesis(high.node_id, "h_high_v2")
    assert "h_high" in hierarchy.nodes[high.node_id].closed_routes
    assert "h_high_v2" in hierarchy.nodes[high.node_id].open_hypotheses

    restored = ProblemHierarchy.from_dict(hierarchy.to_dict())
    assert restored.to_dict() == hierarchy.to_dict()


def test_problem_hierarchy_validates_level_order() -> None:
    hierarchy = ProblemHierarchy()
    task = ProblemNode(level=0, task="A1")
    hierarchy.add_node(task)
    with pytest.raises(ValueError):
        hierarchy.add_node(ProblemNode(level=2, task="A1", bucket="skip"), parent_id=task.node_id)
    with pytest.raises(ValueError):
        ProblemHierarchy().add_node(ProblemNode(level=1, task="A1", pipeline_stage="orphan"))


# ---------------------------------------------------------------- genome


def _base_genome() -> ModelGenome:
    return ModelGenome(layers={f"L{i}": f"desc_{i}" for i in range(10)})


def test_genome_delta_rejects_two_primary_changes() -> None:
    parent = _base_genome()
    child = ModelGenome(layers={**parent.layers, "L3": "new_feature", "L5": "new_model"})
    report = validate_genome_delta(parent, child)
    assert report["status"] == "violation"
    assert any("multiple_primary_changes" in v for v in report["violations"])


def test_genome_delta_accepts_one_primary_plus_safety() -> None:
    parent = _base_genome()
    child = ModelGenome(layers={**parent.layers, "L5": "new_model", "L8": "calibrated_threshold"})
    report = validate_genome_delta(parent, child)
    assert report["status"] == "ok"
    assert report["primary_changed_layers"] == ["L5"]
    assert report["safety_adjusted_layers"] == ["L8"]

    no_change = validate_genome_delta(parent, _base_genome())
    assert no_change["status"] == "violation"
    assert "no_primary_change" in no_change["violations"]


def test_genome_id_stable_from_layers() -> None:
    assert _base_genome().genome_id == _base_genome().genome_id
    assert _base_genome().genome_id != ModelGenome(layers={"L0": "x"}).genome_id


# ---------------------------------------------------------------- registry


def test_registry_availability_bias_with_lightgbm_missing() -> None:
    registry = default_registry()
    report = registry.select("gbdt")
    assert report.status == "selected"
    assert report.best_scientific_option == "gbdt_lightgbm"
    assert report.selected_option == "gbdt_hist_sklearn"
    assert report.availability_bias is True
    assert report.selection_gap > 0.0
    # The report never claims the fallback is scientifically optimal.
    assert report.selected_option != report.best_scientific_option


def test_registry_waiting_for_adapter_when_no_available_option() -> None:
    registry = default_registry()
    report = registry.select("gcn")
    assert report.status == "waiting_for_adapter"
    assert report.selected_option is None
    assert report.availability_bias is False


def test_registry_query_filters_by_family() -> None:
    registry = default_registry()
    retrieval = registry.query(family="retrieval_itemcf")
    assert len(retrieval) == 1 and retrieval[0].available is True
    missing = registry.records["ranker_lightgbm_lambdarank"]
    assert missing.implemented is False and missing.available is False
    assert missing.missing_dependency == ["lightgbm"]


# ---------------------------------------------------------------- controller


def test_exploration_controller_escalates_to_global_explore() -> None:
    history = [{"round": 1, "gain": 0.0}, {"round": 2, "gain": 0.0}]
    first = decide(history, {"mode": "exploit"}, None)
    assert first["mode"] == "adjacent_explore"
    assert "stagnation" in first["reasons"]
    second = decide(history, {"mode": first["mode"]}, None)
    assert second["mode"] == "global_explore"
    assert "reset_to_anchor" in second["actions"]


def test_exploration_controller_stays_exploit_when_progressing() -> None:
    history = [{"round": 1, "gain": 0.01}, {"round": 2, "gain": 0.02}]
    result = decide(history, {"mode": "exploit"}, None)
    assert result["mode"] == "exploit"
    assert result["reasons"] == []


# ---------------------------------------------------------------- MLE-STAR


def test_mlestar_enforces_stage_order() -> None:
    loop = MLESTARLoop()
    assert loop.current_stage() == MLESTAR_STAGES[0]
    with pytest.raises(ValueError):
        loop.advance({"stage": "component_ablation", "component": "x", "delta": -0.01})
    loop.advance({"notes": "strong baseline"})
    loop.advance({})
    with pytest.raises(ValueError):
        loop.advance({"component": "features"})  # missing delta
    loop.advance({"component": "features", "delta": -0.02})
    assert loop.current_stage() == "bottleneck_selection"
    plan = loop.plan_next()
    assert plan["stage"] == "bottleneck_selection"
    assert plan["required_inputs"] == ["ablation_results"]
    for _ in range(3):
        loop.advance({})
    assert loop.current_stage() is None
    assert loop.plan_next() == {"stage": None, "required_inputs": []}


# ---------------------------------------------------------------- cards


def test_card_retrieval_ranks_specific_above_generic() -> None:
    store = CardStore()
    generic = MethodCard(task_regime="tabular_classification", content={"name": "generic"})
    specific = CompetitionSolutionCard(
        task_regime="tabular_classification",
        error_mechanism="underfit",
        content={"name": "specific"},
    )
    store.add(generic)
    store.add(specific)
    results = store.retrieve(task_regime="tabular_classification", error_mechanism="underfit")
    assert [card.card_id for card in results] == [specific.card_id, generic.card_id]

    restored = CardStore.from_dict(store.to_dict())
    assert {c.card_id for c in restored.cards.values()} == {generic.card_id, specific.card_id}
