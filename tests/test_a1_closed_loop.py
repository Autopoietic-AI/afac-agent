# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from afac_agent.a1_closed_loop import run_a1_closed_loop


def _one_hot(labels: np.ndarray, wrong_mask: np.ndarray | None = None) -> np.ndarray:
    pred = labels.copy()
    if wrong_mask is not None:
        pred[wrong_mask] = (pred[wrong_mask] + 1) % 10
    proba = np.full((len(labels), 10), 0.001, dtype=np.float64)
    proba[np.arange(len(labels)), pred] = 0.991
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype(np.float32)


def _policy() -> dict[str, object]:
    return {
        "analysis_levels": ["global", "bucket", "bucket_class", "cross_bucket_class", "error_mechanism"],
        "bucket_taxonomy": ["isolated", "one_hop_available", "exact2_only", "exact3_4_only", "graph_visible"],
        "mechanism_taxonomy": ["information_source_missing", "neighbor_unreliability", "uniform_smoothing_damage", "multi_hop_signal_opportunity"],
        "priority_weights": {"affected_count": 0.35, "evidence_strength": 0.3, "method_availability": 0.25, "risk": -0.1},
        "max_global_deep_research": 1,
        "max_bucket_deep_research": 2,
        "max_bucket_class_deep_research": 2,
        "allow_internal_model_knowledge_as_source": False,
        "allow_unverified_method_promotion": False,
        "allow_automatic_branch_reopen": False,
        "allow_automatic_experiment_execution": False,
    }


def _bundle(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    anchor_dir = project / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1"
    config = project / "config"
    history = project / "history"
    anchor_dir.mkdir(parents=True)
    config.mkdir(parents=True)
    history.mkdir(parents=True)
    n = 11001
    train_idx = np.arange(n, dtype=np.int64)
    labels = train_idx % 10
    fold = train_idx % 5
    isolated = np.zeros(n, dtype=bool)
    isolated[:2213] = True
    connectivity = np.where(isolated, "isolated", "graph_visible")
    reachability = np.where(isolated, "no_visible_train_within_4_hops", "one_hop_available")
    degree_band = np.where(isolated, "isolated", "degree_2_5")
    base = _one_hot(labels, wrong_mask=isolated)
    expert = _one_hot(labels)
    composed = base.copy()
    composed[isolated] = expert[isolated]
    pred = composed.argmax(axis=1).astype(np.int64)
    np.savez(
        anchor_dir / "A1_EVAL_ANCHOR_V1_oof.npz",
        train_idx=train_idx,
        labels=labels,
        proba=composed,
        pred=pred,
        fold=fold,
        connectivity_visibility=connectivity,
        train_label_reachability=reachability,
        degree_band=degree_band,
        isolated_mask=isolated,
    )
    (anchor_dir / "A1_EVAL_ANCHOR_V1_manifest.json").write_text(
        json.dumps({"evaluation_anchor_identity": "A1_EVAL_ANCHOR_V1", "status": "materialized_from_verified_components"}),
        encoding="utf-8",
    )
    v43 = tmp_path / "correct_smooth_v1_oof_proba.npz"
    np.savez(v43, proba=base, base_proba=base, train_idx=train_idx, labels=labels)
    v46 = tmp_path / "two_seed_balanced_candidate_oof.npz"
    np.savez(v46, proba=composed, expert_proba=expert, train_idx=train_idx, labels=labels, isolated_mask=isolated)
    (config / "project_state.json").write_text(json.dumps({"task": "A1", "online_version": "v53Q-1", "online_score": 0.78, "closed_branches": ["generic_confidence_router"]}), encoding="utf-8")
    (config / "tool_registry.json").write_text(json.dumps({"tools": [{"name": "fusion-controller", "adapter_bound": True}]}), encoding="utf-8")
    (config / "research_policy.json").write_text(json.dumps(_policy()), encoding="utf-8")
    (history / "confirmed_experiments_a1.json").write_text(json.dumps({"experiments": []}), encoding="utf-8")
    return {"project": project, "anchor_dir": anchor_dir, "v43": v43, "v46": v46}


def _run(paths: dict[str, Path], **kwargs: object) -> dict[str, object]:
    return run_a1_closed_loop(
        project_root=paths["project"],
        anchor_dir=paths["anchor_dir"],
        v43c_oof=paths["v43"],
        v46a_oof=paths["v46"],
        out_root=paths["project"] / "artifacts" / "a1_closed_loop_runs",
        **kwargs,
    )


def test_a1_closed_loop_real_state_machine_and_safety(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    before_state = (paths["project"] / "config" / "project_state.json").read_text(encoding="utf-8")
    result = _run(paths)
    assert result["status"] == "completed"
    assert result["scientific_rounds_used"] == 1
    run_dir = (paths["project"] / result["artifacts"]["run_manifest"].replace("/", "\\")).parent
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_dir / "trajectory_A1.json").read_text(encoding="utf-8"))
    assert manifest["uses_test_truth"] is False
    assert manifest["creates_submission"] is False
    assert manifest["generates_test_prediction"] is False
    assert manifest["online_champion_mutated"] is False
    assert manifest["scientific_rounds_used"] == 1
    assert trajectory["rounds"][0]["m5_admission"]["status"] == "admitted"
    assert trajectory["rounds"][0]["execution_status"] == "completed"
    assert trajectory["rounds"][1]["execution_status"] == "not_started_due_to_stop"
    assert (paths["project"] / "config" / "project_state.json").read_text(encoding="utf-8") == before_state


def test_a1_closed_loop_waiting_for_input_consumes_no_round(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    paths["v43"].unlink()
    result = _run(paths)
    assert result["status"] == "waiting_for_input"
    assert "v43c_oof" in result["missing_inputs"]
    assert result["artifacts"] == {}


def test_a1_closed_loop_resume_does_not_repeat_round(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    first = _run(paths)
    second = _run(paths)
    assert first["run_id"] == second["run_id"]
    assert second["status"] == "completed"
    run_dir = (paths["project"] / first["artifacts"]["run_manifest"].replace("/", "\\")).parent
    memory = (run_dir / "research_memory" / "research_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([line for line in memory if line.strip()]) == 1


def test_a1_closed_loop_best_candidate_and_macro_protection(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    result = _run(paths)
    run_dir = (paths["project"] / result["artifacts"]["run_manifest"].replace("/", "\\")).parent
    registry = json.loads((run_dir / "best_candidate_registry.json").read_text(encoding="utf-8"))
    portfolio = registry["portfolio_candidates"][0]
    assert portfolio["status"] in {"accepted_portfolio", "promoted_best", "rejected"}
    if portfolio["macro_gain"] is not None and portfolio["macro_gain"] < 0:
        assert portfolio["status"] in {"accepted_portfolio", "rejected"}
        assert registry["best_candidate"]["candidate_id"] == "A1_EVAL_ANCHOR_V1"


def test_a1_closed_loop_report_package_and_no_candidate_csv(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    result = _run(paths)
    run_dir = (paths["project"] / result["artifacts"]["run_manifest"].replace("/", "\\")).parent
    assert (run_dir / "REPORT_PACKAGE" / "trajectory_A1.json").exists()
    assert (run_dir / "round_01" / "round_state.json").exists()
    assert (run_dir / "round_02" / "round_state.json").exists()
    assert (run_dir / "round_03" / "round_state.json").exists()
    assert not (run_dir / "candidate_A1.csv").exists()
