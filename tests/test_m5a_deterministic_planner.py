from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT


CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
PROJECT_STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"
TOOL_REGISTRY = PROJECT_ROOT / "config" / "tool_registry.json"
POLICY = PROJECT_ROOT / "config" / "planner_policy.json"
PROBLEM_MAP = PROJECT_ROOT / "artifacts" / "data_profile" / "a1_m2_v1" / "a1_problem_map.json"
FEEDBACKS = sorted((PROJECT_ROOT / "artifacts" / "feedback_runs").rglob("experiment_feedback.json"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _minimal_problem_map(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "problem_map.json",
        {
            "analysis_tier": "dataset_only",
            "problems": [
                {
                    "problem_id": "structure::no_visible_train_within_4_hops",
                    "scope": "no_visible_train_within_4_hops",
                    "node_count": 10,
                    "structure_evidence": "observed",
                    "signal_status": {
                        "canonical_fold": "missing",
                        "v53q1_anchor_oof": "missing",
                    },
                }
            ],
            "rankings": {
                "train_test_shift": {
                    "status": "observed",
                    "unavailable": [
                        {
                            "metric": "prediction_probability_shift",
                            "reason": "missing_anchor_oof",
                        }
                    ],
                }
            },
        },
    )


def _feedback(tool: str, *, feedback_id: str | None = None) -> dict:
    base = {
        "feedback_version": "m4a_v1",
        "feedback_id": feedback_id or f"{tool}_feedback",
        "feedback_kind": "signal_audit",
        "evaluation_tier": "signal_evidence",
        "task": "A1",
        "tool_name": tool,
        "adapter_id": tool,
        "adapter_version": "test_v1",
        "execution_identity_hash": f"{tool}_identity",
        "execution_result_ref": "execution_result.json",
        "execution_result_hash": f"{tool}_result_hash",
        "status": "completed",
        "validity": {
            "status": "valid",
            "audit_success_is_model_gain": False,
            "replay_success_is_new_experiment": False,
        },
        "evidence": {},
        "metrics": {},
        "class_metrics": {"status": "unavailable"},
        "bucket_metrics": {"status": "unavailable"},
        "candidate_changes": {"status": "unavailable"},
        "overlap_conflict": {"status": "unavailable"},
        "runtime": {"duration_seconds": 0, "returncode": 0},
        "resource_usage": {"counts_as_experiment_round": False, "requires_gpu": False},
        "safety": {
            "read_only": True,
            "mutates_predictions": False,
            "mutates_project_state": False,
            "submission_creating": False,
            "test_truth_metrics_emitted": False,
        },
        "limitations": [],
        "warnings": [],
        "recommendation": "informational_only",
    }
    return base


def _synthetic_inputs(tmp_path: Path) -> dict:
    problem = _minimal_problem_map(tmp_path)
    replay = _feedback("A1_V53Q1_PATCH_REPLAY_SAFE")
    replay.update(
        {
            "feedback_kind": "candidate_replay",
            "evaluation_tier": "candidate_replay",
            "evidence": {
                "semantic_replay_pass": True,
                "byte_replay_pass": True,
                "replay_to_champion_diff_count": 0,
            },
            "candidate_changes": {"status": "observed", "count": 4},
            "recommendation": "accept_as_reference",
            "safety": {
                **replay["safety"],
                "read_only": False,
                "mutates_predictions": True,
                "submission_ready": False,
                "registered_as_champion": False,
            },
        }
    )
    signal = _feedback("A1_V49A_EDGE_UTILITY_AUDIT")
    signal.update(
        {
            "feedback_kind": "signal_audit",
            "evaluation_tier": "signal_evidence",
            "evidence": {
                "gate_recomputed_pass": True,
                "oof_selected_count": 55,
                "test_selected_count": 8,
                "champion_patch_coverage": {"selected": 4, "covered_by_test_meta": 4, "gate_allowed": 4},
                "test_truth_usage_pass": True,
            },
            "recommendation": "proceed_to_candidate_generation",
        }
    )
    scope = _feedback("A1_V46A1_ISOLATED_AUDIT")
    scope.update(
        {
            "feedback_kind": "expert_scope_audit",
            "evaluation_tier": "expert_scope",
            "evidence": {
                "isolated_only_pass": True,
                "diff_count": 44,
                "graph_visible_diff_count": 0,
                "oof_status": {"status": "unavailable", "reason": "missing_candidate_oof_npz"},
            },
            "metrics": {
                "oof_evaluation": {"status": "unavailable", "reason": "missing_candidate_oof_npz"}
            },
        }
    )
    oof = _feedback("A1_OOF_CANDIDATE_EVALUATOR")
    oof.update(
        {
            "feedback_kind": "model_experiment",
            "evaluation_tier": "oof_comparison",
            "evidence": {
                "analysis_tier": "self_contained_oof_comparison",
                "comparison_scope": "embedded_parent_unverified",
                "parent_identity_status": "unverified",
                "test_truth_used": False,
            },
            "metrics": {
                "overall": {"status": "observed", "gain": -0.01},
                "macro": {"status": "observed", "gain": -0.02},
                "rescue_damage": {"status": "observed", "rescue": 8, "damage": 15, "net": -7},
                "oracle": {"status": "observed", "oracle_gain": 0.1},
            },
        }
    )
    feedback_paths = [
        _write_json(tmp_path / "feedback" / "replay.json", replay),
        _write_json(tmp_path / "feedback" / "signal.json", signal),
        _write_json(tmp_path / "feedback" / "scope.json", scope),
        _write_json(tmp_path / "feedback" / "oof.json", oof),
    ]
    return {
        "problem_map": problem,
        "feedbacks": feedback_paths,
        "tool_registry": TOOL_REGISTRY,
        "project_state": PROJECT_STATE,
        "history": HISTORY,
        "policy": POLICY,
    }


def _plan(inputs: dict, out_root: Path):
    from afac_agent.planning.deterministic_planner import DeterministicPlanner

    return DeterministicPlanner(project_root=PROJECT_ROOT).plan(
        problem_map_path=inputs["problem_map"],
        feedback_paths=inputs["feedbacks"],
        tool_registry_path=inputs["tool_registry"],
        project_state_path=inputs["project_state"],
        history_path=inputs["history"],
        policy_path=inputs["policy"],
        out_root=out_root,
    )


def test_plan_id_deterministic_and_changes_with_input_hash(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    first = _plan(inputs, tmp_path / "plans 中文 with space")
    second = _plan(inputs, tmp_path / "plans 中文 with space")
    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    assert first["plan_id"] == second["plan_id"]
    assert first["core_plan_hash"] == second["core_plan_hash"]

    problem_payload = json.loads(inputs["problem_map"].read_text(encoding="utf-8"))
    problem_payload["problems"][0]["node_count"] = 11
    changed = _write_json(tmp_path / "changed_problem.json", problem_payload)
    changed_inputs = {**inputs, "problem_map": changed}
    third = _plan(changed_inputs, tmp_path / "other_plans")
    assert third["plan_id"] != first["plan_id"]


def test_missing_and_bad_inputs_are_reported_without_artifact(tmp_path: Path) -> None:
    from afac_agent.planning.deterministic_planner import DeterministicPlanner

    inputs = _synthetic_inputs(tmp_path)
    planner = DeterministicPlanner(project_root=PROJECT_ROOT)
    missing_problem = planner.plan(
        problem_map_path=tmp_path / "missing.json",
        feedback_paths=inputs["feedbacks"],
        tool_registry_path=inputs["tool_registry"],
        project_state_path=inputs["project_state"],
        history_path=inputs["history"],
        policy_path=inputs["policy"],
        out_root=tmp_path / "plans",
    )
    assert missing_problem["status"] == "waiting_for_input"
    assert "problem_map" in missing_problem["missing_inputs"]

    no_feedback = planner.plan(
        problem_map_path=inputs["problem_map"],
        feedback_paths=[],
        tool_registry_path=inputs["tool_registry"],
        project_state_path=inputs["project_state"],
        history_path=inputs["history"],
        policy_path=inputs["policy"],
        out_root=tmp_path / "plans2",
    )
    assert no_feedback["status"] == "waiting_for_input"
    assert "feedback" in no_feedback["missing_inputs"]

    bad_feedback = _write_json(tmp_path / "bad_feedback.json", {"tool_name": "bad"})
    bad = planner.plan(
        problem_map_path=inputs["problem_map"],
        feedback_paths=[bad_feedback],
        tool_registry_path=inputs["tool_registry"],
        project_state_path=inputs["project_state"],
        history_path=inputs["history"],
        policy_path=inputs["policy"],
        out_root=tmp_path / "plans3",
    )
    assert bad["status"] == "failed"
    assert bad["failure_reason"] == "input_validation_failed"


def test_real_feedback_rules_selected_missing_input_and_blocks(tmp_path: Path) -> None:
    assert FEEDBACKS, "real feedback artifacts should be available from M4"
    inputs = {
        "problem_map": PROBLEM_MAP,
        "feedbacks": FEEDBACKS,
        "tool_registry": TOOL_REGISTRY,
        "project_state": PROJECT_STATE,
        "history": HISTORY,
        "policy": POLICY,
    }
    result = _plan(inputs, tmp_path / "plans")
    decision = result["plan_decision"]

    assert decision["status"] == "waiting_for_input"
    assert decision["primary_problem"] == "missing_full_anchor_evaluation_inputs"
    assert decision["selected_action"] == "request_missing_input"
    assert decision["selected_tool"] is None
    assert {
        "canonical_fold_assignment",
        "final_v53q1_oof_proba",
        "final_v53q1_oof_identity_or_manifest",
    }.issubset(set(decision["missing_inputs"]))
    assert decision["requires_human_approval"] is False
    assert decision["auto_execution_allowed"] is False
    assert "candidate_already_materialized" in decision["reason_codes"]
    assert "reference_already_verified" in decision["reason_codes"]
    assert "negative_oof_gain" in decision["reason_codes"]
    assert "negative_net" in decision["reason_codes"]
    assert "unverified_parent" in decision["reason_codes"]
    assert "missing_policy" in decision["reason_codes"]
    blocked_reasons = {
        reason
        for action in decision["blocked_actions"]
        for reason in action.get("reason_codes", [])
    }
    assert "adapter_unbound" in blocked_reasons
    assert "branch_closed" in blocked_reasons
    assert "submission_auto_forbidden" in blocked_reasons
    deferred_reasons = {
        reason
        for action in decision["deferred_actions"]
        for reason in action.get("reason_codes", [])
    }
    assert "candidate_already_materialized" in deferred_reasons
    assert "reference_already_verified" in deferred_reasons


def test_negative_oof_oracle_gain_does_not_promote_candidate(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    result = _plan(inputs, tmp_path / "plans")
    decision = result["plan_decision"]
    assert "negative_oof_gain" in decision["reason_codes"]
    assert "negative_net" in decision["reason_codes"]
    assert "unverified_parent" in decision["reason_codes"]
    assert not any(
        action["action"] == "generate_candidate"
        and action["status"] == "ranked"
        for action in decision["ranked_actions"]
    )


def test_policy_safety_and_budget_exhaustion(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    policy_payload = json.loads(POLICY.read_text(encoding="utf-8"))
    policy_payload["human_approval_policy"]["online_submission_requires_human_approval"] = False
    unsafe_policy = _write_json(tmp_path / "unsafe_policy.json", policy_payload)
    unsafe = {**inputs, "policy": unsafe_policy}
    from afac_agent.planning.policy import load_policy

    with pytest.raises(ValueError, match="online_submission_requires_human_approval"):
        load_policy(unsafe_policy)

    state = json.loads(PROJECT_STATE.read_text(encoding="utf-8"))
    state["budget"]["rounds_used"] = state["budget"]["max_rounds"]
    exhausted_state = _write_json(tmp_path / "state.json", state)
    exhausted = _plan({**inputs, "project_state": exhausted_state}, tmp_path / "plans")
    assert exhausted["plan_decision"]["status"] == "budget_exhausted"
    assert "budget_blocked" in exhausted["plan_decision"]["reason_codes"]


def test_cli_dry_run_writes_nothing_and_normal_run_preserves_frozen_files(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    frozen_before = {p: _sha256(p) for p in [CHAMPION, PROJECT_STATE, HISTORY]}
    command = [
        sys.executable,
        "-m",
        "afac_agent.main",
        "plan-next",
        "--project_root",
        str(PROJECT_ROOT),
        "--problem-map",
        str(inputs["problem_map"]),
        "--tool-registry",
        str(inputs["tool_registry"]),
        "--project-state",
        str(inputs["project_state"]),
        "--history",
        str(inputs["history"]),
        "--policy",
        str(inputs["policy"]),
        "--out-root",
        str(tmp_path / "plans cli 中文 with space"),
    ]
    for feedback in inputs["feedbacks"]:
        command.extend(["--feedback", str(feedback)])
    dry = subprocess.run([*command, "--dry-run"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert dry.returncode == 0, dry.stderr
    dry_result = json.loads(dry.stdout)
    assert dry_result["status"] == "dry_run"
    assert not (tmp_path / "plans cli 中文 with space").exists()

    normal = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert normal.returncode == 0, normal.stderr
    result = json.loads(normal.stdout)
    assert result["status"] == "completed"
    run_dir = Path(result["artifacts"]["run_dir"])
    assert (run_dir / "plan_decision.json").exists()
    assert not list(run_dir.rglob("A1*.csv"))
    assert not list(run_dir.rglob("submission*.zip"))
    assert not list(run_dir.rglob("*.npz"))
    assert {p: _sha256(p) for p in frozen_before} == frozen_before
    assert json.loads(PROJECT_STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == 0


def test_unknown_feedback_tool_and_missing_input_actions(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    unknown = _feedback("UNKNOWN_TOOL")
    unknown_path = _write_json(tmp_path / "unknown.json", unknown)
    result = _plan({**inputs, "feedbacks": [unknown_path]}, tmp_path / "plans")
    decision = result["plan_decision"]
    assert "unregistered_tool" in {
        reason
        for action in decision["blocked_actions"]
        for reason in action.get("reason_codes", [])
    }
    assert any(action["action"] == "request_missing_input" for action in decision["ranked_actions"])
