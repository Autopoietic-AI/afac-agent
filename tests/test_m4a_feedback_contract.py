from __future__ import annotations

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_execution(tool_name: str, metrics: dict) -> dict:
    return {
        "tool_name": tool_name,
        "adapter_id": tool_name,
        "adapter_version": "test_v1",
        "status": "completed",
        "target_problem": tool_name,
        "execution_mode": "audit",
        "read_only": True,
        "counts_as_experiment_round": False,
        "mutates_predictions": False,
        "mutates_project_state": False,
        "requires_gpu": False,
        "identity_hash": "abc123",
        "started_at": "2030-01-01T00:00:00+00:00",
        "finished_at": "2030-01-01T00:00:01+00:00",
        "duration_seconds": 1.0,
        "command": [],
        "returncode": 0,
        "input_manifest": {"some_abs_path": "C:/tmp/not-in-feedback"},
        "input_hashes": {"x": "hash"},
        "metrics": metrics,
        "bucket_metrics": [],
        "class_metrics": [],
        "artifacts": {"run_dir": "C:/tmp/not-in-feedback"},
        "stdout_log": "C:/tmp/stdout.log",
        "stderr_log": "C:/tmp/stderr.log",
        "warnings": [],
        "failure_reason": "",
        "missing_inputs": [],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_feedback(execution_result: Path, out_root: Path, *, dry_run: bool = False):
    from afac_agent.feedback.builder import FeedbackBuilder

    return FeedbackBuilder(project_root=PROJECT_ROOT).build(
        execution_result_path=execution_result,
        out_root=out_root,
        dry_run=dry_run,
    )


def test_feedback_schema_file_and_registered_normalizers() -> None:
    from afac_agent.feedback.normalizers import NORMALIZER_REGISTRY

    schema = json.loads((PROJECT_ROOT / "schemas" / "experiment_feedback.schema.json").read_text(encoding="utf-8"))
    assert "feedback_id" in schema["required"]
    assert schema["properties"]["recommendation"]["enum"] == [
        "accept_as_reference",
        "proceed_to_oof_evaluation",
        "proceed_to_candidate_generation",
        "proceed_to_training",
        "informational_only",
        "reject",
        "blocked",
    ]
    assert {
        "A1_V46A1_ISOLATED_AUDIT",
        "A1_V49A_EDGE_UTILITY_AUDIT",
        "A1_V53Q1_PATCH_REPLAY_SAFE",
        "A1_OOF_CANDIDATE_EVALUATOR",
    }.issubset(NORMALIZER_REGISTRY)


def test_missing_invalid_and_unregistered_execution_result(tmp_path: Path) -> None:
    missing = _build_feedback(tmp_path / "missing.json", tmp_path / "out")
    assert missing["status"] == "unavailable"
    assert missing["failure_reason"] == "execution_result_missing"

    invalid_path = _write_json(tmp_path / "invalid.json", {"tool_name": "A1_V46A1_ISOLATED_AUDIT"})
    invalid = _build_feedback(invalid_path, tmp_path / "out_invalid")
    assert invalid["status"] == "failed"
    assert invalid["failure_reason"] == "execution_result_schema_invalid"

    unknown_path = _write_json(
        tmp_path / "unknown.json",
        _base_execution("UNKNOWN_TOOL", {"x": 1}),
    )
    unknown = _build_feedback(unknown_path, tmp_path / "out_unknown")
    assert unknown["status"] == "blocked"
    assert unknown["failure_reason"] == "normalizer_not_registered"


def test_m3b_normalizer_missing_oof_does_not_emit_accuracy_or_zero(tmp_path: Path) -> None:
    execution = _base_execution(
        "A1_V46A1_ISOLATED_AUDIT",
        {
            "isolated_only_pass": True,
            "diff_count": 11,
            "isolated_diff_count": 11,
            "graph_visible_diff_count": 0,
            "oof_status": "unavailable",
            "oof_unavailable_reason": "missing_final_oof",
            "parent_identity_status": "verified",
        },
    )
    path = _write_json(tmp_path / "m3b.json", execution)
    result = _build_feedback(path, tmp_path / "反馈 输出 with space")

    assert result["status"] == "completed"
    feedback = json.loads(Path(result["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8"))
    assert feedback["feedback_kind"] == "expert_scope_audit"
    assert feedback["evaluation_tier"] == "expert_scope"
    assert feedback["metrics"]["oof_evaluation"] == {
        "status": "unavailable",
        "reason": "missing_final_oof",
    }
    assert "accuracy" not in json.dumps(feedback, ensure_ascii=False).lower()
    assert feedback["recommendation"] in {"informational_only", "proceed_to_oof_evaluation"}


def test_m3c_and_m3d_normalizers_and_replay_recommendations(tmp_path: Path) -> None:
    m3c_path = _write_json(
        tmp_path / "m3c.json",
        _base_execution(
            "A1_V49A_EDGE_UTILITY_AUDIT",
            {
                "oof_selected_count": 55,
                "test_selected_count": 8,
                "gate_config": {"minimum_support": 3},
                "gate_recomputed_pass": True,
                "champion_patches_covered_by_test_meta": 4,
                "champion_patches_selected": 4,
                "champion_patches_gate_allowed": 4,
                "test_truth_usage_pass": True,
            },
        ),
    )
    m3c = _build_feedback(m3c_path, tmp_path / "out_m3c")
    m3c_feedback = json.loads(Path(m3c["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8"))
    assert m3c_feedback["feedback_kind"] == "signal_audit"
    assert m3c_feedback["evaluation_tier"] == "signal_evidence"
    assert m3c_feedback["evidence"]["test_truth_usage_pass"] is True
    assert "model_score_gain" not in json.dumps(m3c_feedback, ensure_ascii=False)

    m3d_path = _write_json(
        tmp_path / "m3d.json",
        _base_execution(
            "A1_V53Q1_PATCH_REPLAY_SAFE",
            {
                "semantic_replay_pass": True,
                "byte_replay_pass": True,
                "base_to_replay_diff_count": 4,
                "replay_to_champion_differing_row_count": 0,
                "registered_as_champion": False,
                "submission_ready": False,
                "base_to_replay_diff": [{"test_idx": 1, "old_label": 0, "new_label": 1}],
            },
        ),
    )
    m3d = _build_feedback(m3d_path, tmp_path / "out_m3d")
    m3d_feedback = json.loads(Path(m3d["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8"))
    assert m3d_feedback["feedback_kind"] == "candidate_replay"
    assert m3d_feedback["evaluation_tier"] == "candidate_replay"
    assert m3d_feedback["recommendation"] == "accept_as_reference"
    assert m3d_feedback["candidate_changes"]["count"] == 4

    failed_replay = dict(m3d_path=json.loads(m3d_path.read_text(encoding="utf-8")))
    payload = failed_replay["m3d_path"]
    payload["metrics"]["byte_replay_pass"] = False
    payload["metrics"]["replay_to_champion_differing_row_count"] = 1
    reject_path = _write_json(tmp_path / "m3d_reject.json", payload)
    rejected = _build_feedback(reject_path, tmp_path / "out_reject")
    rejected_feedback = json.loads(Path(rejected["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8"))
    assert rejected_feedback["recommendation"] == "reject"


def test_feedback_id_deterministic_and_dry_run_writes_nothing(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "m3d.json",
        _base_execution(
            "A1_V53Q1_PATCH_REPLAY_SAFE",
            {
                "semantic_replay_pass": True,
                "byte_replay_pass": True,
                "base_to_replay_diff_count": 4,
                "replay_to_champion_differing_row_count": 0,
                "registered_as_champion": False,
                "submission_ready": False,
            },
        ),
    )
    dry = _build_feedback(path, tmp_path / "out", dry_run=True)
    assert dry["status"] == "dry_run"
    assert not (tmp_path / "out").exists()

    first = _build_feedback(path, tmp_path / "out")
    second = _build_feedback(path, tmp_path / "out")
    assert second["status"] == "duplicate"
    assert first["feedback_id"] == second["feedback_id"]
    assert first["core_feedback_hash"] == second["core_feedback_hash"]
    feedback_text = Path(first["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8")
    assert "2030-01-01" not in feedback_text
    assert "C:/tmp" not in feedback_text


def test_cli_build_feedback_and_protected_files_unchanged(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "m3c.json",
        _base_execution(
            "A1_V49A_EDGE_UTILITY_AUDIT",
            {
                "oof_selected_count": 55,
                "test_selected_count": 8,
                "gate_recomputed_pass": True,
                "champion_patches_covered_by_test_meta": 4,
                "champion_patches_selected": 4,
                "champion_patches_gate_allowed": 4,
                "test_truth_usage_pass": True,
            },
        ),
    )
    frozen_before = {p: _sha256(p) for p in [CHAMPION, PROJECT_STATE, HISTORY]}
    command = [
        sys.executable,
        "-m",
        "afac_agent.main",
        "build-feedback",
        "--project_root",
        str(PROJECT_ROOT),
        "--execution-result",
        str(path),
        "--out-root",
        str(tmp_path / "out cli 中文"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "completed"
    assert {p: _sha256(p) for p in frozen_before} == frozen_before
    assert json.loads(PROJECT_STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == 0
    run_dir = Path(result["artifacts"]["run_dir"])
    assert not list(run_dir.rglob("*.csv"))
    assert not list(run_dir.rglob("submission*.zip"))
    assert not list(run_dir.rglob("*.npz"))


@pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "adapter_runs").exists(),
    reason="local adapter run artifacts are not available",
)
def test_real_m3b_m3c_m3d_feedback_smoke(tmp_path: Path) -> None:
    from afac_agent.feedback.builder import FeedbackBuilder

    builder = FeedbackBuilder(project_root=PROJECT_ROOT)
    wanted = {
        "A1_V46A1_ISOLATED_AUDIT": "expert_scope_audit",
        "A1_V49A_EDGE_UTILITY_AUDIT": "signal_audit",
        "A1_V53Q1_PATCH_REPLAY_SAFE": "candidate_replay",
    }
    found = {}
    for path in (PROJECT_ROOT / "artifacts" / "adapter_runs").rglob("execution_result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tool = payload.get("tool_name")
        if tool in wanted and payload.get("status") == "completed":
            found.setdefault(tool, path)
    missing = sorted(set(wanted) - set(found))
    if missing:
        pytest.skip(f"missing real execution results: {missing}")

    for tool, kind in wanted.items():
        first = builder.build(
            execution_result_path=found[tool],
            out_root=tmp_path / "real feedback 中文 with space",
        )
        second = builder.build(
            execution_result_path=found[tool],
            out_root=tmp_path / "real feedback 中文 with space",
        )
        assert first["status"] in {"completed", "duplicate"}
        assert second["status"] == "duplicate"
        assert first["feedback_id"] == second["feedback_id"]
        feedback = json.loads(Path(first["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8"))
        assert feedback["feedback_kind"] == kind
        if tool == "A1_V53Q1_PATCH_REPLAY_SAFE":
            assert feedback["recommendation"] == "accept_as_reference"
