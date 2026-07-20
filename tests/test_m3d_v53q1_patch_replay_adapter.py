from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from afac_agent.adapters.runner import AdapterRunner
from afac_agent.registry import ToolRegistry
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


def _write_prediction_csv(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["test_idx", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _write_patch_script(path: Path, *, gate_precision: str = "2.0 / 3.0") -> None:
    path.write_text(
        f"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

MODEL_NAME = "confidence_plus_edge"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--oof_meta_csv", required=True)
    parser.add_argument("--test_meta_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_name", default="replay_candidate.csv")
    parser.add_argument("--minimum_support", type=int, default=3)
    parser.add_argument("--minimum_precision", type=float, default={gate_precision})
    parser.add_argument("--minimum_net", type=int, default=1)
    parser.add_argument("--minimum_folds", type=int, default=2)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_csv)
    oof = pd.read_csv(args.oof_meta_csv)
    test = pd.read_csv(args.test_meta_csv)
    selected_oof = oof[(oof["model_name"] == MODEL_NAME) & (oof["selected"].astype(bool))].copy()
    selected_oof["transition"] = selected_oof["base_pred"].astype(int).astype(str) + "->" + selected_oof["h2_pred"].astype(int).astype(str)
    allowed = []
    summary_rows = []
    for transition, group in selected_oof.groupby("transition"):
        rescue = int(group["rescue"].sum())
        damage = int(group["damage"].sum())
        neutral = int(group["neutral"].sum())
        support = int(len(group))
        folds = int(group["fold"].nunique())
        decisive = rescue + damage
        precision = rescue / decisive if decisive else 0.0
        net = rescue - damage
        passed = support >= args.minimum_support and precision >= args.minimum_precision and net >= args.minimum_net and folds >= args.minimum_folds
        if passed:
            allowed.append(transition)
        summary_rows.append({{"transition": transition, "support": support, "rescue": rescue, "damage": damage, "neutral": neutral, "net": net, "decisive_precision": precision, "folds_observed": folds, "passed": passed}})
    selected_test = test[(test["model_name"] == MODEL_NAME) & (test["selected"].astype(bool))].copy()
    selected_test["transition"] = selected_test["base_pred"].astype(int).astype(str) + "->" + selected_test["h2_pred"].astype(int).astype(str)
    accepted = selected_test[selected_test["transition"].isin(set(allowed))]
    patch = {{int(row.global_idx): int(row.h2_pred) for row in accepted.itertuples()}}
    replay = base.copy()
    replay["label"] = [patch.get(int(idx), int(label)) for idx, label in zip(replay["test_idx"], replay["label"])]
    replay.to_csv(out_dir / args.output_name, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_dir / "v53q1_transition_oof_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{{"held_fold": 0, "changed": int(len(accepted))}}]).to_csv(out_dir / "v53q1_crossfit_fold_results.csv", index=False, encoding="utf-8-sig")
    audit = {{
        "transition_gate": {{
            "minimum_support": args.minimum_support,
            "minimum_precision": args.minimum_precision,
            "minimum_net": args.minimum_net,
            "minimum_folds": args.minimum_folds,
        }},
        "allowed_transitions": sorted(allowed),
        "test_changes": int(len(accepted)),
        "test_patch": [
            {{"test_idx": int(row.global_idx), "base_label": int(row.base_pred), "patched_label": int(row.h2_pred), "transition": row.transition}}
            for row in accepted.sort_values("global_idx").itertuples()
        ],
        "test_labels_used": False,
        "submission_created": False,
    }}
    (out_dir / "v53q1_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )


def _write_bundle(tmp_path: Path, *, gate_precision: str = "2.0 / 3.0") -> dict[str, str]:
    input_dir = tmp_path / "输入 replay with space"
    input_dir.mkdir(parents=True, exist_ok=True)
    base_csv = input_dir / "v46_base.csv"
    champion_csv = input_dir / "champion.csv"
    oof = input_dir / "v49a_oof_meta_scores.csv"
    test = input_dir / "v49a_test_meta_scores.csv"
    patch_py = input_dir / "patch.py"
    _write_prediction_csv(
        base_csv,
        [
            {"test_idx": 6, "label": 0},
            {"test_idx": 7, "label": 2},
            {"test_idx": 8, "label": 1},
            {"test_idx": 9, "label": 4},
        ],
    )
    _write_prediction_csv(
        champion_csv,
        [
            {"test_idx": 6, "label": 1},
            {"test_idx": 7, "label": 2},
            {"test_idx": 8, "label": 1},
            {"test_idx": 9, "label": 4},
        ],
    )
    pd.DataFrame(
        [
            {"fold": 0, "global_idx": 0, "true_label": 1, "base_correct": 0, "h2_correct": 1, "rescue": 1, "damage": 0, "neutral": 0, "target_rescue": 1, "base_pred": 0, "h2_pred": 1, "model_name": "confidence_plus_edge", "rescue_score": 0.9, "selected": True},
            {"fold": 1, "global_idx": 1, "true_label": 1, "base_correct": 0, "h2_correct": 1, "rescue": 1, "damage": 0, "neutral": 0, "target_rescue": 1, "base_pred": 0, "h2_pred": 1, "model_name": "confidence_plus_edge", "rescue_score": 0.8, "selected": True},
            {"fold": 2, "global_idx": 2, "true_label": 0, "base_correct": 1, "h2_correct": 0, "rescue": 0, "damage": 1, "neutral": 0, "target_rescue": 0, "base_pred": 0, "h2_pred": 1, "model_name": "confidence_plus_edge", "rescue_score": 0.7, "selected": True},
            {"fold": 3, "global_idx": 3, "true_label": 3, "base_correct": 0, "h2_correct": 1, "rescue": 1, "damage": 0, "neutral": 0, "target_rescue": 1, "base_pred": 2, "h2_pred": 3, "model_name": "confidence_plus_edge", "rescue_score": 0.7, "selected": True},
        ]
    ).to_csv(oof, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"global_idx": 6, "base_pred": 0, "h2_pred": 1, "model_name": "confidence_plus_edge", "rescue_score": 0.9, "selected": True},
            {"global_idx": 7, "base_pred": 2, "h2_pred": 3, "model_name": "confidence_plus_edge", "rescue_score": 0.7, "selected": True},
        ]
    ).to_csv(test, index=False, encoding="utf-8-sig")
    _write_patch_script(patch_py, gate_precision=gate_precision)
    audit_md = input_dir / "audit.md"
    audit_md.write_text("v53Q-1 synthetic replay audit\n", encoding="utf-8")
    return {
        "v53q1_base_csv": str(base_csv),
        "v49a_oof_meta_csv": str(oof),
        "v49a_test_meta_csv": str(test),
        "v53q1_patch_py": str(patch_py),
        "current_champion_csv": str(champion_csv),
        "v53q1_audit_md": str(audit_md),
    }


def _tool_registry(path: Path, *, output_root: Path) -> None:
    tool = {
        "name": "A1_V53Q1_PATCH_REPLAY_SAFE",
        "task": "A1",
        "layer": "champion_replay",
        "description": "Synthetic safe v53Q-1 patch replay adapter",
        "action_type": "replay",
        "expected_runtime_seconds": 30,
        "prediction_changing": True,
        "submission_creating": False,
        "read_only": False,
        "counts_as_experiment_round": False,
        "mutates_predictions": True,
        "mutates_project_state": False,
        "requires_gpu": False,
        "required_state": {},
        "required_inputs": {
            "v53q1_base_csv": {"kind": "file"},
            "v49a_oof_meta_csv": {"kind": "file"},
            "v49a_test_meta_csv": {"kind": "file"},
            "v53q1_patch_py": {"kind": "file"},
            "current_champion_csv": {"kind": "file"},
        },
        "forbidden_closed_branches": [],
        "command_template": [],
        "adapter_id": "A1_V53Q1_PATCH_REPLAY_SAFE",
        "adapter_version": "m3d_v1",
        "adapter_entrypoint": "afac_agent.adapters.a1_v53q1_patch_replay_safe:Adapter",
        "execution_mode": "replay",
        "result_schema": "schemas/adapter_execution_result.schema.json",
        "output_policy": {
            "output_root": str(output_root),
            "allow_overwrite": False,
            "forbidden_paths": [str(CHAMPION)],
            "frozen_gate_config": {
                "minimum_support": 3,
                "minimum_precision": 2.0 / 3.0,
                "minimum_net": 1,
                "minimum_folds": 2,
            },
        },
        "identity_hash_fields": [
            "tool_name",
            "adapter_id",
            "adapter_version",
            "execution_mode",
            "normalized_config",
            "input_hashes",
            "output_policy",
            "frozen_gate_config",
        ],
    }
    path.write_text(json.dumps({"tools": [tool]}, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_adapter(
    tmp_path: Path,
    variables: dict[str, str],
    *,
    output_root: Path,
    execute: bool = True,
):
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=output_root)
    tool = ToolRegistry(registry_path).get("A1_V53Q1_PATCH_REPLAY_SAFE")
    return AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=tool,
        variables={"root": str(PROJECT_ROOT), "python": sys.executable, **variables},
        execute=execute,
    )


def test_registry_loads_replay_adapter() -> None:
    from afac_agent.adapters.runner import ALLOWED_ADAPTER_ENTRYPOINTS

    tool = ToolRegistry(PROJECT_ROOT / "config" / "tool_registry.json").get(
        "A1_V53Q1_PATCH_REPLAY_SAFE"
    )
    assert tool.adapter_entrypoint == "afac_agent.adapters.a1_v53q1_patch_replay_safe:Adapter"
    assert tool.adapter_entrypoint in ALLOWED_ADAPTER_ENTRYPOINTS
    assert tool.read_only is False
    assert tool.mutates_predictions is True
    assert tool.counts_as_experiment_round is False
    assert tool.submission_creating is False


def test_prediction_permission_missing_is_blocked(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")
    assert result["status"] == "blocked"
    assert result["failure_reason"] == "prediction_artifact_permission_required"


def test_missing_inputs_wait_for_input(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    variables["v53q1_base_csv"] = str(tmp_path / "missing_base.csv")
    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")
    assert result["status"] == "waiting_for_input"
    assert "v53q1_base_csv" in result["missing_inputs"]


def test_gate_mismatch_and_unsafe_output_are_blocked_or_failed(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path, gate_precision="0.5")
    variables["allow_prediction_artifact"] = "true"
    mismatch = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs_mismatch")
    assert mismatch["status"] == "failed"
    assert mismatch["failure_reason"] == "input_validation_failed"

    variables = _write_bundle(tmp_path / "unsafe")
    variables["allow_prediction_artifact"] = "true"
    unsafe = _run_adapter(
        tmp_path / "unsafe",
        variables,
        output_root=Path(variables["current_champion_csv"]).parent,
    )
    assert unsafe["status"] == "blocked"


def test_replay_completed_duplicate_and_hashes(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    variables["allow_prediction_artifact"] = "true"
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
    first = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs 中文 with space")
    second = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs 中文 with space")

    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    assert first["identity_hash"] == second["identity_hash"]
    assert first["metrics"]["semantic_replay_pass"] is True
    assert first["metrics"]["byte_replay_pass"] is True
    assert first["metrics"]["base_to_replay_diff_count"] == 1
    assert first["metrics"]["replay_to_champion_differing_row_count"] == 0
    assert first["metrics"]["registered_as_champion"] is False
    assert first["metrics"]["submission_ready"] is False
    assert Path(first["artifacts"]["candidate_csv"]).exists()
    assert Path(first["artifacts"]["base_to_replay_diff"]).exists()
    assert Path(first["artifacts"]["non_applied_test_selected_audit"]).exists()
    assert {path: _sha256(path) for path in frozen_before} == frozen_before
    assert not list(Path(first["artifacts"]["run_dir"]).rglob("submission*.zip"))
    budget = json.loads(PROJECT_STATE.read_text(encoding="utf-8"))["budget"]
    assert budget["rounds_used"] == 0


def test_cli_replay_smoke(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=tmp_path / "runs")
    command = [
        sys.executable,
        "-m",
        "afac_agent.main",
        "run-adapter",
        "--project_root",
        str(PROJECT_ROOT),
        "--registry",
        str(registry_path),
        "--tool",
        "A1_V53Q1_PATCH_REPLAY_SAFE",
        "--v53q1_base_csv",
        variables["v53q1_base_csv"],
        "--v49a_oof_meta_csv",
        variables["v49a_oof_meta_csv"],
        "--v49a_test_meta_csv",
        variables["v49a_test_meta_csv"],
        "--v53q1_patch_py",
        variables["v53q1_patch_py"],
        "--current_champion_csv",
        variables["current_champion_csv"],
        "--allow-prediction-artifact",
        "--execute",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "completed"
    assert result["metrics"]["base_to_replay_diff_count"] == 1


@pytest.mark.skipif(
    not (
        Path.home()
        / "agent\u6bd4\u8d5b"
        / "model_pro"
        / "a1"
        / "new-a1"
        / "runs"
        / "v49a_task_relevant_edge_utility"
        / "v49a_oof_meta_scores.csv"
    ).exists(),
    reason="local v53Q-1 replay assets are not available",
)
def test_real_v53q1_patch_replay_safe_smoke(tmp_path: Path) -> None:
    race = Path.home() / "agent\u6bd4\u8d5b"
    run = race / "model_pro" / "a1" / "new-a1" / "runs" / "v49a_task_relevant_edge_utility"
    variables = {
        "v53q1_base_csv": str(
            race
            / "model_pro"
            / "a1"
            / "new-a1"
            / "a1_v46a1_tabm_seed_consensus"
            / "A1_v46a1_balanced_seed_consensus_SAFE.csv"
        ),
        "v49a_oof_meta_csv": str(run / "v49a_oof_meta_scores.csv"),
        "v49a_test_meta_csv": str(run / "v49a_test_meta_scores.csv"),
        "v53q1_patch_py": str(PROJECT_ROOT / "artifacts" / "a1_v53q1_transition_stable_edge_h2_patch.py"),
        "current_champion_csv": str(CHAMPION),
        "v53q1_audit_md": str(PROJECT_ROOT / "artifacts" / "V53Q1_TRANSITION_STABLE_EDGE_H2_AUDIT.md"),
        "allow_prediction_artifact": "true",
    }
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
    first = _run_adapter(tmp_path, variables, output_root=tmp_path / "真实 replay with space")
    second = _run_adapter(tmp_path, variables, output_root=tmp_path / "真实 replay with space")

    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    metrics = first["metrics"]
    assert metrics["base_to_replay_diff_count"] == 4
    assert metrics["replay_to_champion_differing_row_count"] == 0
    assert metrics["semantic_replay_pass"] is True
    assert metrics["byte_replay_pass"] is True
    assert metrics["candidate_sha256"] == metrics["champion_sha256"]
    assert metrics["final_patch_test_idx"] == [1879, 2489, 8190, 8499]
    assert metrics["non_applied_test_selected_count"] == 4
    assert {path: _sha256(path) for path in frozen_before} == frozen_before
