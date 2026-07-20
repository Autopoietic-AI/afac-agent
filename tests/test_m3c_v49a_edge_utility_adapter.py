from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from afac_agent.orchestrator import AgentOrchestrator
from afac_agent.schemas import BudgetState, ProjectState
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


def _write_a1_npz(path: Path) -> None:
    rows = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int32)
    cols = np.array([1, 2, 3, 4, 5, 0, 6, 7], dtype=np.int32)
    adj = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(10, 10))
    attr = sparse.eye(10, 3, dtype=np.float32, format="csr")
    labels = np.array([0, 1, 1, 0, 2, 3, -1, -1, -1, -1], dtype=np.int64)
    np.savez(
        path,
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape, dtype=np.int64),
        attr_data=attr.data,
        attr_indices=attr.indices,
        attr_indptr=attr.indptr,
        attr_shape=np.array(attr.shape, dtype=np.int64),
        labels=labels,
        train_idx=np.arange(6, dtype=np.int64),
        test_idx=np.array([6, 7, 8, 9], dtype=np.int64),
    )


def _write_prediction_csv(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["test_idx", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _base_oof_rows() -> list[dict[str, object]]:
    rows = [
        # Allowed transition 0->1: support=3, rescue=2, damage=1, net=1, folds=3.
        (0, 1, 1, 0, 1, 1, 0, 0, True),
        (1, 2, 1, 0, 1, 1, 0, 0, True),
        (2, 0, 0, 0, 1, 0, 1, 0, True),
        # Not allowed transition 2->3: support=1.
        (3, 5, 3, 2, 3, 1, 0, 0, True),
        # Unselected row should not affect gate.
        (4, 4, 2, 4, 5, 0, 0, 1, False),
    ]
    output: list[dict[str, object]] = []
    for fold, global_idx, true_label, base_pred, h2_pred, rescue, damage, neutral, selected in rows:
        output.append(
            {
                "fold": fold,
                "global_idx": global_idx,
                "true_label": true_label,
                "base_correct": int(base_pred == true_label),
                "h2_correct": int(h2_pred == true_label),
                "rescue": rescue,
                "damage": damage,
                "neutral": neutral,
                "target_rescue": rescue,
                "base_pred": base_pred,
                "h2_pred": h2_pred,
                "model_name": "confidence_plus_edge",
                "rescue_score": 0.8 if selected else 0.1,
                "selected": selected,
            }
        )
    return output


def _base_test_rows() -> list[dict[str, object]]:
    return [
        {
            "global_idx": 6,
            "base_pred": 0,
            "h2_pred": 1,
            "model_name": "confidence_plus_edge",
            "rescue_score": 0.9,
            "selected": True,
        },
        {
            "global_idx": 7,
            "base_pred": 2,
            "h2_pred": 3,
            "model_name": "confidence_plus_edge",
            "rescue_score": 0.7,
            "selected": True,
        },
        {
            "global_idx": 8,
            "base_pred": 1,
            "h2_pred": 2,
            "model_name": "confidence_plus_edge",
            "rescue_score": 0.1,
            "selected": False,
        },
    ]


def _write_bundle(tmp_path: Path) -> dict[str, str]:
    input_dir = tmp_path / "输入 v49 with space"
    input_dir.mkdir(parents=True, exist_ok=True)
    a1_npz = input_dir / "A1.npz"
    _write_a1_npz(a1_npz)
    oof = input_dir / "v49a_oof_meta_scores.csv"
    test = input_dir / "v49a_test_meta_scores.csv"
    pd.DataFrame(_base_oof_rows()).to_csv(oof, index=False, encoding="utf-8-sig")
    pd.DataFrame(_base_test_rows()).to_csv(test, index=False, encoding="utf-8-sig")
    base_csv = input_dir / "v46_base.csv"
    champion_csv = input_dir / "champion.csv"
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
    audit_md = input_dir / "V53Q1.md"
    audit_md.write_text(
        "\n".join(
            [
                "Frozen conditions:",
                "support >= 3",
                "precision >= 2/3",
                "net >= +1",
                "at least 2 folds",
                "`6`: `0 -> 1`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    patch_py = input_dir / "patch.py"
    patch_py.write_text(
        "parser.add_argument('--minimum_support', type=int, default=3)\n"
        "parser.add_argument('--minimum_precision', type=float, default=2.0 / 3.0)\n"
        "parser.add_argument('--minimum_net', type=int, default=1)\n"
        "parser.add_argument('--minimum_folds', type=int, default=2)\n",
        encoding="utf-8",
    )
    report = input_dir / "v49_report.md"
    report.write_text("Selected changes: `4`\nNo prediction was changed.\n", encoding="utf-8")
    config = input_dir / "v49_config.json"
    config.write_text(json.dumps({"test_labels_used": False}), encoding="utf-8")
    return {
        "a1_npz": str(a1_npz),
        "v49a_oof_meta_csv": str(oof),
        "v49a_test_meta_csv": str(test),
        "v46a1_base_csv": str(base_csv),
        "current_champion_csv": str(champion_csv),
        "v53q1_audit_md": str(audit_md),
        "v53q1_patch_source": str(patch_py),
        "v49a_report": str(report),
        "v49a_config": str(config),
    }


def _tool_registry(path: Path, *, output_root: Path) -> None:
    tool = {
        "name": "A1_V49A_EDGE_UTILITY_AUDIT",
        "task": "A1",
        "layer": "edge_utility_audit",
        "description": "Synthetic v49A edge utility audit adapter",
        "action_type": "audit",
        "expected_runtime_seconds": 30,
        "prediction_changing": False,
        "submission_creating": False,
        "read_only": True,
        "counts_as_experiment_round": False,
        "mutates_predictions": False,
        "mutates_project_state": False,
        "requires_gpu": False,
        "required_state": {},
        "required_inputs": {
            "a1_npz": {"kind": "file"},
            "v49a_oof_meta_csv": {"kind": "file"},
            "v49a_test_meta_csv": {"kind": "file"},
        },
        "forbidden_closed_branches": [],
        "command_template": [],
        "adapter_id": "A1_V49A_EDGE_UTILITY_AUDIT",
        "adapter_version": "m3c_v1",
        "adapter_entrypoint": "afac_agent.adapters.a1_v49a_edge_utility_audit:Adapter",
        "execution_mode": "audit",
        "result_schema": "schemas/adapter_execution_result.schema.json",
        "output_policy": {
            "output_root": str(output_root),
            "allow_overwrite": False,
            "forbidden_paths": [str(CHAMPION)],
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


def _run_adapter(tmp_path: Path, variables: dict[str, str], *, output_root: Path):
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=output_root)
    tool = ToolRegistry(registry_path).get("A1_V49A_EDGE_UTILITY_AUDIT")
    return AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=tool,
        variables={"root": str(PROJECT_ROOT), **variables},
        execute=True,
    )


def test_registry_loads_v49a_adapter() -> None:
    from afac_agent.adapters.runner import ALLOWED_ADAPTER_ENTRYPOINTS
    from afac_agent.registry import ToolRegistry

    tool = ToolRegistry(PROJECT_ROOT / "config" / "tool_registry.json").get(
        "A1_V49A_EDGE_UTILITY_AUDIT"
    )
    assert tool.adapter_entrypoint == "afac_agent.adapters.a1_v49a_edge_utility_audit:Adapter"
    assert tool.adapter_entrypoint in ALLOWED_ADAPTER_ENTRYPOINTS
    assert tool.read_only is True
    assert tool.counts_as_experiment_round is False


def test_missing_oof_or_test_meta_waits(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    variables["v49a_oof_meta_csv"] = str(tmp_path / "missing_oof.csv")
    missing_oof = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs1")
    assert missing_oof["status"] == "waiting_for_input"
    assert "v49a_oof_meta_csv" in missing_oof["missing_inputs"]

    variables = _write_bundle(tmp_path / "second")
    variables["v49a_test_meta_csv"] = str(tmp_path / "missing_test.csv")
    missing_test = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs2")
    assert missing_test["status"] == "waiting_for_input"
    assert "v49a_test_meta_csv" in missing_test["missing_inputs"]


def test_schema_idx_fold_label_selected_and_logic_failures(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    pd.DataFrame([{"bad": 1}]).to_csv(variables["v49a_oof_meta_csv"], index=False)
    bad_schema = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs_schema")
    assert bad_schema["status"] == "failed"

    cases = [
        ("oof_idx", "v49a_oof_meta_csv", "global_idx", 9),
        ("test_idx", "v49a_test_meta_csv", "global_idx", 0),
        ("fold", "v49a_oof_meta_csv", "fold", 99),
        ("label", "v49a_oof_meta_csv", "true_label", 9),
        ("selected", "v49a_oof_meta_csv", "selected", "maybe"),
    ]
    for name, key, column, value in cases:
        variables = _write_bundle(tmp_path / name)
        frame = pd.read_csv(variables[key])
        if column == "selected":
            frame[column] = frame[column].astype(object)
        frame.loc[0, column] = value
        frame.to_csv(variables[key], index=False, encoding="utf-8-sig")
        result = _run_adapter(tmp_path / name, variables, output_root=tmp_path / f"runs_{name}")
        assert result["status"] == "failed"

    variables = _write_bundle(tmp_path / "duplicate")
    frame = pd.read_csv(variables["v49a_oof_meta_csv"])
    frame.loc[1, "global_idx"] = frame.loc[0, "global_idx"]
    frame.loc[1, "model_name"] = frame.loc[0, "model_name"]
    frame.to_csv(variables["v49a_oof_meta_csv"], index=False, encoding="utf-8-sig")
    duplicate = _run_adapter(tmp_path / "duplicate", variables, output_root=tmp_path / "runs_duplicate")
    assert duplicate["status"] == "failed"

    variables = _write_bundle(tmp_path / "logic")
    frame = pd.read_csv(variables["v49a_oof_meta_csv"])
    frame.loc[0, ["rescue", "damage"]] = [1, 1]
    frame.to_csv(variables["v49a_oof_meta_csv"], index=False, encoding="utf-8-sig")
    bad_logic = _run_adapter(tmp_path / "logic", variables, output_root=tmp_path / "runs_logic")
    assert bad_logic["status"] == "failed"


def test_transition_gate_champion_relation_outputs_and_no_truth(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")

    assert result["status"] == "completed"
    metrics = result["metrics"]
    assert metrics["oof_selected_count"] == 4
    assert metrics["test_selected_count"] == 2
    assert metrics["transition_count"] == 2
    assert metrics["allowed_transition_count"] == 1
    assert metrics["gate_config"] == {
        "minimum_support": 3,
        "minimum_precision": pytest.approx(2 / 3),
        "minimum_net": 1,
        "minimum_folds": 2,
    }
    assert metrics["gate_recomputed_pass"] is True
    assert metrics["champion_patch_diff_count"] == 1
    assert metrics["champion_patches_covered_by_test_meta"] == 1
    assert metrics["champion_patches_selected"] == 1
    assert metrics["champion_patches_gate_allowed"] == 1
    assert metrics["non_champion_test_selected_count"] == 1
    assert metrics["test_truth_usage_pass"] is True
    assert metrics["edge_utility_integrity_pass"] is True
    assert metrics["counts_as_experiment_round"] is False if "counts_as_experiment_round" in metrics else True
    for artifact in [
        "transition_gate_summary",
        "oof_selected_audit",
        "test_selected_audit",
        "champion_patch_relation",
        "audit_details",
    ]:
        assert Path(result["artifacts"][artifact]).exists()
    transition_rows = list(csv.DictReader(Path(result["artifacts"]["transition_gate_summary"]).open(encoding="utf-8-sig")))
    assert {row["transition"]: row["allowed_by_gate"] for row in transition_rows}["0->1"] == "True"
    patch_rows = list(csv.DictReader(Path(result["artifacts"]["champion_patch_relation"]).open(encoding="utf-8-sig")))
    assert patch_rows[0]["test_idx"] == "6"
    assert patch_rows[0]["covered_by_test_meta"] == "True"
    forbidden = list(Path(result["artifacts"]["run_dir"]).glob("A1*.csv"))
    forbidden += list(Path(result["artifacts"]["run_dir"]).glob("submission*.zip"))
    forbidden += list(Path(result["artifacts"]["run_dir"]).glob("*.npz"))
    assert forbidden == []


def test_test_truth_column_warns_and_marks_usage_failed(tmp_path: Path) -> None:
    variables = _write_bundle(tmp_path)
    frame = pd.read_csv(variables["v49a_test_meta_csv"])
    frame["true_label"] = 0
    frame.to_csv(variables["v49a_test_meta_csv"], index=False, encoding="utf-8-sig")

    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")

    assert result["status"] == "completed"
    assert result["metrics"]["test_truth_usage_pass"] is False
    assert any("test meta contains truth-dependent columns" in warning for warning in result["warnings"])


def test_read_only_duplicate_identity_and_orchestrator_rounds(tmp_path: Path) -> None:
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    variables = _write_bundle(tmp_path)
    output_root = tmp_path / "runs 中文 with space"
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
    first = _run_adapter(tmp_path, variables, output_root=output_root)
    second = _run_adapter(tmp_path, variables, output_root=output_root)
    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    assert first["identity_hash"] == second["identity_hash"]
    assert {path: _sha256(path) for path in frozen_before} == frozen_before

    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=output_root)
    tool = ToolRegistry(registry_path).get("A1_V49A_EDGE_UTILITY_AUDIT")
    runner = AdapterRunner(project_root=PROJECT_ROOT)
    input_hashes = runner.hash_inputs(["a1_npz", "v49a_oof_meta_csv", "v49a_test_meta_csv"], variables)
    assert runner.compute_identity_hash(
        tool=tool,
        normalized_config={"started_at": "a", "stable": "x"},
        input_hashes=input_hashes,
        output_policy={"output_root": "volatile-a"},
        frozen_gate_config={},
    ) == runner.compute_identity_hash(
        tool=tool,
        normalized_config={"started_at": "b", "stable": "x"},
        input_hashes=input_hashes,
        output_policy={"output_root": "volatile-b"},
        frozen_gate_config={},
    )

    payload = json.loads(PROJECT_STATE.read_text(encoding="utf-8"))
    payload.update({"next_required_capability": "v49a_edge_utility_audit"})
    payload["budget"] = BudgetState(**payload["budget"])
    state = ProjectState(**payload)
    state_path = tmp_path / "状态 v49 with space" / "project_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    orchestrator = AgentOrchestrator(
        project_root=PROJECT_ROOT,
        state_path=state_path,
        registry_path=registry_path,
        trajectory_path=tmp_path / "trajectory.json",
    )
    outcome = orchestrator.run_once(
        variables={"root": str(PROJECT_ROOT), "python": sys.executable, **variables},
        execute=True,
    )
    assert outcome["result"]["status"] in {"completed", "duplicate"}
    assert outcome["state"]["budget"]["rounds_used"] == payload["budget"].rounds_used
    assert json.loads(state_path.read_text(encoding="utf-8")) == state.to_dict()


def test_cli_v49a_smoke(tmp_path: Path) -> None:
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
        "A1_V49A_EDGE_UTILITY_AUDIT",
        "--a1_npz",
        variables["a1_npz"],
        "--v49a_oof_meta_csv",
        variables["v49a_oof_meta_csv"],
        "--v49a_test_meta_csv",
        variables["v49a_test_meta_csv"],
        "--v46a1_base_csv",
        variables["v46a1_base_csv"],
        "--current_champion_csv",
        variables["current_champion_csv"],
        "--v53q1_audit_md",
        variables["v53q1_audit_md"],
        "--v53q1_patch_source",
        variables["v53q1_patch_source"],
        "--execute",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "completed"
    assert result["metrics"]["champion_patches_gate_allowed"] == 1


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
    reason="local v49A edge utility assets are not available",
)
def test_real_v49a_edge_utility_audit_smoke(tmp_path: Path) -> None:
    race = Path.home() / "agent\u6bd4\u8d5b"
    run = race / "model_pro" / "a1" / "new-a1" / "runs" / "v49a_task_relevant_edge_utility"
    variables = {
        "a1_npz": str(race / "A\u5206\u7c7b" / "A\u5206\u7c7b" / "A1.npz"),
        "v49a_oof_meta_csv": str(run / "v49a_oof_meta_scores.csv"),
        "v49a_test_meta_csv": str(run / "v49a_test_meta_scores.csv"),
        "v46a1_base_csv": str(
            race
            / "model_pro"
            / "a1"
            / "new-a1"
            / "a1_v46a1_tabm_seed_consensus"
            / "A1_v46a1_balanced_seed_consensus_SAFE.csv"
        ),
        "current_champion_csv": str(CHAMPION),
        "v53q1_audit_md": str(PROJECT_ROOT / "artifacts" / "V53Q1_TRANSITION_STABLE_EDGE_H2_AUDIT.md"),
        "v53q1_patch_source": str(PROJECT_ROOT / "artifacts" / "a1_v53q1_transition_stable_edge_h2_patch.py"),
        "v49a_report": str(run / "V49A_TASK_RELEVANT_EDGE_UTILITY_REPORT.md"),
        "v49a_config": str(run / "v49a_config.json"),
        "v49a_fold_results": str(run / "v49a_h2gcn_meta_fold_results.csv"),
    }
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "真实 v49 smoke with space")

    assert result["status"] == "completed"
    assert result["metrics"]["oof_selected_count"] == 55
    assert result["metrics"]["test_selected_count"] == 8
    assert result["metrics"]["champion_patch_diff_count"] == 4
    assert result["metrics"]["champion_patches_covered_by_test_meta"] == 4
    assert result["metrics"]["champion_patches_selected"] == 4
    assert result["metrics"]["champion_patches_gate_allowed"] == 4
    assert result["metrics"]["non_champion_test_selected_count"] == 4
    assert result["metrics"]["test_truth_usage_pass"] is True
    assert result["metrics"]["edge_utility_integrity_pass"] is True
    assert {path: _sha256(path) for path in frozen_before} == frozen_before
