from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def _write_prediction_csv(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["test_idx", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _write_a1_npz(path: Path) -> dict[str, np.ndarray]:
    # Nodes 0..5 train, 6..10 test. Test 6/7 are isolated; 8/9/10 are graph-visible.
    rows = np.array([0, 8, 9, 10, 8], dtype=np.int32)
    cols = np.array([8, 0, 10, 9, 0], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(11, 11))
    attr = sparse.eye(11, 3, dtype=np.float32, format="csr")
    train_idx = np.arange(6, dtype=np.int64)
    test_idx = np.array([6, 7, 8, 9, 10], dtype=np.int64)
    labels = np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1], dtype=np.int64)
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
        train_idx=train_idx,
        test_idx=test_idx,
    )
    return {"train_idx": train_idx, "test_idx": test_idx, "labels": labels}


def _write_candidate_bundle(tmp_path: Path) -> dict[str, str]:
    input_dir = tmp_path / "输入 with space"
    input_dir.mkdir(parents=True, exist_ok=True)
    a1_npz = input_dir / "A1.npz"
    _write_a1_npz(a1_npz)
    parent_csv = input_dir / "parent_v43c.csv"
    candidate_csv = input_dir / "candidate_v46a1.csv"
    parent_rows = [
        {"test_idx": 6, "label": 0},
        {"test_idx": 7, "label": 1},
        {"test_idx": 8, "label": 2},
        {"test_idx": 9, "label": 3},
        {"test_idx": 10, "label": 4},
    ]
    candidate_rows = [
        {"test_idx": 6, "label": 2},
        {"test_idx": 7, "label": 1},
        {"test_idx": 8, "label": 2},
        {"test_idx": 9, "label": 3},
        {"test_idx": 10, "label": 4},
    ]
    _write_prediction_csv(parent_csv, parent_rows)
    _write_prediction_csv(candidate_csv, candidate_rows)
    audit_report = input_dir / "V46A1_CONSENSUS_REPORT.md"
    audit_report.write_text(
        "\n".join(
            [
                "# v46A-1 synthetic audit",
                "Graph-visible remains v43C.",
                "Rule: `balanced_seed_consensus`",
                "safe_changes: `1`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    champion_csv = input_dir / "champion.csv"
    _write_prediction_csv(champion_csv, candidate_rows)
    oof_npz = input_dir / "candidate_oof.npz"
    proba = np.zeros((6, 10), dtype=np.float32)
    proba[np.arange(6), np.arange(6)] = 1.0
    np.savez(
        oof_npz,
        train_idx=np.arange(6, dtype=np.int64),
        labels=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        proba=proba,
    )
    return {
        "a1_npz": str(a1_npz),
        "candidate_csv": str(candidate_csv),
        "parent_csv": str(parent_csv),
        "audit_report": str(audit_report),
        "current_champion_csv": str(champion_csv),
        "candidate_oof_npz": str(oof_npz),
    }


def _tool_registry(path: Path, *, output_root: Path, entrypoint: str | None = None) -> None:
    tool = {
        "name": "A1_V46A1_ISOLATED_AUDIT",
        "task": "A1",
        "layer": "adapter_audit",
        "description": "Synthetic v46A-1 isolated expert audit adapter",
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
            "candidate_csv": {"kind": "file"},
        },
        "forbidden_closed_branches": [],
        "command_template": [],
        "adapter_id": "A1_V46A1_ISOLATED_AUDIT",
        "adapter_version": "m3b_v1",
        "adapter_entrypoint": (
            entrypoint
            if entrypoint is not None
            else "afac_agent.adapters.a1_v46a1_isolated_audit:Adapter"
        ),
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
    path.write_text(
        json.dumps({"tools": [tool]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_adapter(tmp_path: Path, variables: dict[str, str], *, output_root: Path):
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=output_root)
    tool = ToolRegistry(registry_path).get("A1_V46A1_ISOLATED_AUDIT")
    return AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=tool,
        variables={"root": str(PROJECT_ROOT), **variables},
        execute=True,
    )


def test_registry_loads_v46a1_adapter() -> None:
    from afac_agent.adapters.runner import ALLOWED_ADAPTER_ENTRYPOINTS
    from afac_agent.registry import ToolRegistry

    tool = ToolRegistry(PROJECT_ROOT / "config" / "tool_registry.json").get(
        "A1_V46A1_ISOLATED_AUDIT"
    )
    assert tool.adapter_entrypoint == "afac_agent.adapters.a1_v46a1_isolated_audit:Adapter"
    assert tool.adapter_entrypoint in ALLOWED_ADAPTER_ENTRYPOINTS
    assert tool.counts_as_experiment_round is False
    assert tool.read_only is True


def test_missing_candidate_and_npz_wait_for_input(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    variables["candidate_csv"] = str(tmp_path / "missing_candidate.csv")
    missing_candidate = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs1")
    assert missing_candidate["status"] == "waiting_for_input"
    assert "candidate_csv" in missing_candidate["missing_inputs"]

    variables = _write_candidate_bundle(tmp_path / "second")
    variables["a1_npz"] = str(tmp_path / "missing_a1.npz")
    missing_npz = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs2")
    assert missing_npz["status"] == "waiting_for_input"
    assert "a1_npz" in missing_npz["missing_inputs"]


def test_candidate_and_parent_schema_failures(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    Path(variables["candidate_csv"]).write_text("test_idx,bad\n6,1\n", encoding="utf-8")
    bad_candidate = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs1")
    assert bad_candidate["status"] == "failed"
    assert bad_candidate["failure_reason"] == "input_validation_failed"

    variables = _write_candidate_bundle(tmp_path / "second")
    Path(variables["parent_csv"]).write_text("test_idx,bad\n6,1\n", encoding="utf-8")
    bad_parent = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs2")
    assert bad_parent["status"] == "failed"
    assert bad_parent["failure_reason"] == "input_validation_failed"


def test_npz_and_parent_test_idx_alignment_failures(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    candidate_rows = list(csv.DictReader(Path(variables["candidate_csv"]).open(encoding="utf-8-sig")))
    candidate_rows[0]["test_idx"] = "999"
    _write_prediction_csv(
        Path(variables["candidate_csv"]),
        [{"test_idx": int(row["test_idx"]), "label": int(row["label"])} for row in candidate_rows],
    )
    bad_candidate_alignment = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs1")
    assert bad_candidate_alignment["status"] == "failed"
    assert bad_candidate_alignment["failure_reason"] == "input_validation_failed"

    variables = _write_candidate_bundle(tmp_path / "second")
    parent_rows = list(csv.DictReader(Path(variables["parent_csv"]).open(encoding="utf-8-sig")))
    parent_rows[-1]["test_idx"] = "999"
    _write_prediction_csv(
        Path(variables["parent_csv"]),
        [{"test_idx": int(row["test_idx"]), "label": int(row["label"])} for row in parent_rows],
    )
    bad_parent_alignment = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs2")
    assert bad_parent_alignment["status"] == "failed"
    assert bad_parent_alignment["failure_reason"] == "input_validation_failed"


def test_npz_alignment_and_isolated_only_metrics(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")

    assert result["status"] == "completed"
    metrics = result["metrics"]
    assert metrics["candidate_integrity_pass"] is True
    assert metrics["test_idx_strict_match"] is True
    assert metrics["isolated_test_count"] == 2
    assert metrics["graph_visible_test_count"] == 3
    assert metrics["diff_count"] == 1
    assert metrics["isolated_diff_count"] == 1
    assert metrics["graph_visible_diff_count"] == 0
    assert metrics["isolated_only_pass"] is True
    assert metrics["oof_status"] == "observed"
    assert metrics["overall_oof_accuracy"] == 1.0
    assert metrics["candidate_champion_diff_count"] == 0
    assert "graph_visible_diff_count=0" in result["warnings"]
    diff_rows = list(csv.DictReader(Path(result["artifacts"]["candidate_diff_audit"]).open(encoding="utf-8-sig")))
    assert diff_rows == [
        {
            "test_idx": "6",
            "parent_label": "0",
            "candidate_label": "2",
            "transition": "0->2",
            "is_isolated": "True",
            "is_graph_visible": "False",
        }
    ]
    forbidden = list(Path(result["artifacts"]["run_dir"]).glob("A1*.csv"))
    forbidden += list(Path(result["artifacts"]["run_dir"]).glob("*.npz"))
    forbidden += list(Path(result["artifacts"]["run_dir"]).glob("submission*.zip"))
    assert forbidden == []


def test_graph_visible_change_fails_isolated_only_but_completes(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    parent = list(csv.DictReader(Path(variables["parent_csv"]).open(encoding="utf-8-sig")))
    candidate = [dict(row) for row in parent]
    candidate[-1]["label"] = "8"
    _write_prediction_csv(
        Path(variables["candidate_csv"]),
        [{"test_idx": int(row["test_idx"]), "label": int(row["label"])} for row in candidate],
    )

    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")

    assert result["status"] == "completed"
    assert result["metrics"]["isolated_only_pass"] is False
    assert result["metrics"]["graph_visible_diff_count"] == 1
    assert any("graph_visible_diff_count=1" in warning for warning in result["warnings"])


def test_optional_oof_unavailable_is_not_placeholder_zero(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    variables.pop("candidate_oof_npz")

    result = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs")

    assert result["status"] == "completed"
    assert result["metrics"]["oof_status"] == "unavailable"
    assert result["metrics"]["oof_unavailable_reason"] == "missing_candidate_oof_npz"
    assert "overall_oof_accuracy" not in result["metrics"]


def test_read_only_hashes_duplicate_and_identity_stability(tmp_path: Path) -> None:
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    variables = _write_candidate_bundle(tmp_path)
    output_root = tmp_path / "runs 中文 with space"
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
    first = _run_adapter(tmp_path, variables, output_root=output_root)
    second = _run_adapter(tmp_path, variables, output_root=output_root)

    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    assert second["identity_hash"] == first["identity_hash"]
    assert first["counts_as_experiment_round"] is False
    assert {path: _sha256(path) for path in frozen_before} == frozen_before

    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=output_root)
    tool = ToolRegistry(registry_path).get("A1_V46A1_ISOLATED_AUDIT")
    runner = AdapterRunner(project_root=PROJECT_ROOT)
    input_hashes = runner.hash_inputs(["a1_npz", "candidate_csv"], variables)
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


def test_orchestrator_v46a1_adapter_no_round_or_state_mutation(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=tmp_path / "runs")
    payload = json.loads(PROJECT_STATE.read_text(encoding="utf-8"))
    payload.update(
        {
            "history_imported": True,
            "anchor_registered": True,
            "data_profile_ready": True,
            "next_required_capability": "v46a1_isolated_audit",
        }
    )
    payload["budget"] = BudgetState(**payload["budget"])
    state = ProjectState(**payload)
    state_path = tmp_path / "状态 with space" / "project_state.json"
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

    assert outcome["result"]["status"] == "completed"
    assert outcome["state"]["budget"]["rounds_used"] == 0
    assert json.loads(state_path.read_text(encoding="utf-8")) == state.to_dict()


def test_cli_v46a1_smoke(tmp_path: Path) -> None:
    variables = _write_candidate_bundle(tmp_path)
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
        "A1_V46A1_ISOLATED_AUDIT",
        "--a1_npz",
        variables["a1_npz"],
        "--candidate_csv",
        variables["candidate_csv"],
        "--parent_csv",
        variables["parent_csv"],
        "--candidate_oof_npz",
        variables["candidate_oof_npz"],
        "--audit_report",
        variables["audit_report"],
        "--current_champion_csv",
        variables["current_champion_csv"],
        "--execute",
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "completed"
    assert result["metrics"]["isolated_only_pass"] is True


@pytest.mark.skipif(
    not (
        Path.home()
        / "agent\u6bd4\u8d5b"
        / "model_pro"
        / "a1"
        / "new-a1"
        / "a1_v46a1_tabm_seed_consensus"
        / "A1_v46a1_balanced_seed_consensus_SAFE.csv"
    ).exists(),
    reason="local v46A-1 audit assets are not available",
)
def test_real_v46a1_isolated_audit_smoke(tmp_path: Path) -> None:
    race = Path.home() / "agent\u6bd4\u8d5b"
    variables = {
        "a1_npz": str(race / "A\u5206\u7c7b" / "A\u5206\u7c7b" / "A1.npz"),
        "candidate_csv": str(
            race
            / "model_pro"
            / "a1"
            / "new-a1"
            / "a1_v46a1_tabm_seed_consensus"
            / "A1_v46a1_balanced_seed_consensus_SAFE.csv"
        ),
        "parent_csv": str(
            race
            / "model_pro"
            / "a1"
            / "versions"
            / "\u7279\u5f81\u5c42"
            / "runs"
            / "a1_v43c_3seed_ensemble"
            / "A1_v43C_3seed_ensemble_review.csv"
        ),
        "audit_report": str(
            race
            / "model_pro"
            / "a1"
            / "new-a1"
            / "a1_v46a1_tabm_seed_consensus"
            / "V46A1_CONSENSUS_REPORT.md"
        ),
        "current_champion_csv": str(CHAMPION),
    }
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}

    result = _run_adapter(
        tmp_path,
        variables,
        output_root=tmp_path / "真实 v46 smoke with space",
    )

    assert result["status"] == "completed"
    assert result["metrics"]["candidate_integrity_pass"] is True
    assert result["metrics"]["isolated_test_count"] == 579
    assert result["metrics"]["graph_visible_test_count"] == 2172
    assert result["metrics"]["diff_count"] == 44
    assert result["metrics"]["isolated_diff_count"] == 44
    assert result["metrics"]["graph_visible_diff_count"] == 0
    assert result["metrics"]["isolated_only_pass"] is True
    assert result["metrics"]["oof_status"] == "unavailable"
    assert result["metrics"]["parent_identity_status"] in {"verified", "evidence_backed", "unverified"}
    assert {path: _sha256(path) for path in frozen_before} == frozen_before
