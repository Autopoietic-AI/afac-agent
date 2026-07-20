from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def _write_csv(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["test_idx", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _write_anchor_pair(base_path: Path, champion_path: Path) -> list[dict[str, int]]:
    changes = [
        {"test_idx": 1879, "old_label": 8, "new_label": 6},
        {"test_idx": 2489, "old_label": 3, "new_label": 4},
        {"test_idx": 8190, "old_label": 3, "new_label": 4},
        {"test_idx": 8499, "old_label": 0, "new_label": 3},
    ]
    base_ids = [change["test_idx"] for change in changes]
    base_ids.extend(range(10000, 10000 + 2747))
    rows = [
        {"test_idx": test_idx, "label": test_idx % 10}
        for test_idx in base_ids
    ]
    champion_rows = [dict(row) for row in rows]
    for offset, change in enumerate(changes):
        rows[offset]["label"] = change["old_label"]
        champion_rows[offset]["label"] = change["new_label"]
    _write_csv(base_path, rows)
    _write_csv(champion_path, champion_rows)
    return changes


def _write_meta(path: Path, *, selected_count: int, include_fold: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "global_idx",
        "rescue",
        "damage",
        "neutral",
        "base_pred",
        "h2_pred",
        "model_name",
        "rescue_score",
        "selected",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(selected_count + 3):
            selected = index < selected_count
            writer.writerow(
                {
                    "fold": index % 5 if include_fold else "",
                    "global_idx": 1000 + index,
                    "rescue": 1 if include_fold and selected else 0,
                    "damage": 0,
                    "neutral": 0,
                    "base_pred": index % 10,
                    "h2_pred": (index + 1) % 10,
                    "model_name": "confidence_plus_edge",
                    "rescue_score": "0.75",
                    "selected": "True" if selected else "False",
                }
            )


def _write_audit_md(path: Path, changes: list[dict[str, int]]) -> None:
    lines = ["# synthetic audit", "Changes: `4`"]
    for change in changes:
        lines.append(
            f"- `{change['test_idx']}`: `{change['old_label']} -> "
            f"{change['new_label']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_patch_py(path: Path) -> None:
    path.write_text("# patch source placeholder\n", encoding="utf-8")


def _tool_registry(path: Path, *, output_root: Path, entrypoint: str | None = None) -> None:
    tool = {
        "name": "A1_V53Q1_PATCH_AUDIT",
        "task": "A1",
        "layer": "champion_audit",
        "description": "Synthetic M3A patch audit adapter",
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
            "anchor_csv": {"kind": "file"},
            "v53q1_base_csv": {"kind": "file"},
            "v49a_oof_meta_csv": {"kind": "file"},
            "v49a_test_meta_csv": {"kind": "file"},
            "v53q1_audit_md": {"kind": "file"},
        },
        "forbidden_closed_branches": [],
        "command_template": [],
        "adapter_id": "A1_V53Q1_PATCH_AUDIT",
        "adapter_version": "m3a_v1",
        "adapter_entrypoint": (
            entrypoint
            if entrypoint is not None
            else "afac_agent.adapters.a1_v53q1_patch_audit:Adapter"
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


def _inputs(tmp_path: Path) -> dict[str, str]:
    base = tmp_path / "输入 with space" / "base.csv"
    champion = tmp_path / "输入 with space" / "champion.csv"
    oof_meta = tmp_path / "输入 with space" / "oof_meta.csv"
    test_meta = tmp_path / "输入 with space" / "test_meta.csv"
    audit_md = tmp_path / "输入 with space" / "audit.md"
    patch_py = tmp_path / "输入 with space" / "patch.py"
    changes = _write_anchor_pair(base, champion)
    _write_meta(oof_meta, selected_count=55, include_fold=True)
    _write_meta(test_meta, selected_count=8, include_fold=False)
    _write_audit_md(audit_md, changes)
    _write_patch_py(patch_py)
    return {
        "anchor_csv": str(champion),
        "v53q1_base_csv": str(base),
        "v49a_oof_meta_csv": str(oof_meta),
        "v49a_test_meta_csv": str(test_meta),
        "v53q1_audit_md": str(audit_md),
        "v53q1_patch_py": str(patch_py),
    }


def _run_adapter(tmp_path: Path, variables: dict[str, str], *, output_root: Path):
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=output_root)
    tool = ToolRegistry(registry_path).get("A1_V53Q1_PATCH_AUDIT")
    return AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=tool,
        variables={"root": str(PROJECT_ROOT), **variables},
        execute=True,
    )


def test_v53q1_patch_audit_adapter_read_only_and_duplicate(
    tmp_path: Path,
) -> None:
    variables = _inputs(tmp_path)
    output_root = tmp_path / "adapter runs 中文"
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}

    first = _run_adapter(tmp_path, variables, output_root=output_root)
    second = _run_adapter(tmp_path, variables, output_root=output_root)

    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    assert second["identity_hash"] == first["identity_hash"]
    assert first["read_only"] is True
    assert first["counts_as_experiment_round"] is False
    assert first["mutates_predictions"] is False
    assert first["returncode"] == 0
    assert first["metrics"]["champion_integrity_pass"] is True
    assert first["metrics"]["patch_diff_count"] == 4
    assert first["metrics"]["oof_meta_selected_count"] == 55
    assert first["metrics"]["test_meta_selected_count"] == 8
    assert first["metrics"]["audit_document_consistent"] is True
    assert first["metrics"]["patch_transitions"] == [
        {"test_idx": 1879, "old_label": 8, "new_label": 6, "transition": "8->6"},
        {"test_idx": 2489, "old_label": 3, "new_label": 4, "transition": "3->4"},
        {"test_idx": 8190, "old_label": 3, "new_label": 4, "transition": "3->4"},
        {"test_idx": 8499, "old_label": 0, "new_label": 3, "transition": "0->3"},
    ]
    assert Path(first["stdout_log"]).read_text(encoding="utf-8")
    assert Path(first["stderr_log"]).read_text(encoding="utf-8") == ""
    assert (Path(first["artifacts"]["run_dir"]) / "execution_result.json").exists()
    assert (Path(first["artifacts"]["run_dir"]) / "audit_details.json").exists()
    forbidden_outputs = list(Path(first["artifacts"]["run_dir"]).glob("A1*.csv"))
    forbidden_outputs += list(Path(first["artifacts"]["run_dir"]).glob("*.npz"))
    forbidden_outputs += list(Path(first["artifacts"]["run_dir"]).glob("submission*.zip"))
    assert forbidden_outputs == []
    assert {path: _sha256(path) for path in frozen_before} == frozen_before


def test_adapter_missing_input_invalid_schema_and_forbidden_output(
    tmp_path: Path,
) -> None:
    variables = _inputs(tmp_path)
    variables["v49a_oof_meta_csv"] = str(tmp_path / "missing.csv")
    missing = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs1")
    assert missing["status"] == "waiting_for_input"
    assert "v49a_oof_meta_csv" in missing["missing_inputs"]

    variables = _inputs(tmp_path / "invalid")
    bad_champion = Path(variables["anchor_csv"])
    bad_champion.write_text("test_idx,bad_label\n1,2\n", encoding="utf-8")
    invalid = _run_adapter(tmp_path, variables, output_root=tmp_path / "runs2")
    assert invalid["status"] == "failed"
    assert invalid["failure_reason"] == "input_validation_failed"

    variables = _inputs(tmp_path / "forbidden")
    blocked = _run_adapter(
        tmp_path,
        variables,
        output_root=Path(variables["anchor_csv"]),
    )
    assert blocked["status"] == "blocked"
    assert blocked["failure_reason"] == "unsafe_output_path"


def test_unregistered_adapter_entrypoint_is_blocked(tmp_path: Path) -> None:
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    variables = _inputs(tmp_path)
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(
        registry_path,
        output_root=tmp_path / "runs",
        entrypoint="not_allowed.module:Adapter",
    )
    tool = ToolRegistry(registry_path).get("A1_V53Q1_PATCH_AUDIT")

    result = AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=tool,
        variables={"root": str(PROJECT_ROOT), **variables},
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["failure_reason"] == "unregistered_adapter_entrypoint"


def test_identity_hash_excludes_time_and_log_paths(tmp_path: Path) -> None:
    from afac_agent.adapters.runner import AdapterRunner
    from afac_agent.registry import ToolRegistry

    variables = _inputs(tmp_path)
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=tmp_path / "runs")
    tool = ToolRegistry(registry_path).get("A1_V53Q1_PATCH_AUDIT")
    runner = AdapterRunner(project_root=PROJECT_ROOT)
    input_hashes = runner.hash_inputs(
        ["anchor_csv", "v53q1_base_csv"],
        variables,
    )
    first = runner.compute_identity_hash(
        tool=tool,
        normalized_config={"started_at": "one", "stable": 1},
        input_hashes=input_hashes,
        output_policy={"output_root": "volatile-a"},
        frozen_gate_config={"minimum_support": 3},
    )
    second = runner.compute_identity_hash(
        tool=tool,
        normalized_config={"started_at": "two", "stable": 1},
        input_hashes=input_hashes,
        output_policy={"output_root": "volatile-b"},
        frozen_gate_config={"minimum_support": 3},
    )
    assert first == second


def _state(**overrides) -> ProjectState:
    payload = json.loads(PROJECT_STATE.read_text(encoding="utf-8"))
    payload.update(
        {
            "history_imported": True,
            "anchor_registered": True,
            "data_profile_ready": True,
            "next_required_capability": "v53q1_patch_audit",
        }
    )
    payload.update(overrides)
    payload["budget"] = BudgetState(**payload["budget"])
    return ProjectState(**payload)


def _write_state(path: Path, state: ProjectState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_orchestrator_adapter_success_does_not_consume_round_or_state(
    tmp_path: Path,
) -> None:
    variables = _inputs(tmp_path)
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=tmp_path / "runs")
    state_path = tmp_path / "状态 with space" / "project_state.json"
    before_state = _state()
    _write_state(state_path, before_state)
    orchestrator = AgentOrchestrator(
        project_root=PROJECT_ROOT,
        state_path=state_path,
        registry_path=registry_path,
        trajectory_path=tmp_path / "轨迹 with space" / "trajectory.json",
    )

    outcome = orchestrator.run_once(
        variables={"root": str(PROJECT_ROOT), "python": sys.executable, **variables},
        execute=True,
    )

    assert outcome["result"]["status"] == "completed"
    assert outcome["state"]["budget"]["rounds_used"] == 0
    assert json.loads(state_path.read_text(encoding="utf-8")) == before_state.to_dict()


def test_orchestrator_adapter_waiting_and_failed_do_not_consume_round(
    tmp_path: Path,
) -> None:
    variables = _inputs(tmp_path)
    registry_path = tmp_path / "tool_registry.json"
    _tool_registry(registry_path, output_root=tmp_path / "runs")
    state_path = tmp_path / "状态 waiting failed" / "project_state.json"
    before_state = _state()
    _write_state(state_path, before_state)
    orchestrator = AgentOrchestrator(
        project_root=PROJECT_ROOT,
        state_path=state_path,
        registry_path=registry_path,
        trajectory_path=tmp_path / "trajectory.json",
    )

    waiting_vars = dict(variables)
    waiting_vars["v49a_oof_meta_csv"] = str(tmp_path / "missing.csv")
    waiting = orchestrator.run_once(
        variables={"root": str(PROJECT_ROOT), "python": sys.executable, **waiting_vars},
        execute=True,
    )

    assert waiting["result"]["status"] == "waiting_for_input"
    assert waiting["state"]["budget"]["rounds_used"] == 0
    assert json.loads(state_path.read_text(encoding="utf-8")) == before_state.to_dict()

    bad_champion = Path(variables["anchor_csv"])
    bad_champion.write_text("test_idx,bad_label\n1,2\n", encoding="utf-8")
    failed = orchestrator.run_once(
        variables={"root": str(PROJECT_ROOT), "python": sys.executable, **variables},
        execute=True,
    )

    assert failed["result"]["status"] == "failed"
    assert failed["state"]["budget"]["rounds_used"] == 0
    assert json.loads(state_path.read_text(encoding="utf-8")) == before_state.to_dict()


def test_adapter_cli_dry_run_and_execute(tmp_path: Path) -> None:
    variables = _inputs(tmp_path)
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
        "A1_V53Q1_PATCH_AUDIT",
        "--anchor_csv",
        variables["anchor_csv"],
        "--v53q1_base_csv",
        variables["v53q1_base_csv"],
        "--v49a_oof_meta_csv",
        variables["v49a_oof_meta_csv"],
        "--v49a_test_meta_csv",
        variables["v49a_test_meta_csv"],
        "--v53q1_audit_md",
        variables["v53q1_audit_md"],
        "--v53q1_patch_py",
        variables["v53q1_patch_py"],
    ]
    dry_run = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert json.loads(dry_run.stdout)["status"] == "dry_run"

    executed = subprocess.run(
        [*command, "--execute"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    assert json.loads(executed.stdout)["status"] == "completed"


@pytest.mark.skipif(
    not Path(
        r"C:\Users\李天皓\agent比赛\model_pro\a1\new-a1\a1_v46a1_tabm_seed_consensus\A1_v46a1_balanced_seed_consensus_SAFE.csv"
    ).exists(),
    reason="local v53Q-1 replay audit assets are not available",
)
def test_real_v53q1_patch_audit_smoke(tmp_path: Path) -> None:
    variables = {
        "anchor_csv": str(CHAMPION),
        "v53q1_base_csv": (
            r"C:\Users\李天皓\agent比赛\model_pro\a1\new-a1"
            r"\a1_v46a1_tabm_seed_consensus\A1_v46a1_balanced_seed_consensus_SAFE.csv"
        ),
        "v49a_oof_meta_csv": (
            r"C:\Users\李天皓\agent比赛\model_pro\a1\new-a1\runs"
            r"\v49a_task_relevant_edge_utility\v49a_oof_meta_scores.csv"
        ),
        "v49a_test_meta_csv": (
            r"C:\Users\李天皓\agent比赛\model_pro\a1\new-a1\runs"
            r"\v49a_task_relevant_edge_utility\v49a_test_meta_scores.csv"
        ),
        "v53q1_audit_md": str(
            PROJECT_ROOT / "artifacts" / "V53Q1_TRANSITION_STABLE_EDGE_H2_AUDIT.md"
        ),
        "v53q1_patch_py": str(
            PROJECT_ROOT / "artifacts" / "a1_v53q1_transition_stable_edge_h2_patch.py"
        ),
    }
    frozen_before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}

    result = _run_adapter(
        tmp_path,
        variables,
        output_root=tmp_path / "真实 smoke with space",
    )

    assert result["status"] == "completed"
    assert result["metrics"]["champion_integrity_pass"] is True
    assert result["metrics"]["patch_diff_count"] == 4
    assert result["metrics"]["oof_meta_selected_count"] == 55
    assert result["metrics"]["test_meta_selected_count"] == 8
    assert result["metrics"]["patch_transitions"] == [
        {"test_idx": 1879, "old_label": 8, "new_label": 6, "transition": "8->6"},
        {"test_idx": 2489, "old_label": 3, "new_label": 4, "transition": "3->4"},
        {"test_idx": 8190, "old_label": 3, "new_label": 4, "transition": "3->4"},
        {"test_idx": 8499, "old_label": 0, "new_label": 3, "transition": "0->3"},
    ]
    assert {path: _sha256(path) for path in frozen_before} == frozen_before
