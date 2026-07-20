# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

from afac_agent.orchestrator import AgentOrchestrator
from afac_agent.schemas import BudgetState, ProjectState
from conftest import PROJECT_ROOT
from test_m2_a1_data_profiler import _write_npz


def _state(**overrides) -> ProjectState:
    payload = json.loads(
        (PROJECT_ROOT / "config" / "project_state.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(
        {
            "history_imported": True,
            "anchor_registered": True,
            "anchor_oof_analyzed": False,
            "data_profile_ready": False,
            "next_required_capability": "profile_dataset",
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


def test_profile_tool_success_does_not_consume_round_or_mutate_state(
    tmp_path: Path,
) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "profile"
    _write_npz(npz)

    state_path = tmp_path / "project_state.json"
    before_state = _state()
    _write_state(state_path, before_state)
    orchestrator = AgentOrchestrator(
        project_root=PROJECT_ROOT,
        state_path=state_path,
        registry_path=PROJECT_ROOT / "config" / "tool_registry.json",
        trajectory_path=tmp_path / "trajectory.json",
    )

    outcome = orchestrator.run_once(
        variables={
            "python": sys.executable,
            "root": str(PROJECT_ROOT),
            "npz_path": str(npz),
            "edges_csv": "",
            "fold_file": "",
            "anchor_csv": str(
                PROJECT_ROOT
                / "artifacts"
                / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
            ),
            "anchor_oof_npz": "",
            "reference_oof_npz": "",
            "out_dir": str(out_dir),
        },
        execute=True,
    )

    assert outcome["result"]["status"] == "success"
    assert outcome["state"]["budget"]["rounds_used"] == 0
    assert outcome["state"]["data_profile_ready"] is False
    after_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert after_disk == before_state.to_dict()


def test_legacy_data_profile_ready_without_m2_artifact_can_still_profile(
    tmp_path: Path,
) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "profile"
    _write_npz(npz)

    registry_path = tmp_path / "tool_registry.json"
    registry = json.loads(
        (PROJECT_ROOT / "config" / "tool_registry.json").read_text(
            encoding="utf-8"
        )
    )
    profile_tool = next(
        tool for tool in registry["tools"] if tool["name"] == "PROFILE_A1_DATASET"
    )
    assert profile_tool["required_state"] == {}
    registry_path.write_text(
        json.dumps({"tools": [profile_tool]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state_path = tmp_path / "project_state.json"
    _write_state(
        state_path,
        _state(data_profile_ready=True, next_required_capability="profile_dataset"),
    )
    orchestrator = AgentOrchestrator(
        project_root=PROJECT_ROOT,
        state_path=state_path,
        registry_path=registry_path,
        trajectory_path=tmp_path / "trajectory.json",
    )

    outcome = orchestrator.run_once(
        variables={
            "python": sys.executable,
            "root": str(PROJECT_ROOT),
            "npz_path": str(npz),
            "edges_csv": "",
            "fold_file": "",
            "anchor_csv": "",
            "anchor_oof_npz": "",
            "reference_oof_npz": "",
            "out_dir": str(out_dir),
        },
        execute=True,
    )

    assert outcome["result"]["status"] == "success"
    assert "stale legacy" not in outcome["result"].get("stderr_tail", "").lower()


def test_doctor_reports_stale_legacy_data_profile_flag(
    tmp_path: Path,
) -> None:
    from afac_agent.doctor import build_report

    before_state = (
        PROJECT_ROOT / "config" / "project_state.json"
    ).read_text(encoding="utf-8")
    paths_config = tmp_path / "paths.local.yaml"
    missing_profile_dir = tmp_path / "missing profile"
    paths_config.write_text(
        "\n".join(
            [
                "project_root: .",
                "a1:",
                "  anchor_csv: artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv",
                "runtime:",
                f"  profile_out_dir: {missing_profile_dir.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(
        project_root=PROJECT_ROOT,
        paths_config=str(paths_config),
    )

    warnings = report["checks"]["a1_data_profile"]["warnings"]
    assert "legacy_data_profile_flag_stale" in warnings
    assert (
        PROJECT_ROOT / "config" / "project_state.json"
    ).read_text(encoding="utf-8") == before_state
