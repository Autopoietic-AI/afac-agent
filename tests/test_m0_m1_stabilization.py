from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from afac_agent.memory import ExperimentMemory
from afac_agent.orchestrator import AgentOrchestrator
from afac_agent.schemas import BudgetState, ProjectState

from conftest import PROJECT_ROOT


CHAMPION_SHA256 = (
    "3CD9AE37CF4DAB7EDBB72F91EC16ED943EBD6921F6509D90771AC631278EEFCE"
)
CHAMPION_CSV = (
    PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_project_state(**overrides) -> ProjectState:
    payload = json.loads(
        (PROJECT_ROOT / "config" / "project_state.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(overrides)
    payload["budget"] = BudgetState(**payload["budget"])
    return ProjectState(**payload)


def write_state(path: Path, state: ProjectState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_orchestrator(tmp_path: Path, state: ProjectState) -> AgentOrchestrator:
    state_path = tmp_path / "状态 with space" / "project_state.json"
    write_state(state_path, state)
    return AgentOrchestrator(
        project_root=PROJECT_ROOT,
        state_path=state_path,
        registry_path=PROJECT_ROOT / "config" / "tool_registry.json",
        trajectory_path=tmp_path / "轨迹 with space" / "trajectory.json",
    )


def test_champion_csv_hash_and_shape_are_unchanged() -> None:
    assert CHAMPION_CSV.exists()
    assert sha256(CHAMPION_CSV) == CHAMPION_SHA256
    with CHAMPION_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2751
    assert set(rows[0]) == {"test_idx", "label"}


def test_history_import_is_idempotent_on_chinese_space_path(tmp_path: Path) -> None:
    history = json.loads(
        (PROJECT_ROOT / "history" / "confirmed_experiments_a1.json").read_text(
            encoding="utf-8"
        )
    )["experiments"]
    memory = ExperimentMemory(tmp_path / "中文 路径" / "experiment_memory_a1.jsonl")

    first = memory.import_records(history)
    second = memory.import_records(history)

    assert first == 19
    assert second == 0
    records = memory.read_all()
    assert len(records) == 19
    assert [item["version"] for item in records].count("v53Q-1") == 1


def test_history_import_tool_is_idempotent_and_validates(tmp_path: Path) -> None:
    memory_jsonl = tmp_path / "memory 中文 path" / "experiment_memory_a1.jsonl"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "import_confirmed_history.py"),
        "--history_json",
        str(PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"),
        "--memory_jsonl",
        str(memory_jsonl),
    ]

    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    assert first_payload["imported"] == 19
    assert second_payload["imported"] == 0
    assert second_payload["history_schema"]["passed"] is True
    assert len(ExperimentMemory(memory_jsonl).read_all()) == 19


def test_missing_file_returns_waiting_for_input_without_success_round(
    tmp_path: Path,
) -> None:
    state = load_project_state(
        history_imported=True,
        data_profile_ready=True,
        anchor_registered=False,
        anchor_oof_analyzed=False,
        next_required_capability="register_anchor",
    )
    orchestrator = make_orchestrator(tmp_path, state)
    missing_anchor = tmp_path / "不存在 with space" / "missing_anchor.csv"

    outcome = orchestrator.run_once(
        variables={
            "python": sys.executable,
            "root": str(PROJECT_ROOT),
            "npz_path": "",
            "anchor_csv": str(missing_anchor),
            "anchor_oof_npz": "",
            "reference_oof_npz": "",
        },
        execute=True,
    )

    assert outcome["result"]["status"] == "waiting_for_input"
    assert "anchor_csv" in outcome["result"]["missing_inputs"]
    assert outcome["state"]["budget"]["rounds_used"] == 0
    assert outcome["state"]["anchor_registered"] is False


def test_unbound_registered_tool_waits_instead_of_crashing(tmp_path: Path) -> None:
    state = load_project_state(
        history_imported=True,
        data_profile_ready=True,
        anchor_registered=True,
        anchor_oof_analyzed=False,
        next_required_capability="expert_complementarity",
    )
    orchestrator = make_orchestrator(tmp_path, state)

    outcome = orchestrator.run_once(
        variables={
            "python": sys.executable,
            "root": str(PROJECT_ROOT),
            "npz_path": "",
            "anchor_csv": str(CHAMPION_CSV),
            "anchor_oof_npz": "",
            "reference_oof_npz": "",
        },
        execute=True,
    )

    assert outcome["result"]["status"] == "waiting_for_input"
    assert outcome["result"]["reason"] == "unbound_tool"
    assert outcome["state"]["budget"]["rounds_used"] == 0


def test_failed_tool_does_not_consume_success_round(tmp_path: Path) -> None:
    fail_script = tmp_path / "fail_tool.py"
    fail_script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    registry_path = tmp_path / "tool_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "IMPORT_CONFIRMED_HISTORY",
                        "task": "A1",
                        "layer": "memory",
                        "description": "synthetic failing tool",
                        "action_type": "system",
                        "expected_runtime_seconds": 10,
                        "prediction_changing": False,
                        "submission_creating": False,
                        "required_state": {"history_imported": False},
                        "forbidden_closed_branches": [],
                        "command_template": [
                            "{python}",
                            str(fail_script),
                        ],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "project_state.json"
    write_state(state_path, load_project_state())
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
            "npz_path": "",
            "anchor_csv": "",
            "anchor_oof_npz": "",
            "reference_oof_npz": "",
        },
        execute=True,
    )

    assert outcome["result"]["status"] == "failed"
    assert outcome["state"]["budget"]["rounds_used"] == 0
    assert outcome["state"]["history_imported"] is False


def test_anchor_registration_is_hash_idempotent(tmp_path: Path) -> None:
    out_dir = tmp_path / "online anchor 中文"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "register_online_anchor.py"),
        "--a1_csv",
        str(CHAMPION_CSV),
        "--version",
        "v53Q-1",
        "--online_score",
        "0.7800",
        "--expected_rows",
        "2751",
        "--num_classes",
        "10",
        "--out_dir",
        str(out_dir),
    ]

    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    manifest = json.loads(
        (out_dir / "online_anchor_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sha256"].upper() == CHAMPION_SHA256
    assert manifest["registration_status"] == "unchanged"
    assert manifest["registered_as_anchor"] is True
    assert sha256(CHAMPION_CSV) == CHAMPION_SHA256


def test_path_resolver_defaults_to_packaged_champion() -> None:
    from afac_agent.paths import PathResolver

    resolver = PathResolver(PROJECT_ROOT)

    assert resolver.a1_anchor_csv() == CHAMPION_CSV.resolve()
    assert resolver.a1_npz() is None


def test_anchor_manifest_validation_after_registration(tmp_path: Path) -> None:
    from afac_agent.validation import validate_anchor_manifest_file

    out_dir = tmp_path / "manifest 校验"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "register_online_anchor.py"),
        "--a1_csv",
        str(CHAMPION_CSV),
        "--version",
        "v53Q-1",
        "--online_score",
        "0.7800",
        "--expected_rows",
        "2751",
        "--num_classes",
        "10",
        "--out_dir",
        str(out_dir),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert validate_anchor_manifest_file(
        out_dir / "online_anchor_manifest.json"
    ).passed


def test_validation_accepts_current_contract_files() -> None:
    from afac_agent.validation import (
        validate_memory_records_file,
        validate_project_state_file,
        validate_tool_registry_file,
    )

    assert validate_project_state_file(
        PROJECT_ROOT / "config" / "project_state.json"
    ).passed
    assert validate_tool_registry_file(
        PROJECT_ROOT / "config" / "tool_registry.json"
    ).passed
    assert validate_memory_records_file(
        PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"
    ).passed


def test_doctor_json_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "afac_agent.doctor",
            "--project_root",
            str(PROJECT_ROOT),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["project_root"] == str(PROJECT_ROOT.resolve())
    assert report["checks"]["champion_csv"]["passed"] is True
    assert report["checks"]["tool_registry"]["passed"] is True
    assert report["checks"]["project_state"]["passed"] is True
