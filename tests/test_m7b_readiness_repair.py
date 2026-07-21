# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from afac_agent.m7b_readiness import M7BReadinessRepair
from afac_agent.research.event_store import json_dumps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _problem_map(tmp_path: Path) -> Path:
    payload = {
        "analysis_tier": "dataset_only",
        "problems": [
            {
                "problem_id": "structure::isolated",
                "scope": "isolated",
                "node_count": 2792,
                "error_count": None,
                "error_rate": None,
                "structure_evidence": "observed",
                "confidence_evidence": "unavailable_without_anchor_oof",
                "eligible_tool_families": [],
                "blocked_tool_families": ["closed_branches"],
            },
            {
                "problem_id": "structure::exact2_only",
                "scope": "exact2_only",
                "node_count": 527,
                "error_count": None,
                "error_rate": None,
                "structure_evidence": "observed",
                "confidence_evidence": "unavailable_without_anchor_oof",
                "eligible_tool_families": ["audit"],
                "blocked_tool_families": [],
            },
        ],
    }
    path = tmp_path / "problem_map.json"
    path.write_text(json_dumps(payload), encoding="utf-8")
    return path


def _data_profile(tmp_path: Path) -> Path:
    payload = {
        "analysis_tier": "dataset_only",
        "dataset": {"num_nodes": 13752, "train_nodes": 11001, "test_nodes": 2751, "num_classes": 10},
        "oof_profile": {"status": "missing"},
        "anchor_identity": None,
    }
    path = tmp_path / "data_profile.json"
    path.write_text(json_dumps(payload), encoding="utf-8")
    return path


def _registry(tmp_path: Path) -> Path:
    payload = {"tools": [{"name": "A1_OOF_CANDIDATE_EVALUATOR", "adapter_entrypoint": "afac_agent.adapters.a1_oof_candidate_evaluator:Adapter", "command_template": [], "required_inputs": {}, "expected_outputs": {}, "expected_runtime_seconds": 120, "counts_as_experiment_round": False}]}
    path = tmp_path / "tool_registry.json"
    path.write_text(json_dumps(payload), encoding="utf-8")
    return path


def _state(tmp_path: Path) -> Path:
    path = tmp_path / "project_state.json"
    path.write_text(json_dumps({"budget": {"rounds_used": 0, "max_rounds": 12}, "closed_branches": ["ScaleNet"]}), encoding="utf-8")
    return path


def _method_research(tmp_path: Path) -> Path:
    root = tmp_path / "method_research"
    root.mkdir()
    for name, payload in {
        "research_queries.json": {"items": [{"query_text": "graph node classification"}]},
        "source_relevance_audit.json": {"items": [{"relevance_status": "directly_relevant"}]},
        "method_cards_validated.json": {"items": [{"method_id": "m1"}], "validations": [{"method_id": "m1", "valid": True}]},
        "method_conflicts.json": {"items": []},
        "method_ranking.json": {"items": [{"method_id": "m1", "rank": 1}]},
    }.items():
        (root / name).write_text(json_dumps(payload), encoding="utf-8")
    return root


def _paths_config(tmp_path: Path, *, verified: bool) -> Path:
    fold = tmp_path / "fold.csv"
    oof = tmp_path / "v53q1_final_oof.npz"
    if verified:
        with fold.open("w", encoding="utf-8", newline="") as handle:
            handle.write("train_idx,fold\n")
            for idx in range(11001):
                handle.write(f"{idx},{idx % 5}\n")
        proba = np.full((11001, 10), 0.1, dtype=np.float64)
        np.savez(oof, proba=proba)
    cfg = tmp_path / "paths.local.yaml"
    cfg.write_text(
        "\n".join([
            "a1:",
            f"  fold_file: \"{fold}\"",
            f"  anchor_oof_npz: \"{oof}\"",
            f"  anchor_csv: \"{CHAMPION}\"",
        ]),
        encoding="utf-8",
    )
    return cfg


def test_m7b_missing_anchor_waiting_for_input_and_scientific_queue(tmp_path: Path) -> None:
    result = M7BReadinessRepair(project_root=PROJECT_ROOT, paths_config=str(_paths_config(tmp_path, verified=False))).run(
        problem_map=_problem_map(tmp_path),
        data_profile=_data_profile(tmp_path),
        method_research_run=_method_research(tmp_path),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "out",
    )
    assert result["status"] == "waiting_for_input"
    run_dir = tmp_path / "out" / result["run_id"]
    anchor = json.loads((run_dir / "anchor_recovery_decision.json").read_text(encoding="utf-8"))
    queue = json.loads((run_dir / "scientific_research_queue.json").read_text(encoding="utf-8"))
    adapter = json.loads((run_dir / "adapter_execution_preview.json").read_text(encoding="utf-8"))
    assert anchor["status"] == "rebuild_required"
    assert "canonical_fold_assignment" in anchor["missing_inputs"]
    assert queue["items"][0]["problem_id"] == "anchor_recovery::missing_full_anchor_inputs"
    assert queue["ranking_policy"] == "scientific_priority_not_coverage_only"
    assert adapter["execution_allowed"] is False
    assert adapter["prediction_generated"] is False


def test_m7b_verified_anchor_ready_for_human_approval(tmp_path: Path) -> None:
    result = M7BReadinessRepair(project_root=PROJECT_ROOT, paths_config=str(_paths_config(tmp_path, verified=True))).run(
        problem_map=_problem_map(tmp_path),
        data_profile=_data_profile(tmp_path),
        method_research_run=_method_research(tmp_path),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "out",
    )
    assert result["status"] == "ready_for_human_approval"
    run_dir = tmp_path / "out" / result["run_id"]
    fold = json.loads((run_dir / "canonical_fold_verification.json").read_text(encoding="utf-8"))
    oof = json.loads((run_dir / "v53q1_oof_verification.json").read_text(encoding="utf-8"))
    admission = json.loads((run_dir / "m5_admission.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation_plan_preview.json").read_text(encoding="utf-8"))
    assert fold["status"] == "verified_existing"
    assert oof["status"] == "verified_existing"
    assert admission["status"] == "ready_for_human_approval"
    assert evaluation["full_anchor_evaluator_status"] == "ready"
    assert evaluation["evaluation_executed"] is False


def test_m7b_frozen_hashes_rounds_and_cli_doctor(tmp_path: Path) -> None:
    before = {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]}
    rounds_before = json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"]
    missing = subprocess.run(
        [
            sys.executable, "-m", "afac_agent.main", "m7b-readiness",
            "--project_root", str(PROJECT_ROOT),
            "--paths_config", str(_paths_config(tmp_path, verified=False)),
            "--problem-map", str(_problem_map(tmp_path)),
            "--data-profile", str(_data_profile(tmp_path)),
            "--method-research-run", str(_method_research(tmp_path)),
            "--project-state", str(STATE),
            "--tool-registry", str(PROJECT_ROOT / "config" / "tool_registry.json"),
            "--out-root", str(tmp_path / "out"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["status"] == "waiting_for_input"
    assert {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]} == before
    assert json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == rounds_before
    doctor = subprocess.run([sys.executable, "-m", "afac_agent.doctor", "--project_root", str(PROJECT_ROOT), "--json"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    for key in ["m7b_readiness_manifest_schema", "m7b_readiness_output_root", "m7b_readiness_safety"]:
        assert report["checks"][key]["passed"] is True
