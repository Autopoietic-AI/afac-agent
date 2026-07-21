# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from afac_agent.m7_dry_run import M7DryRunOrchestrator
from afac_agent.research.event_store import json_dumps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(tmp_path: Path, *, admission: str = "admitted", tool: str = "SAFE_AUDIT", round_cost: int = 0, missing: list[str] | None = None) -> Path:
    root = tmp_path / f"decision_{admission}_{tool or 'none'}_{round_cost}"
    root.mkdir(parents=True, exist_ok=True)
    proposal = {
        "proposal_id": "p1",
        "target_problem_ids": ["p_one_hop"],
        "target_scope": {"axis_id": "train_label_reachability", "value_id": "one_hop_available"},
        "target_bucket_axes": [{"axis_id": "train_label_reachability", "value_id": "one_hop_available"}],
        "target_classes": [],
        "target_mechanisms": ["neighborhood_reliability_heterogeneity"],
        "parent_candidate": "v53Q-1",
        "baseline": "frozen champion",
        "adapter_or_tool": tool,
        "required_inputs": ["candidate_oof_npz"] if tool else [],
        "missing_inputs": missing or [],
        "minimal_experiment": {"mode": "diagnostic_only" if admission == "admitted_diagnostic_only" else "adapter_preview"},
        "oof_evaluation_plan": {"fold_aware": True},
        "bucket_metrics": ["bucket_accuracy"],
        "bucket_class_metrics": ["bucket_class_delta"],
        "success_conditions": ["OOF improves safely"],
        "failure_conditions": ["negative net"],
        "stop_conditions": ["requires test truth"],
        "estimated_runtime": "minutes",
        "estimated_gpu_memory": "0GB",
        "round_cost": round_cost,
        "risk_level": "low",
    }
    final = {"status": "accepted", "primary_proposal": proposal, "fallback_proposal": None, "revision_applied": "none"}
    if admission == "blocked":
        final = {"status": "blocked", "primary_proposal": None, "fallback_proposal": None}
    adm = {"status": admission, "reason_codes": [admission], "m5_authoritative": True}
    (root / "decision_manifest.json").write_text(json_dumps({"run_version": "synthetic", "run_id": root.name, "artifacts": {}}), encoding="utf-8")
    (root / "m6c_revised_proposals.json").write_text(json_dumps(final), encoding="utf-8")
    (root / "m5_admission_decision.json").write_text(json_dumps(adm), encoding="utf-8")
    (root / "m6b_method_eligibility.json").write_text(json_dumps({"summary": {}}), encoding="utf-8")
    return root


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json_dumps({"tools": [{"name": "SAFE_AUDIT", "adapter_entrypoint": "mock:Adapter", "required_inputs": {"candidate_oof_npz": {"kind": "file"}}, "command_template": ["python", "-m", "safe"], "expected_outputs": {"report": "json"}, "counts_as_experiment_round": False}]}), encoding="utf-8")
    return path


def _state(tmp_path: Path, *, rounds_used: int = 0, max_rounds: int = 1) -> Path:
    path = tmp_path / "state.json"
    path.write_text(json_dumps({"budget": {"rounds_used": rounds_used, "max_rounds": max_rounds}, "closed_branches": ["correct_smooth"]}), encoding="utf-8")
    return path


def test_m7a_ready_for_human_approval_and_no_execution(tmp_path: Path) -> None:
    result = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(
        decision_run=_decision(tmp_path, admission="admitted", missing=[]),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "out",
    )
    assert result["status"] == "ready_for_human_approval"
    run_dir = tmp_path / "out" / result["run_id"]
    manifest = json.loads((run_dir / "dry_run_manifest.json").read_text(encoding="utf-8"))
    adapter = json.loads((run_dir / "adapter_execution_preview.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_dir / "trajectory_preview.json").read_text(encoding="utf-8"))
    assert manifest["executes_adapter"] is False
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert manifest["creates_submission"] is False
    assert manifest["round_consumed"] is False
    assert adapter["execution_allowed"] is False
    assert trajectory["experiment_executed"] is False


def test_m7a_missing_blocked_diagnostic_unregistered_and_budget(tmp_path: Path) -> None:
    missing = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(
        decision_run=tmp_path / "missing",
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "missing_out",
    )
    assert missing["status"] == "waiting_for_input"
    blocked = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(
        decision_run=_decision(tmp_path, admission="blocked"),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "blocked_out",
    )
    assert blocked["status"] == "blocked"
    diagnostic = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(
        decision_run=_decision(tmp_path, admission="admitted_diagnostic_only", tool="", missing=[]),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "diag_out",
    )
    assert diagnostic["status"] == "completed_dry_run"
    unregistered = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(
        decision_run=_decision(tmp_path, admission="admitted", tool="NOT_REGISTERED", missing=[]),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "unregistered_out",
    )
    assert unregistered["status"] == "waiting_for_input"
    budget = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(
        decision_run=_decision(tmp_path, admission="admitted", round_cost=2, missing=[]),
        project_state=_state(tmp_path, rounds_used=1, max_rounds=1),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "budget_out",
    )
    assert budget["status"] == "blocked"


def test_m7a_deterministic_replay_and_frozen_files(tmp_path: Path) -> None:
    before = {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]}
    rounds_before = json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"]
    kwargs = {
        "decision_run": _decision(tmp_path, admission="admitted_diagnostic_only", tool="", missing=[]),
        "project_state": STATE,
        "tool_registry": PROJECT_ROOT / "config" / "tool_registry.json",
        "out_root": tmp_path / "out",
    }
    first = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(**kwargs)
    second = M7DryRunOrchestrator(project_root=PROJECT_ROOT).run(**kwargs)
    assert first["run_id"] == second["run_id"]
    manifest = json.loads((tmp_path / "out" / first["run_id"] / "dry_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["view_hash"]
    assert {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]} == before
    assert json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == rounds_before


def test_m7a_cli_and_doctor_contracts(tmp_path: Path) -> None:
    missing = subprocess.run(
        [
            sys.executable, "-m", "afac_agent.main", "m7-dry-run",
            "--project_root", str(PROJECT_ROOT),
            "--decision-run", str(tmp_path / "missing"),
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
    doctor = subprocess.run([sys.executable, "-m", "afac_agent.doctor", "--project_root", str(PROJECT_ROOT), "--json"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    for key in ["m7_dry_run_manifest_schema", "m7_dry_run_output_root", "m7_dry_run_safety"]:
        assert report["checks"][key]["passed"] is True
