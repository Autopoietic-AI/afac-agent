# -*- coding: utf-8 -*-
"""Synthetic tests for the v2 runtime wiring: CLI separation, execution
identity, LLM ledger contract, legacy replay detection, completion contract,
and the end-to-end v2 orchestration smoke (with a fake provider — no network).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from afac_agent.llm.base import LLMResponse
from afac_agent.research.event_store import load_json, stable_hash
from afac_agent.v2.completion_contract import (
    MANIFEST_VERSION,
    TRAJECTORY_VERSION,
    check_completion_contract,
)
from afac_agent.v2.execution_identity import (
    build_execution_id,
    build_input_fingerprint,
    llm_provider_config_identity,
)
from afac_agent.v2.legacy_replay_detector import detect_legacy_replay
from afac_agent.v2.llm_ledger import LLMLedger, load_ledger
from afac_agent.v2.orchestrator import V2AutonomousResearchOrchestrator, m5_admission
from afac_agent.v2.budget_scheduler import BudgetState

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fake secrets are built dynamically so the doctor secret-location scan
# (which matches the literal pattern sk-<token>) is not tripped by this file.
FAKE_SECRET = "sk" + "-testsecret123"
FAKE_SECRET_SHORT = "sk" + "-secret"

N_TRAIN = 30
N_TEST = 8
N_ITEMS = 60
TOP_K = 10


# --------------------------------------------------------------------------- fixtures


def _iid(i: int) -> str:
    return f"i{i:04d}"


def _uid(i: int) -> str:
    return f"u{i:04d}"


def _write_b2_dataset(root: Path) -> Path:
    """Minimal legal B2 dataset (schema-compatible with the real one)."""
    data = root / "b2_data"
    data.mkdir(parents=True)
    items = [_iid(i) for i in range(N_ITEMS)]
    with (data / "item.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iid", "i_cat_01", "i_cat_02"])
        for iid in items:
            writer.writerow([iid, "1", "2"])
    with (data / "user.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "u_cat_01"])
        for i in range(N_TRAIN + N_TEST):
            writer.writerow([_uid(i), "1"])
    with (data / "train.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
        for i in range(N_TRAIN):
            uid = _uid(i)
            if i % 3 == 0:
                seq = []
            else:
                seq = [items[(i + j) % 40] for j in range(i % 7)]
            target = items[(i * 3 + 5) % N_ITEMS]
            raw = ",".join(seq)
            counts = ",".join(f"{iid}:1" for iid in seq)
            writer.writerow([uid, target, raw, raw, counts])
    with (data / "test.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
        for i in range(N_TEST):
            uid = _uid(N_TRAIN + i)
            seq = [items[(i + j) % 30] for j in range(3)]
            writer.writerow([uid, ",".join(seq), ",".join(seq), ",".join(f"{s}:1" for s in seq)])
    with (data / "sample_submission.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "prediction"])
        for i in range(N_TEST):
            writer.writerow([_uid(N_TRAIN + i), ",".join(items[:TOP_K])])
    (data / "metadata.json").write_text(json.dumps({"submission_format": {"prediction_format": "top-K csv"}}), encoding="utf-8")
    return data


class FakeProvider:
    """Deterministic fake LLM: canned JSON per stage keyword in the prompt."""

    def __init__(self, *, proposal: dict | None = None, critic: dict | None = None) -> None:
        self.calls: list[str] = []
        self.proposal = proposal or {
            "diagnostic_type": "candidate_recall_diagnostic",
            "hypothesis": "union beats popularity parent",
            "information_sources": ["popularity", "history", "pair_transition"],
            "parent": "popularity",
            "budget_seconds": 60,
            "expected_gain": 0.05,
            "success_condition": "union pool recall@100 > parent",
            "failure_condition": "union pool recall@100 <= parent",
            "stop_condition": "single diagnostic",
        }
        self.critic = critic or {"verdict": "approve", "issues": [], "required_adjustments": [], "rationale": "whitelisted diagnostic"}

    def generate(self, request) -> LLMResponse:
        prompt = request.prompt
        self.calls.append(prompt)
        if "problem-synthesis" in prompt:
            text = json.dumps({"problem_id": "", "rationale": "fake choice"})
        elif "M6B proposal stage" in prompt:
            text = json.dumps(self.proposal)
        elif "M6C counterfactual critic" in prompt:
            text = json.dumps(self.critic)
        else:
            text = json.dumps({"summary": "fake postmortem", "what_we_learned": "x", "next_priority": "y"})
        return LLMResponse(status="completed", provider="fake", model="fake-model", text=text)


class UnavailableProvider:
    def generate(self, request) -> LLMResponse:
        return LLMResponse(status="provider_unavailable", provider="fake", model="fake", failure_reason="no api key " + FAKE_SECRET + " in env")


def _orchestrator(tmp_path: Path, data: Path, provider, **kwargs) -> V2AutonomousResearchOrchestrator:
    defaults = dict(
        project_root=PROJECT_ROOT,
        task="B2",
        data_root=data,
        out_root=tmp_path / "v2_runs",
        max_wall_clock_seconds=300.0,
        require_llm=True,
        smoke=True,
        smoke_max_users=24,
        smoke_max_seconds=240.0,
        no_deployment=True,
        provider=provider,
        force_new_execution=True,
    )
    defaults.update(kwargs)
    return V2AutonomousResearchOrchestrator(**defaults)


@pytest.fixture()
def b2_data(tmp_path: Path) -> Path:
    return _write_b2_dataset(tmp_path)


# --------------------------------------------------------------------------- CLI separation


def test_v2_run_never_imports_legacy_runner() -> None:
    source = (PROJECT_ROOT / "afac_agent" / "v2" / "orchestrator.py").read_text(encoding="utf-8")
    assert "B2ClosedLoopRunner" not in source
    assert "run_b2_closed_loop" not in source


def test_v2_cli_handler_calls_orchestrator() -> None:
    source = (PROJECT_ROOT / "afac_agent" / "main.py").read_text(encoding="utf-8")
    assert 'sys.argv[1] == "v2-run"' in source
    assert 'sys.argv[1] == "legacy-b2-closed-loop"' in source
    v2_section = source.split("def _v2_run")[1]
    assert "V2AutonomousResearchOrchestrator" in v2_section
    assert "run_b2_closed_loop" not in v2_section.split("def ")[1].split("def ")[0]


def test_legacy_refuses_v2_formal_runs_out_root() -> None:
    from afac_agent.main import _legacy_out_root_guard

    assert _legacy_out_root_guard("artifacts/v2_formal_runs/b2", False) is not None
    assert _legacy_out_root_guard("artifacts/v2_formal_runs/b2", True) is None
    assert _legacy_out_root_guard("artifacts/b2_runs", False) is None


def test_legacy_cli_guard_via_subprocess(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "afac_agent.main", "legacy-b2-closed-loop", "--data-root", str(tmp_path), "--out-root", "artifacts/v2_formal_runs/b2"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "refused" in proc.stdout


# --------------------------------------------------------------------------- legacy replay detector


def test_legacy_replay_detector_flags_v1_manifest() -> None:
    result = detect_legacy_replay({"manifest_version": "b2_autonomous_recommendation_loop_v1", "trajectory_version": "b2_autonomous_recommendation_loop_v1"})
    assert result["is_legacy_replay"] is True
    assert result["status"] == "invalid_replay"
    assert "return_completed" in result["forbidden_actions"]


def test_legacy_replay_detector_accepts_v2_manifest() -> None:
    result = detect_legacy_replay({
        "manifest_version": MANIFEST_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "orchestrator_version": "2.0.0",
        "planner_mode": "llm_plus_deterministic_gates",
        "llm_calls_count": 3,
        "artifacts": {"problem_node": "p", "m6b_proposal": "b", "m6c_critic": "c"},
    })
    assert result["is_legacy_replay"] is False


def test_legacy_replay_detector_flags_v1_run_id_collision() -> None:
    result = detect_legacy_replay(
        {"manifest_version": MANIFEST_VERSION, "trajectory_version": TRAJECTORY_VERSION, "orchestrator_version": "2.0.0", "planner_mode": "x", "llm_calls_count": 2, "artifacts": {"problem_node": 1, "m6b_proposal": 2, "m6c_critic": 3}},
        execution_id="fcf5ad3dbcdf9700dd644eff",
        known_v1_run_ids={"fcf5ad3dbcdf9700dd644eff"},
    )
    assert result["is_legacy_replay"] is True


def test_real_invalid_replay_dir_is_flagged() -> None:
    replay_manifest = PROJECT_ROOT / "artifacts" / "v2_formal_runs" / "b2" / "fcf5ad3dbcdf9700dd644eff" / "run_manifest.json"
    if not replay_manifest.is_file():
        pytest.skip("invalid replay dir not present")
    result = detect_legacy_replay(manifest_path=replay_manifest, report_name="B2_CLOSED_LOOP_V1_REPORT.md")
    assert result["is_legacy_replay"] is True


# --------------------------------------------------------------------------- execution identity


def test_input_fingerprint_stable_and_separate() -> None:
    fp1 = build_input_fingerprint(data_hash="a", fold_hash="b", deterministic_config_hash="c", metric_contract_hash="d")
    fp2 = build_input_fingerprint(data_hash="a", fold_hash="b", deterministic_config_hash="c", metric_contract_hash="d")
    assert fp1["input_fingerprint"] == fp2["input_fingerprint"]
    fp3 = build_input_fingerprint(data_hash="changed", fold_hash="b", deterministic_config_hash="c", metric_contract_hash="d")
    assert fp3["input_fingerprint"] != fp1["input_fingerprint"]


def test_execution_id_unique_per_execution(tmp_path: Path) -> None:
    kwargs = dict(
        input_fingerprint="fp",
        project_root=tmp_path,
        planner_policy_hash="p",
        capability_registry_hash="c",
        metric_contract_hash="m",
        validation_contract_hash="v",
        llm_provider_config_hash="l",
    )
    e1 = build_execution_id(**kwargs)
    e2 = build_execution_id(**kwargs)
    assert e1["execution_id"] != e2["execution_id"]
    assert e1["components"]["execution_nonce"] != e2["components"]["execution_nonce"]


def test_execution_id_records_code_commit() -> None:
    identity = build_execution_id(
        input_fingerprint="fp",
        project_root=PROJECT_ROOT,
        planner_policy_hash="p",
        capability_registry_hash="c",
        metric_contract_hash="m",
        validation_contract_hash="v",
        llm_provider_config_hash="l",
    )
    commit = identity["components"]["code_commit"]
    assert commit and commit != "unknown"
    assert len(commit) == 40


def test_provider_config_identity_excludes_secrets() -> None:
    identity = llm_provider_config_identity({"provider": "aliyun_bailian_openai", "model": "qwen3.6-max-preview", "api_key": FAKE_SECRET_SHORT, "base_url_env": "AFAC_BAILIAN_BASE_URL", "temperature": 0, "max_output_tokens": 4096})
    assert "api_key" not in identity
    assert FAKE_SECRET_SHORT not in json.dumps(identity)
    assert identity["endpoint_class"] == "env_var_reference"


# --------------------------------------------------------------------------- LLM ledger


def test_ledger_records_contract_fields(tmp_path: Path) -> None:
    ledger = LLMLedger(run_dir=tmp_path, execution_id="exec1", task="B2", provider=FakeProvider(), provider_name="fake", model="fake-model", planner_policy_version="v1")
    result = ledger.call(stage="M6B_PROPOSAL", prompt_template_id="tmpl", prompt="M6B proposal stage test")
    assert result["ok"] is True
    records = load_ledger(ledger.ledger_path)
    assert len(records) == 1
    record = records[0]
    for field in ("call_id", "execution_id", "task", "stage", "provider", "model", "planner_policy_version", "prompt_template_id", "prompt_hash", "response_hash", "started_at", "ended_at", "latency_ms", "input_tokens", "output_tokens", "status", "error_type", "error_message_sanitized", "retry_count", "fallback_used"):
        assert field in record
    assert record["status"] == "completed"
    assert len(record["prompt_hash"]) == 64


def test_ledger_sanitizes_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_SECRET)
    ledger = LLMLedger(run_dir=tmp_path, execution_id="exec1", task="B2", provider=UnavailableProvider(), provider_name="fake", model="fake", planner_policy_version="v1", max_retries=0)
    result = ledger.call(stage="M6B_PROPOSAL", prompt_template_id="tmpl", prompt="hello")
    assert result["ok"] is False
    record = load_ledger(ledger.ledger_path)[0]
    assert FAKE_SECRET not in record["error_message_sanitized"]
    assert "[REDACTED" in record["error_message_sanitized"]


# --------------------------------------------------------------------------- M5 / completion contract


def test_m5_rejects_forbidden_content_and_llm_cannot_bypass() -> None:
    proposal = {"diagnostic_type": "candidate_recall_diagnostic", "note": "use test_truth to tune", "success_condition": "x", "failure_condition": "y", "budget_seconds": 30}
    critic = {"verdict": "approve"}
    decision = m5_admission(proposal=proposal, critic=critic, budget_state=BudgetState(max_wall_clock_seconds=300), smoke=True)
    assert decision["status"] == "rejected"
    assert decision["llm_can_bypass"] is False


def test_m5_rejects_non_whitelisted_diagnostic() -> None:
    proposal = {"diagnostic_type": "train_deep_model", "success_condition": "x", "failure_condition": "y", "budget_seconds": 30}
    decision = m5_admission(proposal=proposal, critic={"verdict": "approve"}, budget_state=BudgetState(max_wall_clock_seconds=300), smoke=True)
    assert decision["status"] == "rejected"


def test_m5_admits_clean_diagnostic() -> None:
    proposal = {"diagnostic_type": "cached_replay", "success_condition": "x", "failure_condition": "y", "budget_seconds": 30}
    decision = m5_admission(proposal=proposal, critic={"verdict": "approve"}, budget_state=BudgetState(max_wall_clock_seconds=300), smoke=True)
    assert decision["status"] == "admitted_diagnostic_only"


def test_completion_contract_rejects_v1_manifest() -> None:
    contract = check_completion_contract({"manifest_version": "b2_autonomous_recommendation_loop_v1", "status": "completed"}, smoke=True)
    assert contract["status"] == "failed"
    assert contract["missing"]


# --------------------------------------------------------------------------- orchestrator end-to-end (fake provider)


def test_smoke_completes_with_llm_ledger(b2_data: Path, tmp_path: Path) -> None:
    manifest = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    assert manifest["status"] == "completed_smoke"
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["orchestrator_version"].startswith("2")
    assert manifest["trajectory_version"] == TRAJECTORY_VERSION
    assert manifest["execution_id"] != "fcf5ad3dbcdf9700dd644eff"
    assert manifest["planner_mode"] == "llm_plus_deterministic_gates"
    assert manifest["llm_required"] is True
    assert manifest["llm_calls_count"] >= 2
    assert manifest["cache_status"] != "legacy_final_result_reuse"
    assert manifest["completion_contract"]["status"] == "passed"

    run_dir = Path(manifest["_run_dir"]) if "_run_dir" in manifest else (tmp_path / "v2_runs" / manifest["execution_id"])
    for name in ("run_manifest.json", "trajectory_v2.json", "heartbeat.json", "STATUS.md", "run_events.jsonl", "dashboard.html", "llm_calls.jsonl", "problem_node.json", "m6b_proposal.json", "m6c_critic.json", "m5_decision.json", "experiment_genome.json", "budget_decision.json", "no_op_audit.json", "postmortem.json", "V2_SMOKE_REPORT.md"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "TO_UPLOAD" / "candidate_B2.csv").exists()


def test_smoke_m5_and_genome_materialized(b2_data: Path, tmp_path: Path) -> None:
    manifest = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    run_dir = tmp_path / "v2_runs" / manifest["execution_id"]
    m5 = load_json(run_dir / "m5_decision.json")
    assert m5["status"] == "admitted_diagnostic_only"
    assert m5["llm_can_bypass"] is False
    genome = load_json(run_dir / "experiment_genome.json")
    assert genome["genome_id"]
    assert genome["changed_layers"] == ["L5"]
    problem = load_json(run_dir / "problem_node.json")
    assert problem["problem_id"]


def test_missing_llm_blocks_instead_of_completing(b2_data: Path, tmp_path: Path) -> None:
    manifest = _orchestrator(tmp_path, b2_data, UnavailableProvider()).run()
    assert manifest["status"] in {"blocked_missing_llm", "blocked_llm_error"}
    assert manifest["status"] != "completed_smoke"


def test_deterministic_fallback_requires_explicit_flag(b2_data: Path, tmp_path: Path) -> None:
    manifest = _orchestrator(tmp_path, b2_data, UnavailableProvider(), allow_deterministic_fallback=True).run()
    assert manifest["status"] == "degraded_deterministic_fallback"
    assert manifest["planner_mode"] == "deterministic_fallback"
    assert manifest["status"] != "completed_smoke"


def test_force_new_execution_produces_unique_ids(b2_data: Path, tmp_path: Path) -> None:
    m1 = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    m2 = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    assert m1["execution_id"] != m2["execution_id"]
    assert m1["input_fingerprint"] == m2["input_fingerprint"]


def test_resume_keeps_execution_id(b2_data: Path, tmp_path: Path) -> None:
    m1 = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    resumed = _orchestrator(tmp_path, b2_data, FakeProvider(), resume_execution_id=m1["execution_id"], force_new_execution=False).run()
    assert resumed["execution_id"] == m1["execution_id"]
    assert resumed["resume_status"] == "resumed"


def test_m5_rejects_llm_proposal_with_forbidden_content(b2_data: Path, tmp_path: Path) -> None:
    bad = FakeProvider(proposal={"diagnostic_type": "candidate_recall_diagnostic", "note": "tune with test_truth", "success_condition": "x", "failure_condition": "y", "budget_seconds": 30})
    manifest = _orchestrator(tmp_path, b2_data, bad).run()
    assert manifest["status"] == "incomplete"
    run_dir = tmp_path / "v2_runs" / manifest["execution_id"]
    m5 = load_json(run_dir / "m5_decision.json")
    assert m5["status"] == "rejected"


def test_dashboard_shows_v2_identity(b2_data: Path, tmp_path: Path) -> None:
    manifest = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    run_dir = tmp_path / "v2_runs" / manifest["execution_id"]
    html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "V2 RUN" in html
    heartbeat = load_json(run_dir / "heartbeat.json")
    assert heartbeat["execution_id"] == manifest["execution_id"]
    assert heartbeat["orchestrator_version"].startswith("2")


def test_dashboard_marks_legacy_runs_red(tmp_path: Path) -> None:
    from afac_agent.supervisor.dashboard import render_dashboard
    from afac_agent.supervisor.heartbeat import HeartbeatState

    html = render_dashboard(HeartbeatState(run_id="legacy", legacy_mode=True))
    assert "LEGACY EXECUTION — NOT A V2 RUN" in html


def test_frozen_hashes_unchanged_by_smoke(b2_data: Path, tmp_path: Path) -> None:
    frozen = [PROJECT_ROOT / "config" / "project_state.json", PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"]
    before = {str(p): stable_hash(p.read_text(encoding="utf-8")) for p in frozen if p.is_file()}
    manifest = _orchestrator(tmp_path, b2_data, FakeProvider()).run()
    after = {str(p): stable_hash(p.read_text(encoding="utf-8")) for p in frozen if p.is_file()}
    assert before == after
    assert manifest["gates"]["frozen_asset_hash"] is True
    assert manifest["uses_test_truth"] is False
    assert manifest["mutates_frozen_assets"] is False
