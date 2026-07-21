# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from afac_agent.research.event_store import json_dumps
from afac_agent.research.live_method_research import (
    LiveCachedMethodResearchRunner,
    MockLiveProvider,
    MockResearchLLM,
    ProviderRegistry,
    ResearchCache,
    _safe_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = PROJECT_ROOT / "config" / "research_policy.json"
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _brief(tmp_path: Path) -> Path:
    payload = {
        "brief_version": "synthetic",
        "brief_id": "brief_exact2_live",
        "brief_type": "bucket_research_brief",
        "target_problem_ids": ["p_exact2"],
        "scope_level": "bucket",
        "scope_refs": {"axis_id": "train_label_reachability", "value_id": "exact2_only"},
        "primary_research_question": "Find source-grounded public methods for exact2-only multi-hop information.",
        "current_evidence": {},
        "evidence_gaps": ["missing final OOF", "missing canonical Fold"],
        "required_new_information": ["new source-grounded signal"],
        "success_conditions": ["primary source verified"],
        "failure_conditions": ["same information as failed route"],
        "stop_conditions": ["requires test truth"],
    }
    path = tmp_path / "brief.json"
    path.write_text(json_dumps(payload), encoding="utf-8")
    return path


def _memory(tmp_path: Path) -> Path:
    root = tmp_path / "memory"
    root.mkdir()
    attempts = {
        "attempts": [
            {
                "attempt_id": "a1",
                "method_id": "method::exact2_dir",
                "branch_id": "exact2_dup",
                "method_family": "gnn",
                "information_source_type": "directed_path_signal",
                "new_information_status": "new_information",
                "target_scope_refs": {"bucket_axes": [{"axis_id": "train_label_reachability", "value_id": "exact2_only"}]},
                "target_mechanism_ids": ["multi_hop_signal_opportunity"],
                "training_objective": "edge_utility",
                "configuration_identity": "cfg_exact_dup",
                "outcome": "failure",
            }
        ]
    }
    (root / "method_attempt_ledger.json").write_text(json_dumps(attempts), encoding="utf-8")
    (root / "failure_ledger.json").write_text(json_dumps({"failures": [{"failure_type": "negative_net"}]}), encoding="utf-8")
    (root / "research_problem_profile.json").write_text(json_dumps({"memory_id": "mock"}), encoding="utf-8")
    (root / "research_manifest.json").write_text(json_dumps({"memory_id": "mock"}), encoding="utf-8")
    return root


def _runner(*, fail_provider: bool = False) -> tuple[LiveCachedMethodResearchRunner, MockResearchLLM]:
    llm = MockResearchLLM()
    registry = ProviderRegistry([MockLiveProvider(fail=fail_provider)])
    return LiveCachedMethodResearchRunner(project_root=PROJECT_ROOT, registry=registry, llm_provider=llm), llm


def test_network_modes_provider_registry_and_unregistered_provider(tmp_path: Path) -> None:
    runner, llm = _runner()
    assert runner.registry.snapshot()["provider_count"] == 1
    try:
        runner.registry.get("not_registered")
    except ValueError as exc:
        assert "unregistered provider" in str(exc)
    else:
        raise AssertionError("unregistered provider should fail")

    disabled = runner.run(
        research_brief=_brief(tmp_path),
        research_memory_root=_memory(tmp_path),
        research_policy=POLICY,
        out_root=tmp_path / "out_disabled",
        cache_root=tmp_path / "cache",
        network_mode="disabled",
    )
    assert disabled["status"] == "completed"
    manifest = json.loads((tmp_path / "out_disabled" / disabled["run_id"] / "research_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["network_mode"] == "disabled"
    assert manifest["uses_network"] is False
    assert manifest["executes_adapter"] is False
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert llm.calls <= 2


def test_cache_hit_and_network_failure_fallback(tmp_path: Path) -> None:
    cache = ResearchCache(root=tmp_path / "cache")
    query_text = "graph neural network self-supervised multi-hop exact 2-hop node classification"
    source = cache.write_content("mock_live", canonical_url="https://example.org/cached", title="Cached source", source_type="preprint", content="# cached\nprimary source", source_level="primary_source")
    source.update({"url": source["canonical_url"], "authors_or_organization": "cached", "year": 2026, "venue": "cached", "source_level": "primary_source"})
    cache.write_query("mock_live", query_text, {"query_id": "cached", "query_text": query_text, "provider_id": "mock_live", "sources": [source], "cache_hits": 1, "network_requests": 0, "retrieval_status": "completed"})

    runner, _ = _runner(fail_provider=True)
    result = runner.run(
        research_brief=_brief(tmp_path),
        research_memory_root=_memory(tmp_path),
        research_policy=POLICY,
        out_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        network_mode="live_cached",
    )
    assert result["status"] == "completed"
    manifest = json.loads((tmp_path / "out" / result["run_id"] / "research_run_manifest.json").read_text(encoding="utf-8"))
    retrieval = json.loads((tmp_path / "out" / result["run_id"] / "retrieval_log.json").read_text(encoding="utf-8"))
    assert manifest["effective_network_mode"] == "cache_only"
    assert retrieval["fallback_used"] is True
    assert result["primary_source_count"] >= 1


def test_query_sanitization_budget_source_refs_conflict_and_ranking(tmp_path: Path) -> None:
    fake_key = "sk-" + "abcDEF1234567890abcdef"
    assert "sk-[REDACTED]" in _safe_text(f"secret {fake_key}")
    assert "[LOCAL_PATH_REDACTED]" in _safe_text(r"C:\Users\name\secret\file.txt")
    runner, llm = _runner()
    result = runner.run(
        research_brief=_brief(tmp_path),
        research_memory_root=_memory(tmp_path),
        research_policy=POLICY,
        out_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        network_mode="live_cached",
        max_queries=1,
    )
    assert result["status"] == "completed"
    run_dir = tmp_path / "out" / result["run_id"]
    queries = json.loads((run_dir / "research_queries.json").read_text(encoding="utf-8"))["items"]
    assert len(queries) == 1
    assert "query_text" in queries[0] and "exact" in queries[0]["query_text"].lower()
    cards = json.loads((run_dir / "method_cards_validated.json").read_text(encoding="utf-8"))
    assert cards["items"]
    assert cards["items"][0]["source_refs"]
    conflicts = json.loads((run_dir / "method_conflicts.json").read_text(encoding="utf-8"))["items"]
    assert conflicts and conflicts[0]["conflict_status"] in {"new_direction", "partial_overlap", "high_overlap", "exact_duplicate"}
    ranking = json.loads((run_dir / "method_ranking.json").read_text(encoding="utf-8"))
    assert ranking["items"] and "ranking_components" in ranking["items"][0]
    assert llm.calls == 2


def test_cli_mockless_missing_input_and_doctor_contracts(tmp_path: Path) -> None:
    missing = subprocess.run(
        [
            sys.executable, "-m", "afac_agent.main", "live-method-research",
            "--project_root", str(PROJECT_ROOT),
            "--research-brief", str(tmp_path / "missing.json"),
            "--research-memory-root", str(tmp_path / "missing_memory"),
            "--research-policy", str(POLICY),
            "--network-mode", "disabled",
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
    for key in ["live_method_research_run_schema", "research_provider_registry_schema", "research_query_schema", "live_provider_registry", "live_network_modes", "live_research_safety", "research_cache_output_root", "live_method_research_output_root"]:
        assert report["checks"][key]["passed"] is True


def test_frozen_files_and_rounds_unchanged_by_mock_live_run(tmp_path: Path) -> None:
    before = {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]}
    rounds_before = json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"]
    runner, _ = _runner()
    result = runner.run(
        research_brief=_brief(tmp_path),
        research_memory_root=_memory(tmp_path),
        research_policy=POLICY,
        out_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        network_mode="live_cached",
    )
    assert result["status"] == "completed"
    assert {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]} == before
    assert json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == rounds_before
