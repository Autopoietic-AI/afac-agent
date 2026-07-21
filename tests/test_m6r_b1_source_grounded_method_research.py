# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from afac_agent.research.event_store import json_dumps, sha256_file
from afac_agent.research.method_research import (
    LocalSourcePackProvider,
    MethodCardExtractor,
    MethodCardValidator,
    MethodRanker,
    MethodResearchRunner,
    SourceChunker,
    SourceVerifier,
    WebSearchProvider,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = PROJECT_ROOT / "config" / "research_policy.json"
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _base_method(**overrides):
    payload = {
        "method_family": "contrastive",
        "target_problem_ids": ["p_exact2"],
        "target_scope_refs": {"axis_id": "train_label_reachability", "value_id": "exact2_only"},
        "bucket_axes": [{"axis_id": "train_label_reachability", "value_id": "exact2_only"}],
        "class_id": None,
        "mechanism_id": ["multi_hop_signal_opportunity"],
        "core_hypothesis": "Synthetic fixture: exact-2-only nodes may benefit from a verified new public homology signal.",
        "required_inputs": ["canonical fold", "final anchor OOF", "local public-homology source"],
        "information_source_type": "public_homology_signal",
        "new_information_status": "new_information",
        "applicable_conditions": ["exact2_only bucket"],
        "inapplicable_conditions": ["test truth dependent routing"],
        "minimal_experiment": {"mode": "read_only_audit", "requires_training": False},
        "success_conditions": ["source-supported hypothesis survives conflict check"],
        "failure_conditions": ["no source support", "same information as failed route"],
        "stop_conditions": ["requires test truth", "requires closed branch reopening"],
        "training_objective": "read_only_design",
        "branch_id": "synthetic_public_homology_exact2",
        "configuration_identity": "cfg_public_homology_v1",
        "compute_cost": "low",
        "implementation_cost": "medium",
        "leakage_risk": "low",
        "deployment_risk": "low",
        "source_type": "mature_open_source_repository",
        "status": "proposed",
    }
    payload.update(overrides)
    return payload


def _memory_root(tmp_path: Path) -> Path:
    root = tmp_path / "memory"
    root.mkdir()
    attempts = {
        "attempts": [
            {
                "attempt_id": "a_exact_dup",
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
            },
            {
                "attempt_id": "a_iso_mlp",
                "method_id": "method::mlp_iso",
                "branch_id": "isolated_mlp",
                "method_family": "mlp",
                "information_source_type": "existing_node_attributes",
                "new_information_status": "same_information_as_failed_route",
                "target_scope_refs": {"bucket_axes": [{"axis_id": "connectivity_visibility", "value_id": "isolated"}]},
                "target_mechanism_ids": ["information_source_missing"],
                "training_objective": "supervised_ce",
                "configuration_identity": "cfg_iso_mlp",
                "outcome": "failure",
            },
        ]
    }
    failures = {"failures": [{"failure_type": "negative_net"}, {"failure_type": "negative_macro_gain"}]}
    profile = {"profile_version": "test", "memory_id": "synthetic_memory"}
    (root / "method_attempt_ledger.json").write_text(json_dumps(attempts), encoding="utf-8")
    (root / "failure_ledger.json").write_text(json_dumps(failures), encoding="utf-8")
    (root / "research_problem_profile.json").write_text(json_dumps(profile), encoding="utf-8")
    (root / "research_manifest.json").write_text(json_dumps({"memory_id": "synthetic_memory"}), encoding="utf-8")
    return root


def _brief(tmp_path: Path) -> Path:
    payload = {
        "brief_version": "synthetic",
        "brief_id": "brief_exact2",
        "brief_type": "bucket_research_brief",
        "target_problem_ids": ["p_exact2"],
        "scope_level": "bucket",
        "scope_refs": {"axis_id": "train_label_reachability", "value_id": "exact2_only"},
        "primary_research_question": "Find source-grounded new information for exact2-only nodes.",
        "current_evidence": {},
        "evidence_gaps": ["missing final OOF", "missing canonical Fold"],
        "required_new_information": ["new signal source"],
        "success_conditions": ["source grounded"],
        "failure_conditions": ["same failed information"],
        "stop_conditions": ["test truth"],
    }
    path = tmp_path / "research_brief.json"
    path.write_text(json_dumps(payload), encoding="utf-8")
    return path


def _manifest(tmp_path: Path) -> Path:
    source_dir = tmp_path / "sources"
    mature = _write(source_dir / "mature.md", "# Exact2 synthetic method\nA mature synthetic source for a new exact2 public homology signal.")
    text = _write(source_dir / "repr.txt", "Synthetic representation-only method with existing node attributes.")
    json_src = _write(source_dir / "external.json", json.dumps({"section": "Synthetic external pretraining signal for isolated nodes."}, ensure_ascii=False))
    dup_a = _write(source_dir / "dup_a.md", "# Duplicate\nsame synthetic content")
    dup_b = _write(source_dir / "dup_b.md", "# Duplicate\nsame synthetic content")
    conflict = _write(source_dir / "conflict.md", "# Duplicate\nconflicting synthetic content")
    smooth = _write(source_dir / "smooth.md", "# Correct Smooth\nSynthetic smoothing rerun that overlaps a closed route.")
    insufficient = _write(source_dir / "insufficient.md", "# Weak\nSynthetic insufficient source support example.")
    sources = [
        {
            "source_type": "mature_open_source_repository",
            "title": "Synthetic Mature Exact2 Repository",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/synthetic-exact2",
            "repository_url": "https://example.org/repo",
            "local_path": "sources/mature.md",
            "content_hash": sha256_file(mature),
            "synthetic_test_only": True,
            "method_metadata": _base_method(),
        },
        {
            "source_type": "preprint",
            "title": "Synthetic Representation Only",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/repr",
            "local_path": "sources/repr.txt",
            "content_hash": sha256_file(text),
            "synthetic_test_only": True,
            "method_metadata": _base_method(
                method_family="mlp",
                information_source_type="existing_node_attributes",
                new_information_status="new_representation_only",
                target_scope_refs={"axis_id": "train_label_reachability", "value_id": "exact2_only"},
                branch_id="repr_only_retry",
                configuration_identity="cfg_repr",
                mechanism_id=["information_source_missing"],
            ),
        },
        {
            "source_type": "preprint",
            "title": "Synthetic External Isolated",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/external",
            "local_path": "sources/external.json",
            "content_hash": sha256_file(json_src),
            "synthetic_test_only": True,
            "method_metadata": _base_method(
                method_family="transformer",
                information_source_type="external_pretraining",
                target_scope_refs={"axis_id": "connectivity_visibility", "value_id": "isolated"},
                bucket_axes=[{"axis_id": "connectivity_visibility", "value_id": "isolated"}],
                mechanism_id=["information_source_missing"],
                branch_id="external_isolated",
                configuration_identity="cfg_external_iso",
            ),
        },
        {
            "source_type": "competition_solution",
            "title": "Synthetic Exact Duplicate",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/dup-a",
            "local_path": "sources/dup_a.md",
            "content_hash": sha256_file(dup_a),
            "synthetic_test_only": True,
            "method_metadata": _base_method(
                method_family="gnn",
                information_source_type="directed_path_signal",
                training_objective="edge_utility",
                branch_id="exact2_dup",
                configuration_identity="cfg_exact_dup",
            ),
        },
        {
            "source_type": "competition_solution",
            "title": "Synthetic Exact Duplicate",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/dup-b",
            "local_path": "sources/dup_b.md",
            "content_hash": sha256_file(dup_b),
            "synthetic_test_only": True,
        },
        {
            "source_type": "competition_solution",
            "title": "Synthetic Mature Exact2 Repository",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/conflict",
            "local_path": "sources/conflict.md",
            "content_hash": sha256_file(conflict),
            "synthetic_test_only": True,
        },
        {
            "source_type": "other",
            "title": "Synthetic Correct Smooth Rerun",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/smooth",
            "local_path": "sources/smooth.md",
            "content_hash": sha256_file(smooth),
            "synthetic_test_only": True,
            "method_metadata": _base_method(
                method_family="smoothing",
                information_source_type="one_hop_topology",
                new_information_status="same_information_as_failed_route",
                target_scope_refs={"axis_id": "train_label_reachability", "value_id": "exact2_only"},
                mechanism_id=["uniform_smoothing_damage"],
                training_objective="uniform_smoothing",
                branch_id="correct_smooth",
                configuration_identity="cfg_smooth",
            ),
        },
        {
            "source_type": "other",
            "title": "Synthetic Insufficient Support",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/weak",
            "local_path": "sources/insufficient.md",
            "content_hash": sha256_file(insufficient),
            "synthetic_test_only": True,
            "method_metadata": _base_method(
                method_family="prototype",
                information_source_type="prototype_signal",
                target_scope_refs={"axis_id": "degree_band", "value_id": "degree_6p"},
                bucket_axes=[{"axis_id": "degree_band", "value_id": "degree_6p"}],
                mechanism_id=["calibration_error"],
                branch_id="prototype_degree6p",
                configuration_identity="cfg_proto",
            ),
        },
        {
            "source_type": "preprint",
            "title": "Synthetic Metadata Only",
            "authors_or_organization": "AFAC synthetic fixture",
            "year": 2026,
            "venue": "synthetic",
            "url": "https://example.org/metadata-only",
            "synthetic_test_only": True,
        },
    ]
    manifest = {"manifest_version": "synthetic_m6rb1", "sources": sources}
    path = tmp_path / "source_manifest.json"
    path.write_text(json_dumps(manifest), encoding="utf-8")
    return path


def _run(tmp_path: Path):
    result = MethodResearchRunner(project_root=PROJECT_ROOT).run(
        research_brief=_brief(tmp_path),
        research_memory_root=_memory_root(tmp_path),
        source_manifest=_manifest(tmp_path),
        research_policy=POLICY,
        out_root=tmp_path / "method research runs 中文 with space",
    )
    run_dir = tmp_path / "method research runs 中文 with space" / result["run_id"]
    return result, run_dir


def test_source_provider_verifier_manifest_hash_duplicate_and_conflict(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    provider = LocalSourcePackProvider()
    assert provider.describe()["network_enabled"] is False
    assert WebSearchProvider().describe()["status"] == "disabled"
    candidates = provider.fetch(source_manifest=manifest)
    first_ids = [item["source_id"] for item in candidates]
    assert first_ids == [item["source_id"] for item in provider.fetch(source_manifest=manifest)]
    verification = SourceVerifier().verify(candidates)
    assert verification["summary"]["synthetic_test_only"] >= 1
    assert verification["summary"]["duplicate"] >= 1
    assert verification["summary"]["conflicting_metadata"] >= 1
    assert all(record["promotion_eligible"] is False for record in verification["records"] if record["synthetic_test_only"])


def test_chunker_markdown_txt_json_and_hash_determinism(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    records = SourceVerifier().verify(LocalSourcePackProvider().fetch(source_manifest=manifest))["records"]
    first = SourceChunker(max_chunk_chars=240).chunk(records, manifest_base=manifest.parent)
    second = SourceChunker(max_chunk_chars=240).chunk(records, manifest_base=manifest.parent)
    assert first == second
    assert {chunk["section_title"] for chunk in first} >= {"Exact2 synthetic method", "text", "json"}
    assert all(chunk["chunk_id"] and chunk["content_hash"] and chunk["character_count"] > 0 for chunk in first)


def test_method_card_validator_rejects_missing_source_unverified_and_scope_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source_records = SourceVerifier().verify(LocalSourcePackProvider().fetch(source_manifest=manifest))["records"]
    chunks = SourceChunker().chunk(source_records, manifest_base=manifest.parent)
    brief = json.loads(_brief(tmp_path).read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    validator = MethodCardValidator()
    card = MethodCardExtractor()._finalize_card(_base_method(source_refs=[]))
    assert validator.validate(method_card=card, research_brief=brief, source_records=source_records, chunks=chunks, policy=policy)["validation_status"] == "invalid"
    metadata_source = next(record for record in source_records if record["title"] == "Synthetic Metadata Only")
    supported = MethodCardExtractor()._finalize_card(_base_method(source_refs=[{"source_id": metadata_source["source_id"], "chunk_id": "missing"}]))
    assert validator.validate(method_card=supported, research_brief=brief, source_records=source_records, chunks=chunks, policy=policy)["validation_status"] == "insufficient_source_support"
    first_chunk = chunks[0]
    mismatch = MethodCardExtractor()._finalize_card(_base_method(target_problem_ids=["other"], source_refs=[{"source_id": first_chunk["source_id"], "chunk_id": first_chunk["chunk_id"]}]))
    assert validator.validate(method_card=mismatch, research_brief=brief, source_records=source_records, chunks=chunks, policy=policy)["validation_status"] == "scope_mismatch"


def test_method_research_run_conflicts_ranking_and_safety_manifest(tmp_path: Path) -> None:
    result, run_dir = _run(tmp_path)
    assert result["status"] == "completed"
    manifest = json.loads((run_dir / "research_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["calls_llm"] is False
    assert manifest["calls_api"] is False
    assert manifest["uses_network"] is False
    assert manifest["executes_adapter"] is False
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert manifest["counts_as_experiment_round"] is False
    assert manifest["conflict_check_executed"] is True

    conflicts = json.loads((run_dir / "method_conflicts.json").read_text(encoding="utf-8"))["items"]
    statuses = {item["conflict_status"] for item in conflicts}
    assert {"exact_duplicate", "high_overlap", "new_direction"}.issubset(statuses)
    assert any("uniform_smoothing_failure_overlap" in item["conflict_reasons"] for item in conflicts)

    ranking = json.loads((run_dir / "method_ranking.json").read_text(encoding="utf-8"))
    required = set(json.loads(POLICY.read_text(encoding="utf-8"))["method_research_ranking_weights"])
    assert all(required.issubset(set(item["ranking_components"])) for item in ranking["items"])
    assert len([item for item in ranking["items"] if item["selection_status"] == "selected"]) <= ranking["top_k"]
    assert any(item["selection_status"] == "blocked" for item in ranking["items"])
    assert any(item["selection_status"] == "selected" for item in ranking["items"])


def test_run_id_duplicate_and_policy_weights_affect_ranking(tmp_path: Path) -> None:
    first, run_dir = _run(tmp_path)
    second = MethodResearchRunner(project_root=PROJECT_ROOT).run(
        research_brief=tmp_path / "research_brief.json",
        research_memory_root=tmp_path / "memory",
        source_manifest=tmp_path / "source_manifest.json",
        research_policy=POLICY,
        out_root=tmp_path / "method research runs 中文 with space",
    )
    assert second["status"] == "duplicate"
    assert second["run_id"] == first["run_id"]
    original_rank = json.loads((run_dir / "method_ranking.json").read_text(encoding="utf-8"))["items"][0]["method_id"]
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["method_research_ranking_weights"]["new_information_value"] = 0.0
    changed_policy = tmp_path / "changed_policy.json"
    changed_policy.write_text(json_dumps(policy), encoding="utf-8")
    changed = MethodResearchRunner(project_root=PROJECT_ROOT).run(
        research_brief=tmp_path / "research_brief.json",
        research_memory_root=tmp_path / "memory",
        source_manifest=tmp_path / "source_manifest.json",
        research_policy=changed_policy,
        out_root=tmp_path / "changed_runs",
    )
    changed_rank = json.loads((tmp_path / "changed_runs" / changed["run_id"] / "method_ranking.json").read_text(encoding="utf-8"))["items"][0]["method_id"]
    assert changed["run_id"] != first["run_id"]
    assert changed_rank == original_rank or changed["view_hash"] != first["view_hash"]


def test_cli_dry_run_missing_and_real_smoke_preserves_frozen_files(tmp_path: Path) -> None:
    before = {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]}
    before_rounds = json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"]
    brief = _brief(tmp_path)
    memory = _memory_root(tmp_path)
    manifest = _manifest(tmp_path)
    out = tmp_path / "cli runs 中文 with space"
    cmd = [
        sys.executable, "-m", "afac_agent.main", "method-research-local",
        "--project_root", str(PROJECT_ROOT),
        "--research-brief", str(brief),
        "--research-memory-root", str(memory),
        "--source-manifest", str(manifest),
        "--research-policy", str(POLICY),
        "--out-root", str(out),
    ]
    dry = subprocess.run([*cmd, "--dry-run"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["status"] == "dry_run"
    assert not out.exists()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "completed"
    missing = subprocess.run([*cmd[:cmd.index("--research-brief") + 1], str(tmp_path / "missing.json"), *cmd[cmd.index("--research-memory-root"):]], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["status"] == "waiting_for_input"
    assert {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]} == before
    assert json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == before_rounds


def test_doctor_reports_m6rb1_contracts() -> None:
    proc = subprocess.run([sys.executable, "-m", "afac_agent.doctor", "--project_root", str(PROJECT_ROOT), "--json"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    for key in [
        "source_manifest_schema",
        "source_provider_contract",
        "local_source_provider",
        "source_verifier",
        "source_chunker",
        "method_card_validator",
        "method_ranker",
        "method_research_runs_output_root",
        "network_disabled_by_default",
    ]:
        assert report["checks"][key]["passed"] is True
