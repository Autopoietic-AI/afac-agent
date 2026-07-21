# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from afac_agent.research.decision_core import (
    DecisionCoreRunner,
    MethodQualityGate,
    MockDecisionLLM,
)
from afac_agent.research.event_store import json_dumps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _brief(tmp_path: Path) -> Path:
    payload = {
        "brief_version": "synthetic",
        "brief_id": "brief_one_hop",
        "brief_type": "bucket_research_brief",
        "target_problem_ids": ["p_one_hop"],
        "scope_level": "bucket",
        "scope_refs": {"axis_id": "train_label_reachability", "value_id": "one_hop_available"},
        "primary_research_question": "Find source-grounded methods for one-hop available graph node classification.",
        "evidence_gaps": ["missing canonical Fold", "missing final v53Q-1 OOF"],
        "required_new_information": ["source-grounded new signal"],
        "success_conditions": ["eligible Method Card"],
        "failure_conditions": ["inspiration-only source"],
        "stop_conditions": ["requires test truth"],
    }
    path = tmp_path / "brief.json"
    path.write_text(json_dumps(payload), encoding="utf-8")
    return path


def _card(method_id: str, name: str, source_id: str, chunk_id: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "method_id": method_id,
        "method_name": name,
        "method_family": "gnn",
        "information_source_type": "graph_node_classification",
        "new_information_status": "new_information",
        "bucket_axes": [{"axis_id": "train_label_reachability", "value_id": "one_hop_available"}],
        "mechanism_id": ["neighborhood_reliability_heterogeneity"],
        "core_hypothesis": "Use heterophily-aware message passing for graph node classification.",
        "source_refs": [{"source_id": source_id, "chunk_id": chunk_id}],
        "chunk_refs": [chunk_id],
        "training_objective": "diagnostic",
        "branch_id": f"branch_{method_id}",
    }
    payload.update(extra)
    return payload


def _live_run(tmp_path: Path) -> Path:
    root = tmp_path / "live"
    root.mkdir(exist_ok=True)
    sources = [
        {
            "source_id": "src_graph",
            "title": "Modeling Heterophily in Graphs for Node Classification",
            "source_type": "preprint",
            "source_level": "primary_source",
            "verification_status": "verified_local_content",
            "venue": "mock",
        },
        {
            "source_id": "src_routing",
            "title": "Node Disjoint Multipath Routing Considering Link and Node Stability",
            "source_type": "paper",
            "source_level": "primary_source",
            "verification_status": "verified_local_content",
            "venue": "mock",
        },
        {
            "source_id": "src_moe",
            "title": "Task Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers",
            "source_type": "paper",
            "source_level": "primary_source",
            "verification_status": "verified_local_content",
            "venue": "mock",
        },
        {
            "source_id": "src_incomplete",
            "title": "",
            "source_type": "paper",
            "source_level": "primary_source",
            "verification_status": "verified_local_content",
        },
    ]
    chunks = [
        {"chunk_id": "ch_graph", "source_id": "src_graph", "content": "Graph node classification with heterophily-aware message passing improves neighborhood reliability."},
        {"chunk_id": "ch_route", "source_id": "src_routing", "content": "Multipath network routing uses link and node stability for packet delivery."},
        {"chunk_id": "ch_moe", "source_id": "src_moe", "content": "Transformer mixture-of-experts routing signatures allocate prompts to experts."},
        {"chunk_id": "ch_incomplete", "source_id": "src_incomplete", "content": "Unknown source."},
    ]
    cards = [
        _card("eligible_graph", "Heterophily Graph Reliability", "src_graph", "ch_graph"),
        _card("inspiration_moe", "Routing Signature Transfer", "src_moe", "ch_moe"),
        _card("unsupported_route", "Heterophily Claim From Routing", "src_routing", "ch_route", core_hypothesis="Claims graph heterophily node classification from routing paper."),
        _card("incomplete_source", "Incomplete Source Identity", "src_incomplete", "ch_incomplete"),
        _card("closed_scalenet", "ScaleNet Scale Invariance", "src_graph", "ch_graph", branch_id="closed_scalenet"),
        _card("closed_smooth", "Correct Smooth Reopen", "src_graph", "ch_graph", mechanism_id=["uniform_smoothing_damage"]),
    ]
    validations = [{"method_id": card["method_id"], "valid": True, "errors": []} for card in cards]
    (root / "source_verification.json").write_text(json_dumps({"verification_version": "synthetic", "records": sources, "summary": {}}), encoding="utf-8")
    (root / "source_chunks.jsonl").write_text("\n".join(json.dumps(chunk, ensure_ascii=False, sort_keys=True) for chunk in chunks) + "\n", encoding="utf-8")
    (root / "method_cards_validated.json").write_text(json_dumps({"items": cards, "validations": validations}), encoding="utf-8")
    (root / "method_ranking.json").write_text(json_dumps({"items": [{"method_id": "inspiration_moe", "rank": 1}]}), encoding="utf-8")
    return root


def _memory(tmp_path: Path) -> Path:
    root = tmp_path / "memory"
    root.mkdir(exist_ok=True)
    attempts = {
        "attempts": [
            {
                "attempt_id": "smooth_failed",
                "method_id": "correct_smooth",
                "branch_id": "correct_smooth",
                "method_family": "gnn",
                "information_source_type": "graph_node_classification",
                "new_information_status": "same_information_as_failed_route",
                "target_scope_refs": {"bucket_axes": [{"axis_id": "train_label_reachability", "value_id": "one_hop_available"}]},
                "target_mechanism_ids": ["uniform_smoothing_damage"],
                "training_objective": "diagnostic",
                "outcome": "failure",
            }
        ]
    }
    (root / "method_attempt_ledger.json").write_text(json_dumps(attempts), encoding="utf-8")
    (root / "failure_ledger.json").write_text(json_dumps({"failures": [{"failure_type": "negative_net"}]}), encoding="utf-8")
    (root / "success_ledger.json").write_text(json_dumps({"successes": []}), encoding="utf-8")
    (root / "research_problem_profile.json").write_text(json_dumps({"memory_id": "mock"}), encoding="utf-8")
    return root


def _state(tmp_path: Path) -> Path:
    path = tmp_path / "project_state.json"
    path.write_text(json_dumps({"budget": {"rounds_used": 0}, "closed_branches": ["closed_scalenet", "correct_smooth"]}), encoding="utf-8")
    return path


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "tool_registry.json"
    path.write_text(json_dumps({"tools": [{"name": "SAFE_READ_ONLY", "counts_as_experiment_round": False, "risk_level": "low"}]}), encoding="utf-8")
    return path


def test_source_relevance_and_method_quality_statuses(tmp_path: Path) -> None:
    live = _live_run(tmp_path)
    cards = json.loads((live / "method_cards_validated.json").read_text(encoding="utf-8"))
    sources = json.loads((live / "source_verification.json").read_text(encoding="utf-8"))["records"]
    chunks = [json.loads(line) for line in (live / "source_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    gate = MethodQualityGate().evaluate(
        method_cards=cards["items"],
        validations=cards["validations"],
        source_records=sources,
        chunks=chunks,
        research_brief=json.loads(_brief(tmp_path).read_text(encoding="utf-8")),
        attempts=json.loads((_memory(tmp_path) / "method_attempt_ledger.json").read_text(encoding="utf-8")),
        failures={"failures": [{"failure_type": "negative_net"}]},
        project_state=json.loads(_state(tmp_path).read_text(encoding="utf-8")),
    )
    statuses = {row["method_card"]["method_id"]: row["quality"]["final_status"] for row in gate["items"]}
    reasons = {row["method_card"]["method_id"]: row["quality"]["blocked_reasons"] for row in gate["items"]}
    assert statuses["eligible_graph"] == "eligible"
    assert statuses["inspiration_moe"] == "inspiration_only"
    assert "unsupported_claim" in reasons["unsupported_route"]
    assert statuses["incomplete_source"] == "blocked"
    assert "closed_scalenet_route_related" in reasons["closed_scalenet"]
    assert "correct_smooth_closed_route" in reasons["closed_smooth"]


def test_decision_core_mock_run_revise_admission_and_artifacts(tmp_path: Path) -> None:
    runner = DecisionCoreRunner(project_root=PROJECT_ROOT, llm_provider=MockDecisionLLM("revise"))
    result = runner.run(
        research_brief=_brief(tmp_path),
        live_research_run=_live_run(tmp_path),
        research_memory_root=_memory(tmp_path),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "decision",
    )
    assert result["status"] == "completed"
    run_dir = tmp_path / "decision" / result["run_id"]
    manifest = json.loads((run_dir / "decision_manifest.json").read_text(encoding="utf-8"))
    outbound = json.loads((run_dir / "outbound_audit.json").read_text(encoding="utf-8"))
    final = json.loads((run_dir / "experiment_proposals_final.json").read_text(encoding="utf-8"))
    critic = json.loads((run_dir / "critic_review.json").read_text(encoding="utf-8"))
    admission = json.loads((run_dir / "admission_decision.json").read_text(encoding="utf-8"))
    assert manifest["executes_adapter"] is False
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert manifest["counts_as_experiment_round"] is False
    assert manifest["llm_call_count"] == 2
    assert outbound["secret_scan_pass"] is True
    assert outbound["test_truth_present"] is False
    assert (run_dir / "m6b_method_eligibility.json").exists()
    assert (run_dir / "m6b_experiment_proposals.json").exists()
    assert (run_dir / "m6c_critic_review.json").exists()
    assert (run_dir / "m6c_revised_proposals.json").exists()
    assert (run_dir / "m5_admission_decision.json").exists()
    assert critic["verdict"] == "revise"
    assert final["status"] == "revised"
    assert final["primary_proposal"]["minimal_experiment"]["mode"] == "diagnostic_only"
    assert admission["m5_authoritative"] is True


def test_critic_approve_reject_and_m5_authority(tmp_path: Path) -> None:
    approve = DecisionCoreRunner(project_root=PROJECT_ROOT, llm_provider=MockDecisionLLM("approve")).run(
        research_brief=_brief(tmp_path),
        live_research_run=_live_run(tmp_path),
        research_memory_root=_memory(tmp_path),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "approve",
    )
    approve_dir = tmp_path / "approve" / approve["run_id"]
    assert json.loads((approve_dir / "admission_decision.json").read_text(encoding="utf-8"))["m5_authoritative"] is True

    reject = DecisionCoreRunner(project_root=PROJECT_ROOT, llm_provider=MockDecisionLLM("reject")).run(
        research_brief=_brief(tmp_path),
        live_research_run=_live_run(tmp_path),
        research_memory_root=_memory(tmp_path),
        project_state=_state(tmp_path),
        tool_registry=_registry(tmp_path),
        out_root=tmp_path / "reject",
    )
    reject_dir = tmp_path / "reject" / reject["run_id"]
    assert json.loads((reject_dir / "experiment_proposals_final.json").read_text(encoding="utf-8"))["status"] == "blocked"
    assert json.loads((reject_dir / "admission_decision.json").read_text(encoding="utf-8"))["status"] == "blocked"


def test_cli_missing_input_and_doctor_contracts(tmp_path: Path) -> None:
    missing = subprocess.run(
        [
            sys.executable, "-m", "afac_agent.main", "decision-core-run",
            "--project_root", str(PROJECT_ROOT),
            "--research-brief", str(tmp_path / "missing.json"),
            "--live-research-run", str(tmp_path / "missing_live"),
            "--research-memory-root", str(tmp_path / "missing_memory"),
            "--project-state", str(STATE),
            "--tool-registry", str(PROJECT_ROOT / "config" / "tool_registry.json"),
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
    for key in [
        "source_relevance_audit_schema",
        "experiment_proposal_schema",
        "critic_review_schema",
        "decision_core_manifest_schema",
        "decision_core_output_root",
        "decision_core_module",
        "decision_core_safety",
    ]:
        assert report["checks"][key]["passed"] is True


def test_frozen_files_and_rounds_unchanged_by_mock_decision_core(tmp_path: Path) -> None:
    before = {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]}
    rounds_before = json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"]
    result = DecisionCoreRunner(project_root=PROJECT_ROOT, llm_provider=MockDecisionLLM("revise")).run(
        research_brief=_brief(tmp_path),
        live_research_run=_live_run(tmp_path),
        research_memory_root=_memory(tmp_path),
        project_state=STATE,
        tool_registry=PROJECT_ROOT / "config" / "tool_registry.json",
        out_root=tmp_path / "out",
    )
    assert result["status"] == "completed"
    assert {_p: _sha(_p) for _p in [CHAMPION, STATE, HISTORY]} == before
    assert json.loads(STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == rounds_before
