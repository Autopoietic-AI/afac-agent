# -*- coding: utf-8 -*-
"""Tests for the AFAC v2.0 knowledge base, portfolio, and online feedback guardrails."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from afac_agent.research.event_store import sha256_file
from afac_agent.v2.online_feedback import (
    ALLOWED_USES,
    FORBIDDEN_USES,
    OnlineFeedbackMisuse,
    load_baseline_feedback,
    use_feedback,
)
from afac_agent.v2.portfolio import ParentSelectionRejected, Portfolio

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "knowledge" / "v1_6_baseline"
CHAMPION_DIR = REPO_ROOT / "knowledge" / "recommendation" / "champions" / "A2_05093"

BASELINE_JSON_FILES = [
    "baseline_manifest.json",
    "online_results.json",
    "b1_postmortem.json",
    "b2_postmortem.json",
    "capability_gap.json",
    "validation_gap.json",
    "metric_semantics_gap.json",
    "no_op_rounds.json",
    "memory_failures.json",
    "v2_initial_priority_queue.json",
]

CHAMPION_FILES = [
    "champion_manifest.json",
    "artifact_registry.json",
    "pipeline_dag.json",
    "data_view_contract.json",
    "candidate_set_contract.json",
    "model_permission_contract.json",
    "evaluation_contract.json",
    "safety_contract.json",
    "experiment_lineage.json",
    "closed_routes.json",
    "ARCHITECTURE_REVIEW.md",
]

PORTABLE_PRINCIPLES = [
    "anchor_first",
    "bucket_specialist",
    "protected_residual",
    "novel_only_rerank",
    "boundary_admission",
    "source_consensus_gate",
]


# ---------------------------------------------------------------------------
# v1.6 baseline knowledge files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", BASELINE_JSON_FILES)
def test_baseline_knowledge_files_exist_and_parse(name: str) -> None:
    path = BASELINE_DIR / name
    assert path.is_file(), f"missing {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_baseline_postmortem_md_exists() -> None:
    assert (BASELINE_DIR / "V1_6_POSTMORTEM.md").is_file()


def test_baseline_manifest_frozen_facts() -> None:
    payload = json.loads((BASELINE_DIR / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert payload["commit"] == "30efa2cbb4264630e0e1e984d55e3fffeba354e1"
    assert payload["master_run_id"] == "86f75dbc66d25382eb0c6a24"
    assert payload["pytest_baseline"]["passed"] == 207
    assert payload["doctor"] == "PASS"
    assert payload["task_run_ids"]["B1_V1"] == "e95368a24e0780e65e92aceb"
    assert payload["task_run_ids"]["B1_V2"] == "c5e33663d37c6e49004e8213"
    assert payload["task_run_ids"]["B2_V1"] == "fcf5ad3dbcdf9700dd644eff"


def test_online_results_identity_verified_and_hashes_match() -> None:
    payload = json.loads((BASELINE_DIR / "online_results.json").read_text(encoding="utf-8"))
    entries = {entry["task"]: entry for entry in payload["entries"]}
    assert set(entries) == {"B1_V1", "B1_V2", "B2_V1"}
    for task, entry in entries.items():
        assert entry["identity_status"] == "verified", task
        assert entry["score_type"] == "competition_scalar_feedback", task
        submitted = REPO_ROOT / entry["submitted_file_path"]
        assert submitted.is_file(), f"missing submitted file {submitted}"
        assert sha256_file(submitted) == entry["submitted_file_sha256"], task
        assert set(entry["allowed_uses"]) == set(ALLOWED_USES), task
        assert set(entry["forbidden_uses"]) == set(FORBIDDEN_USES), task


def test_offline_online_gaps() -> None:
    payload = json.loads((BASELINE_DIR / "online_results.json").read_text(encoding="utf-8"))
    entries = {entry["task"]: entry for entry in payload["entries"]}
    v1 = entries["B1_V1"]
    assert v1["online_score"] == pytest.approx(0.37908, abs=1e-6)
    assert v1["offline_standard_score"] == pytest.approx(0.49069, abs=1e-6)
    assert v1["absolute_gap"] == pytest.approx(0.11161, abs=1e-6)
    v2 = entries["B1_V2"]
    assert v2["online_score"] == pytest.approx(0.37974, abs=1e-6)
    # Confirmed from the run report: 0.49640522875816995 (master manifest rounds to 0.4964).
    assert v2["offline_standard_score"] == pytest.approx(0.49641, abs=1e-6)
    assert v2["absolute_gap"] == pytest.approx(0.11667, abs=1e-6)
    b2 = entries["B2_V1"]
    assert b2["online_score"] == pytest.approx(0.06838, abs=1e-6)
    assert b2["offline_ndcg_at_10"] == pytest.approx(0.1548, abs=1e-6)
    assert b2["absolute_gap"] == pytest.approx(0.08642, abs=1e-6)


# ---------------------------------------------------------------------------
# A2 champion package
# ---------------------------------------------------------------------------

def test_champion_package_all_files_exist() -> None:
    for name in CHAMPION_FILES:
        assert (CHAMPION_DIR / name).is_file(), f"missing {name}"


@pytest.mark.parametrize("name", [n for n in CHAMPION_FILES if n.endswith(".json")])
def test_champion_json_files_parse(name: str) -> None:
    payload = json.loads((CHAMPION_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_champion_manifest_frozen_and_score() -> None:
    payload = json.loads((CHAMPION_DIR / "champion_manifest.json").read_text(encoding="utf-8"))
    assert payload["frozen"] is True
    assert payload["online_score"] == pytest.approx(0.5093, abs=1e-9)
    assert payload["historical_online_scores"] == [0.5052, 0.5066, 0.5068, 0.5092, 0.5093]
    assert set(payload["portable_principles"]) == set(PORTABLE_PRINCIPLES)


def test_model_permission_contract_forbids_non_len3_modification() -> None:
    payload = json.loads((CHAMPION_DIR / "model_permission_contract.json").read_text(encoding="utf-8"))
    text = json.dumps(payload, ensure_ascii=False)
    assert "non-Len3" in text
    assert "forbidden" in text.lower()
    protected = " ".join(str(p) for p in payload["protected"])
    assert "non-Len3" in protected
    for permission in payload["permissions"]:
        assert "non-Len3" in permission["forbidden"]


def test_architecture_review_mentions_portable_principles() -> None:
    text = (CHAMPION_DIR / "ARCHITECTURE_REVIEW.md").read_text(encoding="utf-8")
    for principle in PORTABLE_PRINCIPLES:
        assert principle in text, f"missing principle {principle}"


def test_champion_contracts_contain_no_raw_a2_ids() -> None:
    long_digit_run = re.compile(r"\d{8,}")
    for path in CHAMPION_DIR.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not long_digit_run.search(text), f"possible raw A2 id in {path.name}"
    # direct transfer of A2-specific material must be explicitly forbidden
    combined = "\n".join(p.read_text(encoding="utf-8") for p in CHAMPION_DIR.glob("*.json"))
    for forbidden in ("user ids", "item ids", "scores", "checkpoints", "weights"):
        assert forbidden in combined


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def _build_portfolio() -> Portfolio:
    portfolio = Portfolio()
    portfolio.register_candidate({"candidate_id": "cand_base", "metrics": {"standard": 0.49, "macro": 0.39}})
    portfolio.register_candidate(
        {
            "candidate_id": "cand_challenger",
            "metrics": {"standard": 0.50, "macro": 0.40},
            "roles": ["best_overall"],
        }
    )
    return portfolio


def test_portfolio_default_parent_is_incumbent() -> None:
    portfolio = _build_portfolio()
    parent = portfolio.get_parent()
    assert parent.candidate_id == "cand_base"
    assert "incumbent" in parent.roles


def test_portfolio_unjustified_non_incumbent_parent_raises() -> None:
    portfolio = _build_portfolio()
    with pytest.raises(ParentSelectionRejected):
        portfolio.get_parent(requested_role="best_overall")
    with pytest.raises(ParentSelectionRejected):
        portfolio.get_parent(requested_role="best_overall", orthogonal_reason="orthogonal signal")
    with pytest.raises(ParentSelectionRejected):
        portfolio.get_parent(
            requested_role="best_overall",
            orthogonal_reason="orthogonal signal",
            scope_difference="different scope",
        )


def test_portfolio_justified_non_incumbent_parent_returns_role_candidate() -> None:
    portfolio = _build_portfolio()
    parent = portfolio.get_parent(
        requested_role="best_overall",
        orthogonal_reason="different model family",
        expected_complementarity="covers class-7 residual",
        scope_difference="target bucket only",
    )
    assert parent.candidate_id == "cand_challenger"
    assert "best_overall" in parent.roles


# ---------------------------------------------------------------------------
# Online feedback guardrails
# ---------------------------------------------------------------------------

def test_load_baseline_feedback_verifies_identity() -> None:
    records = load_baseline_feedback(REPO_ROOT)
    assert len(records) == 3
    for record in records:
        assert record.identity_status == "verified"
        assert record.score_type == "competition_scalar_feedback"


def test_use_feedback_forbidden_purposes_raise() -> None:
    records = load_baseline_feedback(REPO_ROOT)
    for record in records:
        for purpose in ("node_level_correction", "item_level_correction", "test_label_inference"):
            with pytest.raises(OnlineFeedbackMisuse):
                use_feedback(record, purpose)


def test_use_feedback_allowed_purposes_pass() -> None:
    records = load_baseline_feedback(REPO_ROOT)
    for record in records:
        for purpose in ALLOWED_USES:
            assert use_feedback(record, purpose) is record
