# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from afac_agent.evaluation_anchor_bootstrap import EvaluationAnchorBootstrap
from afac_agent.m7b_readiness import M7BReadinessRepair
from afac_agent.research.event_store import json_dumps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"


def _paths_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "paths.local.yaml"
    cfg.write_text("\n".join(["a1:", f"  anchor_csv: \"{CHAMPION}\""]), encoding="utf-8")
    return cfg


def _synthetic_inputs(tmp_path: Path, *, v43_source_matches: bool) -> dict[str, Path]:
    train_idx = np.arange(11001, dtype=np.int64)
    test_idx = np.arange(11001, 13752, dtype=np.int64)
    labels = train_idx % 10
    all_labels = np.concatenate([labels, np.full(len(test_idx), -1, dtype=np.int64)])
    a1 = tmp_path / "A1.npz"
    np.savez(a1, train_idx=train_idx, test_idx=test_idx, labels=all_labels)

    fold = tmp_path / "fold_assignment_5.csv"
    with fold.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "label", "fold"])
        writer.writeheader()
        for idx, label in zip(train_idx.tolist(), labels.tolist(), strict=True):
            writer.writerow({"node_id": idx, "label": label, "fold": idx % 5})

    base = np.full((11001, 10), 0.1, dtype=np.float64)
    expert = np.zeros((11001, 10), dtype=np.float64)
    expert[np.arange(11001), labels] = 1.0
    isolated_mask = (train_idx % 7) == 0
    composed = base.copy()
    composed[isolated_mask] = expert[isolated_mask]

    source_base = base.copy()
    if not v43_source_matches:
        source_base = np.zeros((11001, 10), dtype=np.float64)
        source_base[:, 0] = 1.0
    source = tmp_path / "v43c_3seed_oof_proba.npz"
    np.savez(source, train_idx=train_idx, y=labels, anchor_proba=source_base, ensemble_proba=source_base)

    v43 = tmp_path / "correct_smooth_v1_oof_proba.npz"
    np.savez(v43, proba=base, base_proba=base, train_idx=train_idx, labels=labels, bucket=np.zeros(11001, dtype=np.int64))
    (tmp_path / "correct_smooth_v1_decision.json").write_text(
        json_dumps({"base_oof_sources": [str(source)]}),
        encoding="utf-8",
    )

    v46 = tmp_path / "two_seed_balanced_candidate_oof.npz"
    np.savez(v46, proba=composed, expert_proba=expert, train_idx=train_idx, labels=labels, isolated_mask=isolated_mask)
    return {"a1": a1, "fold": fold, "v43": v43, "v46": v46}


def test_evaluation_anchor_bootstrap_materializes_with_verified_synthetic_components(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _synthetic_inputs(tmp_path, v43_source_matches=True)
    result = EvaluationAnchorBootstrap(project_root=project, paths_config=str(_paths_config(tmp_path))).run(
        a1_npz=inputs["a1"],
        fold_candidate=inputs["fold"],
        v43c_oof=inputs["v43"],
        v46a_oof=inputs["v46"],
        out_root=project / "artifacts" / "evaluation_anchor",
    )
    assert result["status"] == "materialized_from_verified_components"
    out = project / "artifacts" / "evaluation_anchor"
    manifest = json.loads((out / "evaluation_anchor_bootstrap_manifest.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((out / "A1_EVAL_ANCHOR_V1_manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_v53q1_oof_status"] == "not_materialized"
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert eval_manifest["evaluation_anchor_identity"] == "A1_EVAL_ANCHOR_V1"
    assert eval_manifest["safety_status"]["test_truth_used"] is False


def test_evaluation_anchor_bootstrap_rebuild_required_when_v43_identity_unverified(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _synthetic_inputs(tmp_path, v43_source_matches=False)
    result = EvaluationAnchorBootstrap(project_root=project, paths_config=str(_paths_config(tmp_path))).run(
        a1_npz=inputs["a1"],
        fold_candidate=inputs["fold"],
        v43c_oof=inputs["v43"],
        v46a_oof=inputs["v46"],
        out_root=project / "artifacts" / "evaluation_anchor",
    )
    assert result["status"] == "rebuild_required"
    out = project / "artifacts" / "evaluation_anchor"
    materialization = json.loads((out / "evaluation_anchor_materialization.json").read_text(encoding="utf-8"))
    fold_manifest = json.loads((out / "AFAC_A1_FOLD_V1_manifest.json").read_text(encoding="utf-8"))
    assert "verified_v43c_base_oof_identity" in materialization["missing_evidence"]
    assert materialization["historical_v53q1_oof_status"] == "not_materialized"
    assert fold_manifest["historical_v53q1_fold_equivalence"] == "unverified"
    assert fold_manifest["future_evaluation_protocol"] == "verified"
    assert not (out / "A1_EVAL_ANCHOR_V1_oof.npz").exists()
    assert (out / "evaluation_anchor_rebuild_spec.json").exists()
    assert (out / "evaluation_anchor_rebuild_plan.md").exists()


def test_champion_csv_is_not_accepted_as_oof_evaluation_anchor(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    verifier = M7BReadinessRepair(project_root=project, paths_config=str(_paths_config(tmp_path)))
    result = verifier._verify_oof(str(CHAMPION))
    assert result["status"] == "conflicting_candidates"
    assert result["is_oof_evaluation_anchor"] is False
    assert result["reason"].startswith("verification_error:")
