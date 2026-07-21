# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from afac_agent.evaluation_anchor_bootstrap import EvaluationAnchorBootstrap
from afac_agent.m7b_readiness import M7BReadinessRepair
from afac_agent.research.event_store import json_dumps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"


def _paths_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "paths.local.yaml"
    cfg.write_text("\n".join(["a1:", f"  anchor_csv: \"{CHAMPION}\""]), encoding="utf-8")
    return cfg


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_inputs(tmp_path: Path, *, composition_valid: bool = True) -> dict[str, Path]:
    train_idx = np.arange(11001, dtype=np.int64)
    test_idx = np.arange(11001, 13752, dtype=np.int64)
    labels = train_idx % 10
    all_labels = np.concatenate([labels, np.full(len(test_idx), -1, dtype=np.int64)])
    adj = sp.csr_matrix((13752, 13752), dtype=np.float32)
    a1 = tmp_path / "A1.npz"
    np.savez(
        a1,
        train_idx=train_idx,
        test_idx=test_idx,
        labels=all_labels,
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=np.asarray(adj.shape),
    )

    fold = tmp_path / "fold_assignment_5.csv"
    with fold.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "label", "fold"])
        writer.writeheader()
        for idx, label in zip(train_idx.tolist(), labels.tolist(), strict=True):
            writer.writerow({"node_id": idx, "label": label, "fold": idx % 5})

    base = np.full((11001, 10), 0.1, dtype=np.float64)
    expert = np.zeros((11001, 10), dtype=np.float64)
    expert[np.arange(11001), labels] = 1.0
    isolated_mask = np.zeros(11001, dtype=bool)
    isolated_mask[:2213] = True
    composed = base.copy()
    composed[isolated_mask] = expert[isolated_mask]
    if not composition_valid:
        composed[~isolated_mask] = expert[~isolated_mask]

    v43 = tmp_path / "correct_smooth_v1_oof_proba.npz"
    np.savez(v43, proba=base, base_proba=base, train_idx=train_idx, labels=labels, bucket=np.zeros(11001, dtype=np.int64))

    v46 = tmp_path / "two_seed_balanced_candidate_oof.npz"
    np.savez(v46, proba=composed, expert_proba=expert, train_idx=train_idx, labels=labels, isolated_mask=isolated_mask)
    return {"a1": a1, "fold": fold, "v43": v43, "v46": v46}


def test_evaluation_anchor_bootstrap_materializes_with_verified_synthetic_components(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _synthetic_inputs(tmp_path)
    result = EvaluationAnchorBootstrap(project_root=project, paths_config=str(_paths_config(tmp_path)), expected_fold_hash=_sha(inputs["fold"])).run(
        a1_npz=inputs["a1"],
        fold_candidate=inputs["fold"],
        v43c_oof=inputs["v43"],
        v46a_oof=inputs["v46"],
        out_root=project / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1",
    )
    assert result["status"] == "materialized_from_verified_components"
    out = project / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1"
    manifest = json.loads((out / "evaluation_anchor_bootstrap_manifest.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((out / "A1_EVAL_ANCHOR_V1_manifest.json").read_text(encoding="utf-8"))
    integrity = json.loads((out / "integrity_audit.json").read_text(encoding="utf-8"))
    assert manifest["historical_v53q1_oof_status"] == "not_materialized"
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert manifest["counts_as_anchor_bootstrap_run"] is True
    assert eval_manifest["evaluation_anchor_identity"] == "A1_EVAL_ANCHOR_V1"
    assert eval_manifest["functional_oof_validity"] == "verified"
    assert eval_manifest["historical_v43c_identity_equivalence"] == "unverified"
    assert integrity["routing_uses_truth"] is False
    assert integrity["output_hash_stable"] is True
    assert (out / "REPORT_PACKAGE" / "EVALUATION_ANCHOR_REPORT.md").exists()


def test_evaluation_anchor_bootstrap_rebuild_required_when_composition_invalid(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _synthetic_inputs(tmp_path, composition_valid=False)
    result = EvaluationAnchorBootstrap(project_root=project, paths_config=str(_paths_config(tmp_path)), expected_fold_hash=_sha(inputs["fold"])).run(
        a1_npz=inputs["a1"],
        fold_candidate=inputs["fold"],
        v43c_oof=inputs["v43"],
        v46a_oof=inputs["v46"],
        out_root=project / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1",
    )
    assert result["status"] == "rebuild_required"
    out = project / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1"
    materialization = json.loads((out / "integrity_audit.json").read_text(encoding="utf-8"))
    fold_manifest = json.loads((out / "AFAC_A1_FOLD_V1_manifest.json").read_text(encoding="utf-8"))
    assert "verified_v46a_composition_rule" in materialization["missing_evidence"]
    assert fold_manifest["historical_v53q1_fold_equivalence"] == "unverified"
    assert fold_manifest["future_evaluation_protocol"] == "verified"
    assert not (out / "A1_EVAL_ANCHOR_V1_oof.npz").exists()
    assert (out / "evaluation_anchor_rebuild_spec.json").exists()


def test_champion_csv_is_not_accepted_as_oof_evaluation_anchor(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    verifier = M7BReadinessRepair(project_root=project, paths_config=str(_paths_config(tmp_path)))
    result = verifier._verify_oof(str(CHAMPION))
    assert result["status"] == "conflicting_candidates"
    assert result["is_oof_evaluation_anchor"] is False
    assert result["reason"].startswith("verification_error:")
