# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from afac_agent.fusion_controller import run_fusion_controller


def _one_hot(labels: np.ndarray, wrong_mask: np.ndarray | None = None) -> np.ndarray:
    pred = labels.copy()
    if wrong_mask is not None:
        pred[wrong_mask] = (pred[wrong_mask] + 1) % 10
    proba = np.full((len(labels), 10), 0.001, dtype=np.float64)
    proba[np.arange(len(labels)), pred] = 0.991
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype(np.float32)


def _bundle(tmp_path: Path, *, mismatch_v43_train_idx: bool = False) -> dict[str, Path]:
    project = tmp_path / "project"
    anchor_dir = project / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1"
    anchor_dir.mkdir(parents=True)
    n = 11001
    train_idx = np.arange(n, dtype=np.int64)
    labels = train_idx % 10
    fold = train_idx % 5
    isolated = np.zeros(n, dtype=bool)
    isolated[:2213] = True
    connectivity = np.where(isolated, "isolated", "graph_visible")
    reachability = np.where(isolated, "no_visible_train_within_4_hops", "one_hop_available")
    degree_band = np.where(isolated, "isolated", "degree_2_5")

    base = _one_hot(labels, wrong_mask=isolated)
    expert = _one_hot(labels)
    composed = base.copy()
    composed[isolated] = expert[isolated]
    pred = composed.argmax(axis=1).astype(np.int64)
    np.savez(
        anchor_dir / "A1_EVAL_ANCHOR_V1_oof.npz",
        train_idx=train_idx,
        labels=labels,
        proba=composed,
        pred=pred,
        fold=fold,
        connectivity_visibility=connectivity,
        train_label_reachability=reachability,
        degree_band=degree_band,
        isolated_mask=isolated,
        component_source=np.where(isolated, "expert", "base"),
        anchor_version=np.asarray(["A1_EVAL_ANCHOR_V1"] * n),
    )
    (anchor_dir / "A1_EVAL_ANCHOR_V1_manifest.json").write_text(
        json.dumps({"evaluation_anchor_identity": "A1_EVAL_ANCHOR_V1", "sha256": "synthetic", "status": "materialized_from_verified_components"}),
        encoding="utf-8",
    )
    v43_train = train_idx.copy()
    if mismatch_v43_train_idx:
        v43_train = np.roll(v43_train, 1)
    v43 = tmp_path / "correct_smooth_v1_oof_proba.npz"
    np.savez(v43, proba=base, base_proba=base, train_idx=v43_train, labels=labels, bucket=np.asarray(["x"] * n))
    v46 = tmp_path / "two_seed_balanced_candidate_oof.npz"
    np.savez(v46, proba=composed, expert_proba=expert, train_idx=train_idx, labels=labels, isolated_mask=isolated)
    return {"project": project, "anchor_dir": anchor_dir, "v43": v43, "v46": v46}


def test_fusion_controller_smoke_no_training_no_test_prediction(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    result = run_fusion_controller(
        project_root=paths["project"],
        anchor_dir=paths["anchor_dir"],
        v43_oof=paths["v43"],
        v46_oof=paths["v46"],
        out_root=paths["project"] / "artifacts" / "fusion_runs",
    )
    assert result["status"] == "completed"
    run_dir = paths["project"] / result["artifacts"]["fusion_manifest"].replace("/", "\\")
    run_dir = run_dir.parent
    manifest = json.loads((run_dir / "fusion_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((run_dir / "asset_verification.json").read_text(encoding="utf-8"))
    candidates = json.loads((run_dir / "fusion_candidates.json").read_text(encoding="utf-8"))
    assert manifest["trains_model"] is False
    assert manifest["generates_test_prediction"] is False
    assert manifest["creates_submission"] is False
    assert manifest["scientific_rounds_used"] == 0
    assert manifest["candidate_count"] <= 16
    assert candidates["budget"]["generated"] <= 16
    assert all(item["verification_status"] == "verified_oof" for item in verification["assets"])
    assert all(item["test_asset_mixed_with_oof"] is False for item in verification["assets"])
    assert (run_dir / "REPORT_PACKAGE" / "fusion_manifest.json").exists()


def test_train_idx_mismatch_marks_asset_invalid(tmp_path: Path) -> None:
    paths = _bundle(tmp_path, mismatch_v43_train_idx=True)
    result = run_fusion_controller(
        project_root=paths["project"],
        anchor_dir=paths["anchor_dir"],
        v43_oof=paths["v43"],
        v46_oof=paths["v46"],
        out_root=paths["project"] / "artifacts" / "fusion_runs",
    )
    assert result["status"] == "failed"
    run_dir = (paths["project"] / result["artifacts"]["fusion_manifest"].replace("/", "\\")).parent
    verification = json.loads((run_dir / "asset_verification.json").read_text(encoding="utf-8"))
    v43 = next(item for item in verification["assets"] if item["model_id"] == "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF")
    assert v43["verification_status"] == "invalid"


def test_complementarity_oracle_diagnostic_and_duplicate_anchor_rejected(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    result = run_fusion_controller(
        project_root=paths["project"],
        anchor_dir=paths["anchor_dir"],
        v43_oof=paths["v43"],
        v46_oof=paths["v46"],
        out_root=paths["project"] / "artifacts" / "fusion_runs",
    )
    run_dir = (paths["project"] / result["artifacts"]["fusion_manifest"].replace("/", "\\")).parent
    comp = json.loads((run_dir / "complementarity_report.json").read_text(encoding="utf-8"))
    decisions = json.loads((run_dir / "fusion_decisions.json").read_text(encoding="utf-8"))
    crossfit = json.loads((run_dir / "crossfit_assignments.json").read_text(encoding="utf-8"))
    base_vs_composed = next(pair for pair in comp["pairs"] if pair["model_a"] == "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF" and pair["model_b"] == "V46A_ISOLATED_COMPOSED_OOF")
    assert comp["oracle_is_diagnostic_only"] is True
    assert base_vs_composed["prediction_complementarity"]["rescue"] == 2213
    duplicate = next(item for item in decisions["items"] if item["candidate_id"] == "duplicate_anchor_route_base_to_expert_isolated")
    assert duplicate["status"] == "rejected"
    assert "duplicate_anchor" in duplicate["reason_codes"]
    confidence = next(item for item in crossfit["items"] if item["candidate_id"] == "confidence_gate_base_expert_isolated")
    assert confidence["mode"] == "strict_outer_fold_cross_fit"
    assert all(row["held_out_labels_used_for_selection"] is False for row in confidence["fold_assignments"])


def test_fusion_run_id_is_deterministic(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    kwargs = dict(
        project_root=paths["project"],
        anchor_dir=paths["anchor_dir"],
        v43_oof=paths["v43"],
        v46_oof=paths["v46"],
        out_root=paths["project"] / "artifacts" / "fusion_runs",
    )
    first = run_fusion_controller(**kwargs)
    second = run_fusion_controller(**kwargs)
    assert first["run_id"] == second["run_id"]
