# -*- coding: utf-8 -*-
"""A2 dual-anchor contracts.

Online Deployment Anchor ``A2_ONLINE_CHAMPION_05093``
    Records the frozen online strategy identity and score only.  It is never
    usable as an offline OOF artifact.

Offline Evaluation Anchor ``A2_EVAL_ANCHOR_V1``
    Materialized from an explicit, verified OOF asset (fold protocol
    AFAC_A2_FOLD_V1, explicit user order, explicit candidate set and per-user
    scores).  ``deployment_equivalent`` may be false.  Test scores are never
    accepted as OOF; assets are never selected by mtime or guessed from file
    names.  If no explicit asset qualifies, the anchor status is
    ``waiting_for_explicit_asset`` and names the single missing asset.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..research.event_store import json_dumps, rel_ref, sha256_file, stable_hash
from .assets import SCOPE_OFFLINE_OOF, ScoreAsset
from .evaluator import A2Evaluator
from .fold import AFAC_A2_FOLD_V1
from .task_adapter import A2Dataset

A2_ONLINE_ANCHOR_ID = "A2_ONLINE_CHAMPION_05093"
A2_EVAL_ANCHOR_ID = "A2_EVAL_ANCHOR_V1"
ANCHOR_VERSION = "a2_dual_anchor_v1"

ONLINE_STRATEGY = {
    "online_score": 0.5093,
    "strategy": [
        "fixed V23 Top10 candidate set (teammate-verified)",
        "only exact Len3 users are modified",
        "rerank strictly inside the original V23 Top7 set",
        "v42c-DIN alpha=0.50",
        "C_all Novel slot alpha=0.50",
        "Top10 candidate set membership fixed",
        "history items and non-Len3 users protected",
    ],
    "historical_online_scores": [0.5052, 0.5066, 0.5068, 0.5092, 0.5093],
}


def materialize_online_anchor(*, out_dir: str | Path, project_root: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": ANCHOR_VERSION,
        "anchor_id": A2_ONLINE_ANCHOR_ID,
        "anchor_role": "online_deployment_anchor",
        "task": "A2",
        "created_at_epoch_seconds": time.time(),
        "frozen": True,
        "deployment_equivalent": True,
        "usable_as_offline_oof": False,
        "offline_or_test": "online_test_score",
        "contains_test_truth": False,
        **ONLINE_STRATEGY,
        "notes": "identity and deployment record only; never an offline evaluation artifact",
    }
    manifest["view_hash"] = stable_hash({k: v for k, v in manifest.items() if k != "created_at_epoch_seconds"})
    path = out_dir / f"{A2_ONLINE_ANCHOR_ID}_manifest.json"
    path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    manifest["manifest_path"] = rel_ref(path, project_root)
    path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def materialize_evaluation_anchor(
    *,
    dataset: A2Dataset,
    fold_manifest: dict[str, Any],
    oof_asset: ScoreAsset | None,
    out_dir: str | Path,
    project_root: Path,
    expected_source_asset_id: str,
) -> dict[str, Any]:
    """Materialize A2_EVAL_ANCHOR_V1 from a verified OOF asset.

    Returns a manifest dict.  Status is ``materialized`` on success or
    ``waiting_for_explicit_asset`` / ``validation_failed`` otherwise.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base: dict[str, Any] = {
        "manifest_version": ANCHOR_VERSION,
        "anchor_id": A2_EVAL_ANCHOR_ID,
        "anchor_role": "offline_evaluation_anchor",
        "task": "A2",
        "created_at_epoch_seconds": time.time(),
        "frozen": False,
        "usable_as_offline_oof": True,
        "offline_or_test": "offline_oof",
        "contains_test_truth": False,
        "fold_protocol": AFAC_A2_FOLD_V1,
        "deployment_equivalent": False,
        "test_scores_used_as_oof": False,
        "selected_by_mtime": False,
        "identity_guessed_from_filename": False,
    }

    if oof_asset is None or not str(oof_asset.path):
        manifest = base | {
            "status": "waiting_for_explicit_asset",
            "missing_asset": expected_source_asset_id,
            "deployment_equivalent": False,
        }
        return _write_manifest(out_dir, project_root, manifest)
    if oof_asset.verification_status != "verified_oof":
        manifest = base | {
            "status": "validation_failed",
            "source_asset_id": oof_asset.asset_id,
            "verification_errors": oof_asset.verification_errors,
        }
        return _write_manifest(out_dir, project_root, manifest)
    if fold_manifest.get("status") != "validated":
        manifest = base | {
            "status": "validation_failed",
            "source_asset_id": oof_asset.asset_id,
            "verification_errors": ["fold_manifest_not_validated"],
        }
        return _write_manifest(out_dir, project_root, manifest)

    asset = oof_asset.aligned_to(dataset.train_uids)
    evaluator = A2Evaluator(dataset, fold_manifest.get("fold_map"))
    evaluation = evaluator.evaluate_scores(
        asset_id=asset.asset_id,
        uids=asset.uids,
        item_ids=asset.item_ids,
        scores=asset.scores,
        targets=asset.targets,
    )

    # Compact anchor artifact: per-user identity/bucket/rank arrays only.
    anchor_npz = out_dir / f"{A2_EVAL_ANCHOR_ID}_oof_summary.npz"
    len_bucket_codes = np.array([dataset.len_bucket(uid) for uid in asset.uids])
    target_types = np.array([dataset.target_type(uid) for uid in asset.uids])
    folds = np.array([fold_manifest["fold_map"][uid] for uid in asset.uids], dtype=np.int64)
    np.savez(
        anchor_npz,
        uids=np.array(asset.uids),
        targets=np.array(asset.targets or [dataset.train_targets[u] for u in asset.uids]),
        fold=folds,
        sequence_length_bucket=len_bucket_codes,
        target_type=target_types,
        target_rank=evaluation.ranks.astype(np.int64),
        in_candidates=evaluation.in_candidates.astype(bool),
    )

    manifest = base | {
        "status": "materialized",
        "source_asset_id": asset.asset_id,
        "source_artifact": rel_ref(asset.path, project_root),
        "source_artifact_hash": asset.artifact_hash,
        "anchor_summary_npz": rel_ref(anchor_npz, project_root),
        "anchor_summary_hash": sha256_file(anchor_npz),
        "user_order": {"explicit": True, "source": "npz_uids_aligned_to_train_csv", "user_count": len(asset.uids)},
        "candidate_set_identity": stable_hash({"item_ids": dataset.item_ids}),
        "candidate_set_size": len(dataset.item_ids),
        "fold_hash": fold_manifest.get("sha256", ""),
        "metrics": evaluation.summary(),
        "len_bucket_metrics": evaluation.len_bucket_metrics,
        "type_bucket_metrics": evaluation.type_bucket_metrics,
        "fold_metrics": evaluation.fold_metrics,
    }
    return _write_manifest(out_dir, project_root, manifest)


def _write_manifest(out_dir: Path, project_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["view_hash"] = stable_hash({k: v for k, v in manifest.items() if k != "created_at_epoch_seconds"})
    path = out_dir / f"{A2_EVAL_ANCHOR_ID}_manifest.json"
    manifest["manifest_path"] = rel_ref(path, project_root)
    path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest
