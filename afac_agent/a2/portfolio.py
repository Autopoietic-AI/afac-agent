# -*- coding: utf-8 -*-
"""A2 model portfolio: asset registration and verification.

Every registered asset records:
asset_id, prediction_type, fold_protocol, user_order, candidate_set_identity,
score_scope, sequence_length_scope, history_novel_scope, artifact_hash,
verification_status, offline_or_test, provenance.

OOF assets and Test-scope assets are strictly separated; a Test-scope asset
can never be marked ``verified_oof`` and is never used for offline metrics.
Assets without an explicit local artifact are registered as
``declared_unmaterialized`` (never fabricated).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..research.event_store import rel_ref, sha256_file, stable_hash
from .anchors import A2_EVAL_ANCHOR_ID, A2_ONLINE_ANCHOR_ID
from .assets import SCOPE_OFFLINE_OOF, SCOPE_TEST, ScoreAsset
from .fold import AFAC_A2_FOLD_V1
from .task_adapter import A2Dataset

PORTFOLIO_VERSION = "a2_portfolio_v1"

ASSET_V42C_DIN = "A2_V42C_DIN_OOF"
ASSET_V48A = "A2_V48A_SASREC_OOF"
ASSET_V42C_TEST = "A2_V42C_DIN_TEST_CVMEAN"
ASSET_V48A_TEST = "A2_V48A_SASREC_TEST_CVMEAN"
ASSET_V23_TOP10 = "A2_V23_TOP10_FIXED_CANDIDATES"
ASSET_CALL_NOVEL = "A2_CALL_NOVEL_SLOT_MODEL"
ASSET_DCN_MIX = "A2_DCN_MIX_RANKER"
ASSET_LAMBDARANK = "A2_LAMBDARANK_RANKER"
ASSET_LEN0_EXPERTS = "A2_LEN0_EXPERTS"
ASSET_LEN3_EXPERTS = "A2_LEN3_EXPERTS"
ASSET_CHAMPION_STRATEGY = "A2_CHAMPION_05093_COMPOSITION"


def _record(
    *,
    asset_id: str,
    prediction_type: str,
    score_scope: str,
    offline_or_test: str,
    verification_status: str,
    artifact_path: str = "",
    artifact_hash: str = "",
    user_order: str = "unknown",
    candidate_set_identity: str = "unknown",
    sequence_length_scope: str = "all",
    history_novel_scope: str = "all",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "prediction_type": prediction_type,
        "fold_protocol": AFAC_A2_FOLD_V1 if offline_or_test == "offline_oof" else "not_applicable",
        "user_order": user_order,
        "candidate_set_identity": candidate_set_identity,
        "score_scope": score_scope,
        "sequence_length_scope": sequence_length_scope,
        "history_novel_scope": history_novel_scope,
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "verification_status": verification_status,
        "offline_or_test": offline_or_test,
        "provenance": provenance or {},
    }


def build_portfolio(
    *,
    dataset: A2Dataset,
    project_root: Path,
    v42_oof: ScoreAsset | None,
    v48_oof: ScoreAsset | None,
    v42_test_path: Path | None,
    v48_test_path: Path | None,
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    catalog_hash = stable_hash({"item_ids": dataset.item_ids})

    for asset, asset_id, model_label in (
        (v42_oof, ASSET_V42C_DIN, "v42c-DIN"),
        (v48_oof, ASSET_V48A, "v48a-clean-SASRec"),
    ):
        if asset is None:
            assets.append(_record(
                asset_id=asset_id,
                prediction_type="ranking_score",
                score_scope="all_items",
                offline_or_test="offline_oof",
                verification_status="declared_unmaterialized",
                provenance={"model": model_label, "missing": "explicit OOF npz not supplied"},
            ))
            continue
        assets.append(_record(
            asset_id=asset_id,
            prediction_type="ranking_score",
            score_scope="all_items",
            offline_or_test="offline_oof",
            verification_status=asset.verification_status,
            artifact_path=rel_ref(asset.path, project_root),
            artifact_hash=asset.artifact_hash,
            user_order="explicit_npz_uids_aligned_to_train_csv",
            candidate_set_identity=catalog_hash,
            provenance={
                "model": model_label,
                "merge_source": str(asset.path.parent.name),
                "verification_errors": asset.verification_errors,
                "test_truth_used": False,
            },
        ))

    # Test-scope assets are registered for identity only; never mixed into OOF.
    for path, asset_id, model_label in (
        (v42_test_path, ASSET_V42C_TEST, "v42c-DIN"),
        (v48_test_path, ASSET_V48A_TEST, "v48a-clean-SASRec"),
    ):
        if path is None or not Path(path).is_file():
            continue
        assets.append(_record(
            asset_id=asset_id,
            prediction_type="ranking_score",
            score_scope="test_users",
            offline_or_test="test_scope",
            verification_status="verified_test_scope_not_for_oof",
            artifact_path=rel_ref(Path(path), project_root),
            artifact_hash=sha256_file(Path(path)),
            user_order="explicit_npz_uids_test_order",
            candidate_set_identity=catalog_hash,
            provenance={"model": model_label, "mixing_with_oof_forbidden": True},
        ))

    # Declared strategy/portfolio assets without a local materialized artifact.
    declared = [
        (ASSET_V23_TOP10, "ranked_candidates", "top10_fixed", "all", "all",
         {"source": "teammate_verified_v23_top10", "used_by": A2_ONLINE_ANCHOR_ID}),
        (ASSET_CALL_NOVEL, "ranking_score", "novel_slots", "all", "novel_only",
         {"source": "C_all novel slot model", "alpha_in_champion": 0.50}),
        (ASSET_DCN_MIX, "ranking_score", "candidate_set", "all", "all",
         {"source": "DCN-Mix candidate ranker"}),
        (ASSET_LAMBDARANK, "ranking_score", "candidate_set", "all", "all",
         {"source": "LambdaRank candidate ranker"}),
        (ASSET_LEN0_EXPERTS, "ranking_score", "candidate_set", "len0", "all",
         {"source": "Len0 specialist experts"}),
        (ASSET_LEN3_EXPERTS, "ranking_score", "top7_rerank", "exact_len3", "all",
         {"source": "exact Len3 Top7 rerank experts"}),
        (ASSET_CHAMPION_STRATEGY, "composition", "top10_fixed", "exact_len3_modified_only", "mixed",
         {"source": "0.5093 online composition strategy", "online_score": 0.5093,
          "components": [ASSET_V23_TOP10, "A2_V42C_DIN", ASSET_CALL_NOVEL]}),
    ]
    for asset_id, prediction_type, score_scope, len_scope, hn_scope, provenance in declared:
        assets.append(_record(
            asset_id=asset_id,
            prediction_type=prediction_type,
            score_scope=score_scope,
            offline_or_test="unknown_scope_declared_only",
            verification_status="declared_unmaterialized",
            sequence_length_scope=len_scope,
            history_novel_scope=hn_scope,
            provenance=provenance | {"local_artifact": "not_supplied_not_fabricated"},
        ))

    return {
        "portfolio_version": PORTFOLIO_VERSION,
        "task": "A2",
        "evaluation_anchor": A2_EVAL_ANCHOR_ID,
        "online_anchor": A2_ONLINE_ANCHOR_ID,
        "assets": assets,
        "view_hash": stable_hash(assets),
    }


def build_asset_verification(portfolio: dict[str, Any]) -> dict[str, Any]:
    rows = []
    oof_test_isolation = "pass"
    for asset in portfolio["assets"]:
        is_oof = asset["offline_or_test"] == "offline_oof"
        is_test = asset["offline_or_test"] == "test_scope"
        mixed = bool(is_oof and "test" in asset["verification_status"]) or bool(
            is_test and asset["verification_status"] == "verified_oof"
        )
        if mixed:
            oof_test_isolation = "fail"
        rows.append({
            "asset_id": asset["asset_id"],
            "verification_status": asset["verification_status"],
            "offline_or_test": asset["offline_or_test"],
            "is_oof_asset": is_oof,
            "is_test_asset": is_test,
            "oof_test_mixed": mixed,
            "artifact_hash": asset["artifact_hash"],
            "materialized": bool(asset["artifact_path"]),
        })
    return {
        "verification_version": PORTFOLIO_VERSION,
        "assets": rows,
        "oof_test_isolation": oof_test_isolation,
        "declared_unmaterialized_count": sum(1 for r in rows if r["verification_status"] == "declared_unmaterialized"),
        "view_hash": stable_hash(rows),
    }
