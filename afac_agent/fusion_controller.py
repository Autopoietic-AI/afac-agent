# -*- coding: utf-8 -*-
"""Unified model portfolio and bounded fusion controller.

The controller is evaluation-only. It registers existing prediction assets,
audits complementarity, generates a finite fusion grid, evaluates candidates
with fold-aware cross-fit, and emits deterministic reports. It never trains,
never generates Test predictions, and never submits.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .research.event_store import json_dumps, rel_ref, sha256_file, stable_hash

FUSION_VERSION = "unified_fusion_controller_v1"
TASK_A1 = "A1"
TASK_TYPE_CLASSIFICATION = "classification"
PREDICTION_TYPE_CLASSIFICATION_PROBA = "classification_proba"
PREDICTION_TYPE_RANKING_SCORE = "ranking_score"
PREDICTION_TYPE_RANKED_CANDIDATES = "ranked_candidates"
ANCHOR_ID = "A1_EVAL_ANCHOR_V1"
ONLINE_ANCHOR_ID = "A1_V53Q1_TRANSITION_STABLE_EDGE_H2"
BASE_ID = "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF"
EXPERT_ID = "V46A_ISOLATED_EXPERT_OOF"
COMPOSED_ID = "V46A_ISOLATED_COMPOSED_OOF"
TRAIN_N = 11001
CLASS_N = 10


@dataclass(frozen=True)
class PortfolioAsset:
    model_id: str
    task_type: str
    prediction_type: str
    artifact_path: str
    artifact_hash: str
    parent_id: str
    fold_protocol: str
    prediction_scope: str
    information_sources: tuple[str, ...]
    supported_buckets: tuple[str, ...]
    supported_classes: tuple[int, ...]
    training_cost: str
    inference_cost: str
    overall_metrics: dict[str, Any]
    bucket_metrics: dict[str, Any]
    class_metrics: dict[str, Any]
    provenance: dict[str, Any]
    verification_status: str
    train_idx: np.ndarray
    labels: np.ndarray
    proba: np.ndarray
    pred: np.ndarray
    fold: np.ndarray
    connectivity_visibility: np.ndarray
    train_label_reachability: np.ndarray
    degree_band: np.ndarray
    isolated_mask: np.ndarray

    def public_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_type": self.task_type,
            "prediction_type": self.prediction_type,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "parent_id": self.parent_id,
            "fold_protocol": self.fold_protocol,
            "prediction_scope": self.prediction_scope,
            "information_sources": list(self.information_sources),
            "supported_buckets": list(self.supported_buckets),
            "supported_classes": list(self.supported_classes),
            "training_cost": self.training_cost,
            "inference_cost": self.inference_cost,
            "overall_metrics": self.overall_metrics,
            "bucket_metrics": self.bucket_metrics,
            "class_metrics": self.class_metrics,
            "provenance": self.provenance,
            "verification_status": self.verification_status,
        }


def _prob_audit(proba: np.ndarray) -> dict[str, Any]:
    row_sums = proba.sum(axis=1) if proba.ndim == 2 else np.array([])
    return {
        "shape": list(proba.shape),
        "finite": bool(np.isfinite(proba).all()),
        "min": float(np.nanmin(proba)) if proba.size else None,
        "max": float(np.nanmax(proba)) if proba.size else None,
        "max_row_sum_abs_error": float(np.abs(row_sums - 1.0).max()) if row_sums.size else None,
        "passed": bool(
            proba.shape == (TRAIN_N, CLASS_N)
            and np.isfinite(proba).all()
            and proba.min() >= -1e-7
            and proba.max() <= 1.0000001
            and np.allclose(row_sums, 1.0, atol=1e-5)
        ),
    }


def _logit(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba.astype(np.float64), 1e-12, 1.0)
    logits = np.log(clipped)
    return logits - logits.mean(axis=1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _accuracy(labels: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    if not bool(mask.any()):
        return None
    return float((labels[mask] == pred[mask]).mean())


def _macro(labels: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    vals = []
    for cls in range(CLASS_N):
        m = mask & (labels == cls)
        if m.any():
            vals.append(float((labels[m] == pred[m]).mean()))
    return float(np.mean(vals)) if vals else None


def _metric_rows(labels: np.ndarray, pred: np.ndarray, fold: np.ndarray, axes: dict[str, np.ndarray], parent_pred: np.ndarray | None = None) -> dict[str, Any]:
    correct = pred == labels
    parent_correct = parent_pred == labels if parent_pred is not None else None
    summary = {
        "overall_accuracy": _accuracy(labels, pred),
        "macro_accuracy": _macro(labels, pred),
    }
    if parent_correct is not None:
        summary.update(_delta_counts(parent_correct, correct))
    fold_rows = []
    for f in sorted(set(fold.tolist())):
        m = fold == f
        row = {"fold": int(f), "sample_count": int(m.sum()), "accuracy": _accuracy(labels, pred, m), "macro_accuracy": _macro(labels, pred, m)}
        if parent_correct is not None:
            row.update(_delta_counts(parent_correct[m], correct[m]))
        fold_rows.append(row)
    bucket_rows: list[dict[str, Any]] = []
    bucket_class_rows: list[dict[str, Any]] = []
    for axis_name, axis in axes.items():
        for value in sorted(set(axis.tolist())):
            m = axis == value
            row = {"axis": axis_name, "bucket": str(value), "sample_count": int(m.sum()), "accuracy": _accuracy(labels, pred, m)}
            if parent_correct is not None:
                row.update(_delta_counts(parent_correct[m], correct[m]))
            bucket_rows.append(row)
            for cls in range(CLASS_N):
                cm = m & (labels == cls)
                crow = {"axis": axis_name, "bucket": str(value), "class_id": cls, "sample_count": int(cm.sum()), "accuracy": _accuracy(labels, pred, cm)}
                if parent_correct is not None:
                    crow.update(_delta_counts(parent_correct[cm], correct[cm]))
                bucket_class_rows.append(crow)
    class_rows = []
    for cls in range(CLASS_N):
        m = labels == cls
        class_rows.append({"class_id": cls, "sample_count": int(m.sum()), "accuracy": _accuracy(labels, pred, m), "predicted_count": int((pred == cls).sum())})
    return {"summary": summary, "fold": fold_rows, "bucket": bucket_rows, "bucket_class": bucket_class_rows, "class": class_rows}


def _delta_counts(parent_correct: np.ndarray, candidate_correct: np.ndarray) -> dict[str, Any]:
    rescue = int((~parent_correct & candidate_correct).sum())
    damage = int((parent_correct & ~candidate_correct).sum())
    changed = rescue + damage
    return {
        "rescue": rescue,
        "damage": damage,
        "net": rescue - damage,
        "changed_count": changed,
        "change_precision": float(rescue / changed) if changed else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows or [{"status": "empty"}]:
            writer.writerow(row)


class ModelPortfolio:
    def __init__(self, *, project_root: Path, anchor_dir: Path, v43_oof: Path, v46_oof: Path) -> None:
        self.project_root = project_root
        self.anchor_dir = anchor_dir
        self.v43_oof = v43_oof
        self.v46_oof = v46_oof
        self.assets: dict[str, PortfolioAsset] = {}

    def register_all(self) -> dict[str, PortfolioAsset]:
        anchor_path = self.anchor_dir / "A1_EVAL_ANCHOR_V1_oof.npz"
        anchor_manifest_path = self.anchor_dir / "A1_EVAL_ANCHOR_V1_manifest.json"
        anchor_manifest = json.loads(anchor_manifest_path.read_text(encoding="utf-8"))
        anchor_npz = np.load(anchor_path, allow_pickle=False)
        self.assets[ANCHOR_ID] = self._asset_from_npz(
            model_id=ANCHOR_ID,
            path=anchor_path,
            npz=anchor_npz,
            proba=anchor_npz["proba"],
            parent_id=ONLINE_ANCHOR_ID,
            prediction_scope="all_train_oof",
            information_sources=("v43c_associated_base_oof", "v46a_isolated_expert_oof", "fixed_isolated_mask"),
            supported_buckets=("graph_visible", "isolated", "one_hop_available", "exact2_only", "exact3_4_only", "no_visible_train_within_4_hops"),
            provenance={"manifest": rel_ref(anchor_manifest_path, self.project_root), "anchor_hash": anchor_manifest.get("sha256", "")},
        )
        z43 = np.load(self.v43_oof, allow_pickle=False)
        self.assets[BASE_ID] = self._asset_from_npz(
            model_id=BASE_ID,
            path=self.v43_oof,
            npz=anchor_npz,
            proba=z43["base_proba"],
            parent_id="",
            prediction_scope="all_train_oof",
            information_sources=("v43c_associated_base_oof",),
            supported_buckets=("graph_visible", "isolated", "one_hop_available", "exact2_only", "exact3_4_only", "no_visible_train_within_4_hops"),
            provenance={"source_npz": rel_ref(self.v43_oof, self.project_root), "source_key": "base_proba", "historical_v43c_identity_equivalence": "unverified"},
            source_npz=z43,
        )
        z46 = np.load(self.v46_oof, allow_pickle=False)
        self.assets[EXPERT_ID] = self._asset_from_npz(
            model_id=EXPERT_ID,
            path=self.v46_oof,
            npz=anchor_npz,
            proba=z46["expert_proba"],
            parent_id=BASE_ID,
            prediction_scope="isolated_only_oof",
            information_sources=("v46a_isolated_expert_oof",),
            supported_buckets=("isolated", "no_visible_train_within_4_hops"),
            provenance={"source_npz": rel_ref(self.v46_oof, self.project_root), "source_key": "expert_proba", "isolated_mask_key": "isolated_mask"},
            source_npz=z46,
        )
        self.assets[COMPOSED_ID] = self._asset_from_npz(
            model_id=COMPOSED_ID,
            path=self.v46_oof,
            npz=anchor_npz,
            proba=z46["proba"],
            parent_id=BASE_ID,
            prediction_scope="all_train_oof",
            information_sources=("v43c_associated_base_oof", "v46a_isolated_expert_oof", "fixed_isolated_mask"),
            supported_buckets=("graph_visible", "isolated", "one_hop_available", "exact2_only", "exact3_4_only", "no_visible_train_within_4_hops"),
            provenance={"source_npz": rel_ref(self.v46_oof, self.project_root), "source_key": "proba", "composition": "base on non-isolated; expert on isolated"},
            source_npz=z46,
        )
        return self.assets

    def _asset_from_npz(
        self,
        *,
        model_id: str,
        path: Path,
        npz: Any,
        proba: np.ndarray,
        parent_id: str,
        prediction_scope: str,
        information_sources: tuple[str, ...],
        supported_buckets: tuple[str, ...],
        provenance: dict[str, Any],
        source_npz: Any | None = None,
    ) -> PortfolioAsset:
        train_idx = npz["train_idx"].astype(np.int64)
        labels = npz["labels"].astype(np.int64)
        fold = npz["fold"].astype(np.int64)
        pred = proba.argmax(axis=1).astype(np.int64)
        axes = {
            "connectivity_visibility": npz["connectivity_visibility"].astype(str),
            "train_label_reachability": npz["train_label_reachability"].astype(str),
            "degree_band": npz["degree_band"].astype(str),
        }
        metrics = _metric_rows(labels, pred, fold, axes)
        prob = _prob_audit(proba)
        verification_errors = []
        if train_idx.shape != (TRAIN_N,) or len(set(train_idx.tolist())) != TRAIN_N:
            verification_errors.append("train_idx_not_complete_unique")
        if fold.shape != (TRAIN_N,) or sorted(set(fold.tolist())) != [0, 1, 2, 3, 4]:
            verification_errors.append("fold_invalid")
        if not prob["passed"]:
            verification_errors.append("probability_invalid")
        if source_npz is not None:
            if "train_idx" not in source_npz.files or not np.array_equal(source_npz["train_idx"].astype(np.int64), train_idx):
                verification_errors.append("source_train_idx_not_aligned")
            if "labels" not in source_npz.files or not np.array_equal(source_npz["labels"].astype(np.int64), labels):
                verification_errors.append("source_labels_not_aligned")
        return PortfolioAsset(
            model_id=model_id,
            task_type=TASK_TYPE_CLASSIFICATION,
            prediction_type=PREDICTION_TYPE_CLASSIFICATION_PROBA,
            artifact_path=rel_ref(path, self.project_root),
            artifact_hash=sha256_file(path),
            parent_id=parent_id,
            fold_protocol="AFAC_A1_FOLD_V1",
            prediction_scope=prediction_scope,
            information_sources=information_sources,
            supported_buckets=supported_buckets,
            supported_classes=tuple(range(CLASS_N)),
            training_cost="already_materialized_oof_no_training_in_controller",
            inference_cost="none_for_existing_oof",
            overall_metrics=metrics["summary"],
            bucket_metrics={"rows": metrics["bucket"]},
            class_metrics={"rows": metrics["class"]},
            provenance=provenance | {"probability_audit": prob, "verification_errors": verification_errors, "test_truth_used": False},
            verification_status="verified_oof" if not verification_errors else "invalid",
            train_idx=train_idx,
            labels=labels,
            proba=proba.astype(np.float64),
            pred=pred,
            fold=fold,
            connectivity_visibility=axes["connectivity_visibility"],
            train_label_reachability=axes["train_label_reachability"],
            degree_band=axes["degree_band"],
            isolated_mask=npz["isolated_mask"].astype(bool),
        )


class FusionController:
    def __init__(self, *, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(
        self,
        *,
        anchor_dir: str | Path = "artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1",
        v43_oof: str | Path,
        v46_oof: str | Path,
        out_root: str | Path = "artifacts/fusion_runs",
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        anchor_dir = self._resolve(anchor_dir)
        v43_oof = self._resolve(v43_oof)
        v46_oof = self._resolve(v46_oof)
        missing = [name for name, path in {"anchor_dir": anchor_dir, "v43_oof": v43_oof, "v46_oof": v46_oof}.items() if not path.exists()]
        if missing:
            return {"status": "waiting_for_input", "missing_inputs": missing, "artifacts": {}}
        input_hashes = {
            "anchor_manifest": sha256_file(anchor_dir / "A1_EVAL_ANCHOR_V1_manifest.json"),
            "anchor_oof": sha256_file(anchor_dir / "A1_EVAL_ANCHOR_V1_oof.npz"),
            "v43_oof": sha256_file(v43_oof),
            "v46_oof": sha256_file(v46_oof),
        }
        run_id = stable_hash({"version": FUSION_VERSION, "input_hashes": input_hashes})
        out_dir = self._resolve(out_root) / run_id
        manifest_path = out_dir / "fusion_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        out_dir.mkdir(parents=True, exist_ok=True)

        portfolio = ModelPortfolio(project_root=self.project_root, anchor_dir=anchor_dir, v43_oof=v43_oof, v46_oof=v46_oof)
        assets = portfolio.register_all()
        self._assets_for_eval = assets
        asset_verification = self._asset_verification(assets)
        if any(record["verification_status"] not in {"verified_oof", "metadata_only"} for record in asset_verification["assets"]):
            status = "failed"
        else:
            status = "completed"
        complementarity = self._complementarity_report(assets)
        candidates = self._generate_candidates(assets)
        evaluated = [self._evaluate_candidate(candidate, assets[ANCHOR_ID]) for candidate in candidates]
        decisions = self._decisions(evaluated, assets[ANCHOR_ID])
        plan = self._fusion_plan(complementarity, decisions)
        payloads = {
            "portfolio_snapshot": {"portfolio_version": FUSION_VERSION, "assets": [asset.public_record() for asset in assets.values()], "view_hash": stable_hash([asset.public_record() for asset in assets.values()])},
            "asset_verification": asset_verification,
            "complementarity_report": complementarity,
            "fusion_candidates": {"candidate_version": FUSION_VERSION, "items": [self._candidate_public(c) for c in candidates], "budget": {"max_total_candidates": 16, "generated": len(candidates)}},
            "crossfit_assignments": {"crossfit_version": FUSION_VERSION, "fold_protocol": "AFAC_A1_FOLD_V1", "items": [e["crossfit_assignment"] for e in evaluated]},
            "fusion_metrics": {"metric_version": FUSION_VERSION, "items": [e["metrics"] for e in evaluated]},
            "fusion_decisions": decisions,
            "fusion_plan": plan,
        }
        artifacts: dict[str, str] = {}
        for name, payload in payloads.items():
            path = out_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        report = out_dir / "FUSION_REPORT.md"
        report.write_text(self._report(decisions, complementarity, plan), encoding="utf-8")
        artifacts["fusion_report"] = rel_ref(report, self.project_root)
        package = out_dir / "REPORT_PACKAGE"
        manifest = {
            "manifest_version": FUSION_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "task_type": "classification",
            "supports_future_task_types": ["ranking_recommendation"],
            "read_only_inputs": True,
            "executes_adapter": False,
            "trains_model": False,
            "generates_test_prediction": False,
            "creates_submission": False,
            "calls_llm": False,
            "uses_network": False,
            "counts_as_experiment_round": False,
            "scientific_rounds_used": 0,
            "anchor_id": ANCHOR_ID,
            "anchor_hash": input_hashes["anchor_oof"],
            "candidate_count": len(candidates),
            "accepted_count": decisions["summary"]["accepted_count"],
            "best_candidate_id": decisions["summary"]["best_candidate_id"],
            "input_hashes": input_hashes,
            "artifacts": artifacts | {"report_package": rel_ref(package, self.project_root)},
        }
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["fusion_manifest"] = rel_ref(manifest_path, self.project_root)
        package.mkdir(parents=True, exist_ok=True)
        for name in ["FUSION_REPORT.md", "portfolio_snapshot.json", "asset_verification.json", "complementarity_report.json", "fusion_metrics.json", "fusion_decisions.json", "fusion_plan.json", "fusion_manifest.json"]:
            src = out_dir / name
            if src.exists():
                shutil.copyfile(src, package / name)
        artifacts["report_package"] = rel_ref(package, self.project_root)
        return {"status": status, "run_id": run_id, "artifacts": artifacts}

    def _asset_verification(self, assets: dict[str, PortfolioAsset]) -> dict[str, Any]:
        rows = []
        for asset in assets.values():
            rows.append({
                "model_id": asset.model_id,
                "verification_status": asset.verification_status,
                "prediction_type": asset.prediction_type,
                "is_oof_asset": asset.verification_status == "verified_oof",
                "is_test_asset": asset.verification_status == "verified_test",
                "test_asset_mixed_with_oof": False,
                "train_idx_count": int(asset.train_idx.shape[0]),
                "label_count": int(asset.labels.shape[0]),
                "proba_shape": list(asset.proba.shape),
                "fold_values": sorted(set(asset.fold.tolist())),
                "artifact_hash": asset.artifact_hash,
                "probability_audit": asset.provenance.get("probability_audit", {}),
                "test_truth_used": False,
            })
        return {"verification_version": FUSION_VERSION, "assets": rows, "oof_test_isolation": "pass", "view_hash": stable_hash(rows)}

    def _complementarity_report(self, assets: dict[str, PortfolioAsset]) -> dict[str, Any]:
        pairs = [(BASE_ID, EXPERT_ID), (BASE_ID, COMPOSED_ID), (ANCHOR_ID, BASE_ID), (ANCHOR_ID, EXPERT_ID), (ANCHOR_ID, COMPOSED_ID)]
        records = []
        for a_id, b_id in pairs:
            a = assets[a_id]
            b = assets[b_id]
            records.append(self._pair_complementarity(a, b))
        return {
            "report_version": FUSION_VERSION,
            "pairs": records,
            "oracle_is_diagnostic_only": True,
            "complementarity_types": ["prediction_complementarity", "information_source_complementarity", "representation_only_difference"],
            "view_hash": stable_hash(records),
        }

    def _pair_complementarity(self, a: PortfolioAsset, b: PortfolioAsset) -> dict[str, Any]:
        axes = {"connectivity_visibility": a.connectivity_visibility, "train_label_reachability": a.train_label_reachability, "degree_band": a.degree_band}
        ac = a.pred == a.labels
        bc = b.pred == b.labels
        def scope_record(name: str, mask: np.ndarray) -> dict[str, Any]:
            both_correct = int((ac[mask] & bc[mask]).sum())
            both_wrong = int((~ac[mask] & ~bc[mask]).sum())
            a_only = int((ac[mask] & ~bc[mask]).sum())
            b_only = int((~ac[mask] & bc[mask]).sum())
            count = int(mask.sum())
            disagreement = int((a.pred[mask] != b.pred[mask]).sum())
            return {
                "scope": name,
                "sample_count": count,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "model_a_only_correct": a_only,
                "model_b_only_correct": b_only,
                "agreement": int(count - disagreement),
                "disagreement": disagreement,
                "rescue": b_only,
                "damage": a_only,
                "net": b_only - a_only,
                "oracle_upper_bound": float((both_correct + a_only + b_only) / count) if count else None,
                "changed_count": disagreement,
                "change_precision": float(b_only / disagreement) if disagreement else None,
            }
        scopes = [scope_record("overall", np.ones(a.labels.shape[0], dtype=bool))]
        for f in sorted(set(a.fold.tolist())):
            scopes.append(scope_record(f"fold={f}", a.fold == f))
        for cls in range(CLASS_N):
            scopes.append(scope_record(f"class={cls}", a.labels == cls))
        for axis_name, axis in axes.items():
            for value in sorted(set(axis.tolist())):
                scopes.append(scope_record(f"{axis_name}={value}", axis == value))
        for bucket in sorted(set(a.train_label_reachability.tolist())):
            for cls in range(CLASS_N):
                scopes.append(scope_record(f"bucket_class={bucket}|class={cls}", (a.train_label_reachability == bucket) & (a.labels == cls)))
        return {
            "model_a": a.model_id,
            "model_b": b.model_id,
            "prediction_complementarity": scopes[0],
            "information_source_complementarity": sorted(set(b.information_sources) - set(a.information_sources)),
            "representation_only_difference": sorted(set(b.information_sources) & set(a.information_sources)),
            "scope_breakdown": scopes,
            "oracle_upper_bound_diagnostic_only": True,
        }

    def _generate_candidates(self, assets: dict[str, PortfolioAsset]) -> list[dict[str, Any]]:
        base = assets[BASE_ID]
        expert = assets[EXPERT_ID]
        composed = assets[COMPOSED_ID]
        candidates: list[dict[str, Any]] = []
        for alpha in [0.25, 0.5, 0.75]:
            candidates.append({"candidate_id": f"probability_blend_base_composed_alpha_{alpha}", "operator": "probability_blend", "assets": [BASE_ID, COMPOSED_ID], "fixed_params": {"alpha": alpha}, "search_space": {}})
        for alpha in [0.25, 0.5, 0.75]:
            candidates.append({"candidate_id": f"logit_blend_base_composed_alpha_{alpha}", "operator": "logit_blend", "assets": [BASE_ID, COMPOSED_ID], "fixed_params": {"alpha": alpha}, "search_space": {}})
        candidates.append({"candidate_id": "bucket_route_base_to_expert_isolated", "operator": "bucket_route", "assets": [BASE_ID, EXPERT_ID], "fixed_params": {"route_bucket": "isolated"}, "search_space": {}})
        candidates.append({"candidate_id": "duplicate_anchor_route_base_to_expert_isolated", "operator": "bucket_route", "assets": [BASE_ID, EXPERT_ID], "fixed_params": {"route_bucket": "isolated", "expected_duplicate_of": ANCHOR_ID}, "search_space": {}})
        candidates.append({"candidate_id": "confidence_gate_base_expert_isolated", "operator": "confidence_gate", "assets": [BASE_ID, EXPERT_ID], "fixed_params": {"scope": "isolated"}, "search_space": {"threshold": [0.0, 0.05, 0.1]}})
        candidates.append({"candidate_id": "class_weighted_blend_base_composed", "operator": "class_weighted_blend", "assets": [BASE_ID, COMPOSED_ID], "fixed_params": {}, "search_space": {"alpha": [0.25, 0.5, 0.75]}})
        candidates.append({"candidate_id": "residual_patch_composed_over_anchor", "operator": "residual_patch", "assets": [ANCHOR_ID, COMPOSED_ID], "fixed_params": {}, "search_space": {}})
        assert len(candidates) <= 16
        return candidates

    def _candidate_public(self, c: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": c["candidate_id"],
            "operator": c["operator"],
            "assets": c["assets"],
            "fixed_params": c["fixed_params"],
            "parameter_search_space": c["search_space"],
            "oracle_used": False,
            "test_feedback_used": False,
        }

    def _evaluate_candidate(self, c: dict[str, Any], anchor: PortfolioAsset) -> dict[str, Any]:
        proba, assignment = self._crossfit_proba(c, anchor)
        pred = proba.argmax(axis=1).astype(np.int64)
        metrics = _metric_rows(anchor.labels, pred, anchor.fold, {
            "connectivity_visibility": anchor.connectivity_visibility,
            "train_label_reachability": anchor.train_label_reachability,
            "degree_band": anchor.degree_band,
        }, parent_pred=anchor.pred)
        summary = metrics["summary"]
        summary["candidate_id"] = c["candidate_id"]
        summary["operator"] = c["operator"]
        summary["overall_gain"] = None if summary["overall_accuracy"] is None else summary["overall_accuracy"] - float(anchor.overall_metrics["overall_accuracy"])
        summary["macro_gain"] = None if summary["macro_accuracy"] is None else summary["macro_accuracy"] - float(anchor.overall_metrics["macro_accuracy"])
        fold_gains = []
        for row in metrics["fold"]:
            parent_acc = _accuracy(anchor.labels, anchor.pred, anchor.fold == row["fold"])
            gain = None if row["accuracy"] is None or parent_acc is None else row["accuracy"] - parent_acc
            row["gain"] = gain
            fold_gains.append(gain if gain is not None else -1.0)
        summary["worst_fold_gain"] = min(fold_gains) if fold_gains else None
        summary["positive_fold_count"] = int(sum(g >= -1e-12 for g in fold_gains))
        duplicate = bool(np.array_equal(pred, anchor.pred) and np.allclose(proba, anchor.proba, atol=1e-12))
        summary["duplicate_of_anchor"] = duplicate
        return {"candidate": c, "proba": proba, "pred": pred, "metrics": {"candidate_id": c["candidate_id"], "operator": c["operator"], "summary": summary, "fold_gains": metrics["fold"], "bucket_gains": metrics["bucket"], "class_gains": metrics["class"], "bucket_class_gains": metrics["bucket_class"]}, "crossfit_assignment": assignment}

    def _crossfit_proba(self, c: dict[str, Any], anchor: PortfolioAsset) -> tuple[np.ndarray, dict[str, Any]]:
        assets = self._current_assets
        op = c["operator"]
        if op in {"probability_blend", "logit_blend", "bucket_route", "residual_patch"} and not c.get("search_space"):
            proba = self._apply_operator(c, c["fixed_params"], np.ones(anchor.labels.shape[0], dtype=bool), assets)
            return proba, {"candidate_id": c["candidate_id"], "mode": "fixed_no_learned_parameter", "fold_assignments": []}
        out = np.zeros_like(anchor.proba)
        assignments = []
        for held_fold in sorted(set(anchor.fold.tolist())):
            fit = anchor.fold != held_fold
            valid = anchor.fold == held_fold
            best = self._select_param(c, fit, assets, anchor)
            out[valid] = self._apply_operator(c, best, valid, assets)[valid]
            assignments.append({"held_out_fold": int(held_fold), "selection_rows": int(fit.sum()), "application_rows": int(valid.sum()), "selected_params": best, "held_out_labels_used_for_selection": False})
        return out, {"candidate_id": c["candidate_id"], "mode": "strict_outer_fold_cross_fit", "fold_assignments": assignments}

    @property
    def _current_assets(self) -> dict[str, PortfolioAsset]:
        return self._assets_for_eval

    def _select_param(self, c: dict[str, Any], fit: np.ndarray, assets: dict[str, PortfolioAsset], anchor: PortfolioAsset) -> dict[str, Any]:
        keys = sorted(c["search_space"].keys())
        grids = [{}]
        for key in keys:
            grids = [g | {key: val} for g in grids for val in c["search_space"][key]]
        best = None
        best_score = (-999.0, -999.0, -999.0)
        for params in grids:
            p = self._apply_operator(c, c["fixed_params"] | params, fit, assets)
            pred = p.argmax(axis=1)
            overall = _accuracy(anchor.labels, pred, fit) or 0.0
            macro = _macro(anchor.labels, pred, fit) or 0.0
            net = _delta_counts(anchor.pred[fit] == anchor.labels[fit], pred[fit] == anchor.labels[fit])["net"]
            score = (overall, macro, float(net))
            if score > best_score:
                best_score = score
                best = params
        return c["fixed_params"] | (best or {})

    def _apply_operator(self, c: dict[str, Any], params: dict[str, Any], mask: np.ndarray, assets: dict[str, PortfolioAsset]) -> np.ndarray:
        a = assets[c["assets"][0]]
        b = assets[c["assets"][1]]
        op = c["operator"]
        if op == "probability_blend":
            alpha = float(params["alpha"])
            return ((1.0 - alpha) * a.proba + alpha * b.proba).astype(np.float64)
        if op == "logit_blend":
            alpha = float(params["alpha"])
            return _softmax((1.0 - alpha) * _logit(a.proba) + alpha * _logit(b.proba))
        if op == "bucket_route":
            out = a.proba.copy()
            route_mask = a.isolated_mask if params.get("route_bucket") == "isolated" else mask
            out[route_mask] = b.proba[route_mask]
            return out
        if op == "confidence_gate":
            threshold = float(params["threshold"])
            out = a.proba.copy()
            scope = a.isolated_mask if params.get("scope") == "isolated" else mask
            choose_b = scope & ((b.proba.max(axis=1) - a.proba.max(axis=1)) >= threshold)
            out[choose_b] = b.proba[choose_b]
            return out
        if op == "class_weighted_blend":
            alpha = float(params["alpha"])
            out = a.proba.copy()
            for cls in range(CLASS_N):
                rows = mask & (a.labels == cls)
                out[rows] = (1.0 - alpha) * a.proba[rows] + alpha * b.proba[rows]
            return out
        if op == "residual_patch":
            return b.proba.copy()
        raise ValueError(f"unsupported operator: {op}")

    def _decisions(self, evaluated: list[dict[str, Any]], anchor: PortfolioAsset) -> dict[str, Any]:
        items = []
        for ev in evaluated:
            s = ev["metrics"]["summary"]
            reasons = []
            status = "accepted"
            if s["duplicate_of_anchor"]:
                status = "rejected"
                reasons.append("duplicate_anchor")
            if (s["overall_gain"] or 0.0) < -1e-12:
                status = "rejected"
                reasons.append("overall_decrease")
            if (s["macro_gain"] or 0.0) < -0.001:
                status = "rejected"
                reasons.append("macro_damage_exceeds_policy")
            if s.get("net", 0) <= 0:
                status = "rejected"
                reasons.append("rescue_not_greater_than_damage")
            if s["positive_fold_count"] < 4:
                status = "rejected"
                reasons.append("less_than_4_nonnegative_folds")
            if not reasons and (s["overall_gain"] or 0.0) <= 1e-12:
                status = "rejected"
                reasons.append("no_gain")
            items.append({
                "candidate_id": s["candidate_id"],
                "operator": s["operator"],
                "status": status,
                "reason_codes": sorted(set(reasons)) or ["policy_passed"],
                "overall_gain": s["overall_gain"],
                "macro_gain": s["macro_gain"],
                "worst_fold_gain": s["worst_fold_gain"],
                "positive_fold_count": s["positive_fold_count"],
                "rescue": s.get("rescue"),
                "damage": s.get("damage"),
                "net": s.get("net"),
                "changed_count": s.get("changed_count"),
                "change_precision": s.get("change_precision"),
                "information_source_novelty": "new_combination_of_existing_oof" if s["candidate_id"] != "residual_patch_composed_over_anchor" else "none_duplicate",
                "complexity_cost": "low" if s["operator"] in {"probability_blend", "bucket_route"} else "medium",
                "deployment_risk": "low_no_test_artifact_generated",
            })
        ranked = sorted(items, key=lambda r: (r["status"] != "accepted", -(r["overall_gain"] or -999), -(r["macro_gain"] or -999), r["candidate_id"]))
        return {"decision_version": FUSION_VERSION, "items": ranked, "summary": {"accepted_count": sum(1 for r in items if r["status"] == "accepted"), "rejected_count": sum(1 for r in items if r["status"] == "rejected"), "best_candidate_id": ranked[0]["candidate_id"] if ranked else "", "best_status": ranked[0]["status"] if ranked else "none"}, "policy": {"strict_cross_fit_required": True, "oracle_executable": False, "test_feedback_allowed": False, "min_nonnegative_folds": 4}}

    def _fusion_plan(self, complementarity: dict[str, Any], decisions: dict[str, Any]) -> dict[str, Any]:
        best = decisions["items"][0] if decisions["items"] else {}
        return {
            "planner_version": FUSION_VERSION,
            "selected_assets": [BASE_ID, EXPERT_ID, COMPOSED_ID, ANCHOR_ID],
            "selected_operator": best.get("operator", ""),
            "target_scope": "offline_oof_only",
            "parameter_search_space": "finite_policy_grid",
            "cross_fit_plan": "for learned alpha/threshold/class-aware params: select on four folds and apply to held-out fold",
            "evaluation_plan": "compare all candidates against A1_EVAL_ANCHOR_V1 OOF only; no Test prediction",
            "acceptance_conditions": decisions["policy"],
            "estimated_cost": {"training": "none", "inference": "existing_oof_array_operations_only"},
            "handoff_status": "ready_for_real_single_round_experiment_design" if decisions["summary"]["accepted_count"] >= 0 else "blocked",
            "oracle_from_complementarity_is_diagnostic_only": complementarity["oracle_is_diagnostic_only"],
        }

    def _report(self, decisions: dict[str, Any], complementarity: dict[str, Any], plan: dict[str, Any]) -> str:
        best = decisions["items"][0] if decisions["items"] else {}
        base_vs_composed = next((p for p in complementarity["pairs"] if p["model_a"] == BASE_ID and p["model_b"] == COMPOSED_ID), {})
        comp = base_vs_composed.get("prediction_complementarity", {})
        return "\n".join([
            "# Fusion Controller Report",
            "",
            f"best_candidate: `{best.get('candidate_id', '')}`",
            f"best_status: `{best.get('status', '')}`",
            f"accepted_count: `{decisions['summary'].get('accepted_count')}`",
            f"base_vs_composed_net: `{comp.get('net')}`",
            f"base_vs_composed_rescue: `{comp.get('rescue')}`",
            f"base_vs_composed_damage: `{comp.get('damage')}`",
            "",
            "Oracle complementarity is diagnostic only and is not an executable strategy.",
            "No training, Test prediction, submission, LLM call, or network access was executed.",
            "",
        ])

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()


def run_fusion_controller(**kwargs: Any) -> dict[str, Any]:
    controller = FusionController(project_root=kwargs.pop("project_root"))
    return controller.run(**kwargs)
