# -*- coding: utf-8 -*-
"""Evaluation anchor bootstrap and dual-anchor contract repair.

This module separates:
- online_deployment_anchor: the immutable Test-side champion CSV identity.
- oof_evaluation_anchor: a reproducible offline OOF baseline for future
  candidate evaluation.

It may freeze a verified future evaluation fold, and may materialize an
evaluation OOF anchor only when all component provenance checks pass. It never
trains models, generates Test predictions, or consumes experiment rounds.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .paths import PathResolver
from .research.event_store import json_dumps, rel_ref, sha256_file, stable_hash

BOOTSTRAP_VERSION = "evaluation_anchor_bootstrap_v1"
ONLINE_ANCHOR_IDENTITY = "A1_V53Q1_TRANSITION_STABLE_EDGE_H2"
EVAL_ANCHOR_IDENTITY = "A1_EVAL_ANCHOR_V1"
FOLD_IDENTITY = "AFAC_A1_FOLD_V1"
TRAIN_N = 11001
TEST_N = 2751
CLASS_N = 10


def _prob_norm(arr: np.ndarray) -> dict[str, Any]:
    sums = arr.sum(axis=1)
    return {
        "row_sum_min": float(sums.min()),
        "row_sum_max": float(sums.max()),
        "passed": bool(np.all((sums > 0.999) & (sums < 1.001))),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class EvaluationAnchorBootstrap:
    def __init__(self, *, project_root: str | Path, paths_config: str = "") -> None:
        self.project_root = Path(project_root).resolve()
        self.resolver = PathResolver(self.project_root, paths_config or None)

    def run(
        self,
        *,
        a1_npz: str | Path,
        fold_candidate: str | Path,
        v43c_oof: str | Path,
        v46a_oof: str | Path,
        out_root: str | Path = "artifacts/evaluation_anchor",
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        paths = {
            "a1_npz": self._resolve(a1_npz),
            "fold_candidate": self._resolve(fold_candidate),
            "v43c_oof": self._resolve(v43c_oof),
            "v46a_oof": self._resolve(v46a_oof),
            "online_anchor_csv": self.resolver.a1_anchor_csv(),
        }
        missing = [name for name, path in paths.items() if not path or not path.exists()]
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        run_id = stable_hash({"version": BOOTSTRAP_VERSION, "inputs": {k: sha256_file(v) for k, v in paths.items() if v and v.is_file()}})
        out_dir = self._resolve(out_root)
        manifest_path = out_dir / "evaluation_anchor_bootstrap_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = _read_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": manifest.get("run_id", run_id), "artifacts": manifest.get("artifacts", {})}
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._execute(paths=paths, out_dir=out_dir, run_id=run_id)
        manifest = _read_json(manifest_path)
        return {"status": manifest.get("status"), "run_id": run_id, "artifacts": artifacts}

    def _execute(self, *, paths: dict[str, Path], out_dir: Path, run_id: str) -> dict[str, str]:
        started = time.time()
        a1 = np.load(paths["a1_npz"], allow_pickle=False)
        online = self._online_anchor(paths["online_anchor_csv"])
        fold_audit = self._fold_audit(paths["fold_candidate"], a1)
        fold_artifacts = self._materialize_fold(paths["fold_candidate"], fold_audit, out_dir)
        v43_audit = self._v43_audit(paths["v43c_oof"])
        v46_audit = self._v46_audit(paths["v46a_oof"], paths["v43c_oof"], fold_audit)
        materialization = self._maybe_materialize_eval_anchor(paths, fold_audit, fold_artifacts, v43_audit, v46_audit, out_dir)
        status = materialization["status"]
        payloads = {
            "dual_anchor_contract": {
                "contract_version": BOOTSTRAP_VERSION,
                "online_deployment_anchor": online,
                "oof_evaluation_anchor": {
                    "identity": EVAL_ANCHOR_IDENTITY,
                    "status": status,
                    "deployment_equivalent": False,
                    "historical_v53q1_oof_status": "not_materialized",
                    "historical_v53q1_oof_equivalent": False,
                },
                "m5_contract_repair": {
                    "required": [
                        "verified_oof_evaluation_anchor",
                        "verified_canonical_evaluation_fold",
                        "candidate_oof",
                        "same_evaluation_protocol",
                        "registered_adapter",
                        "complete_oof_plan",
                    ],
                    "no_longer_required": [
                        "final_v53q1_oof",
                        "final_v53q1_oof_manifest",
                    ],
                    "deployment_stage_requires": ["online_deployment_anchor"],
                },
            },
            "fold_candidate_audit": fold_audit,
            "v43c_oof_identity_audit": v43_audit,
            "v46a_oof_identity_audit": v46_audit,
            "evaluation_anchor_materialization": materialization,
        }
        if status == "rebuild_required":
            payloads["evaluation_anchor_rebuild_spec"] = self._rebuild_spec(fold_artifacts, v43_audit, v46_audit)
        artifacts: dict[str, str] = {}
        for name, payload in payloads.items():
            path = out_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        if status == "rebuild_required":
            plan = out_dir / "evaluation_anchor_rebuild_plan.md"
            plan.write_text(self._rebuild_plan_text(payloads["evaluation_anchor_rebuild_spec"]), encoding="utf-8")
            artifacts["evaluation_anchor_rebuild_plan"] = rel_ref(plan, self.project_root)
        report = out_dir / "EVALUATION_ANCHOR_REPORT.md"
        report.write_text(self._report(status, fold_audit, v43_audit, v46_audit, materialization), encoding="utf-8")
        artifacts["evaluation_anchor_report"] = rel_ref(report, self.project_root)
        manifest = {
            "manifest_version": BOOTSTRAP_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "read_only_inputs": True,
            "trains_model": False,
            "generates_prediction": False,
            "creates_submission": False,
            "counts_as_experiment_round": False,
            "online_deployment_anchor": ONLINE_ANCHOR_IDENTITY,
            "evaluation_anchor_identity": EVAL_ANCHOR_IDENTITY,
            "historical_v53q1_oof_status": "not_materialized",
            "fold_status": fold_audit["status"],
            "evaluation_anchor_status": materialization["status"],
            "view_hash": stable_hash(payloads),
            "artifacts": artifacts,
        }
        manifest_path = out_dir / "evaluation_anchor_bootstrap_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["evaluation_anchor_bootstrap_manifest"] = rel_ref(manifest_path, self.project_root)
        return artifacts

    def _online_anchor(self, path: Path) -> dict[str, Any]:
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames or []
        return {
            "identity": ONLINE_ANCHOR_IDENTITY,
            "artifact": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "columns": columns,
            "usage": ["current_online_champion", "test_deployment_reference", "online_score_identity"],
            "requires_oof_proba": False,
            "requires_canonical_fold": False,
            "deployment_equivalent": True,
        }

    def _fold_audit(self, path: Path, a1: Any) -> dict[str, Any]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = list(rows[0]) if rows else []
        node_ids = np.array([int(row["node_id"]) for row in rows], dtype=np.int64)
        labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
        folds = np.array([int(row["fold"]) for row in rows], dtype=np.int64)
        a1_train = a1["train_idx"]
        a1_test = set(map(int, a1["test_idx"].tolist()))
        a1_labels = a1["labels"]
        class_by_fold = {str(fold): np.bincount(labels[folds == fold], minlength=CLASS_N).astype(int).tolist() for fold in sorted(set(folds.tolist()))}
        errors: list[str] = []
        if len(rows) != TRAIN_N:
            errors.append("row_count_not_11001")
        if columns != ["node_id", "label", "fold"]:
            errors.append("columns_not_node_id_label_fold")
        if len(set(node_ids.tolist())) != TRAIN_N:
            errors.append("duplicate_or_missing_node_id")
        if set(node_ids.tolist()) != set(map(int, a1_train.tolist())):
            errors.append("node_ids_not_equal_a1_train_idx")
        if set(node_ids.tolist()) & a1_test:
            errors.append("contains_test_idx")
        if not np.array_equal(labels, a1_labels[node_ids]):
            errors.append("labels_not_equal_a1_npz")
        if set(folds.tolist()) != {0, 1, 2, 3, 4}:
            errors.append("fold_range_not_0_4")
        status = "verified_future_protocol" if not errors else "rejected"
        return {
            "audit_version": BOOTSTRAP_VERSION,
            "identity": FOLD_IDENTITY,
            "source_path": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "status": status,
            "rows": len(rows),
            "columns": columns,
            "train_idx_coverage": len(set(node_ids.tolist())),
            "contains_test_idx": bool(set(node_ids.tolist()) & a1_test),
            "label_consistency_with_a1_npz": "pass" if "labels_not_equal_a1_npz" not in errors else "fail",
            "fold_values": sorted(set(folds.tolist())),
            "class_distribution_by_fold": class_by_fold,
            "historical_v53q1_fold_equivalence": "unverified",
            "future_evaluation_protocol": "verified" if not errors else "unavailable",
            "errors": errors,
        }

    def _materialize_fold(self, src: Path, audit: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        if audit["status"] != "verified_future_protocol":
            return {"status": "unavailable"}
        fold_csv = out_dir / f"{FOLD_IDENTITY}.csv"
        shutil.copyfile(src, fold_csv)
        manifest = {
            "manifest_version": BOOTSTRAP_VERSION,
            "fold_identity": FOLD_IDENTITY,
            "artifact": rel_ref(fold_csv, self.project_root),
            "sha256": sha256_file(fold_csv),
            "rows": TRAIN_N,
            "fold_count": 5,
            "historical_v53q1_fold_equivalence": "unverified",
            "future_evaluation_protocol": "verified",
            "source_hash": audit["sha256"],
        }
        manifest_path = out_dir / f"{FOLD_IDENTITY}_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        return {"status": "materialized", "fold_csv": rel_ref(fold_csv, self.project_root), "fold_manifest": rel_ref(manifest_path, self.project_root), "fold_hash": sha256_file(fold_csv)}

    def _v43_audit(self, path: Path) -> dict[str, Any]:
        z = np.load(path, allow_pickle=False)
        keys = list(z.files)
        result: dict[str, Any] = {
            "audit_version": BOOTSTRAP_VERSION,
            "component": "v43C_candidate",
            "path": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "keys": keys,
            "required_keys_present": all(key in keys for key in ["proba", "base_proba", "train_idx", "labels", "bucket"]),
            "proba_shape": list(z["proba"].shape) if "proba" in keys else [],
            "base_proba_shape": list(z["base_proba"].shape) if "base_proba" in keys else [],
            "proba_normalization": _prob_norm(z["proba"]) if "proba" in keys else {},
            "base_proba_normalization": _prob_norm(z["base_proba"]) if "base_proba" in keys else {},
            "correct_smooth_final_proba_forbidden_as_anchor": True,
            "base_identity_status": "unverified",
            "usable_as_evaluation_base": False,
            "evidence": [],
            "missing_evidence": [],
        }
        decision = _read_json(path.with_name("correct_smooth_v1_decision.json"))
        sources = decision.get("base_oof_sources", []) if isinstance(decision.get("base_oof_sources"), list) else []
        result["decision_base_oof_sources"] = sources
        if not sources:
            result["missing_evidence"].append("base_oof_source_manifest")
            return result
        source_path = self._normalize_external_path(str(sources[0]))
        if not source_path.exists():
            result["missing_evidence"].append("base_oof_source_file_not_found")
            result["base_source_path"] = str(source_path)
            return result
        source = np.load(source_path, allow_pickle=False)
        result["base_source_sha256"] = sha256_file(source_path)
        result["base_source_keys"] = list(source.files)
        comparisons = {}
        for key in ["anchor_proba", "ensemble_proba", "proba"]:
            if key in source.files and "base_proba" in keys:
                comparisons[f"base_proba_equals_{key}"] = bool(np.array_equal(z["base_proba"], source[key]))
        result["base_source_comparisons"] = comparisons
        if any(comparisons.values()):
            result["base_identity_status"] = "verified_existing"
            result["usable_as_evaluation_base"] = True
        else:
            result["missing_evidence"].append("base_proba_not_equal_declared_base_source")
        return result

    def _v46_audit(self, path: Path, v43_path: Path, fold: dict[str, Any]) -> dict[str, Any]:
        z46 = np.load(path, allow_pickle=False)
        z43 = np.load(v43_path, allow_pickle=False)
        keys = list(z46.files)
        mask = z46["isolated_mask"] if "isolated_mask" in keys else np.zeros((0,), dtype=bool)
        result = {
            "audit_version": BOOTSTRAP_VERSION,
            "component": "v46A_candidate",
            "path": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "keys": keys,
            "required_keys_present": all(key in keys for key in ["proba", "expert_proba", "train_idx", "labels", "isolated_mask"]),
            "proba_shape": list(z46["proba"].shape) if "proba" in keys else [],
            "expert_proba_shape": list(z46["expert_proba"].shape) if "expert_proba" in keys else [],
            "proba_normalization": _prob_norm(z46["proba"]) if "proba" in keys else {},
            "expert_proba_normalization": _prob_norm(z46["expert_proba"]) if "expert_proba" in keys else {},
            "train_idx_aligned_with_v43": bool(np.array_equal(z46["train_idx"], z43["train_idx"])) if "train_idx" in keys and "train_idx" in z43.files else False,
            "labels_aligned_with_v43": bool(np.array_equal(z46["labels"], z43["labels"])) if "labels" in keys and "labels" in z43.files else False,
            "fold_protocol_status": fold["status"],
            "isolated_true_count": int(mask.sum()) if mask.size else 0,
            "composition_rule": "proba = expert_proba on isolated_mask; otherwise v43 base_proba",
            "composition_rule_verified": bool(
                mask.size
                and np.allclose(z46["proba"][mask], z46["expert_proba"][mask])
                and np.allclose(z46["proba"][~mask], z43["base_proba"][~mask])
            ) if all(key in keys for key in ["proba", "expert_proba", "isolated_mask"]) and "base_proba" in z43.files else False,
            "usable_as_evaluation_anchor_component": False,
            "missing_evidence": [],
        }
        if not result["composition_rule_verified"]:
            result["missing_evidence"].append("composition_rule_not_verified")
        return result

    def _maybe_materialize_eval_anchor(self, paths: dict[str, Path], fold: dict[str, Any], fold_artifacts: dict[str, Any], v43: dict[str, Any], v46: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        missing = []
        if fold["status"] != "verified_future_protocol":
            missing.append("verified_canonical_evaluation_fold")
        if not v43.get("usable_as_evaluation_base"):
            missing.append("verified_v43c_base_oof_identity")
        if not v46.get("composition_rule_verified"):
            missing.append("verified_v46a_composition_rule")
        if missing:
            return {
                "status": "rebuild_required",
                "materialized": False,
                "missing_evidence": missing,
                "historical_v53q1_oof_status": "not_materialized",
                "deployment_equivalent": False,
                "historical_v53q1_oof_equivalent": False,
            }
        z46 = np.load(paths["v46a_oof"], allow_pickle=False)
        fold_map = {}
        with (out_dir / f"{FOLD_IDENTITY}.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                fold_map[int(row["node_id"])] = int(row["fold"])
        train_idx = z46["train_idx"]
        fold_arr = np.array([fold_map[int(idx)] for idx in train_idx], dtype=np.int64)
        proba = z46["proba"]
        pred = proba.argmax(axis=1).astype(np.int64)
        labels = z46["labels"]
        acc = float((pred == labels).mean())
        class_acc = []
        for cls in range(CLASS_N):
            mask = labels == cls
            class_acc.append(float((pred[mask] == labels[mask]).mean()) if mask.any() else None)
        oof_path = out_dir / f"{EVAL_ANCHOR_IDENTITY}_oof.npz"
        np.savez(
            oof_path,
            train_idx=train_idx,
            labels=labels,
            proba=proba,
            pred=pred,
            fold=fold_arr,
            isolation_mask=z46["isolated_mask"],
            component_source=np.array(["v46a_composed_from_v43_base_and_isolated_expert"]),
            anchor_version=np.array([EVAL_ANCHOR_IDENTITY]),
        )
        manifest = {
            "manifest_version": BOOTSTRAP_VERSION,
            "online_anchor_identity": ONLINE_ANCHOR_IDENTITY,
            "evaluation_anchor_identity": EVAL_ANCHOR_IDENTITY,
            "artifact": rel_ref(oof_path, self.project_root),
            "sha256": sha256_file(oof_path),
            "deployment_equivalent": False,
            "historical_v53q1_oof_equivalent": False,
            "historical_v53q1_oof_status": "not_materialized",
            "component_hashes": {"v43c_oof": v43["sha256"], "v46a_oof": v46["sha256"]},
            "fold_hash": fold_artifacts.get("fold_hash", ""),
            "composition_rule": v46["composition_rule"],
            "oof_accuracy": acc,
            "macro_accuracy": float(np.mean([x for x in class_acc if x is not None])),
            "class_accuracy": class_acc,
            "bucket_metrics": {"status": "not_generated_in_bootstrap"},
            "safety_status": {
                "test_truth_used": False,
                "test_prediction_generated": False,
                "training_executed": False,
                "counts_as_experiment_round": False,
            },
        }
        manifest_path = out_dir / f"{EVAL_ANCHOR_IDENTITY}_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        return {"status": "materialized_from_verified_components", "materialized": True, "oof_npz": rel_ref(oof_path, self.project_root), "manifest": rel_ref(manifest_path, self.project_root), "oof_accuracy": acc}

    def _rebuild_spec(self, fold_artifacts: dict[str, Any], v43: dict[str, Any], v46: dict[str, Any]) -> dict[str, Any]:
        return {
            "spec_version": BOOTSTRAP_VERSION,
            "status": "rebuild_required",
            "allowed_actions": [
                "reproduce_v43c_main_oof",
                "reproduce_v46a_isolated_expert_oof",
                "apply_deterministic_oof_composition_rule",
            ],
            "forbidden_actions": [
                "rebuild_test_only_four_node_patch",
                "use_champion_csv_as_oof",
                "use_correct_smooth_final_proba_as_anchor",
                "use_h2gcn_component_oof_as_anchor",
                "use_test_proba_as_oof",
            ],
            "inputs": {
                "fold_manifest": fold_artifacts.get("fold_manifest", ""),
                "v43_missing_evidence": v43.get("missing_evidence", []),
                "v46_missing_evidence": v46.get("missing_evidence", []),
            },
            "fold": FOLD_IDENTITY,
            "model_configs": ["v43c_3seed_reproducible_config", "v46a_isolated_expert_reproducible_config"],
            "seeds": ["v43c seeds from source run", "v46A old_seed=1402401", "v46A fresh_seed=1462517"],
            "training_commands": ["TO_BE_APPROVED: reproduce v43C OOF", "TO_BE_APPROVED: reproduce v46A isolated OOF"],
            "estimated_runtime": "requires approval; expected GPU training time unknown from current verified manifests",
            "estimated_gpu_memory": "requires approval; likely single-GPU feasible but not verified here",
            "outputs": [f"{EVAL_ANCHOR_IDENTITY}_oof.npz", f"{EVAL_ANCHOR_IDENTITY}_manifest.json"],
            "success_conditions": ["OOF rows=11001", "10-class normalized proba", "train_idx/labels/fold aligned", "no test truth", "complete manifest"],
            "failure_conditions": ["component provenance mismatch", "fold incompatibility", "negative safety audit"],
            "stop_conditions": ["requires Test patch rebuild", "requires unverified Fold", "requires final v53Q-1 historical OOF claim"],
            "counts_as_experiment_round": True,
        }

    def _rebuild_plan_text(self, spec: dict[str, Any]) -> str:
        return "\n".join([
            "# Evaluation Anchor Rebuild Plan",
            "",
            "Status: `rebuild_required`",
            "",
            "This plan requires explicit approval before any training.",
            "It must not rebuild Test-only patches or claim historical v53Q-1 OOF equivalence.",
            "",
            f"Fold: `{spec.get('fold')}`",
            f"Counts as experiment round: `{spec.get('counts_as_experiment_round')}`",
            "",
        ])

    def _report(self, status: str, fold: dict[str, Any], v43: dict[str, Any], v46: dict[str, Any], materialization: dict[str, Any]) -> str:
        return "\n".join([
            "# Evaluation Anchor Bootstrap Report",
            "",
            f"status: `{status}`",
            f"fold_status: `{fold.get('status')}`",
            f"v43_base_identity: `{v43.get('base_identity_status')}`",
            f"v46_composition_rule_verified: `{v46.get('composition_rule_verified')}`",
            f"missing_evidence: `{materialization.get('missing_evidence', [])}`",
            "",
            "No training, prediction, or submission was executed.",
            "",
        ])

    def _normalize_external_path(self, value: str) -> Path:
        # Some legacy JSON paths were written under a mojibake user directory.
        path = Path(value)
        if path.exists():
            return path
        text = value
        replacements = {
            "?????": "???",
            "???": "??",
            "?????": "???",
            "A???": "A??",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return Path(text)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()
