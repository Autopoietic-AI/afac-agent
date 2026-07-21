# -*- coding: utf-8 -*-
"""Direct materialization of the A1 offline evaluation anchor.

This module separates two anchors:

- online_deployment_anchor: immutable Test-side champion CSV identity.
- oof_evaluation_anchor: strict offline OOF baseline for future evaluation.

The direct materialization path uses only explicitly supplied existing OOF
assets. It does not train, does not infer Test labels, does not generate a Test
submission, and does not claim historical equivalence to v43C or v53Q-1.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .paths import PathResolver
from .research.event_store import json_dumps, rel_ref, sha256_file, stable_hash

BOOTSTRAP_VERSION = "evaluation_anchor_direct_materialization_v1"
ONLINE_ANCHOR_IDENTITY = "A1_V53Q1_TRANSITION_STABLE_EDGE_H2"
EVAL_ANCHOR_IDENTITY = "A1_EVAL_ANCHOR_V1"
FOLD_IDENTITY = "AFAC_A1_FOLD_V1"
TRAIN_N = 11001
TEST_N = 2751
CLASS_N = 10
EXPECTED_FOLD_HASH = "1c9dbaf6718eef8e19acec57106122a4d74e3ecbf5481b7d8898e91e658bccb3"


def _as_bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _prob_audit(arr: np.ndarray) -> dict[str, Any]:
    finite = bool(np.isfinite(arr).all())
    row_sums = arr.sum(axis=1) if arr.ndim == 2 else np.array([])
    return {
        "shape": list(arr.shape),
        "finite": finite,
        "min": float(np.nanmin(arr)) if arr.size else None,
        "max": float(np.nanmax(arr)) if arr.size else None,
        "row_sum_min": float(row_sums.min()) if row_sums.size else None,
        "row_sum_max": float(row_sums.max()) if row_sums.size else None,
        "max_row_sum_abs_error": float(np.abs(row_sums - 1.0).max()) if row_sums.size else None,
        "passed": bool(
            arr.ndim == 2
            and arr.shape == (TRAIN_N, CLASS_N)
            and finite
            and arr.min() >= -1e-7
            and arr.max() <= 1.0000001
            and np.allclose(row_sums, 1.0, atol=1e-5)
        ),
    }


def _macro_accuracy(labels: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for cls in range(CLASS_N):
        mask = labels == cls
        if mask.any():
            vals.append(float((pred[mask] == labels[mask]).mean()))
    return float(np.mean(vals)) if vals else 0.0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_npz_deterministic(path: Path, **arrays: Any) -> None:
    """Write an NPZ with stable zip metadata so file hash is rerun-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, buf.getvalue())


class EvaluationAnchorBootstrap:
    def __init__(self, *, project_root: str | Path, paths_config: str = "", expected_fold_hash: str = EXPECTED_FOLD_HASH) -> None:
        self.project_root = Path(project_root).resolve()
        self.resolver = PathResolver(self.project_root, paths_config or None)
        self.expected_fold_hash = expected_fold_hash

    def run(
        self,
        *,
        a1_npz: str | Path,
        fold_candidate: str | Path,
        v43c_oof: str | Path,
        v46a_oof: str | Path,
        out_root: str | Path = "artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1",
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
        input_hashes = {k: sha256_file(v) for k, v in paths.items() if v and v.is_file()}
        run_id = stable_hash({"version": BOOTSTRAP_VERSION, "inputs": input_hashes})
        out_dir = self._resolve(out_root)
        manifest_path = out_dir / f"{EVAL_ANCHOR_IDENTITY}_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": {"evaluation_anchor_manifest": rel_ref(manifest_path, self.project_root)}}
        out_dir.mkdir(parents=True, exist_ok=True)
        return self._execute(paths=paths, input_hashes=input_hashes, out_dir=out_dir, run_id=run_id)

    def _execute(self, *, paths: dict[str, Path], input_hashes: dict[str, str], out_dir: Path, run_id: str) -> dict[str, Any]:
        started = time.time()
        a1 = np.load(paths["a1_npz"], allow_pickle=False)
        fold = self._fold_audit(paths["fold_candidate"], a1)
        fold_artifacts = self._materialize_fold(paths["fold_candidate"], fold, out_dir)
        structure = self._structure_axes(a1, fold["node_ids"])
        v43 = self._v43_audit(paths["v43c_oof"], fold, a1)
        v46 = self._v46_audit(paths["v46a_oof"], paths["v43c_oof"], fold, a1)
        materialization = self._materialize(paths, input_hashes, fold, fold_artifacts, structure, v43, v46, out_dir)
        status = materialization["status"]

        artifacts: dict[str, str] = {}
        payloads = {
            "component_registry": self._component_registry(input_hashes, fold, v43, v46, materialization),
            "integrity_audit": materialization["integrity_audit"],
            "provenance_graph": self._provenance_graph(input_hashes, materialization),
        }
        for name, payload in payloads.items():
            path = out_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)

        if status == "materialized_from_verified_components":
            artifacts["evaluation_anchor_oof"] = materialization.get("oof_npz", "")
            artifacts["evaluation_anchor_replay_oof"] = materialization.get("replay_npz", "")
            metrics = materialization["metrics"]
            for name, rows in {
                "fold_metrics": metrics["fold_metrics"],
                "class_metrics": metrics["class_metrics"],
                "bucket_metrics": metrics["bucket_metrics"],
                "bucket_class_metrics": metrics["bucket_class_metrics"],
                "degree_band_metrics": metrics["degree_band_metrics"],
                "reachability_metrics": metrics["reachability_metrics"],
            }.items():
                if rows:
                    _write_csv(out_dir / f"{name}.csv", list(rows[0].keys()), rows)
                else:
                    _write_csv(out_dir / f"{name}.csv", ["status"], [{"status": "empty"}])
                artifacts[name] = rel_ref(out_dir / f"{name}.csv", self.project_root)
        else:
            spec = self._rebuild_spec(fold, v43, v46, materialization)
            spec_path = out_dir / "evaluation_anchor_rebuild_spec.json"
            spec_path.write_text(json_dumps(spec) + "\n", encoding="utf-8")
            artifacts["evaluation_anchor_rebuild_spec"] = rel_ref(spec_path, self.project_root)

        report = out_dir / "EVALUATION_ANCHOR_REPORT.md"
        report.write_text(self._report(status, fold, v43, v46, materialization), encoding="utf-8")
        artifacts["evaluation_anchor_report"] = rel_ref(report, self.project_root)

        manifest = {
            "manifest_version": BOOTSTRAP_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "read_only_inputs": True,
            "training_executed": False,
            "trains_model": False,
            "generates_prediction": False,
            "creates_submission": False,
            "counts_as_experiment_round": False,
            "counts_as_anchor_bootstrap_run": status == "materialized_from_verified_components",
            "scientific_rounds_used": 0,
            "online_deployment_anchor": ONLINE_ANCHOR_IDENTITY,
            "evaluation_anchor_identity": EVAL_ANCHOR_IDENTITY,
            "artifact": materialization.get("oof_npz", ""),
            "sha256": materialization.get("oof_sha256", ""),
            "anchor_content_hash": materialization.get("anchor_content_hash", ""),
            "deployment_equivalent": False,
            "historical_v53q1_oof_equivalent": False,
            "historical_v53q1_oof_status": "not_materialized",
            "historical_v43c_identity_equivalence": "unverified",
            "functional_oof_validity": "verified" if status == "materialized_from_verified_components" else "unavailable",
            "rebuild_route": materialization.get("route", ""),
            "fold_identity": FOLD_IDENTITY,
            "fold_hash": fold.get("sha256", ""),
            "fold_status": fold["status"],
            "evaluation_anchor_status": status,
            "metrics_summary": materialization.get("metrics_summary", {}),
            "input_hashes": input_hashes,
            "view_hash": stable_hash({k: v for k, v in payloads.items()} | {"materialization": materialization.get("stable_summary", {})}),
            "artifacts": artifacts,
        }
        manifest_path = out_dir / f"{EVAL_ANCHOR_IDENTITY}_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["evaluation_anchor_manifest"] = rel_ref(manifest_path, self.project_root)

        bootstrap_manifest = {
            "manifest_version": BOOTSTRAP_VERSION,
            "run_id": run_id,
            "status": status,
            "read_only_inputs": True,
            "training_executed": False,
            "trains_model": False,
            "generates_prediction": False,
            "creates_submission": False,
            "counts_as_experiment_round": False,
            "counts_as_anchor_bootstrap_run": status == "materialized_from_verified_components",
            "scientific_rounds_used": 0,
            "online_deployment_anchor": ONLINE_ANCHOR_IDENTITY,
            "evaluation_anchor_identity": EVAL_ANCHOR_IDENTITY,
            "historical_v53q1_oof_status": "not_materialized",
            "fold_status": fold["status"],
            "evaluation_anchor_status": status,
            "artifacts": artifacts,
        }
        bootstrap_manifest_path = out_dir / "evaluation_anchor_bootstrap_manifest.json"
        bootstrap_manifest_path.write_text(json_dumps(bootstrap_manifest) + "\n", encoding="utf-8")
        artifacts["evaluation_anchor_bootstrap_manifest"] = rel_ref(bootstrap_manifest_path, self.project_root)

        self._report_package(out_dir, artifacts)
        artifacts["report_package"] = rel_ref(out_dir / "REPORT_PACKAGE", self.project_root)
        return {"status": status, "run_id": run_id, "artifacts": artifacts}

    def _fold_audit(self, path: Path, a1: Any) -> dict[str, Any]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = list(rows[0]) if rows else []
        node_ids = np.array([int(row["node_id"]) for row in rows], dtype=np.int64)
        labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
        folds = np.array([int(row["fold"]) for row in rows], dtype=np.int64)
        a1_train = a1["train_idx"].astype(np.int64)
        a1_test = set(map(int, a1["test_idx"].tolist()))
        a1_labels = a1["labels"].astype(np.int64)
        errors: list[str] = []
        if len(rows) != TRAIN_N:
            errors.append("row_count_not_11001")
        if not {"node_id", "label", "fold"}.issubset(set(columns)):
            errors.append("missing_required_columns")
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
        expected_hash_match = sha256_file(path) == self.expected_fold_hash
        if not expected_hash_match:
            errors.append("fold_hash_mismatch")
        status = "verified_future_protocol" if not errors else "rejected"
        return {
            "audit_version": BOOTSTRAP_VERSION,
            "identity": FOLD_IDENTITY,
            "source_path": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "expected_hash": self.expected_fold_hash,
            "expected_hash_match": expected_hash_match,
            "status": status,
            "rows": len(rows),
            "columns": columns,
            "node_ids": node_ids,
            "labels": labels,
            "fold": folds,
            "train_idx_coverage": len(set(node_ids.tolist())),
            "contains_test_idx": bool(set(node_ids.tolist()) & a1_test),
            "label_consistency_with_a1_npz": "pass" if "labels_not_equal_a1_npz" not in errors else "fail",
            "fold_values": sorted(set(folds.tolist())),
            "class_distribution_by_fold": {str(f): np.bincount(labels[folds == f], minlength=CLASS_N).astype(int).tolist() for f in sorted(set(folds.tolist()))},
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

    def _v43_audit(self, path: Path, fold: dict[str, Any], a1: Any) -> dict[str, Any]:
        z = np.load(path, allow_pickle=False)
        keys = list(z.files)
        required = all(key in keys for key in ["base_proba", "proba", "train_idx", "labels", "bucket"])
        train_idx = z["train_idx"].astype(np.int64) if "train_idx" in keys else np.array([], dtype=np.int64)
        labels = z["labels"].astype(np.int64) if "labels" in keys else np.array([], dtype=np.int64)
        test_set = set(map(int, a1["test_idx"].tolist()))
        errors: list[str] = []
        if not required:
            errors.append("missing_required_keys")
        if train_idx.shape != (TRAIN_N,) or len(set(train_idx.tolist())) != TRAIN_N:
            errors.append("train_idx_not_complete_unique")
        if set(train_idx.tolist()) & test_set:
            errors.append("contains_test_idx")
        if not np.array_equal(train_idx, fold["node_ids"]):
            errors.append("train_idx_not_aligned_with_fold")
        if not np.array_equal(labels, fold["labels"]):
            errors.append("labels_not_aligned_with_fold")
        base_audit = _prob_audit(z["base_proba"]) if "base_proba" in keys else {"passed": False}
        proba_audit = _prob_audit(z["proba"]) if "proba" in keys else {"passed": False}
        if not base_audit.get("passed"):
            errors.append("base_proba_invalid")
        if not proba_audit.get("passed"):
            errors.append("correct_smooth_proba_invalid")
        return {
            "audit_version": BOOTSTRAP_VERSION,
            "component_identity": "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF",
            "path": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "keys": keys,
            "required_keys_present": required,
            "base_proba_audit": base_audit,
            "correct_smooth_proba_audit": proba_audit,
            "train_idx_aligned_with_fold": "train_idx_not_aligned_with_fold" not in errors,
            "labels_aligned_with_fold": "labels_not_aligned_with_fold" not in errors,
            "contains_test_idx": "contains_test_idx" in errors,
            "asset_declares_oof": "oof" in path.name.lower(),
            "correct_smooth_final_proba_forbidden_as_anchor": True,
            "historical_v43c_identity_equivalence": "unverified",
            "functional_oof_validity": "verified" if not errors else "rejected",
            "usable_as_evaluation_base": not errors,
            "errors": errors,
        }

    def _v46_audit(self, path: Path, v43_path: Path, fold: dict[str, Any], a1: Any) -> dict[str, Any]:
        z46 = np.load(path, allow_pickle=False)
        z43 = np.load(v43_path, allow_pickle=False)
        keys = list(z46.files)
        required = all(key in keys for key in ["proba", "expert_proba", "train_idx", "labels", "isolated_mask"])
        train_idx = z46["train_idx"].astype(np.int64) if "train_idx" in keys else np.array([], dtype=np.int64)
        labels = z46["labels"].astype(np.int64) if "labels" in keys else np.array([], dtype=np.int64)
        mask = z46["isolated_mask"].astype(bool) if "isolated_mask" in keys else np.zeros((0,), dtype=bool)
        test_set = set(map(int, a1["test_idx"].tolist()))
        errors: list[str] = []
        if not required:
            errors.append("missing_required_keys")
        if train_idx.shape != (TRAIN_N,) or len(set(train_idx.tolist())) != TRAIN_N:
            errors.append("train_idx_not_complete_unique")
        if set(train_idx.tolist()) & test_set:
            errors.append("contains_test_idx")
        if not np.array_equal(train_idx, fold["node_ids"]):
            errors.append("train_idx_not_aligned_with_fold")
        if not np.array_equal(labels, fold["labels"]):
            errors.append("labels_not_aligned_with_fold")
        proba_audit = _prob_audit(z46["proba"]) if "proba" in keys else {"passed": False}
        expert_audit = _prob_audit(z46["expert_proba"]) if "expert_proba" in keys else {"passed": False}
        if not proba_audit.get("passed"):
            errors.append("proba_invalid")
        if not expert_audit.get("passed"):
            errors.append("expert_proba_invalid")
        if mask.shape != (TRAIN_N,):
            errors.append("isolated_mask_shape_invalid")
        if int(mask.sum()) != 2213:
            errors.append("isolated_true_count_not_2213")
        train_idx_v43 = z43["train_idx"].astype(np.int64) if "train_idx" in z43.files else np.array([], dtype=np.int64)
        labels_v43 = z43["labels"].astype(np.int64) if "labels" in z43.files else np.array([], dtype=np.int64)
        aligned_with_v43 = bool(np.array_equal(train_idx, train_idx_v43) and np.array_equal(labels, labels_v43))
        if not aligned_with_v43:
            errors.append("not_aligned_with_v43")
        comp_non_iso = bool(mask.size and "base_proba" in z43.files and np.allclose(z46["proba"][~mask], z43["base_proba"][~mask], atol=1e-7))
        comp_iso = bool(mask.size and np.allclose(z46["proba"][mask], z46["expert_proba"][mask], atol=1e-7))
        if not comp_non_iso:
            errors.append("non_isolated_not_equal_v43_base")
        if not comp_iso:
            errors.append("isolated_not_equal_expert")
        return {
            "audit_version": BOOTSTRAP_VERSION,
            "component_identity": "V46A_ISOLATED_COMPOSED_OOF",
            "path": rel_ref(path, self.project_root),
            "sha256": sha256_file(path),
            "keys": keys,
            "required_keys_present": required,
            "proba_audit": proba_audit,
            "expert_proba_audit": expert_audit,
            "train_idx_aligned_with_fold": "train_idx_not_aligned_with_fold" not in errors,
            "labels_aligned_with_fold": "labels_not_aligned_with_fold" not in errors,
            "train_idx_labels_aligned_with_v43": aligned_with_v43,
            "contains_test_idx": "contains_test_idx" in errors,
            "isolated_true_count": int(mask.sum()) if mask.size else 0,
            "composition_rule": "proba = v43c_base_proba on non-isolated rows; proba = expert_proba on isolated_mask rows",
            "composition_rule_verified": comp_non_iso and comp_iso,
            "v46_proba_is_complete_composed_result": comp_non_iso and comp_iso,
            "asset_declares_oof": "oof" in path.name.lower(),
            "functional_oof_validity": "verified" if not errors else "rejected",
            "usable_as_evaluation_anchor_component": not errors,
            "errors": errors,
        }

    def _structure_axes(self, a1: Any, train_idx: np.ndarray) -> dict[str, np.ndarray]:
        n = int(a1["adj_shape"][0])
        adj = sp.csr_matrix((a1["adj_data"], a1["adj_indices"], a1["adj_indptr"]), shape=tuple(a1["adj_shape"]))
        either = ((adj + adj.T) > 0).astype(np.int8).tocsr()
        degree = np.asarray(either.sum(axis=1)).ravel().astype(np.int64)
        train_set = set(map(int, train_idx.tolist()))
        connectivity = []
        reachability = []
        degree_band = []
        for node in train_idx.astype(int).tolist():
            deg = int(degree[node])
            if deg == 0:
                connectivity.append("isolated")
                degree_band.append("isolated")
            else:
                connectivity.append("graph_visible")
                degree_band.append("degree_1" if deg == 1 else "degree_2_5" if deg <= 5 else "degree_6p")
            frontier = {node}
            seen = {node}
            hit = None
            for dist in range(1, 5):
                nxt: set[int] = set()
                for cur in frontier:
                    start, end = either.indptr[cur], either.indptr[cur + 1]
                    nxt.update(map(int, either.indices[start:end]))
                nxt.difference_update(seen)
                seen.update(nxt)
                if any(x in train_set and x != node for x in nxt):
                    hit = dist
                    break
                frontier = nxt
                if not frontier:
                    break
            if hit == 1:
                reachability.append("one_hop_available")
            elif hit == 2:
                reachability.append("exact2_only")
            elif hit in {3, 4}:
                reachability.append("exact3_4_only")
            else:
                reachability.append("no_visible_train_within_4_hops")
        return {
            "connectivity_visibility": np.asarray(connectivity),
            "train_label_reachability": np.asarray(reachability),
            "degree_band": np.asarray(degree_band),
            "isolated_mask_from_graph": np.asarray(connectivity) == "isolated",
        }

    def _materialize(self, paths: dict[str, Path], input_hashes: dict[str, str], fold: dict[str, Any], fold_artifacts: dict[str, Any], structure: dict[str, np.ndarray], v43: dict[str, Any], v46: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        missing = []
        if fold["status"] != "verified_future_protocol":
            missing.append("verified_canonical_evaluation_fold")
        if v43["functional_oof_validity"] != "verified":
            missing.append("verified_v43c_functional_oof")
        if v46["functional_oof_validity"] != "verified":
            missing.append("verified_v46a_functional_oof")
        if not v46.get("composition_rule_verified"):
            missing.append("verified_v46a_composition_rule")
        if missing:
            return self._failed_materialization(missing)

        z43 = np.load(paths["v43c_oof"], allow_pickle=False)
        z46 = np.load(paths["v46a_oof"], allow_pickle=False)
        route = "route_1_v46a_complete_composed_proba"
        proba = z46["proba"].astype(np.float32)
        train_idx = z46["train_idx"].astype(np.int64)
        labels = z46["labels"].astype(np.int64)
        fold_arr = fold["fold"].astype(np.int64)
        isolated_mask = z46["isolated_mask"].astype(bool)
        if not v46.get("v46_proba_is_complete_composed_result"):
            route = "route_2_recompose_v43_base_plus_v46_expert"
            proba = z43["base_proba"].astype(np.float32).copy()
            proba[isolated_mask] = z46["expert_proba"][isolated_mask]
        pred = proba.argmax(axis=1).astype(np.int64)
        component_source = np.where(isolated_mask, "v46a_isolated_expert_oof", "v43c_associated_base_oof")
        anchor_version = np.asarray([EVAL_ANCHOR_IDENTITY] * len(train_idx))
        oof_path = out_dir / f"{EVAL_ANCHOR_IDENTITY}_oof.npz"
        _save_npz_deterministic(
            oof_path,
            anchor_version=anchor_version,
            component_source=component_source,
            connectivity_visibility=structure["connectivity_visibility"],
            degree_band=structure["degree_band"],
            fold=fold_arr,
            isolated_mask=isolated_mask,
            labels=labels,
            pred=pred,
            proba=proba,
            train_idx=train_idx,
            train_label_reachability=structure["train_label_reachability"],
        )
        replay_path = out_dir / "A1_EVAL_ANCHOR_V1_replay_oof.npz"
        _save_npz_deterministic(
            replay_path,
            anchor_version=anchor_version,
            component_source=component_source,
            connectivity_visibility=structure["connectivity_visibility"],
            degree_band=structure["degree_band"],
            fold=fold_arr,
            isolated_mask=isolated_mask,
            labels=labels,
            pred=pred,
            proba=proba,
            train_idx=train_idx,
            train_label_reachability=structure["train_label_reachability"],
        )
        metrics = self._metrics(labels, pred, fold_arr, structure)
        content_hash = stable_hash({
            "train_idx_sha": stable_hash(train_idx.tolist()),
            "labels_sha": stable_hash(labels.tolist()),
            "pred_sha": stable_hash(pred.tolist()),
            "proba_sha": stable_hash(np.round(proba.astype(np.float64), 12).tolist()),
            "fold_sha": stable_hash(fold_arr.tolist()),
            "route": route,
        })
        integrity = {
            "audit_version": BOOTSTRAP_VERSION,
            "status": "pass",
            "route": route,
            "train_idx_complete_unique": len(set(train_idx.tolist())) == TRAIN_N,
            "fold_coverage": sorted(set(fold_arr.tolist())) == [0, 1, 2, 3, 4],
            "labels_aligned": bool(np.array_equal(labels, fold["labels"])),
            "test_idx_overlap": False,
            "nan_or_inf": not bool(np.isfinite(proba).all()),
            "probability_audit": _prob_audit(proba),
            "composition_range_verified": v46["composition_rule_verified"],
            "routing_uses_truth": False,
            "correct_smooth_proba_used_as_anchor": False,
            "test_prediction_used": False,
            "v53q1_test_patch_used": False,
            "replay_pred_equal": bool(np.array_equal(pred, np.load(replay_path, allow_pickle=False)["pred"])),
            "replay_proba_allclose": bool(np.allclose(proba, np.load(replay_path, allow_pickle=False)["proba"], atol=0.0)),
            "anchor_content_hash": content_hash,
            "anchor_file_sha256": sha256_file(oof_path),
            "replay_file_sha256": sha256_file(replay_path),
            "output_hash_stable": sha256_file(oof_path) == sha256_file(replay_path),
        }
        return {
            "status": "materialized_from_verified_components",
            "materialized": True,
            "route": route,
            "oof_npz": rel_ref(oof_path, self.project_root),
            "oof_sha256": sha256_file(oof_path),
            "replay_npz": rel_ref(replay_path, self.project_root),
            "anchor_content_hash": content_hash,
            "metrics": metrics,
            "metrics_summary": metrics["summary"],
            "integrity_audit": integrity,
            "stable_summary": {"route": route, "oof_sha256": sha256_file(oof_path), "content_hash": content_hash, "metrics": metrics["summary"]},
            "missing_evidence": [],
            "historical_v53q1_oof_status": "not_materialized",
            "deployment_equivalent": False,
            "historical_v53q1_oof_equivalent": False,
            "historical_v43c_identity_equivalence": "unverified",
            "functional_oof_validity": "verified",
            "component_hashes": {"v43c_oof": input_hashes["v43c_oof"], "v46a_oof": input_hashes["v46a_oof"]},
            "fold_hash": fold_artifacts.get("fold_hash", ""),
        }

    def _failed_materialization(self, missing: list[str]) -> dict[str, Any]:
        return {
            "status": "rebuild_required",
            "materialized": False,
            "route": "blocked",
            "missing_evidence": sorted(set(missing)),
            "integrity_audit": {"audit_version": BOOTSTRAP_VERSION, "status": "fail", "missing_evidence": sorted(set(missing))},
            "historical_v53q1_oof_status": "not_materialized",
            "deployment_equivalent": False,
            "historical_v53q1_oof_equivalent": False,
            "historical_v43c_identity_equivalence": "unverified",
            "functional_oof_validity": "unavailable",
        }

    def _metrics(self, labels: np.ndarray, pred: np.ndarray, fold: np.ndarray, structure: dict[str, np.ndarray]) -> dict[str, Any]:
        correct = pred == labels
        summary = {
            "overall_accuracy": float(correct.mean()),
            "macro_accuracy": _macro_accuracy(labels, pred),
            "graph_visible_accuracy": self._masked_acc(correct, structure["connectivity_visibility"] == "graph_visible"),
            "isolated_accuracy": self._masked_acc(correct, structure["connectivity_visibility"] == "isolated"),
        }
        fold_rows = []
        for f in sorted(set(fold.tolist())):
            m = fold == f
            fold_rows.append({
                "fold": int(f),
                "sample_count": int(m.sum()),
                "accuracy": self._masked_acc(correct, m),
                "macro_accuracy": _macro_accuracy(labels[m], pred[m]),
                "class_distribution": json.dumps(np.bincount(labels[m], minlength=CLASS_N).astype(int).tolist(), separators=(",", ":")),
                "prediction_distribution": json.dumps(np.bincount(pred[m], minlength=CLASS_N).astype(int).tolist(), separators=(",", ":")),
            })
        class_rows = []
        for cls in range(CLASS_N):
            m = labels == cls
            class_rows.append({"class_id": cls, "sample_count": int(m.sum()), "accuracy": self._masked_acc(correct, m), "predicted_count": int((pred == cls).sum())})
        bucket_rows = []
        for axis_name in ["connectivity_visibility", "train_label_reachability", "degree_band"]:
            axis = structure[axis_name]
            for value in sorted(set(axis.tolist())):
                m = axis == value
                bucket_rows.append({"axis": axis_name, "bucket": value, "sample_count": int(m.sum()), "accuracy": self._masked_acc(correct, m)})
        bucket_class_rows = []
        reach = structure["train_label_reachability"]
        for bucket in sorted(set(reach.tolist())):
            for cls in range(CLASS_N):
                m = (reach == bucket) & (labels == cls)
                bucket_class_rows.append({"bucket": bucket, "class_id": cls, "sample_count": int(m.sum()), "accuracy": self._masked_acc(correct, m)})
        degree_rows = [row for row in bucket_rows if row["axis"] == "degree_band"]
        reach_rows = [row for row in bucket_rows if row["axis"] == "train_label_reachability"]
        return {
            "summary": summary,
            "fold_metrics": fold_rows,
            "class_metrics": class_rows,
            "bucket_metrics": bucket_rows,
            "bucket_class_metrics": bucket_class_rows,
            "degree_band_metrics": degree_rows,
            "reachability_metrics": reach_rows,
        }

    def _masked_acc(self, correct: np.ndarray, mask: np.ndarray) -> float | None:
        return float(correct[mask].mean()) if bool(mask.any()) else None

    def _component_registry(self, input_hashes: dict[str, str], fold: dict[str, Any], v43: dict[str, Any], v46: dict[str, Any], materialization: dict[str, Any]) -> dict[str, Any]:
        return {
            "registry_version": BOOTSTRAP_VERSION,
            "evaluation_anchor_identity": EVAL_ANCHOR_IDENTITY,
            "online_deployment_anchor": ONLINE_ANCHOR_IDENTITY,
            "route": materialization.get("route", ""),
            "components": [
                {"role": "canonical_evaluation_fold", "identity": FOLD_IDENTITY, "status": fold["status"], "sha256": input_hashes["fold_candidate"], "historical_v53q1_fold_equivalence": "unverified", "future_evaluation_protocol": "verified"},
                {"role": "base_oof", "identity": "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF", "status": v43["functional_oof_validity"], "sha256": input_hashes["v43c_oof"], "historical_v43c_identity_equivalence": "unverified"},
                {"role": "isolated_composed_oof", "identity": "V46A_ISOLATED_COMPOSED_OOF", "status": v46["functional_oof_validity"], "sha256": input_hashes["v46a_oof"], "composition_rule_verified": v46["composition_rule_verified"]},
            ],
            "forbidden_components_not_used": ["correct_smooth_candidate_proba", "test_prediction", "v53q1_test_patch", "oracle_router"],
        }

    def _provenance_graph(self, input_hashes: dict[str, str], materialization: dict[str, Any]) -> dict[str, Any]:
        return {
            "graph_version": BOOTSTRAP_VERSION,
            "nodes": [
                {"id": FOLD_IDENTITY, "type": "fold", "sha256": input_hashes["fold_candidate"]},
                {"id": "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF", "type": "oof_component", "sha256": input_hashes["v43c_oof"]},
                {"id": "V46A_ISOLATED_COMPOSED_OOF", "type": "oof_component", "sha256": input_hashes["v46a_oof"]},
                {"id": EVAL_ANCHOR_IDENTITY, "type": "evaluation_anchor", "sha256": materialization.get("oof_sha256", "")},
                {"id": ONLINE_ANCHOR_IDENTITY, "type": "online_deployment_anchor", "deployment_equivalent": False},
            ],
            "edges": [
                {"source": FOLD_IDENTITY, "target": EVAL_ANCHOR_IDENTITY, "relation": "defines_offline_evaluation_protocol"},
                {"source": "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF", "target": EVAL_ANCHOR_IDENTITY, "relation": "base_for_non_isolated_rows"},
                {"source": "V46A_ISOLATED_COMPOSED_OOF", "target": EVAL_ANCHOR_IDENTITY, "relation": "complete_composed_oof_source"},
            ],
            "claims": {
                "historical_v53q1_oof_equivalent": False,
                "historical_v43c_identity_equivalence": "unverified",
                "functional_oof_validity": materialization.get("functional_oof_validity", "unavailable"),
            },
        }

    def _rebuild_spec(self, fold: dict[str, Any], v43: dict[str, Any], v46: dict[str, Any], materialization: dict[str, Any]) -> dict[str, Any]:
        return {
            "spec_version": BOOTSTRAP_VERSION,
            "status": "rebuild_required",
            "unique_blocker": materialization.get("missing_evidence", ["unknown"])[0],
            "fold_status": fold["status"],
            "v43_errors": v43.get("errors", []),
            "v46_errors": v46.get("errors", []),
            "training_allowed_in_this_route": False,
        }

    def _report(self, status: str, fold: dict[str, Any], v43: dict[str, Any], v46: dict[str, Any], materialization: dict[str, Any]) -> str:
        metrics = materialization.get("metrics_summary", {})
        return "\n".join([
            "# A1 Evaluation Anchor Direct Materialization Report",
            "",
            f"status: `{status}`",
            f"route: `{materialization.get('route', '')}`",
            f"fold_status: `{fold.get('status')}`",
            f"v43_functional_oof_validity: `{v43.get('functional_oof_validity')}`",
            f"v46_functional_oof_validity: `{v46.get('functional_oof_validity')}`",
            f"v46_composition_rule_verified: `{v46.get('composition_rule_verified')}`",
            f"overall_accuracy: `{metrics.get('overall_accuracy')}`",
            f"macro_accuracy: `{metrics.get('macro_accuracy')}`",
            f"anchor_hash: `{materialization.get('oof_sha256', '')}`",
            "",
            "Historical v53Q-1 OOF equivalence is false/not materialized.",
            "Historical v43C identity equivalence is unverified.",
            "No training, Test prediction, submission, LLM, or network call was executed.",
            "",
        ])

    def _report_package(self, out_dir: Path, artifacts: dict[str, str]) -> None:
        package = out_dir / "REPORT_PACKAGE"
        package.mkdir(parents=True, exist_ok=True)
        names = [
            "EVALUATION_ANCHOR_REPORT.md",
            f"{EVAL_ANCHOR_IDENTITY}_manifest.json",
            f"{FOLD_IDENTITY}_manifest.json",
            "fold_metrics.csv",
            "class_metrics.csv",
            "bucket_metrics.csv",
            "bucket_class_metrics.csv",
            "integrity_audit.json",
            "component_registry.json",
            "provenance_graph.json",
        ]
        for name in names:
            src = out_dir / name
            if src.exists():
                shutil.copyfile(src, package / name)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()
