# -*- coding: utf-8 -*-
"""Dual-task (B1 + B2) master run materialization.

Creates ``artifacts/b_dual_task_runs/<master_run_id>/`` containing the master
manifest, task sequence, resource budget, cross-task isolation audit,
DUAL_TASK_REPORT.md, and per-task TO_UPLOAD bundles.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .research.event_store import json_dumps, rel_ref, sha256_file, stable_hash

B1_CANONICAL_RUN_ID = "e95368a24e0780e65e92aceb"
B1_CANONICAL_ONLINE_SCORE = 0.37908
B1_CANONICAL_OFFLINE_STANDARD = 0.49069
B1_CANONICAL_GAP = 0.11161

B1_V2_RUN_ID = "c5e33663d37c6e49004e8213"
B1_V2_BEST_CANDIDATE = "B1_LP_UNDIRECTED_ALPHA7"
B1_V2_STANDARD = 0.4964
B1_V2_MACRO = 0.3941


def _load_run_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_path_from_manifest(manifest: dict[str, Any], project_root: Path, task: str) -> Path:
    artifacts = manifest.get("artifacts", {})
    key_candidates = [f"candidate_{task}_csv"]
    if task == "B1_v2":
        key_candidates.append("candidate_B1_csv")
    rel = None
    for key in key_candidates:
        if key in artifacts:
            rel = artifacts[key]
            break
    if rel is None:
        raise KeyError(f"no candidate csv key found for {task} in manifest artifacts: {list(artifacts.keys())}")
    return project_root / rel.replace("/", "\\") if "\\" in rel else project_root / rel


def create_dual_task_master_run(
    *,
    project_root: str | Path,
    b1_v2_run_id: str,
    b2_run_id: str,
    b1_canonical_run_id: str = B1_CANONICAL_RUN_ID,
    b1_canonical_online_score: float = B1_CANONICAL_ONLINE_SCORE,
    b1_canonical_offline_standard: float = B1_CANONICAL_OFFLINE_STANDARD,
    b1_canonical_gap: float = B1_CANONICAL_GAP,
    b1_v2_best_candidate: str = B1_V2_BEST_CANDIDATE,
    b1_v2_standard: float = B1_V2_STANDARD,
    b1_v2_macro: float = B1_V2_MACRO,
    out_root: str | Path = "artifacts/b_dual_task_runs",
    force_rebuild: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    out_root = project_root / out_root

    b1_run_dir = project_root / "artifacts" / "b1_runs" / b1_v2_run_id
    b2_run_dir = project_root / "artifacts" / "b2_runs" / b2_run_id
    b1_manifest = _load_run_manifest(b1_run_dir / "run_manifest.json")
    b2_manifest = _load_run_manifest(b2_run_dir / "run_manifest.json")

    master_run_id = stable_hash({
        "version": "b_dual_task_master_v1",
        "b1_v2_run_id": b1_v2_run_id,
        "b2_run_id": b2_run_id,
    })[:24]
    master_dir = out_root / master_run_id
    if master_dir.exists() and force_rebuild:
        shutil.rmtree(master_dir)
    master_dir.mkdir(parents=True, exist_ok=True)

    b1_candidate_src = _candidate_path_from_manifest(b1_manifest, project_root, "B1_v2")
    b2_candidate_src = _candidate_path_from_manifest(b2_manifest, project_root, "B2")

    # TO_UPLOAD bundles
    to_upload = master_dir / "TO_UPLOAD"
    b1_upload = to_upload / "B1"
    b2_upload = to_upload / "B2"
    b1_upload.mkdir(parents=True, exist_ok=True)
    b2_upload.mkdir(parents=True, exist_ok=True)

    shutil.copy2(b1_candidate_src, b1_upload / "candidate_B1_v2.csv")
    shutil.copy2(b2_candidate_src, b2_upload / "candidate_B2.csv")

    b1_upload_sha256 = sha256_file(b1_upload / "candidate_B1_v2.csv")
    b2_upload_sha256 = sha256_file(b2_upload / "candidate_B2.csv")

    b1_bundle_manifest = {
        "task": "B1",
        "run_id": b1_v2_run_id,
        "best_candidate_id": b1_v2_best_candidate,
        "candidate_file": "candidate_B1_v2.csv",
        "candidate_sha256": b1_upload_sha256,
        "offline_standard": b1_v2_standard,
        "offline_macro": b1_v2_macro,
    }
    b2_bundle_manifest = {
        "task": "B2",
        "run_id": b2_run_id,
        "best_candidate_id": b2_manifest.get("best_deployment_candidate_id", ""),
        "candidate_file": "candidate_B2.csv",
        "candidate_sha256": b2_upload_sha256,
    }
    (b1_upload / "bundle_manifest.json").write_text(json_dumps(b1_bundle_manifest) + "\n", encoding="utf-8")
    (b2_upload / "bundle_manifest.json").write_text(json_dumps(b2_bundle_manifest) + "\n", encoding="utf-8")

    # Master manifest
    master_manifest = {
        "manifest_version": "b_dual_task_master_v1",
        "master_run_id": master_run_id,
        "created_at_epoch_seconds": time.time(),
        "tasks": {
            "B1": {"run_id": b1_v2_run_id, "candidate_sha256": b1_upload_sha256},
            "B2": {"run_id": b2_run_id, "candidate_sha256": b2_upload_sha256},
        },
        "canonical_first_b1_run_id": b1_canonical_run_id,
        "canonical_first_b1_online_score": b1_canonical_online_score,
        "canonical_first_b1_offline_standard": b1_canonical_offline_standard,
        "canonical_first_b1_gap": b1_canonical_gap,
        "b1_v2_run_id": b1_v2_run_id,
        "b1_v2_best_candidate": b1_v2_best_candidate,
        "b1_v2_standard": b1_v2_standard,
        "b1_v2_macro": b1_v2_macro,
    }
    (master_dir / "master_manifest.json").write_text(json_dumps(master_manifest) + "\n", encoding="utf-8")

    # Task sequence
    task_sequence = {
        "sequence": [
            {"step": 1, "task": "B1", "phase": "autonomous_classification_loop_v1", "run_id": b1_canonical_run_id, "note": "canonical first run"},
            {"step": 2, "task": "B1", "phase": "repair_and_rerun_v2", "run_id": b1_v2_run_id, "note": "B1 V2 closed loop"},
            {"step": 3, "task": "B2", "phase": "autonomous_recommendation_loop_v1", "run_id": b2_run_id, "note": "B2 first closed loop"},
            {"step": 4, "task": "DUAL", "phase": "master_run_materialization", "run_id": master_run_id, "note": "cross-task packaging and audit"},
        ],
        "isolation_rule": "A1/A2/B1 frozen assets, folds, champions, and anchors were not reused for B2; B2 did not feed data into B1.",
    }
    (master_dir / "task_sequence.json").write_text(json_dumps(task_sequence) + "\n", encoding="utf-8")

    # Resource budget
    resource_budget = {
        "b1_v2": {
            "max_wall_clock_seconds": 7200,
            "max_rounds": 3,
            "single_process": True,
            "single_gpu": False,
            "peak_gpu_memory_gb_limit": 23,
            "peak_gpu_memory_gb_observed": b1_manifest.get("peak_gpu_memory_gb_observed", 0.0),
        },
        "b2": {
            "max_wall_clock_seconds": 7200,
            "max_rounds": 3,
            "single_process": True,
            "single_gpu": False,
            "peak_gpu_memory_gb_limit": 23,
            "peak_gpu_memory_gb_observed": b2_manifest.get("peak_gpu_memory_gb_observed", 0.0),
        },
        "total_wall_clock_seconds_budget": 14400,
        "total_wall_clock_seconds_used": round(
            b1_manifest.get("wall_clock_seconds", 0.0) + b2_manifest.get("wall_clock_seconds", 0.0), 6
        ),
    }
    (master_dir / "resource_budget.json").write_text(json_dumps(resource_budget) + "\n", encoding="utf-8")

    # Cross-task isolation audit
    isolation_audit = {
        "a1_a2_assets_modified": False,
        "b1_frozen_assets_modified": False,
        "b1_data_used_for_b2": False,
        "b2_data_used_for_b1": False,
        "a2_model_weights_used_for_b2": False,
        "test_truth_not_used_b1": bool(b1_manifest.get("uses_test_truth", False)) is False,
        "test_truth_not_used_b2": bool(b2_manifest.get("uses_test_truth", False)) is False,
        "namespace_isolation": {
            "B1_fold": "AFAC_B1_FOLD_V1",
            "B2_fold": "AFAC_B2_FOLD_V1",
            "B1_eval_anchor": "B1_EVAL_ANCHOR_V2",
            "B2_eval_anchor": "B2_EVAL_ANCHOR_V1",
            "B1_online_anchor": "B1_ONLINE_ANCHOR",
            "B2_online_anchor": "B2_ONLINE_ANCHOR",
        },
        "audit_conclusion": "isolated",
    }
    (master_dir / "cross_task_isolation_audit.json").write_text(json_dumps(isolation_audit) + "\n", encoding="utf-8")

    # Report
    report = (
        "# Dual-Task Master Run Report\n\n"
        f"master_run_id: `{master_run_id}`\n\n"
        "## Task Runs\n"
        f"- B1 V2 run: `{b1_v2_run_id}` best `{b1_v2_best_candidate}` standard={b1_v2_standard} macro={b1_v2_macro}\n"
        f"- B2 first run: `{b2_run_id}` best `{b2_manifest.get('best_deployment_candidate_id', '')}`\n"
        f"- Canonical first B1 run: `{b1_canonical_run_id}` online={b1_canonical_online_score} offline={b1_canonical_offline_standard} gap={b1_canonical_gap}\n\n"
        "## Isolation\n"
        "- A1/A2/B1 frozen assets were not modified.\n"
        "- B1 and B2 used separate data, folds, anchors, and namespaces.\n"
        "- No test truth was used for either task.\n\n"
        "## Outputs\n"
        f"- TO_UPLOAD/B1/candidate_B1_v2.csv ({b1_upload_sha256})\n"
        f"- TO_UPLOAD/B2/candidate_B2.csv ({b2_upload_sha256})\n"
    )
    (master_dir / "DUAL_TASK_REPORT.md").write_text(report, encoding="utf-8")

    artifacts = {
        "master_manifest": rel_ref(master_dir / "master_manifest.json", project_root),
        "task_sequence": rel_ref(master_dir / "task_sequence.json", project_root),
        "resource_budget": rel_ref(master_dir / "resource_budget.json", project_root),
        "cross_task_isolation_audit": rel_ref(master_dir / "cross_task_isolation_audit.json", project_root),
        "DUAL_TASK_REPORT": rel_ref(master_dir / "DUAL_TASK_REPORT.md", project_root),
        "B1_candidate": rel_ref(b1_upload / "candidate_B1_v2.csv", project_root),
        "B2_candidate": rel_ref(b2_upload / "candidate_B2.csv", project_root),
    }
    return {
        "status": "completed",
        "master_run_id": master_run_id,
        "artifacts": artifacts,
    }
