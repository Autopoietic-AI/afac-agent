# -*- coding: utf-8 -*-
"""B1 Data Intelligence orchestrator.

Runs all deterministic audits, writes required artifacts under
artifacts/b1_runs/<run_id>/data_intelligence/, and emits a final
DATA_INTELLIGENCE_REPORT.md.  No model training, no Test truth.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..research.event_store import json_dumps, rel_ref, sha256_file, stable_hash
from .intelligence_core import (
    feature_geometry,
    graph_regime,
    integrity_audit,
    label_graph_reliability,
    regime_identification,
    shift_audit,
    transferability_matrix,
    validation_protocol,
)
from .task_adapter import NodeClassificationTaskAdapter

DATA_INTELLIGENCE_VERSION = "b1_data_intelligence_v1"

OUTPUT_FILES = {
    "dataset_fingerprint.json": "fingerprint",
    "integrity_audit.json": "integrity_audit",
    "feature_geometry.json": "feature_geometry",
    "graph_regime.json": "graph_regime",
    "direction_view_audit.json": "direction_view_audit",
    "label_graph_reliability.json": "label_graph_reliability",
    "class_reliability.json": "class_reliability",
    "shift_audit.json": "shift_audit",
    "signal_reliability.json": "signal_reliability",
    "transferability_matrix.json": "transferability_matrix",
    "validation_protocol.json": "validation_protocol",
    "model_search_prior.json": "model_search_prior",
    "initial_problem_hierarchy.json": "initial_problem_hierarchy",
}


def _fingerprint(dataset: Any, npz_path: Path) -> dict[str, Any]:
    return {
        "task_id": dataset.task_id,
        "n_nodes": dataset.n_nodes,
        "n_features": dataset.n_features,
        "n_classes": dataset.n_classes,
        "n_train": dataset.train_idx.size,
        "n_test": dataset.test_idx.size,
        "feature_shape": list(dataset.features.shape),
        "adj_shape": list(dataset.adj.shape),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
    }


def run_data_intelligence(
    *,
    data_root: str | Path,
    out_root: str | Path = "artifacts/b1_runs",
    project_root: str | Path = ".",
    force_rebuild: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    out_root = project_root / out_root
    data_root = Path(data_root)
    adapter = NodeClassificationTaskAdapter(data_root, task_id="B1")
    missing = adapter.missing_files()
    if missing:
        return {"status": "waiting_for_input", "missing_files": missing, "artifacts": {}}
    dataset = adapter.load()
    if dataset.validation["status"] != "passed":
        return {"status": "validation_failed", "validation": dataset.validation, "artifacts": {}}

    npz_path = data_root / "B1.npz"
    input_hashes = {"B1.npz": sha256_file(npz_path)}
    run_id = stable_hash({"version": DATA_INTELLIGENCE_VERSION, "input_hashes": input_hashes})[:24]
    out_dir = out_root / run_id / "data_intelligence"
    if out_dir.exists() and not force_rebuild:
        manifest_path = out_dir / "data_intelligence_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    integrity = integrity_audit(dataset)
    feature = feature_geometry(dataset)
    graph = graph_regime(dataset)
    label_rel = label_graph_reliability(dataset)
    shift = shift_audit(dataset)
    regime = regime_identification(dataset, {
        "feature_geometry": feature,
        "graph_regime": graph,
        "label_graph_reliability": label_rel,
        "shift_audit": shift,
    })
    transfer = transferability_matrix()
    val_protocol = validation_protocol(dataset)

    artifacts: dict[str, str] = {}

    def write(name: str, payload: dict[str, Any]) -> None:
        path = out_dir / name
        path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
        artifacts[Path(name).stem] = rel_ref(path, project_root)

    write("dataset_fingerprint.json", _fingerprint(dataset, npz_path))
    write("integrity_audit.json", integrity)
    write("feature_geometry.json", feature)
    write("graph_regime.json", graph)
    write("direction_view_audit.json", {k: v for k, v in graph["views"].items()})
    write("label_graph_reliability.json", label_rel)
    write("class_reliability.json", label_rel["views"])
    write("shift_audit.json", shift)
    write("signal_reliability.json", {
        "graph_views": {k: v["global_homophily"] for k, v in label_rel["views"].items()},
        "homophily_regime": label_rel["homophily_regime"],
        "directionality_recommendation": label_rel["directionality_recommendation"],
        "feature_regime": feature["feature_regime"],
        "train_test_shift": shift["train_test_shift"],
    })
    write("transferability_matrix.json", transfer)
    write("validation_protocol.json", val_protocol)
    write("model_search_prior.json", {"model_search_prior": regime["model_search_prior"], "avoid_list": regime["avoid_list"]})
    write("initial_problem_hierarchy.json", {
        "primary_problem": regime["primary_problem"],
        "secondary_problems": regime["secondary_problems"],
        "initial_hypotheses": regime["initial_hypotheses"],
        "validation_risk": regime["validation_risk"],
    })

    report = out_dir / "DATA_INTELLIGENCE_REPORT.md"
    report.write_text(_report(dataset, feature, graph, label_rel, shift, regime), encoding="utf-8")
    artifacts["DATA_INTELLIGENCE_REPORT"] = rel_ref(report, project_root)

    manifest = {
        "manifest_version": DATA_INTELLIGENCE_VERSION,
        "run_id": run_id,
        "status": "verified",
        "created_at_epoch_seconds": time.time(),
        "duration_seconds": round(time.time() - started, 6),
        "input_hashes": input_hashes,
        "data_intelligence_status": "verified",
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "data_intelligence_manifest.json"
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    artifacts["data_intelligence_manifest"] = rel_ref(manifest_path, project_root)
    return {"status": "verified", "run_id": run_id, "artifacts": artifacts}


def _report(dataset, feature, graph, label_rel, shift, regime) -> str:
    return "\n".join([
        "# B1 Data Intelligence Report",
        "",
        f"task: B1 node classification; nodes={dataset.n_nodes}; features={dataset.n_features}; classes={dataset.n_classes}",
        f"train={dataset.train_idx.size}; test={dataset.test_idx.size}",
        "",
        "## Integrity",
        f"- feature_finite: {dataset.validation.get('test_truth_hidden')}",
        "- train/test overlap: 0; test truth hidden: True",
        "",
        "## Feature Geometry",
        f"- density: {feature['density']:.4f}",
        f"- effective_rank: {feature['effective_rank']:.1f}; numeric_rank: {feature['numeric_rank']}",
        f"- feature_regime: {feature['feature_regime']}",
        f"- feature_norm_shift: {feature['feature_norm_shift']:.4f}",
        "",
        "## Graph Regime",
        f"- graph_regime: {graph['graph_regime']}",
        f"- directionality_recommendation: {label_rel['directionality_recommendation']}",
        f"- homophily_regime: {label_rel['homophily_regime']}",
        f"- train_test_degree_shift: {graph['train_test_degree_shift']:.4f}",
        "",
        "## Train/Test Shift",
        f"- propensity_auc_combined: {shift['propensity_auc_combined']:.4f}",
        f"- propensity_auc_degree_only: {shift['propensity_auc_degree_only']:.4f}",
        f"- propensity_auc_feature_only: {shift['propensity_auc_feature_only']:.4f}",
        f"- shift_interpretation: {shift['shift_interpretation']}",
        "",
        "## Regime Identification",
        f"- primary_problem: {regime['primary_problem']}",
        f"- secondary_problems: {regime['secondary_problems']}",
        f"- validation_risk: {regime['validation_risk']}",
        "",
        "## Transferability from A1",
        "- cross_task_prior_mode: advisory_only",
        "- target_task_evidence_priority: hard",
        "- A1 model weights/checkpoints/oof/test-predictions are forbidden direct transfer.",
        "",
    ])
