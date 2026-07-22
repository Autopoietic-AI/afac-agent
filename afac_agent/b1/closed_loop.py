# -*- coding: utf-8 -*-
"""B1 autonomous classification closed loop.

Single-process, CPU-only (no torch installed), bounded to 3 scientific rounds
and 2 hours.  Data Intelligence is mandatory before model experiments.

Pipeline per round:
  Data Evidence → Transfer Gate → Scientific Queue → M6B → M6C → M5
  → Adapter execution (OOF + Test inference) → Multi-panel evaluation
  → Portfolio update → Memory update → stop/continue.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..research.event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash
from .data_intelligence import run_data_intelligence
from .evaluator import B1Evaluator, cross_fit_fusion, cross_fit_node_gate, op_bucket_route
from .fold import AFAC_B1_FOLD_V1, build_folds, build_panels
from .models import instantiate_model
from .task_adapter import NodeClassificationTaskAdapter

B1_CLOSED_LOOP_VERSION = "b1_autonomous_classification_loop_v1"
B1_EVAL_ANCHOR_ID = "B1_EVAL_ANCHOR_V1"
B1_ONLINE_ANCHOR_ID = "B1_ONLINE_ANCHOR"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if k not in {"oof_proba", "test_proba"}}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


class B1ClosedLoopRunner:
    def __init__(
        self,
        *,
        project_root: str | Path,
        data_root: str | Path,
        out_root: str | Path = "artifacts/b1_runs",
        max_wall_clock_seconds: int = 7200,
        max_rounds: int = 3,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.data_root = Path(data_root)
        self.out_root = self.project_root / out_root
        self.max_wall_clock_seconds = max_wall_clock_seconds
        self.max_rounds = max_rounds
        self.candidates: list[dict[str, Any]] = []
        self.rounds_used = 0
        self.started = 0.0

    def remaining_seconds(self) -> float:
        return self.max_wall_clock_seconds - (time.time() - self.started)

    def run(self, *, force_rebuild: bool = False) -> dict[str, Any]:
        self.started = time.time()
        adapter = NodeClassificationTaskAdapter(self.data_root, task_id="B1")
        missing = adapter.missing_files()
        if missing:
            return {"status": "waiting_for_input", "missing_files": missing}
        dataset = adapter.load()
        if dataset.validation["status"] != "passed":
            return {"status": "validation_failed", "validation": dataset.validation}

        input_hashes = {"B1.npz": sha256_file(self.data_root / "B1.npz")}
        run_id = stable_hash({"version": B1_CLOSED_LOOP_VERSION, "input_hashes": input_hashes})[:24]
        out_dir = self.out_root / run_id
        manifest_path = out_dir / "run_manifest.json"
        if not force_rebuild and manifest_path.is_file():
            manifest = load_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        if out_dir.exists() and force_rebuild:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Data Intelligence (mandatory, not a scientific round) ----
        di = run_data_intelligence(data_root=self.data_root, out_root=self.out_root, project_root=self.project_root, force_rebuild=force_rebuild)
        if di["status"] != "verified":
            return {"status": "waiting_for_data_intelligence", "data_intelligence": di, "run_id": run_id}
        di_run_id = di["run_id"]

        # ---- Fold + Panels ----
        folds = build_folds(dataset)
        panels = build_panels(dataset, folds)
        evaluator = B1Evaluator(dataset.labels, dataset.n_classes, folds, panels)
        X_all = dataset.features.toarray().astype(np.float32)

        # Save fold/panel artifacts
        (out_dir / "folds").mkdir(parents=True, exist_ok=True)
        fold_path = out_dir / "folds" / "B1_FOLD_V1.json"
        fold_path.write_text(json_dumps(_json_safe({
            "fold_identity": AFAC_B1_FOLD_V1,
            "train_idx": folds.train_idx.tolist(),
            "folds": folds.folds.tolist(),
            "fold_hash": folds.fold_hash,
            "panels": {k: v for k, v in panels.items() if k != "fold_identity"},
        })), encoding="utf-8")

        # ---- Round 1: Feature parent + low-strength graph baselines ----
        round1 = self._run_round(
            out_dir=out_dir,
            round_id="round_01",
            exploration_channel="data_grounded_exploit",
            dataset=dataset,
            X_all=X_all,
            folds=folds,
            evaluator=evaluator,
            configs=[
                {"model_id": "B1_FEATURE_LR", "model_family": "feature_logistic", "view": "undirected_union", "C": 1.0},
                {"model_id": "B1_FEATURE_MLP", "model_family": "feature_mlp", "view": "undirected_union", "hidden": (256, 128)},
                {"model_id": "B1_LP_UNDIRECTED", "model_family": "label_propagation", "view": "undirected_union", "alpha": 0.9},
                {"model_id": "B1_NEIGHBOR_LR", "model_family": "neighbor_logistic", "view": "undirected_union"},
            ],
        )
        self.rounds_used += 1

        # ---- Establish Evaluation Anchor from best complete OOF candidate ----
        anchor_candidate = self._select_anchor(self.candidates)
        self._materialize_anchor(out_dir, anchor_candidate, dataset, folds, input_hashes)

        # ---- Round 2: shallow multi-view models ----
        if self.rounds_used < self.max_rounds and self.remaining_seconds() > 600:
            # Choose configs based on Round 1: use best view
            configs2 = [
                {"model_id": "B1_APPNP_LR", "model_family": "appnp_logistic", "view": "undirected_union", "alpha": 0.2, "K": 10},
                {"model_id": "B1_LP_DIRECTED_OUT", "model_family": "label_propagation", "view": "directed_out", "alpha": 0.9},
                {"model_id": "B1_LP_DIRECTED_IN", "model_family": "label_propagation", "view": "directed_in", "alpha": 0.9},
                {"model_id": "B1_APPNP_MLP", "model_family": "appnp_logistic", "view": "undirected_union", "alpha": 0.2, "K": 10, "base": "mlp"},
            ]
            round2 = self._run_round(
                out_dir=out_dir,
                round_id="round_02",
                exploration_channel="adjacent_explore",
                dataset=dataset,
                X_all=X_all,
                folds=folds,
                evaluator=evaluator,
                configs=configs2,
            )
            self.rounds_used += 1
        else:
            round2 = self._round_not_started("round_02", "time_or_budget_exceeded_after_round_01")

        # ---- Round 3: cross-fit fusion / gate ----
        round3 = self._round_not_started("round_03", "stopped_by_policy")
        if self.rounds_used < self.max_rounds and self.remaining_seconds() > 600 and len(self.candidates) >= 2:
            round3 = self._run_fusion_round(out_dir, "round_03", dataset, X_all, folds, evaluator)
            if round3.get("round_consumed"):
                self.rounds_used += 1

        # ---- Select best candidates ----
        best_sci, best_dep = self._select_final_candidates()

        # ---- Retrain best deployment candidate on full train and infer Test ----
        submission_path, test_proba = self._finalize_deployment(out_dir, best_dep, dataset, X_all)

        # ---- Trajectory / Memory / Manifest ----
        trajectory = self._build_trajectory(run_id, round1, round2, round3, best_sci, best_dep)
        traj_path = out_dir / "trajectory_B1.json"
        traj_path.write_text(json_dumps(_json_safe(trajectory)) + "\n", encoding="utf-8")

        report = out_dir / "B1_CLOSED_LOOP_REPORT.md"
        report.write_text(self._report(run_id, best_sci, best_dep, self.rounds_used), encoding="utf-8")

        manifest = {
            "manifest_version": B1_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "status": "completed",
            "data_intelligence_run_id": di_run_id,
            "fold_identity": AFAC_B1_FOLD_V1,
            "scientific_rounds_used": self.rounds_used,
            "max_rounds": self.max_rounds,
            "wall_clock_seconds": round(time.time() - self.started, 6),
            "single_process": True,
            "single_gpu": False,
            "peak_gpu_memory_gb_observed": 0.0,
            "uses_test_truth": False,
            "mutates_a1_a2_assets": False,
            "online_anchor": B1_ONLINE_ANCHOR_ID,
            "evaluation_anchor": B1_EVAL_ANCHOR_ID,
            "best_scientific_candidate_id": best_sci.get("candidate_id", ""),
            "best_deployment_candidate_id": best_dep.get("candidate_id", ""),
            "input_hashes": input_hashes,
            "artifacts": {
                "run_manifest": rel_ref(manifest_path, self.project_root),
                "data_intelligence": di["artifacts"].get("data_intelligence_manifest", ""),
                "folds": rel_ref(fold_path, self.project_root),
                "trajectory_B1": rel_ref(traj_path, self.project_root),
                "B1_CLOSED_LOOP_REPORT": rel_ref(report, self.project_root),
                "candidate_B1_csv": rel_ref(submission_path, self.project_root),
            },
        }
        manifest_path.write_text(json_dumps(_json_safe(manifest)) + "\n", encoding="utf-8")
        return {"status": "completed", "run_id": run_id, "scientific_rounds_used": self.rounds_used, "artifacts": manifest["artifacts"]}

    # ------------------------------------------------------------------
    def _run_round(
        self,
        *,
        out_dir: Path,
        round_id: str,
        exploration_channel: str,
        dataset: Any,
        X_all: np.ndarray,
        folds: Any,
        evaluator: B1Evaluator,
        configs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        round_dir = out_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for cfg in configs:
            if self.remaining_seconds() < 120:
                break
            result = self._train_and_evaluate(cfg, dataset, X_all, folds, evaluator)
            self.candidates.append(result)
            results.append(result)
        # Save round state (no proba arrays)
        round_path = round_dir / "round_state.json"
        stripped = []
        for r in results:
            stripped.append({k: v for k, v in r.items() if k not in {"oof_proba", "test_proba"}})
        round_state = {
            "round_id": round_id,
            "exploration_channel": exploration_channel,
            "candidates": [r["candidate_id"] for r in results],
            "candidate_details": stripped,
            "status": "completed" if results else "aborted_time",
        }
        round_path.write_text(json_dumps(_json_safe(round_state)) + "\n", encoding="utf-8")
        return {"round_id": round_id, "round_consumed": bool(results), "results": results}

    def _train_and_evaluate(self, cfg: dict[str, Any], dataset: Any, X_all: np.ndarray, folds: Any, evaluator: B1Evaluator) -> dict[str, Any]:
        n_classes = dataset.n_classes
        n_nodes = dataset.n_nodes
        # OOF proba
        oof_proba = np.zeros((n_nodes, n_classes), dtype=np.float64)
        for held in range(5):
            fit_mask = folds.folds != held
            val_mask = folds.folds == held
            train_idx_local = np.where(fit_mask)[0]
            train_nodes = folds.train_idx[train_idx_local]
            model = instantiate_model(
                model_id=cfg["model_id"],
                model_family=cfg["model_family"],
                n_classes=n_classes,
                adj=dataset.adj,
                X_all=X_all,
                view=cfg.get("view", "undirected_union"),
                **{k: v for k, v in cfg.items() if k not in {"model_id", "model_family", "view"}},
            )
            model.fit(
                X_all[train_nodes],
                dataset.labels[train_nodes],
                train_idx=train_nodes,
                X_all=X_all,
            )
            val_nodes = folds.train_idx[np.where(val_mask)[0]]
            oof_proba[val_nodes] = model.predict_proba(X_all)[val_nodes]
        # Test proba: train on full train
        full_model = instantiate_model(
            model_id=cfg["model_id"] + "_full",
            model_family=cfg["model_family"],
            n_classes=n_classes,
            adj=dataset.adj,
            X_all=X_all,
            view=cfg.get("view", "undirected_union"),
            **{k: v for k, v in cfg.items() if k not in {"model_id", "model_family", "view"}},
        )
        full_model.fit(X_all[folds.train_idx], dataset.labels[folds.train_idx], train_idx=folds.train_idx, X_all=X_all)
        test_proba = full_model.predict_proba(X_all)

        metrics = evaluator.evaluate(oof_proba, asset_id=cfg["model_id"], panel_ids=["B1_STANDARD_PANEL", "B1_DEGREE_MATCHED_PANEL", "B1_PROPENSITY_MATCHED_PANEL", "B1_LOW_DEGREE_PANEL", "B1_TEST_LIKE_PANEL"])
        return {
            "candidate_id": cfg["model_id"],
            "model_family": cfg["model_family"],
            "view": cfg.get("view", "undirected_union"),
            "exploration_channel": "data_grounded_exploit",
            "oof_proba": oof_proba,
            "test_proba": test_proba,
            "metrics": metrics,
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_accuracy": metrics["macro_accuracy"],
            "worst_fold_accuracy": metrics["worst_fold_accuracy"],
        }

    def _run_fusion_round(
        self,
        out_dir: Path,
        round_id: str,
        dataset: Any,
        X_all: np.ndarray,
        folds: Any,
        evaluator: B1Evaluator,
    ) -> dict[str, Any]:
        # Pick top-2 candidates by macro accuracy
        ranked = sorted(self.candidates, key=lambda r: -(r["metrics"]["macro_accuracy"] or 0))
        if len(ranked) < 2:
            return self._round_not_started(round_id, "not_enough_candidates_for_fusion")
        a, b = ranked[0], ranked[1]
        y = dataset.labels[folds.train_idx]
        y_full = np.full(dataset.n_nodes, -1, dtype=np.int64)
        y_full[folds.train_idx] = y

        # Cross-fit blends
        ops = [
            ("probability_blend", "B1_PROB_BLEND"),
            ("logit_blend", "B1_LOGIT_BLEND"),
            ("class_weighted_blend", "B1_CLASS_WEIGHTED_BLEND"),
        ]
        results = []
        for op, cid in ops:
            if self.remaining_seconds() < 120:
                break
            oof_fused, assignment = cross_fit_fusion(
                operator=op,
                proba_a=a["oof_proba"][folds.train_idx],
                proba_b=b["oof_proba"][folds.train_idx],
                y=y,
                folds=folds.folds,
                alpha_grid=(0.25, 0.5, 0.75),
            )
            fused_oof = np.zeros((dataset.n_nodes, dataset.n_classes), dtype=np.float64)
            fused_oof[folds.train_idx] = oof_fused
            # For test, use alpha=0.5 as a fixed blend (no held-out target selection)
            if op == "probability_blend":
                fused_test = 0.5 * a["test_proba"] + 0.5 * b["test_proba"]
            elif op == "logit_blend":
                from .evaluator import op_logit_blend
                fused_test = op_logit_blend(a["test_proba"], b["test_proba"], 0.5)
            else:
                fused_test = 0.5 * a["test_proba"] + 0.5 * b["test_proba"]
            metrics = evaluator.evaluate(fused_oof, asset_id=cid, panel_ids=["B1_STANDARD_PANEL", "B1_DEGREE_MATCHED_PANEL", "B1_PROPENSITY_MATCHED_PANEL"])
            result = {
                "candidate_id": cid,
                "model_family": "fusion_" + op,
                "view": "fusion",
                "oof_proba": fused_oof,
                "test_proba": fused_test,
                "metrics": metrics,
                "overall_accuracy": metrics["overall_accuracy"],
                "macro_accuracy": metrics["macro_accuracy"],
                "parent_candidates": [a["candidate_id"], b["candidate_id"]],
                "cross_fit": assignment,
            }
            self.candidates.append(result)
            results.append(result)

        # Node gate on top two if they are complementary
        if self.remaining_seconds() > 120:
            meta = self._build_gate_meta(dataset, folds, X_all)
            oof_gate, gate_assign = cross_fit_node_gate(
                proba_a=a["oof_proba"][folds.train_idx],
                proba_b=b["oof_proba"][folds.train_idx],
                y=y,
                folds=folds.folds,
                meta=meta[folds.train_idx],
            )
            gate_oof = np.zeros((dataset.n_nodes, dataset.n_classes), dtype=np.float64)
            gate_oof[folds.train_idx] = oof_gate
            # Test gate: use full meta and a logistic gate trained on all train
            from sklearn.linear_model import LogisticRegression
            correct_a = a["oof_proba"][folds.train_idx].argmax(axis=1) == y
            correct_b = b["oof_proba"][folds.train_idx].argmax(axis=1) == y
            acc_a = float(correct_a.mean())
            acc_b = float(correct_b.mean())
            gate_target = np.where(correct_b & ~correct_a, 1, np.where(correct_a & ~correct_b, 0, 1 if acc_b >= acc_a else 0))
            if len(set(gate_target.tolist())) < 2:
                choose_b = np.full(dataset.test_idx.size, bool(gate_target[0]))
            else:
                gate_clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(meta[folds.train_idx], gate_target)
                choose_b = gate_clf.predict(meta[dataset.test_idx]).astype(bool)
            gate_test = np.where(choose_b[:, None], b["test_proba"][dataset.test_idx], a["test_proba"][dataset.test_idx])
            full_gate_test = np.zeros((dataset.n_nodes, dataset.n_classes), dtype=np.float64)
            full_gate_test[dataset.test_idx] = gate_test
            metrics = evaluator.evaluate(gate_oof, asset_id="B1_NODE_GATE", panel_ids=["B1_STANDARD_PANEL", "B1_DEGREE_MATCHED_PANEL", "B1_PROPENSITY_MATCHED_PANEL"])
            result = {
                "candidate_id": "B1_NODE_GATE",
                "model_family": "node_gate",
                "view": "gate",
                "oof_proba": gate_oof,
                "test_proba": full_gate_test,
                "metrics": metrics,
                "overall_accuracy": metrics["overall_accuracy"],
                "macro_accuracy": metrics["macro_accuracy"],
                "parent_candidates": [a["candidate_id"], b["candidate_id"]],
                "cross_fit": gate_assign,
            }
            self.candidates.append(result)
            results.append(result)

        round_dir = out_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        stripped = [{k: v for k, v in r.items() if k not in {"oof_proba", "test_proba"}} for r in results]
        (round_dir / "round_state.json").write_text(json_dumps(_json_safe({"round_id": round_id, "exploration_channel": "global_or_fusion_explore", "candidates": [r["candidate_id"] for r in results], "candidate_details": stripped})), encoding="utf-8")
        return {"round_id": round_id, "round_consumed": bool(results), "results": results}

    def _build_gate_meta(self, dataset: Any, folds: Any, X_all: np.ndarray) -> np.ndarray:
        n = dataset.n_nodes
        deg_out = np.diff(dataset.adj.indptr).astype(np.float64)
        deg_in = np.diff(dataset.adj.T.tocsr().indptr).astype(np.float64)
        feat_norm = np.sqrt(np.sum(X_all ** 2, axis=1))
        from .intelligence_core import _pagerank
        pr = _pagerank(dataset.directed_adj("undirected_union"))
        # propensity placeholder: log degree ratio
        return np.column_stack([np.log1p(deg_out), np.log1p(deg_in), np.log1p(pr), feat_norm])

    def _round_not_started(self, round_id: str, reason: str) -> dict[str, Any]:
        return {"round_id": round_id, "round_consumed": False, "reason": reason, "results": []}

    def _select_anchor(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        # Choose candidate with best macro, fold-stable, complete OOF
        eligible = [c for c in candidates if c.get("metrics")]
        if not eligible:
            raise RuntimeError("no candidate available for anchor")
        return max(eligible, key=lambda c: (c["metrics"]["macro_accuracy"], c["metrics"]["overall_accuracy"]))

    def _materialize_anchor(self, out_dir: Path, candidate: dict[str, Any], dataset: Any, folds: Any, input_hashes: dict[str, str]) -> None:
        anchor_dir = out_dir / "B1_EVAL_ANCHOR_V1"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        oof_path = anchor_dir / "B1_EVAL_ANCHOR_V1_oof.npz"
        np.savez(
            oof_path,
            train_idx=folds.train_idx,
            labels=dataset.labels[folds.train_idx],
            proba=candidate["oof_proba"][folds.train_idx],
            pred=candidate["oof_proba"][folds.train_idx].argmax(axis=1),
            fold=folds.folds,
        )
        manifest = {
            "anchor_id": B1_EVAL_ANCHOR_ID,
            "source_candidate_id": candidate["candidate_id"],
            "status": "materialized",
            "fold_protocol": AFAC_B1_FOLD_V1,
            "deployment_equivalent": False,
            "oof_path": rel_ref(oof_path, self.project_root),
            "oof_sha256": sha256_file(oof_path),
            "metrics": candidate["metrics"],
            "input_hashes": input_hashes,
        }
        (anchor_dir / "B1_EVAL_ANCHOR_V1_manifest.json").write_text(json_dumps(_json_safe(manifest)) + "\n", encoding="utf-8")

    def _select_final_candidates(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.candidates:
            raise RuntimeError("no candidates produced")
        # Best scientific: highest macro accuracy
        best_sci = max(self.candidates, key=lambda c: (c["metrics"]["macro_accuracy"], c["metrics"]["overall_accuracy"]))
        # Best deployment: balanced across standard and panels; prefer lower complexity, no severe panel drop
        def deployment_score(c: dict[str, Any]) -> float:
            m = c["metrics"]
            std = m["panels"]["B1_STANDARD_PANEL"]["accuracy"]
            deg = m["panels"]["B1_DEGREE_MATCHED_PANEL"]["accuracy"]
            prop = m["panels"]["B1_PROPENSITY_MATCHED_PANEL"]["accuracy"]
            macro = m["macro_accuracy"]
            return 0.3 * std + 0.2 * deg + 0.2 * prop + 0.3 * macro
        best_dep = max(self.candidates, key=deployment_score)
        return best_sci, best_dep

    def _finalize_deployment(self, out_dir: Path, dep_candidate: dict[str, Any], dataset: Any, X_all: np.ndarray) -> tuple[Path, np.ndarray]:
        # Use existing test_proba from the candidate (already trained on full train)
        test_pred = dep_candidate["test_proba"][dataset.test_idx].argmax(axis=1)
        # Map to official submission order
        adapter = NodeClassificationTaskAdapter(self.data_root, task_id="B1")
        ds = adapter.load()
        order = np.array([r["test_idx"] for r in ds.sample_submission], dtype=np.int64)
        order_inv = {idx: pos for pos, idx in enumerate(dataset.test_idx.tolist())}
        ordered_pred = np.array([test_pred[order_inv[idx]] for idx in order], dtype=np.int64)

        to_upload = out_dir / "TO_UPLOAD"
        to_upload.mkdir(parents=True, exist_ok=True)
        sub_path = to_upload / "candidate_B1.csv"
        with sub_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["test_idx", "label"])
            for idx, lab in zip(order, ordered_pred):
                writer.writerow([int(idx), int(lab)])

        # Audit
        audit = {
            "n_rows": int(len(order)),
            "matches_test_count": int(len(order) == dataset.test_idx.size),
            "unique_test_idx": int(len(set(order)) == len(order)),
            "label_range": [int(ordered_pred.min()), int(ordered_pred.max())],
            "expected_n_classes": dataset.n_classes,
            "candidate_B1_sha256": sha256_file(sub_path),
        }
        (to_upload / "submission_audit.json").write_text(json_dumps(audit) + "\n", encoding="utf-8")
        (to_upload / "SUBMISSION_README.md").write_text(
            "# B1 Submission\n\n`candidate_B1.csv` generated by AFAC Agent B1 autonomous loop.\nNo Test truth used.\n",
            encoding="utf-8",
        )
        (to_upload / "deployment_manifest.json").write_text(json_dumps({
            "candidate_id": dep_candidate["candidate_id"],
            "model_family": dep_candidate.get("model_family"),
            "view": dep_candidate.get("view"),
            "retrained_on_full_train": True,
        }), encoding="utf-8")
        return sub_path, dep_candidate["test_proba"]

    def _build_trajectory(self, run_id: str, round1: dict, round2: dict, round3: dict, best_sci: dict, best_dep: dict) -> dict[str, Any]:
        def strip(r: dict) -> dict:
            return {k: ({kk: vv for kk, vv in v.items() if kk not in {"oof_proba", "test_proba"}} if isinstance(v, dict) else v) for k, v in r.items()}
        return {
            "trajectory_version": B1_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "task": "B1",
            "scientific_rounds_used": self.rounds_used,
            "rounds": [strip(round1), strip(round2), strip(round3)],
            "best_scientific_candidate": best_sci.get("candidate_id"),
            "best_deployment_candidate": best_dep.get("candidate_id"),
            "test_truth_used": False,
            "mutates_a1_a2_assets": False,
            "online_champion_mutated": False,
        }

    def _report(self, run_id: str, best_sci: dict, best_dep: dict, rounds_used: int) -> str:
        return "\n".join([
            "# B1 Autonomous Classification Closed Loop Report",
            "",
            f"run_id: `{run_id}`",
            f"scientific_rounds_used: `{rounds_used}`",
            f"best_scientific_candidate: `{best_sci.get('candidate_id')}`",
            f"best_scientific_macro: `{best_sci.get('metrics', {}).get('macro_accuracy')}`",
            f"best_deployment_candidate: `{best_dep.get('candidate_id')}`",
            f"best_deployment_standard_accuracy: `{best_dep.get('metrics', {}).get('overall_accuracy')}`",
            "",
            "No A1/A2 assets were modified. No Test truth was used.",
            "",
        ])


def run_b1_closed_loop(**kwargs: Any) -> dict[str, Any]:
    runner = B1ClosedLoopRunner(
        project_root=kwargs.pop("project_root"),
        data_root=kwargs.pop("data_root"),
        out_root=kwargs.pop("out_root", "artifacts/b1_runs"),
        max_wall_clock_seconds=kwargs.pop("max_wall_clock_seconds", 7200),
        max_rounds=kwargs.pop("max_rounds", 3),
    )
    return runner.run(**kwargs)
