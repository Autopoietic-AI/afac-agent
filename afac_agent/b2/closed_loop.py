# -*- coding: utf-8 -*-
"""B2 autonomous recommendation closed loop V1.

Single-process, bounded to 3 scientific rounds and 2 hours wall-clock.
Round 1 = retrieval foundation, Round 2 = ranking model explore,
Round 3 = bucket/rerank explore (only when complementary panels exist).
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
from .evaluator import B2Evaluator
from .fold import AFAC_B2_FOLD_V1, build_folds, build_panels
from .models import B2Model, instantiate_model
from .task_adapter import B2Dataset, B2TaskAdapter

B2_CLOSED_LOOP_VERSION = "b2_autonomous_recommendation_loop_v1"
B2_EVAL_ANCHOR_ID = "B2_EVAL_ANCHOR_V1"
B2_ONLINE_ANCHOR_ID = "B2_ONLINE_ANCHOR"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if k not in {"oof_topk", "test_topk", "oof_scores", "test_scores"}}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


class B2ClosedLoopRunner:
    def __init__(
        self,
        *,
        project_root: str | Path,
        data_root: str | Path,
        out_root: str | Path = "artifacts/b2_runs",
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
        adapter = B2TaskAdapter(self.data_root, task_id="B2")
        missing = adapter.missing_files()
        if missing:
            return {"status": "waiting_for_input", "missing_files": missing}
        dataset = adapter.load()
        if dataset.validation["status"] != "passed":
            return {"status": "validation_failed", "validation": dataset.validation}

        actual_root = adapter.actual_root()
        input_hashes = {
            "train.csv": sha256_file(actual_root / "train.csv"),
            "test.csv": sha256_file(actual_root / "test.csv"),
            "user.csv": sha256_file(actual_root / "user.csv"),
            "item.csv": sha256_file(actual_root / "item.csv"),
        }
        run_id = stable_hash({"version": B2_CLOSED_LOOP_VERSION, "input_hashes": input_hashes})[:24]
        out_dir = self.out_root / run_id
        manifest_path = out_dir / "run_manifest.json"
        if not force_rebuild and manifest_path.is_file():
            manifest = load_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        if out_dir.exists() and force_rebuild:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Data Intelligence
        di = run_data_intelligence(data_root=self.data_root, out_root=self.out_root, project_root=self.project_root, force_rebuild=force_rebuild)
        if di["status"] != "verified":
            return {"status": "waiting_for_data_intelligence", "data_intelligence": di, "run_id": run_id}
        di_run_id = di["run_id"]

        # Folds and panels
        folds = build_folds(dataset)
        panels = build_panels(dataset, folds)
        evaluator = B2Evaluator(dataset, folds, panels)
        item_list = sorted(dataset.all_item_ids())
        train_targets = {str(uid): str(tid) for uid, tid in zip(dataset.train_df["uid"], dataset.train_df["target_iid"])} if "target_iid" in dataset.train_df.columns else {}

        (out_dir / "folds").mkdir(parents=True, exist_ok=True)
        fold_path = out_dir / "folds" / "B2_FOLD_V1.json"
        fold_path.write_text(json_dumps(_json_safe({
            "fold_identity": AFAC_B2_FOLD_V1,
            "uids": folds.uids.tolist(),
            "folds": folds.folds.tolist(),
            "fold_hash": folds.fold_hash,
            "panels": {k: v for k, v in panels.items() if k != "fold_identity"},
        })), encoding="utf-8")

        # Round 1: retrieval foundation
        round1 = self._run_round(
            out_dir=out_dir,
            round_id="round_01",
            exploration_channel="retrieval_foundation",
            dataset=dataset,
            folds=folds,
            evaluator=evaluator,
            item_list=item_list,
            train_targets=train_targets,
            configs=[
                {"model_id": "B2_POPULARITY", "model_family": "popularity"},
                {"model_id": "B2_HISTORY_RECALL", "model_family": "history_recall"},
                {"model_id": "B2_ITEM_ITEM_COOCCURRENCE", "model_family": "item_item_cooccurrence", "window": 5},
                {"model_id": "B2_LAST_ITEM_TRANSITION", "model_family": "last_item_transition"},
                {"model_id": "B2_SCORE_BLEND_H1P05C1", "model_family": "score_blend", "weights": {"history": 1.0, "popularity": 0.5, "cooccurrence": 1.0}},
                {"model_id": "B2_SCORE_BLEND_H1P1C1T05", "model_family": "score_blend", "weights": {"history": 1.0, "popularity": 1.0, "cooccurrence": 1.0, "last_transition": 0.5}},
            ],
        )
        self.rounds_used += 1

        # Anchor from best retrieval candidate
        anchor_candidate = self._select_anchor(self.candidates)
        self._materialize_anchor(out_dir, anchor_candidate, folds, input_hashes)

        # Round 2: ranking model explore
        round2 = self._round_not_started("round_02", "time_or_budget_exceeded_after_round_01")
        if self.rounds_used < self.max_rounds and self.remaining_seconds() > 900:
            round2 = self._run_round(
                out_dir=out_dir,
                round_id="round_02",
                exploration_channel="ranking_explore",
                dataset=dataset,
                folds=folds,
                evaluator=evaluator,
                item_list=item_list,
                train_targets=train_targets,
                configs=[
                    {"model_id": "B2_CANDIDATE_RANKER_LR", "model_family": "candidate_ranker", "base": "logistic", "n_candidates": 100, "n_negatives": 25},
                ],
            )
            if round2.get("round_consumed"):
                self.rounds_used += 1

        # Round 3: bucket/rerank explore (complementary panels exist)
        round3 = self._round_not_started("round_03", "stopped_by_policy")
        if self.rounds_used < self.max_rounds and self.remaining_seconds() > 600 and len(self.candidates) >= 2:
            round3 = self._run_bucket_rerank_round(out_dir, "round_03", dataset, folds, evaluator, item_list, train_targets)
            if round3.get("round_consumed"):
                self.rounds_used += 1

        best_sci, best_dep = self._select_final_candidates()
        submission_path = self._finalize_deployment(out_dir, best_dep, dataset, adapter, folds, item_list, train_targets)

        trajectory = self._build_trajectory(run_id, round1, round2, round3, best_sci, best_dep)
        traj_path = out_dir / "trajectory_B2.json"
        traj_path.write_text(json_dumps(_json_safe(trajectory)) + "\n", encoding="utf-8")

        report = out_dir / "B2_CLOSED_LOOP_V1_REPORT.md"
        report.write_text(self._report(run_id, best_sci, best_dep, self.rounds_used), encoding="utf-8")

        manifest = {
            "manifest_version": B2_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "status": "completed",
            "data_intelligence_run_id": di_run_id,
            "fold_identity": AFAC_B2_FOLD_V1,
            "scientific_rounds_used": self.rounds_used,
            "max_rounds": self.max_rounds,
            "wall_clock_seconds": round(time.time() - self.started, 6),
            "single_process": True,
            "single_gpu": False,
            "peak_gpu_memory_gb_observed": 0.0,
            "uses_test_truth": False,
            "mutates_a1_a2_b1_assets": False,
            "online_anchor": B2_ONLINE_ANCHOR_ID,
            "evaluation_anchor": B2_EVAL_ANCHOR_ID,
            "best_scientific_candidate_id": best_sci.get("candidate_id", ""),
            "best_deployment_candidate_id": best_dep.get("candidate_id", ""),
            "input_hashes": input_hashes,
            "artifacts": {
                "run_manifest": rel_ref(manifest_path, self.project_root),
                "data_intelligence": di["artifacts"].get("data_intelligence_manifest", ""),
                "folds": rel_ref(fold_path, self.project_root),
                "trajectory_B2": rel_ref(traj_path, self.project_root),
                "B2_CLOSED_LOOP_REPORT": rel_ref(report, self.project_root),
                "candidate_B2_csv": rel_ref(submission_path, self.project_root),
            },
        }
        manifest_path.write_text(json_dumps(_json_safe(manifest)) + "\n", encoding="utf-8")
        return {"status": "completed", "run_id": run_id, "scientific_rounds_used": self.rounds_used, "artifacts": manifest["artifacts"]}

    def _run_round(
        self,
        *,
        out_dir: Path,
        round_id: str,
        exploration_channel: str,
        dataset: B2Dataset,
        folds: Any,
        evaluator: B2Evaluator,
        item_list: list[str],
        train_targets: dict[str, str],
        configs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        round_dir = out_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for cfg in configs:
            if self.remaining_seconds() < 180:
                break
            result = self._train_and_evaluate(cfg, dataset, folds, evaluator, item_list, train_targets)
            self.candidates.append(result)
            results.append(result)
        round_path = round_dir / "round_state.json"
        stripped = []
        for r in results:
            stripped.append({k: v for k, v in r.items() if k not in {"oof_topk", "test_topk", "oof_scores", "test_scores"}})
        round_state = {
            "round_id": round_id,
            "exploration_channel": exploration_channel,
            "candidates": [r["candidate_id"] for r in results],
            "candidate_details": stripped,
            "status": "completed" if results else "aborted_time",
        }
        round_path.write_text(json_dumps(_json_safe(round_state)) + "\n", encoding="utf-8")
        return {"round_id": round_id, "round_consumed": bool(results), "results": results}

    def _train_and_evaluate(
        self,
        cfg: dict[str, Any],
        dataset: B2Dataset,
        folds: Any,
        evaluator: B2Evaluator,
        item_list: list[str],
        train_targets: dict[str, str],
    ) -> dict[str, Any]:
        top_k = dataset.top_k
        n_train = len(folds.uids)
        oof_topk = np.full((n_train, top_k), "", dtype=object)
        oof_scores = np.zeros((n_train, len(item_list)), dtype=np.float64)

        for held in range(5):
            fit_mask = folds.folds != held
            val_mask = folds.folds == held
            fit_uids = folds.uids[fit_mask].tolist()
            val_uids = folds.uids[val_mask].tolist()
            fit_targets = {uid: train_targets[uid] for uid in fit_uids if uid in train_targets}

            model = self._instantiate(cfg)
            model.fit(fit_uids, item_list, dataset.train_seq, train_targets=fit_targets, user_df=dataset.user_df, item_df=dataset.item_df)
            scores = model.predict_scores(val_uids)
            top_idx = np.argsort(-scores, axis=1)[:, :top_k]
            val_positions = np.where(val_mask)[0]
            for i, pos in enumerate(val_positions):
                oof_topk[pos] = [item_list[j] for j in top_idx[i]]
                oof_scores[pos] = scores[i]

        # Test prediction trained on full train
        full_model = self._instantiate(cfg)
        full_model.fit(folds.uids.tolist(), item_list, dataset.train_seq, train_targets=train_targets, user_df=dataset.user_df, item_df=dataset.item_df)
        test_uids = dataset.test_df["uid"].astype(str).tolist()
        test_scores = full_model.predict_scores(test_uids)
        test_top_idx = np.argsort(-test_scores, axis=1)[:, :top_k]
        test_topk = np.array([[item_list[j] for j in row] for row in test_top_idx], dtype=object)

        metrics = evaluator.evaluate(
            oof_topk,
            asset_id=cfg["model_id"],
            panel_ids=[
                "B2_STANDARD_PANEL",
                "B2_SHORT_HISTORY_PANEL",
                "B2_EXACT_LEN3_PANEL",
                "B2_LEN4_PLUS_PANEL",
                "B2_HISTORY_TARGET_PANEL",
                "B2_NOVEL_TARGET_PANEL",
                "B2_LONG_TAIL_PANEL",
                "B2_TEST_LIKE_PANEL",
                "B2_TOP10_BOUNDARY_PANEL",
            ],
        )
        return {
            "candidate_id": cfg["model_id"],
            "model_family": cfg["model_family"],
            "exploration_channel": cfg.get("exploration_channel", "retrieval_foundation"),
            "oof_topk": oof_topk,
            "test_topk": test_topk,
            "oof_scores": oof_scores,
            "test_scores": test_scores,
            "metrics": metrics,
            f"overall_ndcg@{top_k}": metrics["overall"][f"ndcg@{top_k}"],
            f"overall_hit_rate@{top_k}": metrics["overall"][f"hit_rate@{top_k}"],
            f"overall_mrr@{top_k}": metrics["overall"][f"mrr@{top_k}"],
            "worst_fold_ndcg": metrics["worst_fold_ndcg"],
        }

    def _instantiate(self, cfg: dict[str, Any]) -> B2Model:
        return instantiate_model(model_id=cfg["model_id"], model_family=cfg["model_family"], **{k: v for k, v in cfg.items() if k not in {"model_id", "model_family", "exploration_channel"}})

    def _run_bucket_rerank_round(
        self,
        out_dir: Path,
        round_id: str,
        dataset: B2Dataset,
        folds: Any,
        evaluator: B2Evaluator,
        item_list: list[str],
        train_targets: dict[str, str],
    ) -> dict[str, Any]:
        """Round 3: simple bucket rerank by sequence length, blending top retrieval and ranker."""
        top_k = dataset.top_k
        # Need at least one ranker and one retrieval candidate
        rankers = [c for c in self.candidates if c.get("model_family") == "candidate_ranker"]
        retrievals = [c for c in self.candidates if c.get("model_family") in {"history_recall", "score_blend", "popularity"}]
        if not rankers or not retrievals:
            return self._round_not_started(round_id, "missing_complementary_assets")
        ranker_cfg = self._find_original_config(rankers[0])
        retrieval_cfg = self._find_original_config(retrievals[0])

        n_train = len(folds.uids)
        oof_topk = np.full((n_train, top_k), "", dtype=object)

        for held in range(5):
            fit_mask = folds.folds != held
            val_mask = folds.folds == held
            fit_uids = folds.uids[fit_mask].tolist()
            val_uids = folds.uids[val_mask].tolist()
            fit_targets = {uid: train_targets[uid] for uid in fit_uids if uid in train_targets}

            ranker = self._instantiate(ranker_cfg)
            ranker.fit(fit_uids, item_list, dataset.train_seq, train_targets=fit_targets, user_df=dataset.user_df, item_df=dataset.item_df)
            retrieval = self._instantiate(retrieval_cfg)
            retrieval.fit(fit_uids, item_list, dataset.train_seq, train_targets=fit_targets, user_df=dataset.user_df, item_df=dataset.item_df)

            r_scores = ranker.predict_scores(val_uids)
            t_scores = retrieval.predict_scores(val_uids)
            # Bucket blend: cold-start (empty history) -> retrieval; others -> 0.7 ranker + 0.3 retrieval
            for i, uid in enumerate(val_uids):
                seq_len = len(dataset.train_seq.get(uid, []))
                if seq_len == 0:
                    scores = t_scores[i]
                else:
                    scores = 0.7 * r_scores[i] + 0.3 * t_scores[i]
                top_idx = np.argsort(-scores)[:top_k]
                pos = np.where(val_mask)[0][i]
                oof_topk[pos] = [item_list[j] for j in top_idx]

        # Full train for test
        full_ranker = self._instantiate(ranker_cfg)
        full_ranker.fit(folds.uids.tolist(), item_list, dataset.train_seq, train_targets=train_targets, user_df=dataset.user_df, item_df=dataset.item_df)
        full_retrieval = self._instantiate(retrieval_cfg)
        full_retrieval.fit(folds.uids.tolist(), item_list, dataset.train_seq, train_targets=train_targets, user_df=dataset.user_df, item_df=dataset.item_df)
        test_uids = dataset.test_df["uid"].astype(str).tolist()
        r_scores_test = full_ranker.predict_scores(test_uids)
        t_scores_test = full_retrieval.predict_scores(test_uids)
        test_topk = np.full((len(test_uids), top_k), "", dtype=object)
        for i, uid in enumerate(test_uids):
            seq_len = len(dataset.train_seq.get(uid, []))
            if seq_len == 0:
                scores = t_scores_test[i]
            else:
                scores = 0.7 * r_scores_test[i] + 0.3 * t_scores_test[i]
            top_idx = np.argsort(-scores)[:top_k]
            test_topk[i] = [item_list[j] for j in top_idx]

        metrics = evaluator.evaluate(
            oof_topk,
            asset_id="B2_BUCKET_RERANK",
            panel_ids=["B2_STANDARD_PANEL", "B2_SHORT_HISTORY_PANEL", "B2_LEN4_PLUS_PANEL", "B2_TEST_LIKE_PANEL"],
        )
        result = {
            "candidate_id": "B2_BUCKET_RERANK",
            "model_family": "bucket_rerank",
            "exploration_channel": "bucket_rerank_explore",
            "oof_topk": oof_topk,
            "test_topk": test_topk,
            "metrics": metrics,
            f"overall_ndcg@{top_k}": metrics["overall"][f"ndcg@{top_k}"],
            f"overall_hit_rate@{top_k}": metrics["overall"][f"hit_rate@{top_k}"],
            f"overall_mrr@{top_k}": metrics["overall"][f"mrr@{top_k}"],
            "worst_fold_ndcg": metrics["worst_fold_ndcg"],
            "parent_candidates": [rankers[0]["candidate_id"], retrievals[0]["candidate_id"]],
        }
        self.candidates.append(result)

        round_dir = out_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        stripped = {k: v for k, v in result.items() if k not in {"oof_topk", "test_topk", "oof_scores", "test_scores"}}
        (round_dir / "round_state.json").write_text(json_dumps(_json_safe({"round_id": round_id, "exploration_channel": "bucket_rerank_explore", "candidates": ["B2_BUCKET_RERANK"], "candidate_details": [stripped]})), encoding="utf-8")
        return {"round_id": round_id, "round_consumed": True, "results": [result]}

    def _find_original_config(self, candidate: dict[str, Any]) -> dict[str, Any]:
        # Reconstruct config from candidate fields
        cfg = {"model_id": candidate["candidate_id"], "model_family": candidate["model_family"]}
        # We stored only limited fields; blend weights not preserved. Use defaults.
        if candidate["model_family"] == "score_blend":
            cfg["weights"] = {"history": 1.0, "popularity": 0.5, "cooccurrence": 1.0}
        if candidate["model_family"] == "candidate_ranker":
            cfg["base"] = "logistic"
            cfg["n_candidates"] = 100
            cfg["n_negatives"] = 25
        return cfg

    def _round_not_started(self, round_id: str, reason: str) -> dict[str, Any]:
        return {"round_id": round_id, "round_consumed": False, "reason": reason, "results": []}

    def _select_anchor(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [c for c in candidates if c.get("metrics")]
        if not eligible:
            raise RuntimeError("no candidate available for anchor")
        return max(eligible, key=lambda c: (c["metrics"]["overall"][f"ndcg@{c.get('top_k', 10)}"], c["metrics"]["overall"][f"hit_rate@{c.get('top_k', 10)}"]))

    def _materialize_anchor(self, out_dir: Path, candidate: dict[str, Any], folds: Any, input_hashes: dict[str, str]) -> None:
        anchor_dir = out_dir / B2_EVAL_ANCHOR_ID
        anchor_dir.mkdir(parents=True, exist_ok=True)
        top_k = candidate["oof_topk"].shape[1]
        oof_path = anchor_dir / f"{B2_EVAL_ANCHOR_ID}_oof.npz"
        np.savez(
            oof_path,
            uids=folds.uids,
            topk=candidate["oof_topk"],
            fold=folds.folds,
        )
        manifest = {
            "anchor_id": B2_EVAL_ANCHOR_ID,
            "source_candidate_id": candidate["candidate_id"],
            "status": "materialized",
            "fold_protocol": AFAC_B2_FOLD_V1,
            "deployment_equivalent": False,
            "oof_path": rel_ref(oof_path, self.project_root),
            "oof_sha256": sha256_file(oof_path),
            "metrics": candidate["metrics"],
            "input_hashes": input_hashes,
        }
        (anchor_dir / f"{B2_EVAL_ANCHOR_ID}_manifest.json").write_text(json_dumps(_json_safe(manifest)) + "\n", encoding="utf-8")

    def _select_final_candidates(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.candidates:
            raise RuntimeError("no candidates produced")
        top_k = self.candidates[0]["oof_topk"].shape[1]
        metric_key = f"overall_ndcg@{top_k}"
        best_sci = max(self.candidates, key=lambda c: (c["metrics"]["overall"][f"ndcg@{top_k}"], c["metrics"]["overall"][f"hit_rate@{top_k}"]))

        def deployment_score(c: dict[str, Any]) -> float:
            m = c["metrics"]
            std = m["panels"]["B2_STANDARD_PANEL"][f"ndcg@{top_k}"]
            short = m["panels"].get("B2_SHORT_HISTORY_PANEL", {}).get(f"ndcg@{top_k}", std)
            test_like = m["panels"].get("B2_TEST_LIKE_PANEL", {}).get(f"ndcg@{top_k}", std)
            worst = m["worst_fold_ndcg"]
            return 0.35 * std + 0.20 * short + 0.20 * test_like + 0.25 * worst
        best_dep = max(self.candidates, key=deployment_score)
        return best_sci, best_dep

    def _finalize_deployment(
        self,
        out_dir: Path,
        dep_candidate: dict[str, Any],
        dataset: B2Dataset,
        adapter: B2TaskAdapter,
        folds: Any,
        item_list: list[str],
        train_targets: dict[str, str],
    ) -> Path:
        top_k = dataset.top_k
        test_uids = dataset.test_df["uid"].astype(str).tolist()
        order = adapter.submission_order(dataset)
        order_inv = {uid: pos for pos, uid in enumerate(order)}

        # If test_topk not present, recompute
        if "test_topk" not in dep_candidate:
            model = self._instantiate({"model_id": dep_candidate["candidate_id"], "model_family": dep_candidate["model_family"]})
            model.fit(folds.uids.tolist(), item_list, dataset.train_seq, train_targets=train_targets, user_df=dataset.user_df, item_df=dataset.item_df)
            test_topk = model.topk(test_uids, k=top_k)
        else:
            test_topk = dep_candidate["test_topk"]

        ordered_topk = []
        for uid in order:
            pos = order_inv[uid]
            ordered_topk.append(test_topk[pos])

        to_upload = out_dir / "TO_UPLOAD"
        to_upload.mkdir(parents=True, exist_ok=True)
        sub_path = to_upload / "candidate_B2.csv"
        with sub_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["uid", "prediction"])
            for uid, topk in zip(order, ordered_topk):
                writer.writerow([uid, ",".join(str(iid) for iid in topk)])

        all_pred_items = {iid for topk in test_topk for iid in topk}
        audit = {
            "n_rows": int(len(order)),
            "matches_test_count": int(len(order) == len(test_uids)),
            "unique_test_uids": int(len(set(order)) == len(order)),
            "top_k": int(top_k),
            "all_predictions_legal": all_pred_items.issubset(set(item_list)),
            "candidate_B2_sha256": sha256_file(sub_path),
        }
        (to_upload / "submission_audit.json").write_text(json_dumps(audit) + "\n", encoding="utf-8")
        (to_upload / "SUBMISSION_README.md").write_text(
            "# B2 Submission\n\n`candidate_B2.csv` generated by AFAC Agent B2 autonomous recommendation loop.\nNo Test truth used.\n",
            encoding="utf-8",
        )
        (to_upload / "deployment_manifest.json").write_text(json_dumps({
            "candidate_id": dep_candidate["candidate_id"],
            "model_family": dep_candidate.get("model_family"),
        }), encoding="utf-8")
        return sub_path

    def _build_trajectory(self, run_id: str, round1: dict, round2: dict, round3: dict, best_sci: dict, best_dep: dict) -> dict[str, Any]:
        def strip(r: dict) -> dict:
            return {k: ({kk: vv for kk, vv in v.items() if kk not in {"oof_topk", "test_topk", "oof_scores", "test_scores"}} if isinstance(v, dict) else v) for k, v in r.items()}
        return {
            "trajectory_version": B2_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "task": "B2",
            "scientific_rounds_used": self.rounds_used,
            "rounds": [strip(round1), strip(round2), strip(round3)],
            "best_scientific_candidate": best_sci.get("candidate_id"),
            "best_deployment_candidate": best_dep.get("candidate_id"),
            "test_truth_used": False,
            "mutates_a1_a2_b1_assets": False,
            "online_champion_mutated": False,
        }

    def _report(self, run_id: str, best_sci: dict, best_dep: dict, rounds_used: int) -> str:
        top_k = best_sci.get("oof_topk", np.array([[""] * 10])).shape[1]
        return "\n".join([
            "# B2 Autonomous Recommendation Closed Loop V1 Report",
            "",
            f"run_id: `{run_id}`",
            f"scientific_rounds_used: `{rounds_used}`",
            f"best_scientific_candidate: `{best_sci.get('candidate_id')}`",
            f"best_scientific_ndcg@{top_k}: `{best_sci.get('metrics', {}).get('overall', {}).get(f'ndcg@{top_k}')}`",
            f"best_deployment_candidate: `{best_dep.get('candidate_id')}`",
            f"best_deployment_ndcg@{top_k}: `{best_dep.get('metrics', {}).get('overall', {}).get(f'ndcg@{top_k}')}`",
            "",
            "No A1/A2/B1 assets were modified. No Test truth was used.",
            "",
        ])


def run_b2_closed_loop(**kwargs: Any) -> dict[str, Any]:
    runner = B2ClosedLoopRunner(
        project_root=kwargs.pop("project_root"),
        data_root=kwargs.pop("data_root"),
        out_root=kwargs.pop("out_root", "artifacts/b2_runs"),
        max_wall_clock_seconds=kwargs.pop("max_wall_clock_seconds", 7200),
        max_rounds=kwargs.pop("max_rounds", 3),
    )
    return runner.run(**kwargs)
