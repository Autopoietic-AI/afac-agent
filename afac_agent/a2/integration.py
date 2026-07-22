# -*- coding: utf-8 -*-
"""A2 integration dry-run orchestrator.

Chains the full A2 integration pipeline without executing any real
experiment:

A2 Profile → Scientific Queue → Portfolio → Complementarity → Fusion Planner
→ M6B proposal → M6C review → M5 admission → Execution Preview
→ Evaluation Preview → Trajectory Preview

Guarantees:
- no training, no GPU, no Test prediction, no submission, no LLM, no network;
- no consumption of A2 scientific rounds (scientific_rounds_used stays 0);
- no mutation of A1 champion/anchor/fold/history/project-state;
- Test truth isolation: test.csv targets are never read (they do not exist);
- deterministic resume: same inputs produce the same run_id and artifacts.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..research.event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash
from .anchors import (
    A2_EVAL_ANCHOR_ID,
    A2_ONLINE_ANCHOR_ID,
    materialize_evaluation_anchor,
    materialize_online_anchor,
)
from .assets import SCOPE_OFFLINE_OOF, ScoreAsset, load_score_asset
from .complementarity import build_complementarity_report
from .evaluator import A2Evaluator, ndcg_at_k
from .fold import AFAC_A2_FOLD_V1, validate_fold_csv
from .fusion import (
    ALPHA_GRID,
    build_history_mask,
    cross_fit_scores,
    op_rank_fusion,
    op_retriever_ranker_composition,
    op_slot_protected_rerank,
    op_topk_protected_rerank,
)
from .portfolio import (
    ASSET_V42C_DIN,
    ASSET_V48A,
    build_asset_verification,
    build_portfolio,
)
from .profiler import build_bucket_registry, build_profile
from .task_adapter import A2TaskAdapter

A2_INTEGRATION_VERSION = "a2_integration_v1"

V42_OOF_REL = "runs/C2_v42_merged/oof_scores_full.npz"
V48_OOF_REL = "runs/C2_v48_merged/oof_scores_full.npz"
V42_TEST_REL = "runs/C2_v42_merged/test_scores_cvmean.npz"
V48_TEST_REL = "runs/C2_v48_merged/test_scores_cvmean.npz"
FOLD_REL = "runs/stageA/train_folds.csv"

PACKAGE_FILES = (
    "A2_INTEGRATION_REPORT.md",
    "integration_manifest.json",
    "a2_profile.json",
    "a2_bucket_registry.json",
    "a2_fold_manifest.json",
    "online_anchor_manifest.json",
    "evaluation_anchor_manifest.json",
    "portfolio_snapshot.json",
    "asset_verification.json",
    "complementarity_report.json",
    "fusion_plan.json",
    "m6b_proposal.json",
    "m6c_review.json",
    "m5_admission.json",
    "execution_preview.json",
    "evaluation_preview.json",
    "trajectory_preview.json",
)


class A2IntegrationRunner:
    def __init__(self, *, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def run(
        self,
        *,
        a2_data_dir: str | Path = "",
        a2_runs_root: str | Path = "",
        out_root: str | Path = "artifacts/a2_integration",
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        out_root = self._resolve(out_root)

        # ---- input resolution: report the single missing item, never scan ----
        missing_item = ""
        data_dir = self._resolve(a2_data_dir) if a2_data_dir else None
        runs_root = self._resolve(a2_runs_root) if a2_runs_root else None
        required: dict[str, Path | None] = {}
        if data_dir is None:
            missing_item = "a2_data_dir"
        else:
            missing_files = A2TaskAdapter(data_dir).missing_files()
            if missing_files:
                missing_item = f"a2_data_file:{missing_files[0]}"
        if not missing_item:
            if runs_root is None:
                missing_item = "a2_runs_root"
            else:
                for rel in (FOLD_REL, V42_OOF_REL, V48_OOF_REL):
                    found = self._find_asset(runs_root, rel)
                    if found is None:
                        missing_item = f"a2_asset:{rel}"
                        break
                    required[rel] = found
        if missing_item:
            return self._waiting(out_root, missing_item, started)

        required[V42_TEST_REL] = self._find_asset(runs_root, V42_TEST_REL)
        required[V48_TEST_REL] = self._find_asset(runs_root, V48_TEST_REL)

        input_hashes = {
            "train_csv": sha256_file(data_dir / "train.csv"),
            "test_csv": sha256_file(data_dir / "test.csv"),
            "item_csv": sha256_file(data_dir / "item.csv"),
            "user_csv": sha256_file(data_dir / "user.csv"),
            "fold_csv": sha256_file(required[FOLD_REL]),
            "v42_oof": sha256_file(required[V42_OOF_REL]),
            "v48_oof": sha256_file(required[V48_OOF_REL]),
        }
        run_id = stable_hash({"version": A2_INTEGRATION_VERSION, "input_hashes": input_hashes})[:24]
        out_dir = out_root / run_id
        manifest_path = out_dir / "integration_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        if out_dir.exists() and force_rebuild:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return self._execute(
            data_dir=data_dir,
            required=required,
            input_hashes=input_hashes,
            out_dir=out_dir,
            run_id=run_id,
            started=started,
        )

    def _find_asset(self, runs_root: Path, rel: str) -> Path | None:
        parts = Path(rel).parts
        candidates = [runs_root / rel]
        if parts and parts[0] == "runs":
            candidates.append(runs_root.joinpath(*parts[1:]))
            candidates.append(runs_root / "runs" / Path(*parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _waiting(self, out_root: Path, missing_item: str, started: float) -> dict[str, Any]:
        out_root.mkdir(parents=True, exist_ok=True)
        return {
            "status": "waiting_for_explicit_asset",
            "missing_asset": missing_item,
            "scientific_rounds_used": 0,
            "duration_seconds": round(time.time() - started, 6),
            "artifacts": {},
        }

    # ------------------------------------------------------------------
    def _execute(
        self,
        *,
        data_dir: Path,
        required: dict[str, Path | None],
        input_hashes: dict[str, str],
        out_dir: Path,
        run_id: str,
        started: float,
    ) -> dict[str, Any]:
        artifacts: dict[str, str] = {}

        def write(name: str, payload: dict[str, Any]) -> None:
            path = out_dir / name
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[Path(name).stem] = rel_ref(path, self.project_root)

        # ---- stage 1: task adapter + fold + profile ----
        dataset = A2TaskAdapter(data_dir).load()
        fold_result = validate_fold_csv(required[FOLD_REL], dataset)
        fold_map_full = fold_result.get("fold_map", {})
        fold_manifest = {k: v for k, v in fold_result.items() if k != "fold_map"}
        write("a2_fold_manifest.json", fold_manifest)

        online_anchor = materialize_online_anchor(out_dir=out_dir / "anchors" / A2_ONLINE_ANCHOR_ID, project_root=self.project_root)
        write("online_anchor_manifest.json", online_anchor)

        # ---- stage 2: portfolio assets (OOF + test-scope identity) ----
        v42_oof = load_score_asset(asset_id=ASSET_V42C_DIN, path=required[V42_OOF_REL], dataset=dataset, expected_scope=SCOPE_OFFLINE_OOF, project_root=self.project_root)
        v48_oof = load_score_asset(asset_id=ASSET_V48A, path=required[V48_OOF_REL], dataset=dataset, expected_scope=SCOPE_OFFLINE_OOF, project_root=self.project_root)
        portfolio = build_portfolio(
            dataset=dataset,
            project_root=self.project_root,
            v42_oof=v42_oof,
            v48_oof=v48_oof,
            v42_test_path=required.get(V42_TEST_REL),
            v48_test_path=required.get(V48_TEST_REL),
        )
        write("portfolio_snapshot.json", portfolio)
        asset_verification = build_asset_verification(portfolio)
        write("asset_verification.json", asset_verification)

        # ---- stage 3: evaluation anchor from the verified v42c OOF route ----
        eval_anchor = materialize_evaluation_anchor(
            dataset=dataset,
            fold_manifest=fold_result,
            oof_asset=v42_oof,
            out_dir=out_dir / "anchors" / A2_EVAL_ANCHOR_ID,
            project_root=self.project_root,
            expected_source_asset_id=ASSET_V42C_DIN,
        )
        write("evaluation_anchor_manifest.json", eval_anchor)

        if eval_anchor.get("status") != "materialized":
            return self._finalize_blocked(out_dir, run_id, artifacts, input_hashes, started, eval_anchor)

        # ---- stage 4: evaluator + profile ----
        evaluator = A2Evaluator(dataset, fold_map_full)
        baseline_asset = v42_oof.aligned_to(dataset.train_uids)
        baseline_eval = evaluator.evaluate_scores(
            asset_id=ASSET_V42C_DIN,
            uids=dataset.train_uids,
            item_ids=baseline_asset.item_ids,
            scores=baseline_asset.scores,
            targets=baseline_asset.targets,
        )
        profile = build_profile(dataset, evaluations=[baseline_eval])
        write("a2_profile.json", profile)
        write("a2_bucket_registry.json", build_bucket_registry())

        # ---- stage 5: complementarity audit ----
        complementarity = build_complementarity_report(dataset=dataset, evaluator=evaluator, asset_a=v42_oof, asset_b=v48_oof)
        write("complementarity_report.json", complementarity)

        # ---- stage 6: fusion planner over bounded candidates ----
        fusion_plan = self._fusion_plan(dataset, evaluator, fold_map_full, v42_oof, v48_oof, baseline_eval)
        write("fusion_plan.json", fusion_plan)

        # ---- stage 7: M6B proposal / M6C review / M5 admission ----
        proposal = self._m6b_proposal(run_id, fusion_plan, eval_anchor)
        write("m6b_proposal.json", proposal)
        review = self._m6c_review(proposal)
        write("m6c_review.json", review)
        admission = self._m5_admission(proposal, review)
        write("m5_admission.json", admission)

        # ---- stage 8: previews (no execution) ----
        execution_preview = self._execution_preview(proposal, admission, fusion_plan)
        write("execution_preview.json", execution_preview)
        evaluation_preview = self._evaluation_preview(proposal, eval_anchor)
        write("evaluation_preview.json", evaluation_preview)
        trajectory_preview = self._trajectory_preview(run_id)
        write("trajectory_preview.json", trajectory_preview)

        # ---- report + package + manifest ----
        report = out_dir / "A2_INTEGRATION_REPORT.md"
        report.write_text(self._report(run_id, eval_anchor, portfolio, complementarity, fusion_plan, admission), encoding="utf-8")
        artifacts["A2_INTEGRATION_REPORT"] = rel_ref(report, self.project_root)

        status = "ready_for_experiment_design"
        manifest = {
            "manifest_version": A2_INTEGRATION_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "task": "A2",
            "fold_protocol": AFAC_A2_FOLD_V1,
            "online_anchor": A2_ONLINE_ANCHOR_ID,
            "evaluation_anchor": A2_EVAL_ANCHOR_ID,
            "evaluation_anchor_status": eval_anchor.get("status"),
            "deployment_equivalent": eval_anchor.get("deployment_equivalent"),
            "scientific_rounds_used": 0,
            "counts_as_experiment_round": False,
            "trains_model": False,
            "uses_gpu": False,
            "generates_test_prediction": False,
            "creates_submission": False,
            "uses_llm": False,
            "uses_network": False,
            "uses_test_truth": False,
            "mutates_a1_champion_or_anchor": False,
            "mutates_project_state": False,
            "mutates_confirmed_history": False,
            "input_hashes": input_hashes,
            "blocking_item": "",
            "next_stage": "final_a2_real_closed_loop_requires_explicit_next_module_approval",
            "artifacts": artifacts,
        }
        manifest_path = out_dir / "integration_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["integration_manifest"] = rel_ref(manifest_path, self.project_root)

        package = out_dir / "REPORT_PACKAGE"
        package.mkdir(parents=True, exist_ok=True)
        for name in PACKAGE_FILES:
            src = out_dir / name
            if src.exists():
                shutil.copyfile(src, package / name)
        artifacts["report_package"] = rel_ref(package, self.project_root)
        # refresh manifest with final artifacts map
        manifest["artifacts"] = artifacts
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        shutil.copyfile(manifest_path, package / "integration_manifest.json")
        return {"status": status, "run_id": run_id, "scientific_rounds_used": 0, "artifacts": artifacts}

    def _finalize_blocked(self, out_dir, run_id, artifacts, input_hashes, started, eval_anchor) -> dict[str, Any]:
        status = "waiting_for_explicit_asset" if eval_anchor.get("status") == "waiting_for_explicit_asset" else "validation_failed"
        manifest = {
            "manifest_version": A2_INTEGRATION_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "task": "A2",
            "scientific_rounds_used": 0,
            "blocking_item": eval_anchor.get("missing_asset") or ";".join(eval_anchor.get("verification_errors", [])),
            "input_hashes": input_hashes,
            "artifacts": artifacts,
        }
        path = out_dir / "integration_manifest.json"
        path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["integration_manifest"] = rel_ref(path, self.project_root)
        return {"status": status, "run_id": run_id, "scientific_rounds_used": 0, "artifacts": artifacts}

    # ------------------------------------------------------------------
    def _fusion_plan(self, dataset, evaluator, fold_map, v42_oof, v48_oof, baseline_eval) -> dict[str, Any]:
        a = v42_oof.aligned_to(dataset.train_uids)
        b = v48_oof.aligned_to(dataset.train_uids)
        uids = dataset.train_uids
        folds = np.array([fold_map[uid] for uid in uids], dtype=np.int64)
        len3_mask = np.array([dataset.len_bucket(uid) == "exact_len3" for uid in uids], dtype=bool)
        history_mask = build_history_mask(dataset, uids)
        col_of = {iid: pos for pos, iid in enumerate(a.item_ids)}
        target_col = np.array([col_of.get(dataset.train_targets.get(uid, ""), -1) for uid in uids], dtype=np.int64)

        candidates: list[dict[str, Any]] = [
            {"candidate_id": "score_blend_v42_v48", "operator": "score_blend", "learned": True},
            {"candidate_id": "rank_fusion_rrf_v42_v48", "operator": "rank_fusion", "learned": False},
            {"candidate_id": "bucket_route_exact_len3_to_v48", "operator": "bucket_route", "learned": True},
            {"candidate_id": "topk_protected_rerank_v48_over_v42", "operator": "topk_protected_rerank", "learned": False},
            {"candidate_id": "slot_protected_rerank_history_protected", "operator": "slot_protected_rerank", "learned": False},
            {"candidate_id": "retriever_v42_ranker_v48_pool20", "operator": "retriever_ranker_composition", "learned": False},
        ]

        evaluated: list[dict[str, Any]] = []
        for candidate in candidates:
            op = candidate["operator"]
            if op == "rank_fusion":
                scores = op_rank_fusion(a.scores, b.scores)
                assignment = {"mode": "fixed_no_learned_parameter", "fold_assignments": []}
            elif op == "topk_protected_rerank":
                scores = op_topk_protected_rerank(a.scores, b.scores)
                assignment = {"mode": "fixed_no_learned_parameter", "fold_assignments": []}
            elif op == "slot_protected_rerank":
                scores = op_slot_protected_rerank(a.scores, b.scores, history_mask)
                assignment = {"mode": "fixed_no_learned_parameter", "fold_assignments": []}
            elif op == "retriever_ranker_composition":
                scores = op_retriever_ranker_composition(a.scores, b.scores, pool_size=20)
                assignment = {"mode": "fixed_no_learned_parameter", "fold_assignments": []}
            else:
                scores, assignment = cross_fit_scores(
                    operator=op,
                    scores_a=a.scores,
                    scores_b=b.scores,
                    folds=folds,
                    route_mask=len3_mask,
                    history_mask=history_mask,
                    alpha_grid=ALPHA_GRID,
                    selection_targets=target_col,
                )
            ev = evaluator.evaluate_scores(
                asset_id=candidate["candidate_id"],
                uids=uids,
                item_ids=a.item_ids,
                scores=scores,
                targets=a.targets,
            )
            comparison = evaluator.compare(baseline=baseline_eval, candidate=ev)
            bucket_comparison = evaluator.compare_by_bucket(baseline=baseline_eval, candidate=ev, uids=uids)
            per_user_gain = ndcg_at_k(ev.ranks) - ndcg_at_k(baseline_eval.ranks)
            fold_rows = []
            for f in sorted(set(folds.tolist())):
                mask = folds == f
                fold_rows.append({"fold": int(f), "ndcg_at_10_gain": float(per_user_gain[mask].mean()), "user_count": int(mask.sum())})
            positive_folds = sum(1 for row in fold_rows if row["ndcg_at_10_gain"] >= -1e-12)
            duplicate = bool(np.array_equal(ev.top10, baseline_eval.top10))
            evaluated.append({
                "candidate_id": candidate["candidate_id"],
                "operator": op,
                "cross_fit": assignment,
                "metrics": ev.summary(),
                "comparison_vs_anchor": comparison,
                "fold_gains": fold_rows,
                "positive_fold_count": positive_folds,
                "duplicate_of_anchor": duplicate,
                "bucket_comparison": bucket_comparison,
            })

        decisions = []
        for ev in evaluated:
            cmp_ = ev["comparison_vs_anchor"]
            reasons = []
            if ev["duplicate_of_anchor"]:
                reasons.append("duplicate_anchor")
            if cmp_["ndcg_at_10_gain"] <= 1e-12:
                reasons.append("no_ndcg_gain")
            if cmp_["hit_rate_at_10_gain"] < -1e-12:
                reasons.append("hit_rate_decrease")
            if (cmp_["net"] or 0) <= 0:
                reasons.append("rescue_not_greater_than_damage")
            if ev["positive_fold_count"] < 4:
                reasons.append("less_than_4_nonnegative_folds")
            decisions.append({
                "candidate_id": ev["candidate_id"],
                "operator": ev["operator"],
                "status": "accepted" if not reasons else "rejected",
                "reason_codes": sorted(reasons) or ["policy_passed"],
                "ndcg_at_10_gain": cmp_["ndcg_at_10_gain"],
                "hit_rate_at_10_gain": cmp_["hit_rate_at_10_gain"],
                "rescue": cmp_["rescue"],
                "damage": cmp_["damage"],
                "net": cmp_["net"],
                "changed_user_count": cmp_["changed_user_count"],
                "change_precision": cmp_["change_precision"],
                "positive_fold_count": ev["positive_fold_count"],
            })
        ranked = sorted(decisions, key=lambda r: (r["status"] != "accepted", -(r["ndcg_at_10_gain"] or -999), r["candidate_id"]))
        return {
            "planner_version": A2_INTEGRATION_VERSION,
            "task": "A2",
            "baseline_anchor": A2_EVAL_ANCHOR_ID,
            "fold_protocol": AFAC_A2_FOLD_V1,
            "scientific_queue": [
                {
                    "problem_id": "scientific::a2_bounded_portfolio_fusion",
                    "selected_problem": "Can verified v42c/v48 OOF assets produce a bounded, deployment-safe ranking candidate without training?",
                    "priority": 1,
                    "information_source": "new_combination_of_existing_verified_oof_assets",
                }
            ],
            "operators_enabled": ["score_blend", "rank_fusion", "candidate_union", "bucket_route", "topk_protected_rerank", "slot_protected_rerank", "retriever_ranker_composition"],
            "protection_rules": {
                "top10_set_membership_fixed": ["topk_protected_rerank", "slot_protected_rerank"],
                "top7_rerank_only": ["topk_protected_rerank"],
                "history_slots_protected": ["slot_protected_rerank"],
                "exact_len3_routing": ["bucket_route"],
            },
            "candidates": [
                {k: ev[k] for k in ("candidate_id", "operator", "cross_fit", "metrics", "comparison_vs_anchor", "fold_gains", "positive_fold_count", "duplicate_of_anchor")}
                for ev in evaluated
            ],
            "decisions": {"items": ranked, "accepted_count": sum(1 for r in ranked if r["status"] == "accepted"), "best_candidate_id": ranked[0]["candidate_id"] if ranked else ""},
            "acceptance_policy": {
                "strict_cross_fit_for_learned_params": True,
                "oracle_executable": False,
                "test_feedback_allowed": False,
                "min_nonnegative_folds": 4,
                "rescue_must_exceed_damage": True,
            },
            "budget": {"max_candidates": 8, "generated": len(candidates)},
            "bucket_comparisons": {ev["candidate_id"]: ev["bucket_comparison"] for ev in evaluated},
        }

    # ------------------------------------------------------------------
    def _m6b_proposal(self, run_id: str, fusion_plan: dict[str, Any], eval_anchor: dict[str, Any]) -> dict[str, Any]:
        best = fusion_plan["decisions"]["items"][0] if fusion_plan["decisions"]["items"] else {}
        return {
            "proposal_version": A2_INTEGRATION_VERSION,
            "proposal_id": stable_hash({"run_id": run_id, "stage": "m6b"}),
            "problem_id": "scientific::a2_bounded_portfolio_fusion",
            "target_scope": "offline_oof_only",
            "parent_candidate": A2_EVAL_ANCHOR_ID,
            "selected_assets": [ASSET_V42C_DIN, ASSET_V48A],
            "best_offline_candidate": best.get("candidate_id", ""),
            "configuration": {"operator_family": "bounded_ranking_fusion_grid", "max_candidates": 8, "strict_cross_fit": True},
            "fold_protocol": AFAC_A2_FOLD_V1,
            "round_cost": 1,
            "estimated_runtime": "minutes_cpu_oof_array_operations",
            "success_conditions": ["strict OOF verification passes", "ndcg@10 gain positive", "fold stability policy satisfied", "no Test truth", "Top10 set protection honored where required"],
            "failure_conditions": ["asset verification failure", "hit@10 decrease", "rescue not greater than damage", "duplicate anchor"],
            "stop_conditions": ["accepted bounded candidate found", "no non-repeating executable follow-up with expected gain"],
            "explicitly_not_in_scope": ["training", "test_prediction", "submission", "online_champion_mutation"],
        }

    def _m6c_review(self, proposal: dict[str, Any]) -> dict[str, Any]:
        return {
            "critic_version": A2_INTEGRATION_VERSION,
            "decision": "approve_for_bounded_offline_execution_in_future_round",
            "risk": "low",
            "reason_codes": [
                "uses_verified_oof_assets_only",
                "strict_oof_only",
                "no_training",
                "no_test_truth",
                "no_champion_mutation",
                "deployment_protection_rules_encoded",
            ],
            "concerns": [
                "declared_unmaterialized assets (V23 Top10, C_all, DCN-Mix, LambdaRank, Len0/Len3 experts) cannot enter OOF evaluation until explicit artifacts are supplied",
                "online champion strategy is not deployment_equivalent to the offline anchor",
            ],
        }

    def _m5_admission(self, proposal: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        return {
            "admission_version": A2_INTEGRATION_VERSION,
            "status": "admitted_for_experiment_design_only",
            "m5_authoritative": True,
            "round_cost_reserved": proposal["round_cost"],
            "scientific_rounds_used_now": 0,
            "reason_codes": ["dry_run_no_execution", "bounded_oof_design_admitted", "scientific_round_budget_untouched"],
        }

    def _execution_preview(self, proposal: dict[str, Any], admission: dict[str, Any], fusion_plan: dict[str, Any]) -> dict[str, Any]:
        return {
            "preview_version": A2_INTEGRATION_VERSION,
            "status": "preview_only_not_executed",
            "would_execute": {
                "adapter_id": "a2-fusion-controller",
                "problem_id": proposal["problem_id"],
                "candidates": [c["candidate_id"] for c in fusion_plan["candidates"]],
                "round_cost": proposal["round_cost"],
            },
            "guards": {
                "trains_model": False,
                "uses_gpu": False,
                "generates_test_prediction": False,
                "creates_submission": False,
                "uses_test_truth": False,
                "mutates_online_champion": False,
                "requires_explicit_next_module_approval": True,
            },
            "admission_status": admission["status"],
        }

    def _evaluation_preview(self, proposal: dict[str, Any], eval_anchor: dict[str, Any]) -> dict[str, Any]:
        return {
            "preview_version": A2_INTEGRATION_VERSION,
            "status": "preview_only_not_executed",
            "baseline_anchor": A2_EVAL_ANCHOR_ID,
            "metrics_protocol": ["ndcg_at_10", "hit_rate_at_10", "mrr_at_10", "candidate_recall", "target_rank", "rescue_damage_net", "changed_user_count", "change_precision"],
            "bucket_axes": ["sequence_length", "candidate_item_type", "ranking_stage", "fold"],
            "retrieval_vs_ranking_failure_separated": True,
            "test_scores_never_used_as_oof": True,
            "anchor_metrics_snapshot": eval_anchor.get("metrics", {}),
        }

    def _trajectory_preview(self, run_id: str) -> dict[str, Any]:
        stages = [
            "a2_profile", "scientific_queue", "portfolio", "complementarity",
            "fusion_planner", "m6b_proposal", "m6c_review", "m5_admission",
            "execution_preview", "evaluation_preview", "trajectory_preview",
        ]
        return {
            "trajectory_version": A2_INTEGRATION_VERSION,
            "run_id": run_id,
            "task": "A2",
            "status": "preview_only_not_executed",
            "stages": [{"stage": name, "status": "completed_offline_analysis" if name not in {"execution_preview", "evaluation_preview", "trajectory_preview"} else "preview_only"} for name in stages],
            "test_truth_used": False,
            "online_champion_mutated": False,
            "scientific_rounds_used": 0,
        }

    def _report(self, run_id, eval_anchor, portfolio, complementarity, fusion_plan, admission) -> str:
        decisions = fusion_plan["decisions"]
        best = decisions["items"][0] if decisions["items"] else {}
        rank_comp = complementarity["ranker_complementarity_at_10"]
        lines = [
            "# A2 Integration Report",
            "",
            f"run_id: `{run_id}`",
            f"status: `ready_for_experiment_design`",
            f"fold_protocol: `{AFAC_A2_FOLD_V1}`",
            f"online_anchor: `{A2_ONLINE_ANCHOR_ID}` (identity only, score 0.5093, not an OOF artifact)",
            f"evaluation_anchor: `{A2_EVAL_ANCHOR_ID}` status=`{eval_anchor.get('status')}` source=`{eval_anchor.get('source_asset_id', '')}` deployment_equivalent=`{eval_anchor.get('deployment_equivalent')}`",
            "",
            "## Evaluation anchor metrics (v42c OOF)",
            "",
            f"ndcg@10: `{eval_anchor.get('metrics', {}).get('ndcg_at_10')}`",
            f"hit@10: `{eval_anchor.get('metrics', {}).get('hit_rate_at_10')}`",
            f"mrr@10: `{eval_anchor.get('metrics', {}).get('mrr_at_10')}`",
            f"candidate_recall: `{eval_anchor.get('metrics', {}).get('candidate_recall')}`",
            "",
            "## Complementarity (v42c vs v48a)",
            "",
            f"a_only_hit: `{rank_comp['a_only_hit']}` b_only_hit: `{rank_comp['b_only_hit']}` net_b_over_a: `{rank_comp['net_b_over_a']}`",
            f"oracle_union_hit_rate: `{rank_comp['oracle_union_hit_rate']}` (diagnostic only)",
            "",
            "## Fusion planner",
            "",
            f"best_candidate: `{best.get('candidate_id', '')}` status=`{best.get('status', '')}`",
            f"accepted_count: `{decisions['accepted_count']}`",
            f"ndcg@10_gain: `{best.get('ndcg_at_10_gain')}` rescue/damage/net: `{best.get('rescue')}/{best.get('damage')}/{best.get('net')}`",
            "",
            f"admission: `{admission['status']}`, scientific_rounds_used: `0`",
            "",
            "No training, GPU use, Test prediction, submission, LLM call, network access, or A1 state mutation was performed.",
            "",
        ]
        return "\n".join(lines)


def run_a2_integration(**kwargs: Any) -> dict[str, Any]:
    runner = A2IntegrationRunner(project_root=kwargs.pop("project_root"))
    return runner.run(**kwargs)
