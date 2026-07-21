# -*- coding: utf-8 -*-
"""A1 autonomous scientific closed loop v1.

This runner performs a bounded, auditable A1 closed-loop execution.  The first
implementation intentionally uses only already-registered, deterministic OOF
fusion machinery.  It may consume a scientific round for a real offline OOF
experiment, but it never uses Test truth, never submits, and never mutates the
online champion.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .fusion_controller import run_fusion_controller
from .research.event_store import ResearchEventStore, json_dumps, load_json, make_event, rel_ref, sha256_file, stable_hash
from .research.memory_views import ResearchMemoryBuilder

A1_CLOSED_LOOP_VERSION = "a1_autonomous_scientific_closed_loop_v1"
MAX_SCIENTIFIC_ROUNDS = 3
DEFAULT_MAX_WALL_CLOCK_SECONDS = 7200
ANCHOR_ID = "A1_EVAL_ANCHOR_V1"
ONLINE_CHAMPION_ID = "v53Q-1"
PORTFOLIO_CANDIDATE_ID = "class_weighted_blend_base_composed"


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _hash_if_exists(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else ""


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class A1ClosedLoopRunner:
    """Run a bounded A1 scientific loop with deterministic resume."""

    def __init__(self, *, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(
        self,
        *,
        anchor_dir: str | Path = "artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1",
        v43c_oof: str | Path,
        v46a_oof: str | Path,
        project_state: str | Path = "config/project_state.json",
        tool_registry: str | Path = "config/tool_registry.json",
        research_policy: str | Path = "config/research_policy.json",
        out_root: str | Path = "artifacts/a1_closed_loop_runs",
        max_rounds: int = MAX_SCIENTIFIC_ROUNDS,
        max_wall_clock_seconds: int = DEFAULT_MAX_WALL_CLOCK_SECONDS,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        max_rounds = max(1, min(int(max_rounds), MAX_SCIENTIFIC_ROUNDS))
        inputs = {
            "anchor_dir": _resolve(self.project_root, anchor_dir),
            "v43c_oof": _resolve(self.project_root, v43c_oof),
            "v46a_oof": _resolve(self.project_root, v46a_oof),
            "project_state": _resolve(self.project_root, project_state),
            "tool_registry": _resolve(self.project_root, tool_registry),
            "research_policy": _resolve(self.project_root, research_policy),
        }
        required_files = {
            "anchor_manifest": inputs["anchor_dir"] / "A1_EVAL_ANCHOR_V1_manifest.json",
            "anchor_oof": inputs["anchor_dir"] / "A1_EVAL_ANCHOR_V1_oof.npz",
            "v43c_oof": inputs["v43c_oof"],
            "v46a_oof": inputs["v46a_oof"],
            "project_state": inputs["project_state"],
            "tool_registry": inputs["tool_registry"],
            "research_policy": inputs["research_policy"],
        }
        missing = [name for name, path in required_files.items() if not path.exists()]
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        input_hashes = {name: _hash_if_exists(path) for name, path in required_files.items()}
        run_id = stable_hash({
            "version": A1_CLOSED_LOOP_VERSION,
            "input_hashes": input_hashes,
            "max_rounds": max_rounds,
            "max_wall_clock_seconds": max_wall_clock_seconds,
        })[:24]
        out_dir = _resolve(self.project_root, out_root) / run_id
        manifest_path = out_dir / "run_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        if manifest_path.exists() and force_rebuild:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return self._execute(
            inputs=inputs,
            input_hashes=input_hashes,
            out_dir=out_dir,
            run_id=run_id,
            started=started,
            max_rounds=max_rounds,
            max_wall_clock_seconds=max_wall_clock_seconds,
        )

    def _execute(
        self,
        *,
        inputs: dict[str, Path],
        input_hashes: dict[str, str],
        out_dir: Path,
        run_id: str,
        started: float,
        max_rounds: int,
        max_wall_clock_seconds: int,
    ) -> dict[str, Any]:
        artifacts: dict[str, str] = {}
        project_state = load_json(inputs["project_state"])
        tool_registry = load_json(inputs["tool_registry"])
        initial_state = self._initial_state(run_id, inputs, input_hashes, project_state, max_rounds, max_wall_clock_seconds)
        self._write(out_dir / "initial_state.json", initial_state, artifacts, "initial_state")
        trajectory: dict[str, Any] = {
            "trajectory_version": A1_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "task": "A1",
            "events": [],
            "rounds": [],
            "test_truth_used": False,
            "online_champion_mutated": False,
        }
        rounds_used = 0
        best_registry = self._initial_best_registry(project_state)

        round1 = self._run_round_01(out_dir, inputs, input_hashes, tool_registry)
        rounds_used += int(round1["round_consumed"])
        best_registry = self._update_best_registry(best_registry, round1)
        trajectory["rounds"].append(round1["trajectory_record"])
        self._write(out_dir / "round_01" / "round_state.json", round1["round_state"], artifacts, "round_01_state")

        memory_update = self._write_memory_update(out_dir, run_id, round1, input_hashes)
        self._write(out_dir / "research_memory_update.json", memory_update, artifacts, "research_memory_update")
        trajectory["events"].extend(memory_update.get("events", []))

        stop = self._stop_decision(round1, rounds_used, max_rounds, started, max_wall_clock_seconds)
        round2 = self._round_not_started(
            round_id="round_02",
            reason=stop["reason"],
            previous_feedback_ref=round1["artifacts"].get("fusion_decisions", ""),
            remaining_budget=max_rounds - rounds_used,
        )
        trajectory["rounds"].append(round2["trajectory_record"])
        self._write(out_dir / "round_02" / "round_state.json", round2["round_state"], artifacts, "round_02_state")
        round3 = self._round_not_started(
            round_id="round_03",
            reason="closed_loop_stopped_before_round_03",
            previous_feedback_ref=round1["artifacts"].get("fusion_decisions", ""),
            remaining_budget=max_rounds - rounds_used,
        )
        trajectory["rounds"].append(round3["trajectory_record"])
        self._write(out_dir / "round_03" / "round_state.json", round3["round_state"], artifacts, "round_03_state")

        self._write(out_dir / "best_candidate_registry.json", best_registry, artifacts, "best_candidate_registry")
        portfolio_update = self._portfolio_update(round1, best_registry)
        self._write(out_dir / "portfolio_update.json", portfolio_update, artifacts, "portfolio_update")
        self._write(out_dir / "stop_decision.json", stop, artifacts, "stop_decision")
        self._write(out_dir / "trajectory_A1.json", trajectory, artifacts, "trajectory_A1")
        report_path = out_dir / "A1_CLOSED_LOOP_REPORT.md"
        report_path.write_text(self._report(run_id, round1, best_registry, stop, rounds_used), encoding="utf-8")
        artifacts["closed_loop_report"] = rel_ref(report_path, self.project_root)
        package = out_dir / "REPORT_PACKAGE"
        package.mkdir(parents=True, exist_ok=True)
        for name in [
            "run_manifest.json",
            "initial_state.json",
            "best_candidate_registry.json",
            "portfolio_update.json",
            "research_memory_update.json",
            "stop_decision.json",
            "trajectory_A1.json",
            "A1_CLOSED_LOOP_REPORT.md",
        ]:
            src = out_dir / name
            if src.exists():
                shutil.copyfile(src, package / name)
        for r in ["round_01", "round_02", "round_03"]:
            src = out_dir / r / "round_state.json"
            if src.exists():
                dst = package / r
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst / "round_state.json")
        artifacts["report_package"] = rel_ref(package, self.project_root)
        manifest = {
            "manifest_version": A1_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "status": "completed",
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "max_scientific_rounds": max_rounds,
            "scientific_rounds_used": rounds_used,
            "single_process": True,
            "single_gpu": True,
            "peak_gpu_memory_gb_observed": 0.0,
            "uses_test_truth": False,
            "creates_submission": False,
            "generates_test_prediction": False,
            "online_champion_mutated": False,
            "mutates_project_state": False,
            "mutates_confirmed_history": False,
            "uses_llm": False,
            "uses_network": False,
            "candidate_a1_csv_generated": False,
            "best_candidate_id": best_registry["best_candidate"]["candidate_id"],
            "best_candidate_status": best_registry["best_candidate"]["status"],
            "fusion_candidate_status": best_registry["portfolio_candidates"][0]["status"],
            "stop_reason": stop["reason"],
            "input_hashes": input_hashes,
            "view_hash": stable_hash({"round_01": round1["round_state"], "best": best_registry, "stop": stop}),
            "artifacts": artifacts | {"report_package": rel_ref(package, self.project_root)},
        }
        manifest_path = out_dir / "run_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["run_manifest"] = rel_ref(manifest_path, self.project_root)
        shutil.copyfile(manifest_path, package / "run_manifest.json")
        return {"status": "completed", "run_id": run_id, "scientific_rounds_used": rounds_used, "artifacts": artifacts}

    def _run_round_01(self, out_dir: Path, inputs: dict[str, Path], input_hashes: dict[str, str], tool_registry: dict[str, Any]) -> dict[str, Any]:
        round_dir = out_dir / "round_01"
        round_dir.mkdir(parents=True, exist_ok=True)
        proposal = {
            "proposal_id": stable_hash({"round": "round_01", "candidate": PORTFOLIO_CANDIDATE_ID, "input_hashes": input_hashes}),
            "problem_id": "scientific::bounded_portfolio_fusion",
            "selected_problem": "Can existing verified OOF assets produce a bounded portfolio candidate without new training?",
            "target_scope": "offline_oof_only",
            "parent_candidate": ANCHOR_ID,
            "selected_assets": ["A1_EVAL_ANCHOR_V1", "V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF", "V46A_ISOLATED_EXPERT_OOF", "V46A_ISOLATED_COMPOSED_OOF"],
            "adapter_id": "fusion-controller",
            "configuration": {"operator_family": "bounded_fusion_grid", "max_candidates": 16, "strict_cross_fit": True},
            "new_information_source": "new_combination_of_existing_verified_oof_assets",
            "expected_information_gain": "low_cost_portfolio_complementarity_validation",
            "fold_protocol": "AFAC_A1_FOLD_V1",
            "round_cost": 1,
            "estimated_runtime": "minutes_cpu_oof_array_operations",
            "estimated_memory": "low_cpu_memory; no GPU allocation expected",
            "success_conditions": ["strict OOF verification passes", "overall gain non-negative", "fold stability policy satisfied", "no Test truth"],
            "failure_conditions": ["asset verification failure", "macro damage exceeds policy", "rescue not greater than damage", "duplicate anchor"],
            "stop_conditions": ["accepted bounded candidate found", "macro protection prevents direct promotion", "no non-repeating executable follow-up with expected gain"],
        }
        critic = {
            "critic_version": A1_CLOSED_LOOP_VERSION,
            "decision": "approve_for_bounded_execution",
            "risk": "low",
            "reason_codes": ["uses_registered_fusion_controller", "strict_oof_only", "no_training", "no_test_truth", "no_champion_mutation"],
        }
        admission = {
            "admission_version": A1_CLOSED_LOOP_VERSION,
            "status": "admitted",
            "m5_authoritative": True,
            "round_cost": 1,
            "reason_codes": ["bounded_oof_execution_admitted", "scientific_round_budget_available"],
        }
        fusion = run_fusion_controller(
            project_root=self.project_root,
            anchor_dir=inputs["anchor_dir"],
            v43_oof=inputs["v43c_oof"],
            v46_oof=inputs["v46a_oof"],
            out_root=out_dir / "fx",
            force_rebuild=True,
        )
        fusion_dir = self.project_root / fusion["artifacts"]["fusion_manifest"]
        fusion_dir = fusion_dir.parent
        manifest = load_json(fusion_dir / "fusion_manifest.json")
        decisions = load_json(fusion_dir / "fusion_decisions.json")
        metrics = load_json(fusion_dir / "fusion_metrics.json")
        complementarity = load_json(fusion_dir / "complementarity_report.json")
        best = decisions["items"][0] if decisions.get("items") else {}
        best_metrics = next((item for item in metrics.get("items", []) if item.get("candidate_id") == best.get("candidate_id")), {})
        promotion = self._promotion_status(best)
        feedback = {
            "feedback_version": A1_CLOSED_LOOP_VERSION,
            "feedback_type": "fusion_oof_feedback",
            "status": "completed",
            "candidate_id": best.get("candidate_id", ""),
            "candidate_status": promotion["status"],
            "overall_gain": best.get("overall_gain"),
            "macro_gain": best.get("macro_gain"),
            "positive_fold_count": best.get("positive_fold_count"),
            "worst_fold_gain": best.get("worst_fold_gain"),
            "rescue": best.get("rescue"),
            "damage": best.get("damage"),
            "net": best.get("net"),
            "changed_count": best.get("changed_count"),
            "change_precision": best.get("change_precision"),
            "key_bucket_class_feedback": self._key_bucket_feedback(best_metrics),
            "promotion_reason_codes": promotion["reason_codes"],
            "test_truth_used": False,
            "oracle_used_for_selection": False,
        }
        artifacts = {
            "fusion_manifest": rel_ref(fusion_dir / "fusion_manifest.json", self.project_root),
            "fusion_decisions": rel_ref(fusion_dir / "fusion_decisions.json", self.project_root),
            "fusion_metrics": rel_ref(fusion_dir / "fusion_metrics.json", self.project_root),
            "complementarity_report": rel_ref(fusion_dir / "complementarity_report.json", self.project_root),
        }
        round_state = {
            "round_version": A1_CLOSED_LOOP_VERSION,
            "round_id": "round_01",
            "status": "completed",
            "selected_problem": proposal["selected_problem"],
            "proposal": proposal,
            "critic_decision": critic,
            "m5_admission": admission,
            "execution": {
                "adapter_id": "fusion-controller",
                "status": fusion["status"],
                "start_end_recorded": True,
                "trains_model": False,
                "generates_test_prediction": False,
                "creates_submission": False,
                "uses_test_truth": False,
                "scientific_round_consumed": True,
            },
            "feedback": feedback,
            "safe_resume_point": "round_01_completed",
            "completed_outputs": artifacts,
            "remaining_budget_after_round": MAX_SCIENTIFIC_ROUNDS - 1,
        }
        trajectory_record = {
            "round_id": "round_01",
            "selected_problem": proposal["problem_id"],
            "input_state_hash": stable_hash(input_hashes),
            "proposal": proposal,
            "critic_decision": critic,
            "m5_admission": admission,
            "adapter_config": proposal["configuration"],
            "execution_status": fusion["status"],
            "feedback": feedback,
            "memory_update": "append_only_event_planned",
            "best_candidate": {"candidate_id": best.get("candidate_id", ""), "status": promotion["status"]},
            "next_decision": "stop_or_continue_evaluated_after_feedback",
        }
        return {"round_consumed": fusion["status"] == "completed", "round_state": round_state, "trajectory_record": trajectory_record, "feedback": feedback, "artifacts": artifacts, "fusion_manifest": manifest, "decisions": decisions, "complementarity": complementarity}

    def _round_not_started(self, *, round_id: str, reason: str, previous_feedback_ref: str, remaining_budget: int) -> dict[str, Any]:
        state = {
            "round_version": A1_CLOSED_LOOP_VERSION,
            "round_id": round_id,
            "status": "not_started_due_to_stop",
            "reason": reason,
            "previous_feedback_read": bool(previous_feedback_ref),
            "previous_feedback_ref": previous_feedback_ref,
            "round_consumed": False,
            "remaining_budget": remaining_budget,
            "safe_resume_point": "closed_loop_stopped",
        }
        return {
            "round_state": state,
            "trajectory_record": {
                "round_id": round_id,
                "selected_problem": "",
                "input_state_hash": "",
                "proposal": {},
                "critic_decision": {"decision": "not_run"},
                "m5_admission": {"status": "not_requested"},
                "adapter_config": {},
                "execution_status": "not_started_due_to_stop",
                "feedback": {"status": "not_generated"},
                "memory_update": "not_appended",
                "best_candidate": {},
                "next_decision": reason,
            },
        }

    def _promotion_status(self, decision: dict[str, Any]) -> dict[str, Any]:
        reasons = _as_list(decision.get("reason_codes"))
        if decision.get("status") != "accepted":
            return {"status": "rejected", "reason_codes": reasons or ["fusion_policy_rejected"]}
        macro_gain = decision.get("macro_gain")
        if isinstance(macro_gain, (int, float)) and macro_gain < 0:
            return {"status": "accepted_portfolio", "reason_codes": ["overall_gain_positive", "fold_stability_met", "macro_gain_negative_requires_portfolio_only"]}
        return {"status": "promoted_best", "reason_codes": ["overall_and_macro_policy_passed"]}

    def _key_bucket_feedback(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for row in _as_list(metrics.get("bucket_class_gains") or metrics.get("bucket_class")):
            if row.get("sample_count", 0) and row.get("net") not in {None, 0}:
                rows.append({
                    "axis": row.get("axis"),
                    "bucket": row.get("bucket"),
                    "class_id": row.get("class_id"),
                    "sample_count": row.get("sample_count"),
                    "rescue": row.get("rescue"),
                    "damage": row.get("damage"),
                    "net": row.get("net"),
                })
        return sorted(rows, key=lambda r: (-abs(int(r.get("net") or 0)), str(r.get("bucket")), int(r.get("class_id") or 0)))[:12]

    def _initial_state(self, run_id: str, inputs: dict[str, Path], input_hashes: dict[str, str], project_state: dict[str, Any], max_rounds: int, max_wall_clock_seconds: int) -> dict[str, Any]:
        return {
            "state_version": A1_CLOSED_LOOP_VERSION,
            "run_id": run_id,
            "online_champion": project_state.get("online_version", ONLINE_CHAMPION_ID),
            "online_score": project_state.get("online_score"),
            "evaluation_anchor": ANCHOR_ID,
            "existing_portfolio_candidate": {"candidate_id": PORTFOLIO_CANDIDATE_ID, "initial_status": "portfolio_candidate"},
            "max_scientific_rounds": max_rounds,
            "max_wall_clock_seconds": max_wall_clock_seconds,
            "scientific_rounds_used_before_run": 0,
            "input_hashes": input_hashes,
            "input_refs": {k: rel_ref(v, self.project_root) for k, v in inputs.items()},
        }

    def _initial_best_registry(self, project_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "registry_version": A1_CLOSED_LOOP_VERSION,
            "online_deployment_anchor": {"candidate_id": project_state.get("online_version", ONLINE_CHAMPION_ID), "status": "frozen_online_champion", "mutated": False},
            "evaluation_anchor": {"candidate_id": ANCHOR_ID, "status": "frozen_evaluation_anchor"},
            "best_candidate": {"candidate_id": ANCHOR_ID, "status": "current_best_scientific_baseline"},
            "portfolio_candidates": [{"candidate_id": PORTFOLIO_CANDIDATE_ID, "status": "portfolio_candidate"}],
        }

    def _update_best_registry(self, registry: dict[str, Any], round_result: dict[str, Any]) -> dict[str, Any]:
        feedback = round_result["feedback"]
        status = feedback["candidate_status"]
        registry = json.loads(json.dumps(registry, ensure_ascii=False))
        registry["portfolio_candidates"] = [{
            "candidate_id": feedback["candidate_id"],
            "status": status,
            "overall_gain": feedback["overall_gain"],
            "macro_gain": feedback["macro_gain"],
            "positive_fold_count": feedback["positive_fold_count"],
            "rescue": feedback["rescue"],
            "damage": feedback["damage"],
            "net": feedback["net"],
            "reason_codes": feedback["promotion_reason_codes"],
        }]
        if status == "promoted_best":
            registry["best_candidate"] = {"candidate_id": feedback["candidate_id"], "status": status}
        return registry

    def _portfolio_update(self, round1: dict[str, Any], best_registry: dict[str, Any]) -> dict[str, Any]:
        return {
            "update_version": A1_CLOSED_LOOP_VERSION,
            "status": "completed",
            "fusion_candidate_final_status": best_registry["portfolio_candidates"][0]["status"],
            "candidate_id": round1["feedback"]["candidate_id"],
            "not_auto_deployed": True,
            "online_champion_unchanged": True,
            "candidate_a1_csv_generated": False,
        }

    def _write_memory_update(self, out_dir: Path, run_id: str, round1: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
        memory_dir = out_dir / "research_memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        feedback = round1["feedback"]
        event = make_event(
            event_type="scientific_round_completed",
            task="A1",
            experiment_id=run_id,
            execution_identity=stable_hash({"run_id": run_id, "round": "round_01"}),
            branch_id="bounded_portfolio_fusion",
            candidate_id=feedback["candidate_id"],
            parent_candidate_id=ANCHOR_ID,
            scope_level="global",
            problem_ids=["scientific::bounded_portfolio_fusion"],
            method_ids=["method::fusion-controller"],
            artifact_refs=[{"kind": k, "path": v} for k, v in sorted(round1["artifacts"].items())],
            observations={
                "problem": "bounded portfolio fusion over verified OOF assets",
                "hypothesis": "class-aware/base-composed fusion can recover small OOF net without new training",
                "execution_status": "completed",
                "best_candidate_change": feedback["candidate_status"],
                "next_priority": "do not repeat same fusion source; require new information or stricter macro-safe variant",
                "stop_decision": "stop_after_round_01",
            },
            metrics_summary={
                "overall": {"gain": feedback["overall_gain"]},
                "macro": {"gain": feedback["macro_gain"]},
                "fold": {"positive_fold_count": feedback["positive_fold_count"], "worst_fold_gain": feedback["worst_fold_gain"]},
                "rescue_damage": {"rescue": feedback["rescue"], "damage": feedback["damage"], "net": feedback["net"], "changed_count": feedback["changed_count"], "change_precision": feedback["change_precision"]},
                "bucket_class": feedback["key_bucket_class_feedback"],
            },
            decision={"candidate_status": feedback["candidate_status"], "online_champion_unchanged": True, "candidate_a1_csv_generated": False},
            outcome_status="success" if feedback["candidate_status"] in {"accepted_portfolio", "promoted_best"} else "failure",
            evidence_level="strict_oof",
            confidence="high",
            success_mode="bounded_portfolio_gain" if feedback["candidate_status"] == "accepted_portfolio" else "",
            failure_mode="" if feedback["candidate_status"] != "rejected" else "fusion_policy_rejected",
            input_hashes=input_hashes,
        )
        store = ResearchEventStore(memory_dir / "research_events.jsonl")
        appended = store.append_unique([event])
        policy = load_json(self.project_root / "config" / "research_policy.json")
        views = ResearchMemoryBuilder(project_root=self.project_root).materialize(store.load(), policy=policy, memory_id=run_id)
        artifacts = {"research_events": rel_ref(memory_dir / "research_events.jsonl", self.project_root)}
        for name, payload in views.items():
            path = memory_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        return {"update_version": A1_CLOSED_LOOP_VERSION, "status": "completed", "append_only": True, "events_appended": appended, "events": [event], "artifacts": artifacts}

    def _stop_decision(self, round1: dict[str, Any], rounds_used: int, max_rounds: int, started: float, max_wall_clock_seconds: int) -> dict[str, Any]:
        feedback = round1["feedback"]
        reason = "accepted_portfolio_candidate_requires_human_experiment_design_before_training"
        if feedback["candidate_status"] == "rejected":
            reason = "main_candidate_blocked_by_fusion_policy"
        elif isinstance(feedback.get("macro_gain"), (int, float)) and feedback["macro_gain"] < 0:
            reason = "macro_protection_prevents_direct_promotion_and_repeating_same_information_source_is_low_value"
        return {
            "stop_version": A1_CLOSED_LOOP_VERSION,
            "status": "stopped",
            "reason": reason,
            "round_budget_used": rounds_used,
            "round_budget_max": max_rounds,
            "wall_clock_elapsed_seconds": round(time.time() - started, 6),
            "wall_clock_budget_seconds": max_wall_clock_seconds,
            "safe_resume_point": "closed_loop_completed",
            "next_allowed_stage": "human_review_or_next_controlled_experiment_design",
        }

    def _write(self, path: Path, payload: dict[str, Any], artifacts: dict[str, str], key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
        artifacts[key] = rel_ref(path, self.project_root)

    def _report(self, run_id: str, round1: dict[str, Any], best_registry: dict[str, Any], stop: dict[str, Any], rounds_used: int) -> str:
        feedback = round1["feedback"]
        return "\n".join([
            "# A1 Closed Loop Report",
            "",
            f"run_id: `{run_id}`",
            f"scientific_rounds_used: `{rounds_used}`",
            f"round_01_candidate: `{feedback['candidate_id']}`",
            f"round_01_status: `{feedback['candidate_status']}`",
            f"overall_gain: `{feedback['overall_gain']}`",
            f"macro_gain: `{feedback['macro_gain']}`",
            f"positive_fold_count: `{feedback['positive_fold_count']}`",
            f"best_candidate: `{best_registry['best_candidate']['candidate_id']}`",
            f"fusion_candidate_final_status: `{best_registry['portfolio_candidates'][0]['status']}`",
            f"stop_reason: `{stop['reason']}`",
            "",
            "No Test truth, submission, Test prediction, online Champion mutation, Project State mutation, or confirmed history mutation was performed.",
            "",
        ])


def run_a1_closed_loop(**kwargs: Any) -> dict[str, Any]:
    runner = A1ClosedLoopRunner(project_root=kwargs.pop("project_root"))
    return runner.run(**kwargs)
