# -*- coding: utf-8 -*-
"""M7B readiness repair: anchor recovery and scientific priority repair.

This is a read-only readiness layer. It inventories and verifies existing
anchor inputs, ranks scientific problems with blocker-aware priority, and emits
proposal/admission/dry-run previews. It never rebuilds folds, materializes OOF,
executes adapters, trains models, generates predictions, or consumes rounds.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .paths import PathResolver
from .research.event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash

M7B_VERSION = "m7b_readiness_repair_v1"
TRAIN_NODE_COUNT = 11001
TEST_NODE_COUNT = 2751
NUM_CLASSES = 10


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class M7BReadinessRepair:
    def __init__(self, *, project_root: str | Path, paths_config: str = "") -> None:
        self.project_root = Path(project_root).resolve()
        self.resolver = PathResolver(self.project_root, paths_config or None)

    def run(
        self,
        *,
        problem_map: str | Path = "artifacts/data_profile/a1_m2_v1/a1_problem_map.json",
        data_profile: str | Path = "artifacts/data_profile/a1_m2_v1/a1_data_profile.json",
        method_research_run: str | Path = "",
        decision_run: str | Path = "",
        m7a_run: str | Path = "",
        project_state: str | Path = "config/project_state.json",
        tool_registry: str | Path = "config/tool_registry.json",
        out_root: str | Path = "artifacts/m7b_readiness",
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        inputs = {
            "problem_map": self._resolve(problem_map),
            "data_profile": self._resolve(data_profile),
            "project_state": self._resolve(project_state),
            "tool_registry": self._resolve(tool_registry),
        }
        optional = {
            "method_research_run": self._resolve(method_research_run) if method_research_run else None,
            "decision_run": self._resolve(decision_run) if decision_run else None,
            "m7a_run": self._resolve(m7a_run) if m7a_run else None,
        }
        missing = [name for name, path in inputs.items() if not path.exists()]
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        run_id = stable_hash({
            "version": M7B_VERSION,
            "inputs": {k: self._hash_path(v) for k, v in inputs.items()},
            "optional": {k: self._hash_path(v) for k, v in optional.items() if v and v.exists()},
        })
        out_dir = self._resolve(out_root) / run_id
        manifest_path = out_dir / "readiness_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._execute(inputs=inputs, optional=optional, out_dir=out_dir, run_id=run_id)
        manifest = load_json(manifest_path)
        return {"status": manifest.get("status"), "run_id": run_id, "artifacts": artifacts}

    def _execute(self, *, inputs: dict[str, Path], optional: dict[str, Path | None], out_dir: Path, run_id: str) -> dict[str, str]:
        started = time.time()
        problem_map = load_json(inputs["problem_map"])
        data_profile = load_json(inputs["data_profile"])
        state = load_json(inputs["project_state"])
        registry = load_json(inputs["tool_registry"])
        inventory = self._anchor_inventory(data_profile)
        fold_verification = self._verify_fold(inventory.get("canonical_evaluation_fold", {}).get("path", ""))
        oof_verification = self._verify_oof(inventory.get("oof_evaluation_anchor", {}).get("path", ""))
        anchor_decision = self._anchor_decision(inventory, fold_verification, oof_verification)
        queue = self._scientific_queue(problem_map, data_profile, anchor_decision)
        selected = self._select_problem(queue, anchor_decision)
        research_bundle = self._research_bundle(optional.get("method_research_run"))
        proposal = self._proposal(selected, anchor_decision, research_bundle)
        critic = self._critic(proposal, anchor_decision, research_bundle)
        final_proposal = self._revise(proposal, critic, anchor_decision)
        admission = self._admit(final_proposal, anchor_decision, registry, state)
        adapter_preview = self._adapter_preview(final_proposal, admission, registry)
        evaluation_preview = self._evaluation_preview(final_proposal, anchor_decision)
        trajectory_preview = self._trajectory_preview(run_id, admission, adapter_preview, evaluation_preview)
        status = self._status(anchor_decision, admission, adapter_preview)
        payloads = {
            "anchor_input_inventory": inventory,
            "canonical_fold_verification": fold_verification,
            "v53q1_oof_verification": oof_verification,
            "anchor_recovery_decision": anchor_decision,
            "scientific_research_queue": queue,
            "selected_scientific_problem": selected,
            "research_queries": research_bundle["research_queries"],
            "source_relevance_audit": research_bundle["source_relevance_audit"],
            "method_cards_validated": research_bundle["method_cards_validated"],
            "method_conflicts": research_bundle["method_conflicts"],
            "method_ranking": research_bundle["method_ranking"],
            "m6b_proposals": {"proposal_version": M7B_VERSION, "primary_proposal": proposal, "fallback_proposal": None},
            "m6c_critic_review": critic,
            "m5_admission": admission,
            "adapter_execution_preview": adapter_preview,
            "evaluation_plan_preview": evaluation_preview,
            "trajectory_preview": trajectory_preview,
        }
        artifacts: dict[str, str] = {}
        for name, payload in payloads.items():
            path = out_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        report = out_dir / "M7B_READINESS_REPORT.md"
        report.write_text(self._report(status, anchor_decision, selected, admission, adapter_preview), encoding="utf-8")
        artifacts["readiness_report"] = rel_ref(report, self.project_root)
        manifest = {
            "run_version": M7B_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "read_only": True,
            "executes_adapter": False,
            "trains_model": False,
            "generates_prediction": False,
            "creates_submission": False,
            "mutates_project_state": False,
            "mutates_history": False,
            "counts_as_experiment_round": False,
            "round_consumed": False,
            "anchor_status": anchor_decision["status"],
            "full_anchor_evaluator_status": evaluation_preview["full_anchor_evaluator_status"],
            "view_hash": stable_hash(payloads),
            "artifacts": artifacts,
        }
        manifest_path = out_dir / "readiness_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["readiness_manifest"] = rel_ref(manifest_path, self.project_root)
        return artifacts

    def _anchor_inventory(self, data_profile: dict[str, Any]) -> dict[str, Any]:
        fold = self.resolver.a1_fold_file() or (self.project_root / "artifacts" / "evaluation_anchor" / "AFAC_A1_FOLD_V1.csv")
        eval_manifest = self.project_root / "artifacts" / "evaluation_anchor" / "A1_EVAL_ANCHOR_V1_manifest.json"
        eval_oof = self._eval_oof_from_manifest(eval_manifest)
        ref_oof = self.resolver.a1_reference_oof_npz()
        anchor_csv = self.resolver.a1_anchor_csv()
        candidates = {
            "online_deployment_anchor": self._candidate_record(anchor_csv, "config.a1.anchor_csv"),
            "canonical_evaluation_fold": self._candidate_record(fold, "AFAC_A1_FOLD_V1"),
            "oof_evaluation_anchor_manifest": self._candidate_record(eval_manifest, "A1_EVAL_ANCHOR_V1_manifest"),
            "oof_evaluation_anchor": self._candidate_record(eval_oof, "A1_EVAL_ANCHOR_V1_oof"),
            "reference_oof": self._candidate_record(ref_oof, "config.a1.reference_oof_npz"),
            "champion_csv": self._candidate_record(anchor_csv, "config.a1.anchor_csv"),
        }
        candidates["dataset_profile_anchor_identity"] = data_profile.get("anchor_identity")
        candidates["data_profile_oof_status"] = data_profile.get("oof_profile", {}).get("status")
        candidates["historical_v53q1_oof_status"] = "not_materialized"
        candidates["known_missing_inputs"] = ["verified_oof_evaluation_anchor", "verified_canonical_evaluation_fold"]
        return candidates

    def _eval_oof_from_manifest(self, manifest_path: Path) -> Path | None:
        if not manifest_path.exists():
            return None
        try:
            payload = load_json(manifest_path)
            artifact = payload.get("artifact", "")
            return self._resolve(artifact) if artifact else None
        except Exception:
            return None

    def _candidate_record(self, path: Path | None, source: str) -> dict[str, Any]:
        exists = bool(path and path.exists())
        return {
            "source": source,
            "path": rel_ref(path, self.project_root) if path else "",
            "exists": exists,
            "sha256": sha256_file(path) if exists and path and path.is_file() else "",
            "size_bytes": path.stat().st_size if exists and path and path.is_file() else 0,
        }

    def _verify_fold(self, path_text: str) -> dict[str, Any]:
        path = self._resolve(path_text) if path_text else Path("")
        result: dict[str, Any] = {
            "verification_version": M7B_VERSION,
            "status": "unavailable",
            "path": path_text,
            "is_canonical": False,
            "contains_11001_train_idx": False,
            "contains_fold_assignment": False,
            "fold_count": 0,
            "duplicate_or_missing_rows": "unknown",
            "reason": "canonical_evaluation_fold_missing",
        }
        if not path_text or not path.exists() or not path.is_file():
            return result
        result.update({"status": "candidate_found", "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                headers = set(rows[0]) if rows else set()
                fold_key = next((k for k in ["fold", "fold_id", "fold_idx"] if k in headers), "")
                node_key = next((k for k in ["train_idx", "node_idx", "node_id", "idx"] if k in headers), "")
                result["contains_fold_assignment"] = bool(fold_key)
                result["contains_11001_train_idx"] = len(rows) == TRAIN_NODE_COUNT and bool(node_key)
                result["fold_count"] = len({row.get(fold_key, "") for row in rows}) if fold_key else 0
                result["duplicate_or_missing_rows"] = "pass" if node_key and len({row.get(node_key, "") for row in rows}) == len(rows) == TRAIN_NODE_COUNT else "fail"
            elif suffix in {".npy", ".npz"}:
                import numpy as np
                payload = np.load(path, allow_pickle=False)
                if hasattr(payload, "files"):
                    keys = list(payload.files)
                    arr = payload[keys[0]] if keys else None
                    result["keys"] = keys
                else:
                    arr = payload
                shape = list(getattr(arr, "shape", []))
                result["shape"] = shape
                result["contains_fold_assignment"] = bool(shape and shape[0] == TRAIN_NODE_COUNT)
                result["contains_11001_train_idx"] = bool(shape and shape[0] == TRAIN_NODE_COUNT)
                result["fold_count"] = len(set(map(int, arr.tolist()))) if arr is not None and getattr(arr, "ndim", 0) == 1 else 0
            if result["contains_11001_train_idx"] and result["contains_fold_assignment"] and result["fold_count"] > 1:
                result["status"] = "verified_existing"
                result["is_canonical"] = True
                result["reason"] = "verified_existing_candidate"
            else:
                result["status"] = "conflicting_candidates"
                result["reason"] = "candidate_does_not_satisfy_canonical_fold_contract"
        except Exception as exc:
            result["status"] = "conflicting_candidates"
            result["reason"] = f"verification_error:{type(exc).__name__}"
        return result

    def _verify_oof(self, path_text: str) -> dict[str, Any]:
        path = self._resolve(path_text) if path_text else Path("")
        result: dict[str, Any] = {
            "verification_version": M7B_VERSION,
            "status": "unavailable",
            "path": path_text,
            "is_oof_evaluation_anchor": False,
            "contains_11001_train_idx": False,
            "contains_10_class_proba": False,
            "probability_normalization": "unknown",
            "node_ordering": "unknown",
            "provenance_chain": "missing",
            "reason": "oof_evaluation_anchor_missing",
        }
        if not path_text or not path.exists() or not path.is_file():
            return result
        result.update({"status": "candidate_found", "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        try:
            import numpy as np
            payload = np.load(path, allow_pickle=False)
            keys = list(payload.files) if hasattr(payload, "files") else []
            arr = None
            if hasattr(payload, "files"):
                for key in ["proba", "oof_proba", "pred_proba", "probabilities"]:
                    if key in keys:
                        arr = payload[key]
                        break
                if arr is None and keys:
                    arr = payload[keys[0]]
            else:
                arr = payload
            shape = list(getattr(arr, "shape", []))
            result["keys"] = keys
            result["shape"] = shape
            result["contains_11001_train_idx"] = bool(shape and shape[0] == TRAIN_NODE_COUNT)
            result["contains_10_class_proba"] = bool(len(shape) == 2 and shape[1] == NUM_CLASSES)
            if result["contains_10_class_proba"]:
                sums = arr.sum(axis=1)
                result["probability_normalization"] = "pass" if bool(((sums > 0.999) & (sums < 1.001)).all()) else "fail"
            name_text = path.name.lower()
            result["is_oof_evaluation_anchor"] = "eval_anchor" in name_text and "oof" in name_text and result["contains_10_class_proba"]
            if result["is_oof_evaluation_anchor"] and result["contains_11001_train_idx"] and result["probability_normalization"] == "pass":
                result["status"] = "verified_existing"
                result["reason"] = "verified_existing_candidate"
            else:
                result["status"] = "conflicting_candidates"
                result["reason"] = "candidate_does_not_satisfy_oof_evaluation_anchor_contract"
        except Exception as exc:
            result["status"] = "conflicting_candidates"
            result["reason"] = f"verification_error:{type(exc).__name__}"
        return result

    def _anchor_decision(self, inventory: dict[str, Any], fold: dict[str, Any], oof: dict[str, Any]) -> dict[str, Any]:
        missing = []
        if fold["status"] != "verified_existing":
            missing.append("verified_canonical_evaluation_fold")
        if oof["status"] != "verified_existing":
            missing.append("verified_oof_evaluation_anchor")
        champion_ok = inventory.get("champion_csv", {}).get("exists") is True
        if not champion_ok:
            missing.append("online_deployment_anchor")
        status = "verified_existing" if not missing else "unavailable"
        if missing and champion_ok:
            status = "rebuild_required"
        return {
            "decision_version": M7B_VERSION,
            "status": status,
            "missing_inputs": sorted(set(missing)),
            "verified_inputs": {
                "online_deployment_anchor": champion_ok,
                "canonical_evaluation_fold": fold["status"] == "verified_existing",
                "oof_evaluation_anchor": oof["status"] == "verified_existing",
            },
            "may_clear_m7a_missing_input": status in {"verified_existing", "materialized_from_verified_components"},
            "rebuild_specification": {
                "required": bool(missing),
                "allowed_in_m7b": False,
                "reason": "M7B readiness may specify but must not train or rebuild the evaluation anchor without approval",
            },
            "historical_v53q1_oof_status": "not_materialized",
        }

    def _scientific_queue(self, problem_map: dict[str, Any], data_profile: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
        total = _safe_float(data_profile.get("dataset", {}).get("num_nodes"), 1.0) or 1.0
        items = []
        infrastructure_blockers = []
        if anchor["status"] != "verified_existing":
            infrastructure_blockers.append({
                "blocker_id": "anchor_recovery::missing_full_anchor_inputs",
                "scope": "evaluation_infrastructure",
                "status": "waiting_for_anchor_rebuild_approval" if anchor["status"] == "rebuild_required" else "waiting_for_input",
                "missing_inputs": anchor["missing_inputs"],
                "rationale": "Verified offline evaluation anchor is required before executable OOF comparison, but this is infrastructure, not a scientific model problem.",
            })
        for problem in _as_list(problem_map.get("problems")):
            node_count = _safe_float(problem.get("node_count"), 0.0)
            headroom_unknown = problem.get("error_count") is None and problem.get("error_rate") is None
            components = {
                "evidence_strength": 0.8 if problem.get("structure_evidence") == "observed" else 0.3,
                "observed_error_headroom": 0.45 if headroom_unknown else min(1.0, _safe_float(problem.get("error_rate"), 0.0) * 2.0),
                "macro_importance": 0.7 if "class" in str(problem.get("problem_id", "")).lower() else 0.5,
                "mechanism_specificity": 0.8 if "::" in str(problem.get("problem_id", "")) else 0.4,
                "new_information_gap": 0.9 if problem.get("confidence_evidence") == "unavailable_without_anchor_oof" else 0.5,
                "local_failure_explanation": 0.5,
                "method_researchability": 0.65,
                "expected_information_gain": 0.75,
                "implementation_feasibility": 0.55 if problem.get("eligible_tool_families") else 0.35,
                "scope_size": min(1.0, node_count / total),
                "overlap_penalty": 0.25 if problem.get("blocked_tool_families") else 0.0,
            }
            score = self._score_components(components)
            items.append(self._queue_item(str(problem.get("problem_id")), str(problem.get("scope")), score, components, [], "dataset_only structural evidence; OOF headroom unknown is retained as uncertainty, not zeroed."))
        items.sort(key=lambda item: (-item["scientific_priority_score"], item["problem_id"]))
        for rank, item in enumerate(items, 1):
            item["rank"] = rank
        return {
            "queue_version": M7B_VERSION,
            "ranking_policy": "scientific_priority_not_coverage_only",
            "items": items,
            "infrastructure_blockers": infrastructure_blockers,
            "view_hash": stable_hash({"items": items, "infrastructure_blockers": infrastructure_blockers}),
        }

    def _queue_item(self, problem_id: str, scope: str, score: float, components: dict[str, float], missing: list[str], rationale: str) -> dict[str, Any]:
        return {
            "problem_id": problem_id,
            "scope": scope,
            "scientific_priority_score": round(score, 6),
            "component_breakdown": {k: round(v, 6) for k, v in components.items()},
            "missing_inputs": missing,
            "rationale": rationale,
        }

    def _score_components(self, c: dict[str, float]) -> float:
        return (
            0.13 * c["evidence_strength"]
            + 0.11 * c["observed_error_headroom"]
            + 0.09 * c["macro_importance"]
            + 0.12 * c["mechanism_specificity"]
            + 0.13 * c["new_information_gap"]
            + 0.08 * c["local_failure_explanation"]
            + 0.10 * c["method_researchability"]
            + 0.12 * c["expected_information_gain"]
            + 0.08 * c["implementation_feasibility"]
            + 0.04 * c["scope_size"]
            - 0.15 * c["overlap_penalty"]
        )

    def _select_problem(self, queue: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
        items = _as_list(queue.get("items"))
        selected = items[0] if items else {}
        return {
            "selection_version": M7B_VERSION,
            "status": "selected" if selected else "no_scientific_problem_available",
            "selected_problem": selected,
            "selection_reason": "highest scientific priority score; infrastructure blockers are tracked separately from scientific queue",
        }

    def _research_bundle(self, run: Path | None) -> dict[str, Any]:
        def read(name: str, default: Any) -> Any:
            path = (run / name) if run else Path("")
            return load_json(path) if path.exists() else default

        return {
            "research_queries": read("research_queries.json", {"items": [], "status": "unavailable"}),
            "source_relevance_audit": read("source_relevance_audit.json", {"items": [], "status": "unavailable"}),
            "method_cards_validated": read("method_cards_validated.json", {"items": [], "validations": []}),
            "method_conflicts": read("method_conflicts.json", {"items": []}),
            "method_ranking": read("method_ranking.json", {"items": []}),
        }

    def _proposal(self, selected: dict[str, Any], anchor: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
        method_count = len(_as_list(research.get("method_cards_validated", {}).get("items")))
        missing = list(anchor["missing_inputs"])
        return {
            "proposal_id": stable_hash({"m7b": selected, "anchor": anchor["status"]}),
            "target_problem_ids": [selected.get("selected_problem", {}).get("problem_id", "no_scientific_problem_selected")],
            "target_scope": selected.get("selected_problem", {}).get("scope", "dataset_problem"),
            "core_hypothesis": "Use a verified offline evaluation anchor and canonical evaluation fold before any executable OOF model comparison.",
            "expected_new_information": "Whether the candidate OOF improves over the offline evaluation parent under the same verified evaluation protocol.",
            "selected_method_components": [],
            "method_card_count": method_count,
            "adapter_or_tool": "A1_OOF_CANDIDATE_EVALUATOR" if anchor["status"] == "verified_existing" else "",
            "required_inputs": [
                "verified_oof_evaluation_anchor",
                "verified_canonical_evaluation_fold",
                "candidate_oof",
                "same_evaluation_protocol",
                "registered_adapter",
                "complete_oof_plan",
            ],
            "missing_inputs": missing,
            "minimal_experiment": {"mode": "offline_oof_evaluation_preview" if not missing else "evaluation_anchor_rebuild_diagnostic"},
            "oof_evaluation_plan": {
                "fold_aware": True,
                "requires_oof_evaluation_anchor": True,
                "requires_final_v53q1_oof": False,
                "same_evaluation_protocol": True,
                "test_truth_used": False,
                "evaluation_parent": "A1_EVAL_ANCHOR_V1",
                "deployment_parent": "A1_V53Q1_TRANSITION_STABLE_EDGE_H2",
            },
            "bucket_metrics": ["overall", "macro", "bucket", "bucket_class"],
            "success_conditions": ["verified canonical evaluation Fold", "verified offline OOF evaluation anchor", "M5 admits registered read-only evaluator"],
            "failure_conditions": ["conflicting evaluation-anchor candidates", "missing evaluation anchor identity or OOF manifest"],
            "stop_conditions": ["requires rebuilding Fold", "requires generating OOF", "requires test truth", "requires training"],
            "round_cost": 0,
            "risk_level": "low",
        }

    def _critic(self, proposal: dict[str, Any], anchor: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
        issues = []
        verdict = "approve"
        if anchor["status"] != "verified_existing":
            verdict = "revise"
            issues.append("oof_evaluation_anchor_unavailable")
        if not _as_list(research.get("method_cards_validated", {}).get("items")):
            issues.append("method_research_unavailable")
        return {
            "critic_version": M7B_VERSION,
            "verdict": verdict,
            "critical_issues": issues,
            "required_revisions": ["keep diagnostic-only readiness repair"] if issues else [],
            "unsupported_claims": [],
            "closed_branch_conflicts": [],
            "missing_evidence": anchor["missing_inputs"],
            "minimal_safe_revision": {"mode": "readiness_repair_diagnostic"} if issues else {},
        }

    def _revise(self, proposal: dict[str, Any], critic: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
        revised = dict(proposal)
        if critic["verdict"] == "revise" or anchor["status"] != "verified_existing":
            revised["adapter_or_tool"] = ""
            revised["minimal_experiment"] = {"mode": "evaluation_anchor_rebuild_diagnostic"}
            revised["round_cost"] = 0
            revised["revision_applied"] = "diagnostic_only_until_oof_evaluation_anchor_verified"
        else:
            revised["revision_applied"] = "none"
        return revised

    def _admit(self, proposal: dict[str, Any], anchor: dict[str, Any], registry: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        reasons = []
        if anchor["missing_inputs"]:
            reasons.extend(anchor["missing_inputs"])
        tool_name = proposal.get("adapter_or_tool", "")
        registered = {tool.get("name") for tool in registry.get("tools", []) if isinstance(tool, dict)}
        if tool_name and tool_name not in registered:
            reasons.append(f"unregistered_adapter:{tool_name}")
        budget = state.get("budget", {})
        if int(budget.get("rounds_used", 0)) >= int(budget.get("max_rounds", 0)):
            reasons.append("round_budget_exhausted")
        if not reasons and tool_name:
            status = "ready_for_human_approval"
        elif not reasons:
            status = "diagnostic_only"
        elif anchor["status"] == "rebuild_required":
            status = "waiting_for_anchor_rebuild_approval"
        else:
            status = "waiting_for_input"
        return {
            "admission_version": M7B_VERSION,
            "status": status,
            "reason_codes": sorted(set(reasons)) or ["ready"],
            "m5_authoritative": True,
            "requires_human_approval": status == "ready_for_human_approval",
            "auto_execution_allowed": False,
        }

    def _adapter_preview(self, proposal: dict[str, Any], admission: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(proposal.get("adapter_or_tool") or "")
        tool = next((item for item in registry.get("tools", []) if isinstance(item, dict) and item.get("name") == tool_name), {})
        return {
            "preview_version": M7B_VERSION,
            "tool_id": tool_name,
            "adapter_id": tool.get("adapter_entrypoint", ""),
            "command_template_preview": tool.get("command_template", []),
            "required_inputs": proposal.get("required_inputs", []),
            "missing_inputs": proposal.get("missing_inputs", []),
            "expected_outputs": tool.get("expected_outputs", {}),
            "estimated_runtime": tool.get("expected_runtime_seconds", ""),
            "estimated_gpu_memory": "0GB",
            "round_cost": proposal.get("round_cost", 0),
            "admission_status": admission["status"],
            "execution_allowed": False,
            "experiment_executed": False,
            "training_started": False,
            "prediction_generated": False,
            "submission_created": False,
        }

    def _evaluation_preview(self, proposal: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
        ready = anchor["status"] == "verified_existing"
        wait_status = "waiting_for_anchor_rebuild_approval" if anchor["status"] == "rebuild_required" else "waiting_for_input"
        return {
            "preview_version": M7B_VERSION,
            "full_anchor_evaluator_status": "ready" if ready else wait_status,
            "evaluation_anchor_evaluator_status": "ready" if ready else wait_status,
            "evaluation_parent": "A1_EVAL_ANCHOR_V1" if ready else "unverified_until_evaluation_anchor_rebuilt",
            "deployment_parent": "A1_V53Q1_TRANSITION_STABLE_EDGE_H2",
            "parent_identity": "A1_EVAL_ANCHOR_V1" if ready else "unverified_until_evaluation_anchor_rebuilt",
            "fold_plan": "canonical_fold_verified" if ready else "missing",
            "overall_metrics": ["accuracy", "macro_f1"] if ready else [],
            "macro_metrics": ["macro_f1"] if ready else [],
            "bucket_metrics": proposal.get("bucket_metrics", []),
            "bucket_class_metrics": ["bucket_class_rescue_damage_net"] if ready else [],
            "rescue_damage_net": {"status": "preview_only", "test_truth_used": False},
            "evaluation_executed": False,
        }

    def _trajectory_preview(self, run_id: str, admission: dict[str, Any], adapter: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_version": M7B_VERSION,
            "event_id": stable_hash({"m7b": run_id, "admission": admission["status"]}),
            "event_type": "readiness_repair_preview",
            "admission_status": admission["status"],
            "adapter_execution_preview_status": "not_executed",
            "evaluation_preview_status": evaluation["full_anchor_evaluator_status"],
            "experiment_executed": False,
            "feedback_observed": False,
            "prediction_generated": False,
            "round_consumed": False,
            "adapter_preview_hash": stable_hash(adapter),
            "evaluation_preview_hash": stable_hash(evaluation),
        }

    def _status(self, anchor: dict[str, Any], admission: dict[str, Any], adapter: dict[str, Any]) -> str:
        if admission["status"] == "ready_for_human_approval" and not adapter["missing_inputs"]:
            return "ready_for_human_approval"
        if admission["status"] == "diagnostic_only":
            return "diagnostic_only"
        if admission["status"] == "waiting_for_anchor_rebuild_approval":
            return "waiting_for_anchor_rebuild_approval"
        if admission["status"] == "waiting_for_input":
            return "waiting_for_input"
        return "blocked" if anchor["status"] == "conflicting_candidates" else "waiting_for_input"

    def _report(self, status: str, anchor: dict[str, Any], selected: dict[str, Any], admission: dict[str, Any], adapter: dict[str, Any]) -> str:
        return "\n".join([
            "# M7B Readiness Repair Report",
            "",
            f"status: `{status}`",
            f"anchor_status: `{anchor.get('status')}`",
            f"selected_problem: `{selected.get('selected_problem', {}).get('problem_id')}`",
            f"m5_admission: `{admission.get('status')}`",
            f"execution_allowed: `{adapter.get('execution_allowed')}`",
            "",
            "No Adapter/training/prediction/submission was executed.",
            "",
        ])

    def _hash_path(self, path: Path | None) -> str:
        if not path or not path.exists():
            return ""
        if path.is_file():
            return sha256_file(path)
        for name in ["readiness_manifest.json", "dry_run_manifest.json", "decision_manifest.json", "research_run_manifest.json"]:
            candidate = path / name
            if candidate.exists():
                return sha256_file(candidate)
        return stable_hash(sorted(p.name for p in path.iterdir()))

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()
