# -*- coding: utf-8 -*-
"""M7A single-round closed-loop dry-run orchestration.

This module composes already-produced planning/research/decision artifacts into
an auditable one-round preview. It never executes adapters, training, prediction,
submission, state mutation, or history mutation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .research.event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash

M7A_VERSION = "m7a_single_round_dry_run_v1"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class M7DryRunOrchestrator:
    def __init__(self, *, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(
        self,
        *,
        decision_run: str | Path,
        project_state: str | Path = "config/project_state.json",
        tool_registry: str | Path = "config/tool_registry.json",
        out_root: str | Path = "artifacts/m7_dry_runs",
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        inputs = {
            "decision_run": self._resolve(decision_run),
            "project_state": self._resolve(project_state),
            "tool_registry": self._resolve(tool_registry),
        }
        missing = [name for name, path in inputs.items() if not path.exists()]
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        required_decision_files = [
            inputs["decision_run"] / "decision_manifest.json",
            inputs["decision_run"] / "m6c_revised_proposals.json",
            inputs["decision_run"] / "m5_admission_decision.json",
        ]
        missing_files = [rel_ref(path, self.project_root) for path in required_decision_files if not path.exists()]
        if missing_files:
            return {"status": "waiting_for_input", "failure_reason": "missing_decision_artifacts", "missing_inputs": missing_files, "artifacts": {}}
        run_id = stable_hash({"version": M7A_VERSION, "inputs": {k: self._hash_path(v) for k, v in inputs.items()}})
        out_dir = self._resolve(out_root) / run_id
        manifest_path = out_dir / "dry_run_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._execute(inputs=inputs, out_dir=out_dir, run_id=run_id)
        manifest = load_json(out_dir / "dry_run_manifest.json")
        return {"status": manifest.get("status"), "run_id": run_id, "artifacts": artifacts}

    def _execute(self, *, inputs: dict[str, Path], out_dir: Path, run_id: str) -> dict[str, str]:
        started = time.time()
        decision_manifest = load_json(inputs["decision_run"] / "decision_manifest.json")
        final = load_json(inputs["decision_run"] / "m6c_revised_proposals.json")
        admission = load_json(inputs["decision_run"] / "m5_admission_decision.json")
        state = load_json(inputs["project_state"])
        registry = load_json(inputs["tool_registry"])
        proposal = final.get("primary_proposal") if isinstance(final, dict) else None
        tools = {str(tool.get("name")): tool for tool in registry.get("tools", []) if isinstance(tool, dict)}
        tool_name = str((proposal or {}).get("adapter_or_tool") or "")
        tool = tools.get(tool_name) if tool_name else None
        adapter_preview = self._adapter_preview(proposal, admission, tool_name, tool)
        evaluation_preview = self._evaluation_preview(proposal)
        trajectory_preview = self._trajectory_preview(run_id, proposal, admission, adapter_preview, evaluation_preview)
        status = self._status(proposal, admission, adapter_preview, state)
        input_manifest = {
            "manifest_version": M7A_VERSION,
            "run_id": run_id,
            "input_hashes": {name: self._hash_path(path) for name, path in inputs.items()},
            "decision_run": rel_ref(inputs["decision_run"], self.project_root),
            "project_state": rel_ref(inputs["project_state"], self.project_root),
            "tool_registry": rel_ref(inputs["tool_registry"], self.project_root),
            "explicit_replay": True,
        }
        research_memory_preview = {
            "preview_version": M7A_VERSION,
            "would_update_memory": False,
            "feedback_observed": False,
            "source": "M7A dry-run preview only",
            "candidate_record": {
                "decision_run_id": decision_manifest.get("run_id"),
                "admission_status": admission.get("status"),
                "proposal_id": (proposal or {}).get("proposal_id"),
            },
        }
        selected_queue_item = {
            "selection_version": M7A_VERSION,
            "selected_from_existing_decision": True,
            "target_problem_ids": (proposal or {}).get("target_problem_ids", []),
            "target_scope": (proposal or {}).get("target_scope", {}),
            "status": "selected_preview" if proposal else "unavailable",
        }
        method_research_selection = {
            "selection_version": M7A_VERSION,
            "reuses_existing_method_research": True,
            "reuses_existing_m6_decision": True,
            "qwen_called": False,
            "decision_run_id": decision_manifest.get("run_id"),
            "method_eligibility_ref": rel_ref(inputs["decision_run"] / "m6b_method_eligibility.json", self.project_root),
        }
        decision_reference = {
            "reference_version": M7A_VERSION,
            "decision_run_id": decision_manifest.get("run_id"),
            "decision_manifest_ref": rel_ref(inputs["decision_run"] / "decision_manifest.json", self.project_root),
            "m6_final_status": final.get("status"),
            "m5_admission_status": admission.get("status"),
            "m5_authoritative": admission.get("m5_authoritative") is True,
        }
        payloads = {
            "input_manifest": input_manifest,
            "research_memory_update_preview": research_memory_preview,
            "selected_queue_item": selected_queue_item,
            "method_research_selection": method_research_selection,
            "m6_decision_reference": decision_reference,
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
        report = out_dir / "DRY_RUN_REPORT.md"
        report.write_text(self._report(status, adapter_preview, evaluation_preview, trajectory_preview), encoding="utf-8")
        artifacts["dry_run_report"] = rel_ref(report, self.project_root)
        manifest = {
            "run_version": M7A_VERSION,
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
            "experiment_executed": False,
            "feedback_observed": False,
            "prediction_generated": False,
            "round_consumed": False,
            "view_hash": stable_hash(payloads),
            "artifacts": artifacts,
        }
        manifest_path = out_dir / "dry_run_manifest.json"
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["dry_run_manifest"] = rel_ref(manifest_path, self.project_root)
        return artifacts

    def _adapter_preview(self, proposal: dict[str, Any] | None, admission: dict[str, Any], tool_name: str, tool: dict[str, Any] | None) -> dict[str, Any]:
        required = list(_as_list((proposal or {}).get("required_inputs")))
        missing = list(_as_list((proposal or {}).get("missing_inputs")))
        if tool:
            required.extend(sorted((tool.get("required_inputs") or {}).keys()))
        if tool_name and tool is None:
            missing.append(f"unregistered_adapter:{tool_name}")
        return {
            "preview_version": M7A_VERSION,
            "tool_id": tool_name,
            "adapter_id": (tool or {}).get("adapter_entrypoint", ""),
            "command_template_preview": list((tool or {}).get("command_template", [])),
            "required_inputs": sorted(set(map(str, required))),
            "missing_inputs": sorted(set(map(str, missing))),
            "expected_outputs": (tool or {}).get("expected_outputs", {}),
            "estimated_runtime": (proposal or {}).get("estimated_runtime", (tool or {}).get("expected_runtime_seconds", "")),
            "estimated_gpu_memory": (proposal or {}).get("estimated_gpu_memory", "0GB"),
            "round_cost": (proposal or {}).get("round_cost", 0),
            "admission_status": admission.get("status"),
            "execution_allowed": False,
            "experiment_executed": False,
            "training_started": False,
            "prediction_generated": False,
            "submission_created": False,
        }

    def _evaluation_preview(self, proposal: dict[str, Any] | None) -> dict[str, Any]:
        proposal = proposal or {}
        return {
            "preview_version": M7A_VERSION,
            "parent_identity": proposal.get("parent_candidate", ""),
            "candidate_identity_plan": proposal.get("proposal_id", ""),
            "fold_plan": (proposal.get("oof_evaluation_plan") or proposal.get("OOF_evaluation_plan") or {}),
            "overall_metrics": ["accuracy", "macro_f1"],
            "macro_metrics": ["macro_accuracy", "macro_f1"],
            "bucket_metrics": proposal.get("bucket_metrics", []),
            "bucket_class_metrics": proposal.get("bucket_class_metrics", []),
            "rescue_damage_net": {"status": "preview_only", "test_truth_used": False},
            "success_conditions": proposal.get("success_conditions", []),
            "failure_conditions": proposal.get("failure_conditions", []),
            "stop_conditions": proposal.get("stop_conditions", []),
            "evaluation_executed": False,
        }

    def _trajectory_preview(self, run_id: str, proposal: dict[str, Any] | None, admission: dict[str, Any], adapter_preview: dict[str, Any], evaluation_preview: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_version": M7A_VERSION,
            "event_id": stable_hash({"m7a": run_id, "proposal": (proposal or {}).get("proposal_id")}),
            "event_type": "closed_loop_dry_run_preview",
            "proposal_id": (proposal or {}).get("proposal_id", ""),
            "admission_status": admission.get("status"),
            "adapter_execution_preview_status": "not_executed",
            "evaluation_preview_status": "not_executed",
            "experiment_executed": False,
            "feedback_observed": False,
            "prediction_generated": False,
            "round_consumed": False,
            "adapter_preview_hash": stable_hash(adapter_preview),
            "evaluation_preview_hash": stable_hash(evaluation_preview),
        }

    def _status(self, proposal: dict[str, Any] | None, admission: dict[str, Any], adapter_preview: dict[str, Any], state: dict[str, Any]) -> str:
        admission_status = str(admission.get("status") or "")
        if not proposal or admission_status == "blocked":
            return "blocked"
        if adapter_preview["missing_inputs"] or admission_status == "waiting_for_input":
            return "waiting_for_input"
        budget = state.get("budget", {})
        rounds_left = int(budget.get("max_rounds", 0)) - int(budget.get("rounds_used", 0))
        try:
            round_cost = int(adapter_preview.get("round_cost") or 0)
        except (TypeError, ValueError):
            round_cost = 0
        if round_cost > rounds_left:
            return "blocked"
        if admission_status == "admitted_diagnostic_only":
            return "completed_dry_run"
        if admission_status == "admitted":
            return "ready_for_human_approval"
        if admission_status == "deferred":
            return "blocked"
        return "invalid"

    def _report(self, status: str, adapter: dict[str, Any], evaluation: dict[str, Any], trajectory: dict[str, Any]) -> str:
        return "\n".join([
            "# M7A Single-Round Closed Loop Dry-Run",
            "",
            f"status: `{status}`",
            f"tool_id: `{adapter.get('tool_id')}`",
            f"execution_allowed: `{adapter.get('execution_allowed')}`",
            f"evaluation_executed: `{evaluation.get('evaluation_executed')}`",
            f"trajectory_event_type: `{trajectory.get('event_type')}`",
            "",
            "No Adapter/training/prediction/submission was executed.",
            "",
        ])

    def _hash_path(self, path: Path) -> str:
        if path.is_file():
            return sha256_file(path)
        if path.is_dir():
            for name in ["dry_run_manifest.json", "decision_manifest.json", "research_manifest.json"]:
                candidate = path / name
                if candidate.exists():
                    return sha256_file(candidate)
            return stable_hash(sorted(p.name for p in path.iterdir()))
        return ""

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()
