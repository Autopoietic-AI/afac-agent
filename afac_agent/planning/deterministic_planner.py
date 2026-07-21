# -*- coding: utf-8 -*-
"""M5A deterministic, read-only planner.

This planner produces a PlanDecision only. It never executes registered tools,
never calls adapters, and never mutates project state/history/champion files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from afac_agent.planning.duplicate_index import DuplicateIndex
from afac_agent.planning.evidence_loader import load_evidence, missing_paths
from afac_agent.planning.policy import load_policy
from afac_agent.planning.report import render_plan_report


PLANNER_VERSION = "m5a_v1"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_payload(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_stable_payload(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _stable_hash(payload: Any) -> str:
    import hashlib

    encoded = json.dumps(
        _stable_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action(
    *,
    action: str,
    status: str,
    reason_codes: list[str],
    tool: str | None = None,
    risk_level: str = "low",
    requires_human_approval: bool = False,
    auto_execution_allowed: bool = False,
    branch_id: str = "",
    evidence_tier: str = "",
) -> dict[str, Any]:
    return {
        "action": action,
        "status": status,
        "tool": tool,
        "tool_version": None,
        "branch_id": branch_id,
        "parent_branch_id": "",
        "experiment_identity": "",
        "action_identity": _stable_hash(
            {
                "action": action,
                "tool": tool,
                "branch_id": branch_id,
                "reason_codes": sorted(reason_codes),
            }
        ),
        "evidence_tier": evidence_tier,
        "reason_codes": sorted(set(reason_codes)),
        "risk_level": risk_level,
        "requires_human_approval": requires_human_approval,
        "auto_execution_allowed": auto_execution_allowed,
    }


class DeterministicPlanner:
    def __init__(self, *, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def plan(
        self,
        *,
        problem_map_path: str | Path,
        feedback_paths: list[str | Path],
        tool_registry_path: str | Path,
        project_state_path: str | Path,
        history_path: str | Path,
        policy_path: str | Path,
        out_root: str | Path,
        dry_run: bool = False,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        problem_map = Path(problem_map_path)
        feedbacks = [Path(path) for path in feedback_paths]
        tool_registry = Path(tool_registry_path)
        project_state = Path(project_state_path)
        history = Path(history_path)
        policy = Path(policy_path)
        out_root_path = Path(out_root)
        if not out_root_path.is_absolute():
            out_root_path = self.project_root / out_root_path

        missing = missing_paths(
            problem_map_path=problem_map,
            feedback_paths=feedbacks,
            tool_registry_path=tool_registry,
            project_state_path=project_state,
            history_path=history,
            policy_path=policy,
        )
        if missing:
            return {
                "status": "waiting_for_input",
                "failure_reason": "missing_required_inputs",
                "missing_inputs": missing,
                "artifacts": {},
            }
        try:
            policy_payload, policy_hash = load_policy(policy)
            evidence = load_evidence(
                problem_map_path=problem_map,
                feedback_paths=feedbacks,
                tool_registry_path=tool_registry,
                project_state_path=project_state,
                history_path=history,
            )
        except ValueError as exc:
            return {
                "status": "failed",
                "failure_reason": "input_validation_failed",
                "error": str(exc),
                "missing_inputs": [],
                "artifacts": {},
            }

        decision = self._build_decision(
            policy=policy_payload,
            policy_hash=policy_hash,
            evidence=evidence,
        )
        identity_payload = {
            "planner_version": PLANNER_VERSION,
            "policy_hash": policy_hash,
            "input_hashes": evidence.input_hashes,
            "normalized_action_candidates": {
                "ranked_actions": decision["ranked_actions"],
                "blocked_actions": decision["blocked_actions"],
                "deferred_actions": decision["deferred_actions"],
                "selected_action": decision["selected_action"],
                "selected_tool": decision["selected_tool"],
            },
        }
        plan_id = _stable_hash(identity_payload)
        decision["plan_id"] = plan_id
        core_plan_hash = _stable_hash(decision)

        if dry_run:
            return {
                "status": "dry_run",
                "plan_id": plan_id,
                "core_plan_hash": core_plan_hash,
                "selected_action": decision["selected_action"],
                "selected_tool": decision["selected_tool"],
                "blocked_action_count": len(decision["blocked_actions"]),
                "deferred_action_count": len(decision["deferred_actions"]),
                "missing_inputs": decision["missing_inputs"],
                "plan_decision": decision,
                "artifacts": {},
            }

        run_dir = out_root_path / plan_id
        plan_path = run_dir / "plan_decision.json"
        if plan_path.exists() and not force_rebuild:
            try:
                previous = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception:
                previous = {}
            if previous == decision:
                return {
                    "status": "duplicate",
                    "plan_id": plan_id,
                    "core_plan_hash": core_plan_hash,
                    "plan_decision": decision,
                    "artifacts": {
                        "run_dir": str(run_dir),
                        "plan_decision": str(plan_path),
                        "existing_plan_decision": str(plan_path),
                    },
                    "missing_inputs": decision["missing_inputs"],
                }

        run_dir.mkdir(parents=True, exist_ok=True)
        ranked_path = run_dir / "ranked_actions.json"
        blocked_path = run_dir / "blocked_actions.json"
        deferred_path = run_dir / "deferred_actions.json"
        report_path = run_dir / "PLAN_REPORT.md"
        manifest_path = run_dir / "plan_manifest.json"
        plan_path.write_text(_json_dumps(decision) + "\n", encoding="utf-8")
        ranked_path.write_text(_json_dumps(decision["ranked_actions"]) + "\n", encoding="utf-8")
        blocked_path.write_text(_json_dumps(decision["blocked_actions"]) + "\n", encoding="utf-8")
        deferred_path.write_text(_json_dumps(decision["deferred_actions"]) + "\n", encoding="utf-8")
        report_path.write_text(render_plan_report(decision), encoding="utf-8")
        manifest = {
            "planner_version": PLANNER_VERSION,
            "plan_id": plan_id,
            "core_plan_hash": core_plan_hash,
            "created_at_epoch_seconds": time.time(),
            "input_paths": {
                "problem_map": str(problem_map),
                "feedback": [str(path) for path in feedbacks],
                "tool_registry": str(tool_registry),
                "project_state": str(project_state),
                "history": str(history),
                "policy": str(policy),
            },
            "input_hashes": {**evidence.input_hashes, "policy": policy_hash},
            "artifacts": {
                "plan_decision": str(plan_path),
                "plan_manifest": str(manifest_path),
                "plan_report": str(report_path),
                "ranked_actions": str(ranked_path),
                "blocked_actions": str(blocked_path),
                "deferred_actions": str(deferred_path),
            },
        }
        manifest_path.write_text(_json_dumps(manifest) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "plan_id": plan_id,
            "core_plan_hash": core_plan_hash,
            "plan_decision": decision,
            "artifacts": {
                "run_dir": str(run_dir),
                "plan_decision": str(plan_path),
                "plan_manifest": str(manifest_path),
                "plan_report": str(report_path),
                "ranked_actions": str(ranked_path),
                "blocked_actions": str(blocked_path),
                "deferred_actions": str(deferred_path),
            },
            "missing_inputs": decision["missing_inputs"],
        }

    def _build_decision(
        self,
        *,
        policy: dict[str, Any],
        policy_hash: str,
        evidence,
    ) -> dict[str, Any]:
        registry_tools = {
            item.get("name"): item
            for item in evidence.tool_registry.get("tools", [])
            if isinstance(item, dict)
        }
        duplicate_index = DuplicateIndex(
            feedbacks=evidence.feedbacks,
            history=evidence.history,
            project_state=evidence.project_state,
        )
        reason_codes: list[str] = []
        ranked_actions: list[dict[str, Any]] = []
        blocked_actions: list[dict[str, Any]] = []
        deferred_actions: list[dict[str, Any]] = []
        missing_inputs: list[str] = []

        if self._full_anchor_inputs_missing(evidence):
            missing_inputs.extend(
                [
                    "canonical_fold_assignment",
                    "final_v53q1_oof_proba",
                    "final_v53q1_oof_identity_or_manifest",
                ]
            )
            reason_codes.extend(["missing_input", "missing_full_anchor_inputs"])

        reason_codes.extend(duplicate_index.duplicate_reason_codes())

        for feedback in sorted(evidence.feedbacks, key=lambda item: item.get("feedback_id", "")):
            tool_name = str(feedback.get("tool_name", ""))
            if tool_name not in registry_tools:
                blocked_actions.append(
                    _action(
                        action="human_review",
                        status="blocked",
                        tool=tool_name,
                        reason_codes=["unregistered_tool"],
                        risk_level="medium",
                        requires_human_approval=True,
                    )
                )
                reason_codes.append("unregistered_tool")
                continue
            tier = str(feedback.get("evaluation_tier", ""))
            kind = str(feedback.get("feedback_kind", ""))
            if kind == "candidate_replay":
                if duplicate_index.reference_verified():
                    deferred_actions.append(
                        _action(
                            action="accept_reference",
                            status="deferred",
                            tool=tool_name,
                            reason_codes=["reference_already_verified"],
                            evidence_tier="artifact_integrity",
                        )
                    )
                    reason_codes.append("reference_already_verified")
                continue
            if kind == "signal_audit":
                recommendation = str(feedback.get("recommendation", ""))
                if recommendation == "proceed_to_candidate_generation":
                    if duplicate_index.candidate_materialized():
                        deferred_actions.append(
                            _action(
                                action="generate_candidate",
                                status="deferred",
                                tool="A1_V53Q1_PATCH_REPLAY_SAFE",
                                reason_codes=["candidate_already_materialized"],
                                evidence_tier="signal_evidence",
                            )
                        )
                        reason_codes.append("candidate_already_materialized")
                    else:
                        ranked_actions.append(
                            _action(
                                action="generate_candidate",
                                status="ranked",
                                tool="A1_V53Q1_PATCH_REPLAY_SAFE",
                                reason_codes=["signal_evidence_valid"],
                                risk_level="high",
                                requires_human_approval=True,
                                evidence_tier="signal_evidence",
                            )
                        )
                continue
            if kind == "expert_scope_audit":
                metrics = feedback.get("metrics", {})
                oof_eval = metrics.get("oof_evaluation", {})
                evidence_payload = feedback.get("evidence", {})
                if evidence_payload.get("isolated_only_pass") is True:
                    reason_codes.append("scope_evidence_valid")
                if oof_eval.get("status") == "unavailable" or evidence_payload.get("oof_status", {}).get("status") == "unavailable":
                    ranked_actions.append(
                        _action(
                            action="run_oof_evaluation",
                            status="ranked",
                            tool="A1_OOF_CANDIDATE_EVALUATOR",
                            reason_codes=["missing_candidate_final_oof", "missing_input"],
                            evidence_tier="expert_scope",
                        )
                    )
                    reason_codes.append("missing_input")
                continue
            if kind == "model_experiment" or tier == "oof_comparison":
                self._handle_oof_feedback(
                    feedback=feedback,
                    ranked_actions=ranked_actions,
                    blocked_actions=blocked_actions,
                    deferred_actions=deferred_actions,
                    reason_codes=reason_codes,
                    policy=policy,
                )

        for name, tool in sorted(registry_tools.items()):
            if tool.get("task") != evidence.project_state.get("task"):
                continue
            has_binding = bool(tool.get("adapter_entrypoint") or tool.get("command_template"))
            if not has_binding and name != "REVIEW_AND_REPLAN":
                blocked_actions.append(
                    _action(
                        action=str(tool.get("action_type", "run")),
                        status="blocked",
                        tool=name,
                        reason_codes=["adapter_unbound"],
                        risk_level="medium",
                    )
                )
            if tool.get("submission_creating") is True:
                blocked_actions.append(
                    _action(
                        action="online_submission",
                        status="blocked",
                        tool=name,
                        reason_codes=["submission_auto_forbidden"],
                        risk_level="critical",
                        requires_human_approval=True,
                    )
                )
            if tool.get("requires_gpu") or tool.get("action_type") == "training":
                blocked_actions.append(
                    _action(
                        action="run_training",
                        status="blocked",
                        tool=name,
                        reason_codes=["human_approval_required", "missing_policy"],
                        risk_level="high",
                        requires_human_approval=True,
                    )
                )

        for branch in sorted(duplicate_index.closed_branches):
            blocked_actions.append(
                _action(
                    action="keep_branch_closed",
                    status="blocked",
                    tool=None,
                    branch_id=branch,
                    reason_codes=["branch_closed"],
                    risk_level="medium",
                    requires_human_approval=True,
                )
            )

        missing_inputs = sorted(set(missing_inputs))
        reason_codes = sorted(set(reason_codes))
        rounds_used = int(evidence.project_state.get("budget", {}).get("rounds_used", 0))
        max_rounds = int(evidence.project_state.get("budget", {}).get("max_rounds", 0))
        budget_exhausted = max_rounds > 0 and rounds_used >= max_rounds
        if budget_exhausted:
            reason_codes.append("budget_blocked")

        if budget_exhausted:
            status = "budget_exhausted"
            selected_action = "no_safe_action"
            selected_tool = None
        elif missing_inputs:
            status = "waiting_for_input"
            selected_action = "request_missing_input"
            selected_tool = None
            ranked_actions.insert(
                0,
                _action(
                    action="request_missing_input",
                    status="ranked",
                    reason_codes=["missing_input", "missing_full_anchor_inputs"],
                    risk_level="low",
                ),
            )
        elif ranked_actions:
            status = "ready"
            selected = self._sort_actions(ranked_actions, policy)[0]
            selected_action = str(selected["action"])
            selected_tool = selected.get("tool")
        else:
            status = "no_safe_action"
            selected_action = "no_safe_action"
            selected_tool = None

        ranked_actions = self._dedupe_actions(self._sort_actions(ranked_actions, policy))
        blocked_actions = self._dedupe_actions(blocked_actions)
        deferred_actions = self._dedupe_actions(deferred_actions)
        return {
            "planner_version": PLANNER_VERSION,
            "plan_id": "PENDING",
            "task": str(evidence.project_state.get("task", "A1")),
            "status": status,
            "current_stage": str(evidence.project_state.get("active_layer", "")),
            "primary_problem": (
                "missing_full_anchor_evaluation_inputs"
                if missing_inputs
                else self._primary_problem_from_map(evidence.problem_map)
            ),
            "problem_evidence": self._problem_evidence(evidence.problem_map),
            "selected_action": selected_action,
            "selected_tool": selected_tool,
            "selected_tool_version": (
                registry_tools.get(selected_tool, {}).get("adapter_version") if selected_tool else None
            ),
            "ranked_actions": ranked_actions,
            "blocked_actions": blocked_actions,
            "deferred_actions": deferred_actions,
            "evidence_refs": evidence.evidence_refs,
            "feedback_refs": evidence.feedback_refs,
            "required_inputs": [
                "problem_map",
                "feedback",
                "tool_registry",
                "project_state",
                "history",
                "policy",
            ],
            "missing_inputs": missing_inputs,
            "expected_information_gain": "close missing-input and duplicate/materialization decisions without executing tools",
            "expected_model_gain_status": "unavailable_without_full_anchor_oof_or_explicit_promotion_policy",
            "risk_level": "low" if selected_action == "request_missing_input" else "medium",
            "budget_cost": {
                "counts_as_experiment_round": False,
                "rounds_used": rounds_used,
                "max_rounds": max_rounds,
            },
            "stop_conditions": [
                "missing required full-anchor inputs",
                "budget exhausted",
                "closed branch would need reopening",
            ],
            "success_conditions": [
                "PlanDecision generated deterministically",
                "no tools executed",
                "no frozen artifacts mutated",
            ],
            "failure_conditions": [
                "invalid input schema",
                "unsafe planner policy",
                "unknown tool selected for execution",
            ],
            "reason_codes": sorted(set(reason_codes)),
            "human_readable_rationale": self._rationale(reason_codes, missing_inputs),
            "requires_human_approval": False if selected_action == "request_missing_input" else status == "human_approval_required",
            "auto_execution_allowed": False,
            "input_hashes": {**evidence.input_hashes, "policy": policy_hash},
            "policy_hash": policy_hash,
        }

    def _handle_oof_feedback(
        self,
        *,
        feedback: dict[str, Any],
        ranked_actions: list[dict[str, Any]],
        blocked_actions: list[dict[str, Any]],
        deferred_actions: list[dict[str, Any]],
        reason_codes: list[str],
        policy: dict[str, Any],
    ) -> None:
        del ranked_actions, deferred_actions
        metrics = feedback.get("metrics", {})
        evidence = feedback.get("evidence", {})
        overall_gain = metrics.get("overall", {}).get("gain")
        macro_gain = metrics.get("macro", {}).get("gain")
        net = metrics.get("rescue_damage", {}).get("net")
        parent_status = str(evidence.get("parent_identity_status", ""))
        if isinstance(overall_gain, (int, float)) and overall_gain < 0:
            reason_codes.append("negative_oof_gain")
        if isinstance(macro_gain, (int, float)) and macro_gain < 0:
            reason_codes.append("negative_macro_gain")
        if isinstance(net, (int, float)) and net < 0:
            reason_codes.append("negative_net")
        if parent_status == "unverified":
            reason_codes.append("unverified_parent")
        if not policy.get("promotion_policy_refs"):
            reason_codes.append("missing_policy")
        blocked_actions.append(
            _action(
                action="generate_candidate",
                status="blocked",
                tool="A1_V53Q1_PATCH_REPLAY_SAFE",
                reason_codes=[
                    code
                    for code in [
                        "negative_oof_gain" if isinstance(overall_gain, (int, float)) and overall_gain < 0 else "",
                        "negative_net" if isinstance(net, (int, float)) and net < 0 else "",
                        "unverified_parent" if parent_status == "unverified" else "",
                        "missing_policy" if not policy.get("promotion_policy_refs") else "",
                    ]
                    if code
                ],
                risk_level="high",
                requires_human_approval=True,
                evidence_tier="unverified_self_contained_oof",
            )
        )
        blocked_actions.append(
            _action(
                action="keep_branch_closed",
                status="blocked",
                reason_codes=["negative_net", "oracle_gain_not_sufficient_to_reopen"],
                risk_level="medium",
                requires_human_approval=True,
                evidence_tier="unverified_self_contained_oof",
            )
        )

    def _sort_actions(self, actions: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
        priority = {name: index for index, name in enumerate(policy.get("action_priority", []))}
        return sorted(
            actions,
            key=lambda item: (
                priority.get(str(item.get("action")), 999),
                str(item.get("tool") or ""),
                ",".join(item.get("reason_codes", [])),
            ),
        )

    def _dedupe_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for action in actions:
            key = _stable_hash(
                {
                    "action": action.get("action"),
                    "tool": action.get("tool"),
                    "branch_id": action.get("branch_id"),
                    "reason_codes": action.get("reason_codes"),
                }
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(action)
        return result

    def _full_anchor_inputs_missing(self, evidence) -> bool:
        problem_map = evidence.problem_map
        if problem_map.get("anchor_identity") == "unavailable":
            return True
        for problem in problem_map.get("problems", []):
            signal = problem.get("signal_status", {})
            if signal.get("canonical_fold") == "missing":
                return True
            if signal.get("v53q1_anchor_oof") == "missing":
                return True
        for feedback in evidence.feedbacks:
            for limitation in feedback.get("limitations", []):
                if limitation.get("metric_family") in {"current_champion_anchor_gain", "fold_gain"}:
                    return True
        return False

    def _primary_problem_from_map(self, problem_map: dict[str, Any]) -> str:
        problems = problem_map.get("problems", [])
        if not problems:
            return "no_problem_map_entries"
        best = sorted(
            problems,
            key=lambda item: (-int(item.get("node_count") or 0), str(item.get("problem_id", ""))),
        )[0]
        return str(best.get("problem_id", "unknown_problem"))

    def _problem_evidence(self, problem_map: dict[str, Any]) -> dict[str, Any]:
        problems = problem_map.get("problems", [])
        return {
            "analysis_tier": problem_map.get("analysis_tier", ""),
            "problem_count": len(problems),
            "top_problem_ids": [
                str(item.get("problem_id", ""))
                for item in sorted(
                    problems,
                    key=lambda problem: (-int(problem.get("node_count") or 0), str(problem.get("problem_id", ""))),
                )[:5]
            ],
            "train_test_shift_status": problem_map.get("rankings", {}).get("train_test_shift", {}).get("status", "unavailable"),
        }

    def _rationale(self, reason_codes: list[str], missing_inputs: list[str]) -> str:
        if missing_inputs:
            return (
                "Planner selected request_missing_input because full anchor OOF/Fold "
                "inputs are unavailable; lower-tier feedback cannot safely promote a candidate."
            )
        if "budget_blocked" in reason_codes:
            return "Planner found the project budget exhausted and selected no safe action."
        return "Planner produced a conservative deterministic action ordering from explicit inputs."
