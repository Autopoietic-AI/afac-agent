"""M6A LLM Shadow Planner.

The shadow planner produces advisory proposals and comparisons only.  It never
executes adapters or changes the deterministic M5A plan.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .base import LLMRequest, LLMResponse
from .comparator import compare_shadow_plan
from .policy import load_llm_shadow_policy
from .prompt_builder import build_prompt_package
from .providers import provider_config_hash as build_provider_config_hash
from .providers import make_provider
from .report import render_shadow_report
from .utils import pretty_json, stable_hash

PROPOSAL_VERSION = "m6a_v1"
SHADOW_RUNNER_VERSION = "m6a_shadow_runner_v1"


class LLMShadowPlanner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(
        self,
        *,
        deterministic_plan_path: Path,
        problem_map_path: Path,
        feedback_paths: list[Path],
        tool_registry_path: Path,
        project_state_path: Path,
        history_path: Path,
        planner_policy_path: Path,
        llm_policy_path: Path,
        provider_name: str,
        provider_config: str = "",
        mock_mode: str = "agree",
        out_root: Path | None = None,
        dry_run: bool = False,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        del force_rebuild
        paths = [
            deterministic_plan_path,
            problem_map_path,
            tool_registry_path,
            project_state_path,
            history_path,
            planner_policy_path,
            llm_policy_path,
            *feedback_paths,
        ]
        missing = [str(path) for path in paths if not Path(path).exists()]
        if missing:
            return {"status": "waiting_for_input", "missing_inputs": missing}
        llm_policy, llm_policy_hash = load_llm_shadow_policy(Path(llm_policy_path))
        package = build_prompt_package(
            deterministic_plan_path=Path(deterministic_plan_path),
            problem_map_path=Path(problem_map_path),
            feedback_paths=[Path(path) for path in feedback_paths],
            tool_registry_path=Path(tool_registry_path),
            project_state_path=Path(project_state_path),
            history_path=Path(history_path),
            planner_policy_path=Path(planner_policy_path),
            llm_policy=llm_policy,
            llm_policy_hash=llm_policy_hash,
        )
        provider_config_hash = self._provider_config_hash(provider_config)
        deterministic_plan = json.loads(Path(deterministic_plan_path).read_text(encoding="utf-8"))
        shadow_identity = {
            "runner_version": SHADOW_RUNNER_VERSION,
            "deterministic_plan_id": deterministic_plan.get("plan_id", ""),
            "evidence_bundle_hash": package.evidence_bundle_hash,
            "prompt_hash": package.prompt_hash,
            "provider": provider_name,
            "provider_config_hash": provider_config_hash,
            "policy_hash": llm_policy_hash,
            "mock_mode": mock_mode if provider_name == "mock" else "",
        }
        shadow_run_id = stable_hash(shadow_identity)
        if dry_run:
            return {
                "status": "dry_run",
                "deterministic_plan_id": deterministic_plan.get("plan_id", ""),
                "shadow_run_id": shadow_run_id,
                "prompt_hash": package.prompt_hash,
                "evidence_bundle_hash": package.evidence_bundle_hash,
                "provider_config_hash": provider_config_hash,
                "would_call_provider": False,
            }
        provider = make_provider(
            provider_name,
            project_root=self.project_root,
            provider_config=provider_config,
            mock_mode=mock_mode,
        )
        max_calls = int(llm_policy.get("max_calls", 2))
        response = self._generate_with_optional_retry(
            provider=provider,
            request=LLMRequest(
                prompt=package.prompt,
                provider=provider_name,
                model="",
                timeout_seconds=int(llm_policy.get("timeout_seconds", 30)),
                max_output_tokens=int(llm_policy.get("max_output_chars", 4096)),
                temperature=0.0,
                metadata={"evidence_bundle": package.evidence_bundle},
            ),
            max_calls=max_calls,
        )
        raw_response_hash = stable_hash(response.text)
        proposal = self._normalize_response(
            response=response,
            deterministic_plan=deterministic_plan,
            policy=llm_policy,
            tool_registry=json.loads(Path(tool_registry_path).read_text(encoding="utf-8")),
        )
        proposal["proposal_id"] = stable_hash(
            {
                "proposal_version": proposal.get("proposal_version"),
                "deterministic_plan_id": deterministic_plan.get("plan_id", ""),
                "provider": response.provider or provider_name,
                "raw_response_hash": raw_response_hash,
                "normalized": {k: v for k, v in proposal.items() if k != "proposal_id"},
            }
        )
        comparison = compare_shadow_plan(deterministic_plan, proposal)
        comparison_hash = stable_hash(comparison)
        root = Path(out_root) if out_root else self.project_root / "artifacts" / "llm_shadow_runs"
        run_dir = root / str(deterministic_plan.get("plan_id", "unknown_plan")) / shadow_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "evidence_bundle": run_dir / "evidence_bundle.json",
            "prompt": run_dir / "prompt.txt",
            "raw_response": run_dir / "raw_response.txt",
            "llm_plan_proposal": run_dir / "llm_plan_proposal.json",
            "shadow_plan_comparison": run_dir / "shadow_plan_comparison.json",
            "provider_usage": run_dir / "provider_usage.json",
            "shadow_report": run_dir / "SHADOW_REPORT.md",
            "shadow_manifest": run_dir / "shadow_manifest.json",
        }
        artifacts["evidence_bundle"].write_text(pretty_json(package.evidence_bundle) + "\n", encoding="utf-8")
        artifacts["prompt"].write_text(package.prompt + "\n", encoding="utf-8")
        artifacts["raw_response"].write_text(response.text, encoding="utf-8")
        artifacts["llm_plan_proposal"].write_text(pretty_json(proposal) + "\n", encoding="utf-8")
        artifacts["shadow_plan_comparison"].write_text(pretty_json(comparison) + "\n", encoding="utf-8")
        provider_usage = {
            "provider": response.provider or provider_name,
            "model": response.model,
            "status": response.status,
            "failure_reason": response.failure_reason,
            "warnings": response.warnings,
            "audit": response.audit,
        }
        artifacts["provider_usage"].write_text(pretty_json(provider_usage) + "\n", encoding="utf-8")
        artifacts["shadow_report"].write_text(render_shadow_report(proposal, comparison), encoding="utf-8")
        manifest = {
            "shadow_runner_version": SHADOW_RUNNER_VERSION,
            "shadow_run_id": shadow_run_id,
            "deterministic_plan_id": deterministic_plan.get("plan_id", ""),
            "provider": response.provider or provider_name,
            "model": response.model,
            "status": proposal.get("status"),
            "prompt_hash": package.prompt_hash,
            "evidence_bundle_hash": package.evidence_bundle_hash,
            "provider_config_hash": provider_config_hash,
            "raw_response_hash": raw_response_hash,
            "proposal_hash": stable_hash(proposal),
            "comparison_hash": comparison_hash,
            "llm_policy_hash": llm_policy_hash,
            "created_at_epoch_seconds": time.time(),
            "artifacts": {
                key: self._safe_artifact_ref(path)
                for key, path in artifacts.items()
            },
        }
        artifacts["shadow_manifest"].write_text(pretty_json(manifest) + "\n", encoding="utf-8")
        return {
            "status": proposal.get("status"),
            "deterministic_plan_id": deterministic_plan.get("plan_id", ""),
            "shadow_run_id": shadow_run_id,
            "proposal_id": proposal.get("proposal_id"),
            "proposed_action": proposal.get("proposed_action"),
            "proposed_tool": proposal.get("proposed_tool"),
            "research_needed": proposal.get("research_needed"),
            "research_query": proposal.get("research_query"),
            "agreement_level": comparison.get("agreement_level"),
            "llm_novelty": comparison.get("llm_novelty"),
            "llm_safety_status": comparison.get("llm_safety_status"),
            "comparison_summary": comparison.get("comparison_summary"),
            "prompt_hash": package.prompt_hash,
            "evidence_bundle_hash": package.evidence_bundle_hash,
            "provider_config_hash": provider_config_hash,
            "raw_response_hash": raw_response_hash,
            "proposal_hash": stable_hash(proposal),
            "comparison_hash": comparison_hash,
            "artifacts": {key: str(path) for key, path in artifacts.items()},
        }

    def _generate_with_optional_retry(self, *, provider, request: LLMRequest, max_calls: int) -> LLMResponse:
        response = provider.generate(request)
        if response.status != "completed":
            return response
        try:
            json.loads(response.text)
        except json.JSONDecodeError:
            if max_calls < 2:
                return response
        if not self._response_has_required_schema(response.text):
            if max_calls < 2:
                return response
        else:
            return response
        retry = LLMRequest(
            prompt=(
                "Your previous output was not a valid AFAC M6A shadow proposal. "
                "Using the same evidence and facts below, return exactly one JSON object "
                "matching the output_contract. Do not add new evidence, Markdown, prose, "
                "commands, or a top-level key named plan.\n\n"
                f"{request.prompt}"
            ),
            provider=request.provider,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
            max_output_tokens=request.max_output_tokens,
            temperature=0.0,
            metadata=request.metadata,
        )
        return provider.generate(retry)

    @staticmethod
    def _response_has_required_schema(text: str) -> bool:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        required = {
            "proposal_version",
            "status",
            "proposed_action",
            "proposed_tool",
            "research_needed",
            "auto_execution_allowed",
        }
        return required.issubset(payload)

    def _normalize_response(
        self,
        *,
        response: LLMResponse,
        deterministic_plan: dict[str, Any],
        policy: dict[str, Any],
        tool_registry: dict[str, Any],
    ) -> dict[str, Any]:
        if response.status in {"provider_unavailable", "timeout"}:
            return self._base_proposal(
                deterministic_plan,
                status=response.status,
                proposed_action="request_missing_input",
                proposed_tool=None,
                reason_codes=[response.failure_reason or response.status],
            )
        if response.status != "completed":
            return self._base_proposal(
                deterministic_plan,
                status="failed",
                proposed_action="request_missing_input",
                proposed_tool=None,
                reason_codes=[response.failure_reason or "provider_failed"],
            )
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return self._base_proposal(
                deterministic_plan,
                status="invalid_output",
                proposed_action="request_missing_input",
                proposed_tool=None,
                reason_codes=["invalid_json"],
            )
        if not isinstance(payload, dict):
            return self._base_proposal(
                deterministic_plan,
                status="invalid_output",
                proposed_action="request_missing_input",
                proposed_tool=None,
                reason_codes=["json_not_object"],
            )
        if not self._response_has_required_schema(response.text):
            return self._base_proposal(
                deterministic_plan,
                status="invalid_output",
                proposed_action="request_missing_input",
                proposed_tool=None,
                reason_codes=["proposal_schema_invalid"],
            )
        proposal = self._base_proposal(
            deterministic_plan,
            status=str(payload.get("status", "completed")),
            proposed_action=str(payload.get("proposed_action", "")),
            proposed_tool=payload.get("proposed_tool"),
            reason_codes=list(payload.get("reason_codes", [])),
        )
        for key in [
            "problem_interpretation",
            "expected_information_gain",
            "expected_model_gain_status",
            "risk_level",
            "uncertainties",
            "assumptions",
            "stop_conditions",
            "success_conditions",
            "failure_conditions",
            "human_readable_rationale",
            "research_needed",
            "research_query",
        ]:
            if key in payload:
                proposal[key] = payload[key]
        proposal["primary_problem"] = str(payload.get("primary_problem") or deterministic_plan.get("primary_problem", ""))
        proposal["missing_inputs"] = list(payload.get("missing_inputs") or deterministic_plan.get("missing_inputs", []))
        proposal["required_inputs"] = list(payload.get("required_inputs") or deterministic_plan.get("required_inputs", []))
        proposal["evidence_refs"] = list(payload.get("evidence_refs") or deterministic_plan.get("evidence_refs", []))
        proposal["feedback_refs"] = list(payload.get("feedback_refs") or deterministic_plan.get("feedback_refs", []))
        proposal["proposed_tool_version"] = payload.get("proposed_tool_version")
        proposal["auto_execution_allowed"] = False
        proposal["requires_human_approval"] = bool(payload.get("requires_human_approval", False))
        self._apply_safety(proposal, policy, tool_registry)
        return proposal

    def _apply_safety(self, proposal: dict[str, Any], policy: dict[str, Any], tool_registry: dict[str, Any]) -> None:
        allowed = set(map(str, policy.get("allowed_actions", [])))
        forbidden = set(map(str, policy.get("forbidden_actions", [])))
        action = str(proposal.get("proposed_action") or "")
        tool = proposal.get("proposed_tool")
        registered_tools = {
            str(item.get("name"))
            for item in tool_registry.get("tools", [])
            if isinstance(item, dict)
        }
        reasons = set(map(str, proposal.get("reason_codes", [])))
        if action not in allowed:
            reasons.add("action_not_allowed")
            proposal["status"] = "blocked"
        if action in forbidden:
            reasons.add("forbidden_action")
            proposal["status"] = "blocked"
        if action in {"online_submission", "modify_champion", "register_champion", "overwrite_champion"}:
            reasons.add("submission_or_champion_mutation_forbidden")
            proposal["status"] = "blocked"
        if tool is not None and str(tool) not in registered_tools:
            reasons.add("unregistered_tool")
            proposal["status"] = "blocked"
        if action == "run_training":
            proposal["requires_human_approval"] = True
            reasons.add("training_requires_human_approval")
        if action == "generate_candidate":
            proposal["requires_human_approval"] = True
            reasons.add("prediction_requires_human_approval")
        if action == "reopen_branch":
            proposal["requires_human_approval"] = True
            reasons.add("closed_branch_reopen_forbidden")
            proposal["status"] = "blocked"
        if proposal.get("research_needed") is True and not policy.get("allow_research_request", False):
            proposal["research_needed"] = False
            proposal["research_query"] = ""
            reasons.add("research_request_disabled")
        if not proposal.get("research_needed"):
            proposal["research_query"] = ""
        proposal["auto_execution_allowed"] = False
        proposal["reason_codes"] = sorted(reasons)

    def _base_proposal(
        self,
        deterministic_plan: dict[str, Any],
        *,
        status: str,
        proposed_action: str,
        proposed_tool: str | None,
        reason_codes: list[str],
    ) -> dict[str, Any]:
        return {
            "proposal_version": PROPOSAL_VERSION,
            "proposal_id": "PENDING",
            "task": deterministic_plan.get("task", "A1"),
            "status": status,
            "primary_problem": deterministic_plan.get("primary_problem", ""),
            "problem_interpretation": "",
            "proposed_action": proposed_action,
            "proposed_tool": proposed_tool,
            "proposed_tool_version": None,
            "evidence_refs": deterministic_plan.get("evidence_refs", []),
            "feedback_refs": deterministic_plan.get("feedback_refs", []),
            "required_inputs": deterministic_plan.get("required_inputs", []),
            "missing_inputs": deterministic_plan.get("missing_inputs", []),
            "expected_information_gain": "",
            "expected_model_gain_status": "",
            "risk_level": "low",
            "uncertainties": [],
            "assumptions": [],
            "stop_conditions": deterministic_plan.get("stop_conditions", []),
            "success_conditions": deterministic_plan.get("success_conditions", []),
            "failure_conditions": deterministic_plan.get("failure_conditions", []),
            "reason_codes": reason_codes,
            "human_readable_rationale": "",
            "requires_human_approval": False,
            "auto_execution_allowed": False,
            "research_needed": False,
            "research_query": "",
        }

    def _provider_config_hash(self, provider_config: str) -> str:
        return build_provider_config_hash(self.project_root, provider_config)

    def _safe_artifact_ref(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.name
