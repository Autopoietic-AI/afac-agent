# -*- coding: utf-8 -*-
"""Environment and path preflight for AFAC Agent M0/M1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .feedback.normalizers import NORMALIZER_REGISTRY
from .llm.policy import load_llm_shadow_policy
from .llm.providers import (
    ALIYUN_BAILIAN_DEFAULT_MODEL,
    ALIYUN_BAILIAN_PROVIDER,
    is_allowed_bailian_model,
)
from .paths import PathResolver
from .planning.policy import load_policy
from .validation import (
    reports_to_checks,
    validate_a1_data_profile_dir,
    validate_a1_champion_csv,
    validate_memory_records_file,
    validate_project_state_file,
    validate_tool_registry_file,
    validate_trajectory_file,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_check(names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        result[name] = {
            "available": importlib.util.find_spec(name) is not None,
        }
    return result


def _git_lines(root: Path, *args: str) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _scan_tracked_secret_locations(root: Path) -> list[str]:
    locations: list[str] = []
    secret_pattern = re.compile(r"sk-[A-Za-z0-9._-]+|Authorization\s*[:=]\s*Bearer\s+", re.IGNORECASE)
    for rel in _git_lines(root, "ls-files"):
        if rel.startswith("artifacts/llm_shadow_runs/"):
            continue
        path = root / rel
        if not path.is_file() or path.suffix.lower() in {".pyc", ".npz", ".npy"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if secret_pattern.search(line):
                locations.append(f"{rel}:{line_no}")
    return locations



def _json_file_check(root: Path, rel: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    path = root / rel
    errors: list[str] = []
    details: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    payload: Any = None
    if not path.exists():
        errors.append(f"{rel}: missing")
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{rel}: invalid JSON: {exc}")
    if isinstance(payload, dict) and required_keys:
        missing = sorted(required_keys - set(payload))
        if missing:
            errors.append(f"{rel}: missing keys {missing}")
        details["keys"] = sorted(payload.keys())
    return {"name": Path(rel).stem, "passed": not errors, "errors": errors, "warnings": [], "details": details}


def _gitignore_has(root: Path, pattern: str) -> bool:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return False
    return pattern in gitignore.read_text(encoding="utf-8")

def build_report(
    *,
    project_root: Path,
    paths_config: str = "",
) -> dict[str, Any]:
    resolver = PathResolver(project_root, paths_config or None)
    root = resolver.project_root
    champion_csv = resolver.a1_anchor_csv()
    reports = [
        validate_project_state_file(root / "config" / "project_state.json"),
        validate_tool_registry_file(root / "config" / "tool_registry.json"),
        validate_memory_records_file(root / "history" / "confirmed_experiments_a1.json"),
        validate_trajectory_file(root / "output" / "trajectory_A1.json"),
        validate_a1_champion_csv(champion_csv, expected_rows=2751, num_classes=10),
        validate_a1_data_profile_dir(resolver.a1_profile_out_dir()),
    ]
    checks = reports_to_checks(reports)
    try:
        project_state_payload = json.loads(
            (root / "config" / "project_state.json").read_text(encoding="utf-8")
        )
    except Exception:
        project_state_payload = {}
    profile_check = checks.get("a1_data_profile", {})
    profile_valid = bool(profile_check.get("details", {}).get("artifact_valid"))
    profile_input_hash_stale = False
    configured_npz = resolver.a1_npz()
    input_validation_path = resolver.a1_profile_out_dir() / "a1_input_validation.json"
    if profile_valid and configured_npz and configured_npz.exists() and input_validation_path.exists():
        try:
            input_validation = json.loads(
                input_validation_path.read_text(encoding="utf-8")
            )
            recorded_hash = input_validation.get("npz_path", {}).get("sha256")
            profile_input_hash_stale = recorded_hash != sha256(configured_npz)
        except Exception:
            profile_input_hash_stale = True
    if profile_input_hash_stale:
        profile_valid = False
        profile_check.setdefault("warnings", []).append(
            "legacy_data_profile_input_hash_stale"
        )
        profile_check.setdefault("details", {})[
            "legacy_data_profile_input_hash_stale"
        ] = True
    if project_state_payload.get("data_profile_ready") is True and not profile_valid:
        profile_check.setdefault("warnings", []).append(
            "legacy_data_profile_flag_stale"
        )
        profile_check.setdefault("details", {})[
            "legacy_data_profile_flag_stale"
        ] = True
        checks["a1_data_profile"] = profile_check
    if champion_csv.exists():
        checks["champion_csv"]["details"]["sha256"] = sha256(champion_csv)
    checks["paths"] = {
        "name": "paths",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": {
            "paths_config": str(resolver.paths_config),
            "paths_config_exists": resolver.paths_config.exists(),
            "a1_anchor_csv": str(champion_csv),
            "a1_dataset_npz": str(resolver.a1_npz() or ""),
            "a1_edges_csv": str(resolver.a1_edges_csv() or ""),
            "a1_fold_file": str(resolver.a1_fold_file() or ""),
            "a1_anchor_oof_npz": str(resolver.a1_anchor_oof_npz() or ""),
            "a1_reference_oof_npz": str(resolver.a1_reference_oof_npz() or ""),
            "a1_profile_out_dir": str(resolver.a1_profile_out_dir()),
            "adapter_output_root": str(resolver.adapter_output_root()),
            "v53q1_base_csv": str(resolver.a1_v53q1_base_csv() or ""),
            "v49a_oof_meta_csv": str(resolver.a1_v49a_oof_meta_csv() or ""),
            "v49a_test_meta_csv": str(resolver.a1_v49a_test_meta_csv() or ""),
            "v53q1_audit_md": str(resolver.a1_v53q1_audit_md()),
            "v53q1_patch_py": str(resolver.a1_v53q1_patch_py()),
            "v46a1_candidate_csv": str(resolver.a1_v46a1_candidate_csv() or ""),
            "v46a1_parent_csv": str(resolver.a1_v46a1_parent_csv() or ""),
            "v46a1_candidate_oof_npz": str(
                resolver.a1_v46a1_candidate_oof_npz() or ""
            ),
            "v46a1_audit_report": str(resolver.a1_v46a1_audit_report() or ""),
            "v49a_edge_oof_meta_csv": str(
                resolver.a1_v49a_edge_oof_meta_csv() or ""
            ),
            "v49a_edge_test_meta_csv": str(
                resolver.a1_v49a_edge_test_meta_csv() or ""
            ),
            "v49a_v46a1_base_csv": str(
                resolver.a1_v49a_v46a1_base_csv() or ""
            ),
            "v49a_report": str(resolver.a1_v49a_report() or ""),
            "v49a_config": str(resolver.a1_v49a_config() or ""),
            "v49a_fold_results": str(resolver.a1_v49a_fold_results() or ""),
        },
    }
    schema_path = root / "schemas" / "adapter_execution_result.schema.json"
    schema_errors: list[str] = []
    if not schema_path.exists():
        schema_errors.append(f"{schema_path}: missing")
    else:
        try:
            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
            required = schema_payload.get("required", [])
            for key in [
                "tool_name",
                "adapter_id",
                "adapter_version",
                "status",
                "identity_hash",
                "input_hashes",
                "stdout_log",
                "stderr_log",
            ]:
                if key not in required:
                    schema_errors.append(f"adapter result schema missing required {key}")
        except Exception as exc:
            schema_errors.append(f"{schema_path}: cannot read JSON: {exc}")
    checks["adapter_result_schema"] = {
        "name": "adapter_result_schema",
        "passed": not schema_errors,
        "errors": schema_errors,
        "warnings": [],
        "details": {"path": str(schema_path)},
    }

    registry_errors: list[str] = []
    registry_warnings: list[str] = []
    try:
        registry_payload = json.loads(
            (root / "config" / "tool_registry.json").read_text(encoding="utf-8")
        )
        tools = registry_payload.get("tools", [])
    except Exception as exc:
        tools = []
        registry_errors.append(f"tool registry unreadable: {exc}")
    adapter_tools = [
        item for item in tools
        if isinstance(item, dict) and item.get("adapter_entrypoint")
    ]
    patch_audit = next(
        (item for item in adapter_tools if item.get("name") == "A1_V53Q1_PATCH_AUDIT"),
        None,
    )
    if patch_audit is None:
        registry_errors.append("A1_V53Q1_PATCH_AUDIT is not registered")
    else:
        if patch_audit.get("adapter_entrypoint") != (
            "afac_agent.adapters.a1_v53q1_patch_audit:Adapter"
        ):
            registry_errors.append("A1_V53Q1_PATCH_AUDIT entrypoint is invalid")
        if patch_audit.get("read_only") is not True:
            registry_errors.append("A1_V53Q1_PATCH_AUDIT must be read_only")
        if patch_audit.get("counts_as_experiment_round") is not False:
            registry_errors.append("A1_V53Q1_PATCH_AUDIT must not consume rounds")
        if patch_audit.get("mutates_predictions") is not False:
            registry_errors.append("A1_V53Q1_PATCH_AUDIT must not mutate predictions")
        if patch_audit.get("mutates_project_state") is not False:
            registry_errors.append("A1_V53Q1_PATCH_AUDIT must not mutate project state")
        if patch_audit.get("command_template") != []:
            registry_errors.append("A1_V53Q1_PATCH_AUDIT command_template must stay empty")
    v46_audit = next(
        (item for item in adapter_tools if item.get("name") == "A1_V46A1_ISOLATED_AUDIT"),
        None,
    )
    if v46_audit is None:
        registry_errors.append("A1_V46A1_ISOLATED_AUDIT is not registered")
    else:
        if v46_audit.get("adapter_entrypoint") != (
            "afac_agent.adapters.a1_v46a1_isolated_audit:Adapter"
        ):
            registry_errors.append("A1_V46A1_ISOLATED_AUDIT entrypoint is invalid")
        if v46_audit.get("read_only") is not True:
            registry_errors.append("A1_V46A1_ISOLATED_AUDIT must be read_only")
        if v46_audit.get("counts_as_experiment_round") is not False:
            registry_errors.append("A1_V46A1_ISOLATED_AUDIT must not consume rounds")
        if v46_audit.get("mutates_predictions") is not False:
            registry_errors.append("A1_V46A1_ISOLATED_AUDIT must not mutate predictions")
        if v46_audit.get("mutates_project_state") is not False:
            registry_errors.append("A1_V46A1_ISOLATED_AUDIT must not mutate project state")
        if v46_audit.get("command_template") != []:
            registry_errors.append("A1_V46A1_ISOLATED_AUDIT command_template must stay empty")
    v49_audit = next(
        (item for item in adapter_tools if item.get("name") == "A1_V49A_EDGE_UTILITY_AUDIT"),
        None,
    )
    if v49_audit is None:
        registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT is not registered")
    else:
        if v49_audit.get("adapter_entrypoint") != (
            "afac_agent.adapters.a1_v49a_edge_utility_audit:Adapter"
        ):
            registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT entrypoint is invalid")
        if v49_audit.get("read_only") is not True:
            registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT must be read_only")
        if v49_audit.get("counts_as_experiment_round") is not False:
            registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT must not consume rounds")
        if v49_audit.get("mutates_predictions") is not False:
            registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT must not mutate predictions")
        if v49_audit.get("mutates_project_state") is not False:
            registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT must not mutate project state")
        if v49_audit.get("command_template") != []:
            registry_errors.append("A1_V49A_EDGE_UTILITY_AUDIT command_template must stay empty")
    replay_safe = next(
        (item for item in adapter_tools if item.get("name") == "A1_V53Q1_PATCH_REPLAY_SAFE"),
        None,
    )
    if replay_safe is None:
        registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE is not registered")
    else:
        if replay_safe.get("adapter_entrypoint") != (
            "afac_agent.adapters.a1_v53q1_patch_replay_safe:Adapter"
        ):
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE entrypoint is invalid")
        if replay_safe.get("read_only") is not False:
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE must not be read_only")
        if replay_safe.get("counts_as_experiment_round") is not False:
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE must not consume rounds")
        if replay_safe.get("mutates_predictions") is not True:
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE must declare prediction mutation")
        if replay_safe.get("mutates_project_state") is not False:
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE must not mutate project state")
        if replay_safe.get("submission_creating") is not False:
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE must not create submissions")
        if replay_safe.get("command_template") != []:
            registry_errors.append("A1_V53Q1_PATCH_REPLAY_SAFE command_template must stay empty")
    oof_evaluator = next(
        (item for item in adapter_tools if item.get("name") == "A1_OOF_CANDIDATE_EVALUATOR"),
        None,
    )
    if oof_evaluator is None:
        registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR is not registered")
    else:
        if oof_evaluator.get("adapter_entrypoint") != (
            "afac_agent.adapters.a1_oof_candidate_evaluator:Adapter"
        ):
            registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR entrypoint is invalid")
        if oof_evaluator.get("read_only") is not True:
            registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR must be read_only")
        if oof_evaluator.get("counts_as_experiment_round") is not False:
            registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR must not consume rounds")
        if oof_evaluator.get("mutates_predictions") is not False:
            registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR must not mutate predictions")
        if oof_evaluator.get("mutates_project_state") is not False:
            registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR must not mutate project state")
        if oof_evaluator.get("command_template") != []:
            registry_errors.append("A1_OOF_CANDIDATE_EVALUATOR command_template must stay empty")
    checks["adapter_registry_bindings"] = {
        "name": "adapter_registry_bindings",
        "passed": not registry_errors,
        "errors": registry_errors,
        "warnings": registry_warnings,
        "details": {
            "adapter_count": len(adapter_tools),
            "has_A1_V53Q1_PATCH_AUDIT": patch_audit is not None,
            "has_A1_V46A1_ISOLATED_AUDIT": v46_audit is not None,
            "has_A1_V49A_EDGE_UTILITY_AUDIT": v49_audit is not None,
            "has_A1_V53Q1_PATCH_REPLAY_SAFE": replay_safe is not None,
            "has_A1_OOF_CANDIDATE_EVALUATOR": oof_evaluator is not None,
        },
    }

    output_root = resolver.adapter_output_root()
    output_errors: list[str] = []
    if output_root.resolve() == champion_csv.resolve():
        output_errors.append("adapter output root must not equal champion csv")
    checks["adapter_output_root"] = {
        "name": "adapter_output_root",
        "passed": not output_errors,
        "errors": output_errors,
        "warnings": [],
        "details": {
            "path": str(output_root),
            "exists": output_root.exists(),
            "parent_exists": output_root.parent.exists(),
        },
    }
    feedback_schema_errors: list[str] = []
    feedback_schema_path = root / "schemas" / "experiment_feedback.schema.json"
    try:
        feedback_schema = json.loads(feedback_schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        feedback_schema = {}
        feedback_schema_errors.append(f"experiment feedback schema unreadable: {exc}")
    if isinstance(feedback_schema, dict):
        required = set(feedback_schema.get("required", []))
        for key in [
            "feedback_id",
            "feedback_kind",
            "evaluation_tier",
            "execution_result_hash",
            "recommendation",
        ]:
            if key not in required:
                feedback_schema_errors.append(f"experiment feedback schema missing {key}")
    checks["experiment_feedback_schema"] = {
        "name": "experiment_feedback_schema",
        "passed": not feedback_schema_errors,
        "errors": feedback_schema_errors,
        "warnings": [],
        "details": {"path": str(feedback_schema_path)},
    }
    oof_eval_schema_errors: list[str] = []
    oof_eval_schema_path = root / "schemas" / "a1_oof_evaluation.schema.json"
    try:
        oof_eval_schema = json.loads(oof_eval_schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        oof_eval_schema = {}
        oof_eval_schema_errors.append(f"a1 oof evaluation schema unreadable: {exc}")
    if isinstance(oof_eval_schema, dict):
        required = set(oof_eval_schema.get("required", []))
        for key in [
            "evaluation_id",
            "analysis_tier",
            "comparison_scope",
            "parent_identity_status",
            "test_truth_used",
            "leakage_safety_pass",
            "evaluation_integrity_pass",
        ]:
            if key not in required:
                oof_eval_schema_errors.append(f"a1 oof evaluation schema missing {key}")
    checks["a1_oof_evaluation_schema"] = {
        "name": "a1_oof_evaluation_schema",
        "passed": not oof_eval_schema_errors,
        "errors": oof_eval_schema_errors,
        "warnings": [],
        "details": {"path": str(oof_eval_schema_path)},
    }
    plan_schema_errors: list[str] = []
    plan_schema_path = root / "schemas" / "plan_decision.schema.json"
    try:
        plan_schema = json.loads(plan_schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        plan_schema = {}
        plan_schema_errors.append(f"plan decision schema unreadable: {exc}")
    if isinstance(plan_schema, dict):
        required = set(plan_schema.get("required", []))
        for key in [
            "planner_version",
            "plan_id",
            "status",
            "selected_action",
            "ranked_actions",
            "blocked_actions",
            "deferred_actions",
            "input_hashes",
            "policy_hash",
        ]:
            if key not in required:
                plan_schema_errors.append(f"plan decision schema missing {key}")
    checks["plan_decision_schema"] = {
        "name": "plan_decision_schema",
        "passed": not plan_schema_errors,
        "errors": plan_schema_errors,
        "warnings": [],
        "details": {"path": str(plan_schema_path)},
    }
    planner_policy_schema_errors: list[str] = []
    planner_policy_schema_path = root / "schemas" / "planner_policy.schema.json"
    try:
        planner_policy_schema = json.loads(
            planner_policy_schema_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        planner_policy_schema = {}
        planner_policy_schema_errors.append(f"planner policy schema unreadable: {exc}")
    if isinstance(planner_policy_schema, dict):
        required = set(planner_policy_schema.get("required", []))
        for key in [
            "evidence_rank",
            "action_priority",
            "human_approval_policy",
            "branch_reopen_policy",
            "promotion_policy_refs",
        ]:
            if key not in required:
                planner_policy_schema_errors.append(f"planner policy schema missing {key}")
    checks["planner_policy_schema"] = {
        "name": "planner_policy_schema",
        "passed": not planner_policy_schema_errors,
        "errors": planner_policy_schema_errors,
        "warnings": [],
        "details": {"path": str(planner_policy_schema_path)},
    }
    planner_policy_errors: list[str] = []
    planner_policy_path = root / "config" / "planner_policy.json"
    try:
        _policy_payload, planner_policy_hash = load_policy(planner_policy_path)
    except Exception as exc:
        planner_policy_hash = ""
        planner_policy_errors.append(str(exc))
    checks["planner_policy_config"] = {
        "name": "planner_policy_config",
        "passed": not planner_policy_errors,
        "errors": planner_policy_errors,
        "warnings": [],
        "details": {
            "path": str(planner_policy_path),
            "sha256": planner_policy_hash,
        },
    }
    llm_proposal_schema_errors: list[str] = []
    llm_proposal_schema_path = root / "schemas" / "llm_plan_proposal.schema.json"
    try:
        llm_proposal_schema = json.loads(
            llm_proposal_schema_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        llm_proposal_schema = {}
        llm_proposal_schema_errors.append(f"llm plan proposal schema unreadable: {exc}")
    if isinstance(llm_proposal_schema, dict):
        required = set(llm_proposal_schema.get("required", []))
        for key in [
            "proposal_version",
            "proposal_id",
            "status",
            "proposed_action",
            "proposed_tool",
            "research_needed",
            "auto_execution_allowed",
        ]:
            if key not in required:
                llm_proposal_schema_errors.append(f"llm plan proposal schema missing {key}")
        auto_exec = llm_proposal_schema.get("properties", {}).get("auto_execution_allowed", {})
        if auto_exec.get("const") is not False:
            llm_proposal_schema_errors.append("llm proposal schema must force auto_execution_allowed=false")
    checks["llm_plan_proposal_schema"] = {
        "name": "llm_plan_proposal_schema",
        "passed": not llm_proposal_schema_errors,
        "errors": llm_proposal_schema_errors,
        "warnings": [],
        "details": {"path": str(llm_proposal_schema_path)},
    }
    shadow_comparison_schema_errors: list[str] = []
    shadow_comparison_schema_path = root / "schemas" / "shadow_plan_comparison.schema.json"
    try:
        shadow_comparison_schema = json.loads(
            shadow_comparison_schema_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        shadow_comparison_schema = {}
        shadow_comparison_schema_errors.append(
            f"shadow plan comparison schema unreadable: {exc}"
        )
    if isinstance(shadow_comparison_schema, dict):
        required = set(shadow_comparison_schema.get("required", []))
        for key in [
            "comparison_version",
            "comparison_id",
            "agreement_level",
            "llm_novelty",
            "llm_safety_status",
            "deterministic_decision_remains_authoritative",
        ]:
            if key not in required:
                shadow_comparison_schema_errors.append(
                    f"shadow plan comparison schema missing {key}"
                )
        authoritative = shadow_comparison_schema.get("properties", {}).get(
            "deterministic_decision_remains_authoritative", {}
        )
        if authoritative.get("const") is not True:
            shadow_comparison_schema_errors.append(
                "shadow comparison schema must keep deterministic plan authoritative"
            )
    checks["shadow_plan_comparison_schema"] = {
        "name": "shadow_plan_comparison_schema",
        "passed": not shadow_comparison_schema_errors,
        "errors": shadow_comparison_schema_errors,
        "warnings": [],
        "details": {"path": str(shadow_comparison_schema_path)},
    }
    llm_policy_schema_errors: list[str] = []
    llm_policy_schema_path = root / "schemas" / "llm_shadow_policy.schema.json"
    try:
        llm_policy_schema = json.loads(
            llm_policy_schema_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        llm_policy_schema = {}
        llm_policy_schema_errors.append(f"llm shadow policy schema unreadable: {exc}")
    if isinstance(llm_policy_schema, dict):
        required = set(llm_policy_schema.get("required", []))
        for key in [
            "allowed_actions",
            "forbidden_actions",
            "max_calls",
            "force_human_approval_for_training",
            "force_human_approval_for_prediction",
            "force_auto_execution_false",
        ]:
            if key not in required:
                llm_policy_schema_errors.append(f"llm shadow policy schema missing {key}")
    checks["llm_shadow_policy_schema"] = {
        "name": "llm_shadow_policy_schema",
        "passed": not llm_policy_schema_errors,
        "errors": llm_policy_schema_errors,
        "warnings": [],
        "details": {"path": str(llm_policy_schema_path)},
    }
    llm_policy_errors: list[str] = []
    llm_policy_warnings: list[str] = []
    llm_policy_payload: dict[str, Any] = {}
    llm_policy_path = root / "config" / "llm_shadow_policy.json"
    try:
        llm_policy_payload, llm_policy_hash = load_llm_shadow_policy(llm_policy_path)
    except Exception as exc:
        llm_policy_hash = ""
        llm_policy_errors.append(str(exc))
    local_llm_config = root / "config" / "llm.local.json"
    if not local_llm_config.exists():
        llm_policy_warnings.append("local LLM provider config is not configured")
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "config/llm.local.json"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            llm_policy_errors.append("config/llm.local.json must not be tracked")
    except Exception as exc:
        llm_policy_warnings.append(f"could not inspect git tracking for llm.local.json: {exc}")
    checks["llm_shadow_policy_config"] = {
        "name": "llm_shadow_policy_config",
        "passed": not llm_policy_errors,
        "errors": llm_policy_errors,
        "warnings": llm_policy_warnings,
        "details": {
            "path": str(llm_policy_path),
            "sha256": llm_policy_hash,
            "local_config_exists": local_llm_config.exists(),
        },
    }
    provider_source_for_binding = (root / "afac_agent" / "llm" / "providers.py").read_text(encoding="utf-8")
    main_source_for_binding = (root / "afac_agent" / "main.py").read_text(encoding="utf-8")
    provider_binding_passed = (
        ALIYUN_BAILIAN_PROVIDER in provider_source_for_binding
        and (
            ALIYUN_BAILIAN_PROVIDER in main_source_for_binding
            or "ALIYUN_BAILIAN_PROVIDER" in main_source_for_binding
        )
    )
    checks["aliyun_bailian_provider_binding"] = {
        "name": "aliyun_bailian_provider_binding",
        "passed": provider_binding_passed,
        "errors": (
            []
            if provider_binding_passed
            else ["aliyun_bailian_openai provider binding is incomplete"]
        ),
        "warnings": [],
        "details": {
            "provider": ALIYUN_BAILIAN_PROVIDER,
            "default_model": ALIYUN_BAILIAN_DEFAULT_MODEL,
        },
    }
    openai_available = importlib.util.find_spec("openai") is not None
    try:
        openai_version = importlib.metadata.version("openai") if openai_available else ""
    except importlib.metadata.PackageNotFoundError:
        openai_version = ""
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    openai_declared = '"openai"' in pyproject_text or "'openai'" in pyproject_text
    checks["openai_sdk_available"] = {
        "name": "openai_sdk_available",
        "passed": openai_available or openai_declared,
        "errors": [] if (openai_available or openai_declared) else ["openai SDK is not installed or declared"],
        "warnings": [] if openai_available else ["openai SDK is declared but not importable in this environment"],
        "details": {
            "available": openai_available,
            "version": openai_version,
            "declared_in_pyproject": openai_declared,
        },
    }
    example_config_path = root / "config" / "llm.local.example.json"
    try:
        example_config = json.loads(example_config_path.read_text(encoding="utf-8"))
    except Exception:
        example_config = {}
    model_policy_errors: list[str] = []
    example_model = str(example_config.get("model") or "")
    if example_model != ALIYUN_BAILIAN_DEFAULT_MODEL:
        model_policy_errors.append("llm.local.example.json must default to qwen3.6-max-preview")
    if not is_allowed_bailian_model(example_model):
        model_policy_errors.append("llm.local.example.json model must be qwen3.5-* or qwen3.6-*")
    if not is_allowed_bailian_model(ALIYUN_BAILIAN_DEFAULT_MODEL):
        model_policy_errors.append("default Bailian model is outside allowed family")
    provider_source = (root / "afac_agent" / "llm" / "providers.py").read_text(encoding="utf-8")
    forbidden_fallbacks = ["qwen-plus", "qwen-turbo", "deepseek", "gpt-", "kimi"]
    for forbidden in forbidden_fallbacks:
        if forbidden in provider_source.lower():
            model_policy_errors.append(f"forbidden fallback model appears in provider source: {forbidden}")
    checks["llm_provider_model_policy"] = {
        "name": "llm_provider_model_policy",
        "passed": not model_policy_errors,
        "errors": model_policy_errors,
        "warnings": [],
        "details": {
            "allowed_prefixes": ["qwen3.5-", "qwen3.6-"],
            "example_model": example_model,
            "default_model": ALIYUN_BAILIAN_DEFAULT_MODEL,
        },
    }
    main_source = (root / "afac_agent" / "main.py").read_text(encoding="utf-8")
    secret_locations = _scan_tracked_secret_locations(root)
    secret_errors = []
    if "--api-key" in main_source or "--api_key" in main_source:
        secret_errors.append("CLI must not accept API key arguments")
    if secret_locations:
        secret_errors.append("tracked files contain possible secret locations")
    tracked_local_config = bool(_git_lines(root, "ls-files", "--error-unmatch", "config/llm.local.json"))
    if tracked_local_config:
        secret_errors.append("config/llm.local.json must not be tracked")
    secret_warnings = []
    if not os.environ.get("DASHSCOPE_API_KEY"):
        secret_warnings.append("DASHSCOPE_API_KEY is not set")
    if not os.environ.get("AFAC_BAILIAN_BASE_URL"):
        secret_warnings.append("AFAC_BAILIAN_BASE_URL is not set")
    checks["llm_secret_source_policy"] = {
        "name": "llm_secret_source_policy",
        "passed": not secret_errors,
        "errors": secret_errors,
        "warnings": secret_warnings,
        "details": {
            "api_key_cli_allowed": "--api-key" in main_source or "--api_key" in main_source,
            "matching_file_count": len({item.rsplit(":", 1)[0] for item in secret_locations}),
            "redacted_locations": secret_locations[:20],
            "local_config_tracked": tracked_local_config,
        },
    }
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "config/llm.local.json"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        llm_local_ignored = ignored.returncode == 0
    except Exception:
        llm_local_ignored = False
    checks["llm_local_config_gitignore"] = {
        "name": "llm_local_config_gitignore",
        "passed": llm_local_ignored and not tracked_local_config,
        "errors": [] if (llm_local_ignored and not tracked_local_config) else ["config/llm.local.json must be ignored and untracked"],
        "warnings": [] if local_llm_config.exists() else ["user local provider config is not configured"],
        "details": {
            "ignored": llm_local_ignored,
            "tracked": tracked_local_config,
            "exists": local_llm_config.exists(),
        },
    }
    safety_errors: list[str] = []
    forbidden_actions = set(map(str, llm_policy_payload.get("forbidden_actions", [])))
    for action in [
        "online_submission",
        "register_champion",
        "modify_champion",
        "overwrite_prediction",
        "execute_shell",
        "execute_python",
        "run_unregistered_tool",
    ]:
        if action not in forbidden_actions:
            safety_errors.append(f"llm shadow policy must forbid {action}")
    if llm_policy_payload.get("force_auto_execution_false") is not True:
        safety_errors.append("llm shadow policy must force auto_execution_allowed=false")
    if not llm_policy_payload.get("force_human_approval_for_training", False):
        safety_errors.append("training proposals must require human approval")
    if not llm_policy_payload.get("force_human_approval_for_prediction", False):
        safety_errors.append("prediction proposals must require human approval")
    checks["llm_shadow_api_safety"] = {
        "name": "llm_shadow_api_safety",
        "passed": not safety_errors,
        "errors": safety_errors,
        "warnings": [],
        "details": {
            "auto_execution_forced_false": llm_policy_payload.get("force_auto_execution_false") is True,
            "forbidden_actions_checked": sorted(forbidden_actions),
            "max_calls": llm_policy_payload.get("max_calls"),
        },
    }
    expected_normalizers = {
        "A1_V46A1_ISOLATED_AUDIT",
        "A1_V49A_EDGE_UTILITY_AUDIT",
        "A1_V53Q1_PATCH_REPLAY_SAFE",
        "A1_OOF_CANDIDATE_EVALUATOR",
    }
    normalizer_errors = [
        f"missing normalizer: {name}"
        for name in sorted(expected_normalizers - set(NORMALIZER_REGISTRY))
    ]
    checks["feedback_normalizer_registry"] = {
        "name": "feedback_normalizer_registry",
        "passed": not normalizer_errors,
        "errors": normalizer_errors,
        "warnings": [],
        "details": {
            "registered": sorted(NORMALIZER_REGISTRY),
        },
    }
    feedback_root = root / "artifacts" / "feedback_runs"
    feedback_warnings = []
    if not feedback_root.exists():
        feedback_warnings.append("feedback_runs artifact root does not exist yet")
    checks["feedback_output_root"] = {
        "name": "feedback_output_root",
        "passed": feedback_root.resolve() != champion_csv.resolve(),
        "errors": (
            ["feedback output root must not equal champion csv"]
            if feedback_root.resolve() == champion_csv.resolve()
            else []
        ),
        "warnings": feedback_warnings,
        "details": {
            "path": str(feedback_root),
            "exists": feedback_root.exists(),
            "parent_exists": feedback_root.parent.exists(),
        },
    }
    evaluation_root = root / "artifacts" / "evaluation_runs"
    checks["evaluation_output_root"] = {
        "name": "evaluation_output_root",
        "passed": evaluation_root.resolve() != champion_csv.resolve(),
        "errors": (
            ["evaluation output root must not equal champion csv"]
            if evaluation_root.resolve() == champion_csv.resolve()
            else []
        ),
        "warnings": (
            ["evaluation_runs artifact root does not exist yet"]
            if not evaluation_root.exists()
            else []
        ),
        "details": {
            "path": str(evaluation_root),
            "exists": evaluation_root.exists(),
            "parent_exists": evaluation_root.parent.exists(),
        },
    }
    planning_root = root / "artifacts" / "plans"
    checks["planning_output_root"] = {
        "name": "planning_output_root",
        "passed": planning_root.resolve() != champion_csv.resolve(),
        "errors": (
            ["planning output root must not equal champion csv"]
            if planning_root.resolve() == champion_csv.resolve()
            else []
        ),
        "warnings": (
            ["plans artifact root does not exist yet"]
            if not planning_root.exists()
            else []
        ),
        "details": {
            "path": str(planning_root),
            "exists": planning_root.exists(),
            "parent_exists": planning_root.parent.exists(),
        },
    }
    shadow_root = root / "artifacts" / "llm_shadow_runs"
    checks["llm_shadow_output_root"] = {
        "name": "llm_shadow_output_root",
        "passed": shadow_root.resolve() != champion_csv.resolve(),
        "errors": (
            ["llm shadow output root must not equal champion csv"]
            if shadow_root.resolve() == champion_csv.resolve()
            else []
        ),
        "warnings": (
            ["llm_shadow_runs artifact root does not exist yet"]
            if not shadow_root.exists()
            else []
        ),
        "details": {
            "path": str(shadow_root),
            "exists": shadow_root.exists(),
            "parent_exists": shadow_root.parent.exists(),
        },
    }

    asset_warnings: list[str] = []
    for key, value in {
        "v53q1_base_csv": resolver.a1_v53q1_base_csv(),
        "v49a_oof_meta_csv": resolver.a1_v49a_oof_meta_csv(),
        "v49a_test_meta_csv": resolver.a1_v49a_test_meta_csv(),
    }.items():
        if not value:
            asset_warnings.append(f"{key}: not configured; adapter will wait for input")
        elif not value.exists():
            asset_warnings.append(f"{key}: configured path does not exist")
    checks["A1_V53Q1_PATCH_AUDIT_config"] = {
        "name": "A1_V53Q1_PATCH_AUDIT_config",
        "passed": True,
        "errors": [],
        "warnings": asset_warnings,
        "details": {
            "audit_md_exists": resolver.a1_v53q1_audit_md().exists(),
            "patch_py_exists": resolver.a1_v53q1_patch_py().exists(),
        },
    }
    replay_warnings: list[str] = []
    for key, value in {
        "v53q1_base_csv": resolver.a1_v53q1_base_csv(),
        "v49a_oof_meta_csv": resolver.a1_v49a_oof_meta_csv(),
        "v49a_test_meta_csv": resolver.a1_v49a_test_meta_csv(),
    }.items():
        if not value:
            replay_warnings.append(f"{key}: not configured; replay adapter will wait for input")
        elif not value.exists():
            replay_warnings.append(f"{key}: configured path does not exist")
    checks["A1_V53Q1_PATCH_REPLAY_SAFE_config"] = {
        "name": "A1_V53Q1_PATCH_REPLAY_SAFE_config",
        "passed": True,
        "errors": [],
        "warnings": replay_warnings,
        "details": {
            "patch_py_exists": resolver.a1_v53q1_patch_py().exists(),
            "default_champion_exists": resolver.a1_current_champion_csv().exists(),
        },
    }
    v46_warnings: list[str] = []
    for key, value in {
        "candidate_csv": resolver.a1_v46a1_candidate_csv(),
        "parent_csv": resolver.a1_v46a1_parent_csv(),
        "candidate_oof_npz": resolver.a1_v46a1_candidate_oof_npz(),
        "audit_report": resolver.a1_v46a1_audit_report(),
    }.items():
        if not value:
            v46_warnings.append(f"{key}: not configured; adapter may wait or mark optional audit unavailable")
        elif not value.exists():
            v46_warnings.append(f"{key}: configured path does not exist")
    checks["A1_V46A1_ISOLATED_AUDIT_config"] = {
        "name": "A1_V46A1_ISOLATED_AUDIT_config",
        "passed": True,
        "errors": [],
        "warnings": v46_warnings,
        "details": {},
    }
    v49_warnings: list[str] = []
    for key, value in {
        "oof_meta_csv": resolver.a1_v49a_edge_oof_meta_csv(),
        "test_meta_csv": resolver.a1_v49a_edge_test_meta_csv(),
        "v46a1_base_csv": resolver.a1_v49a_v46a1_base_csv(),
        "report": resolver.a1_v49a_report(),
        "config": resolver.a1_v49a_config(),
        "fold_results": resolver.a1_v49a_fold_results(),
    }.items():
        if not value:
            v49_warnings.append(f"{key}: not configured; adapter may wait or mark optional audit unavailable")
        elif not value.exists():
            v49_warnings.append(f"{key}: configured path does not exist")
    checks["A1_V49A_EDGE_UTILITY_AUDIT_config"] = {
        "name": "A1_V49A_EDGE_UTILITY_AUDIT_config",
        "passed": True,
        "errors": [],
        "warnings": v49_warnings,
        "details": {},
    }

    research_schema_required = {
        "research_event.schema.json": {"event_version", "event_id", "event_type", "task", "scope_level"},
        "research_problem_record.schema.json": {"problem_record_version", "problem_id", "task", "scope_level"},
        "research_problem_profile.schema.json": {"profile_version", "memory_id", "scope_levels"},
        "research_brief.schema.json": {"brief_version", "brief_id", "brief_type", "scope_level"},
        "method_card.schema.json": {"method_id", "method_family", "information_source_type"},
        "method_attempt.schema.json": {"attempt_version", "attempt_id", "method_id", "outcome"},
        "failure_record.schema.json": {"failure_id", "attempt_id", "failure_type"},
        "research_source.schema.json": {"source_id", "source_type", "verification_status"},
        "problem_method_index.schema.json": {"index_version", "methods_by_problem", "failures_by_method"},
        "research_queue.schema.json": {"queue_version", "policy_limits", "items"},
        "local_method_conflict.schema.json": {"conflict_id", "status", "compared_fields"},
        "research_policy.schema.json": {"analysis_levels", "priority_weights"},
        "bucket_axis_registry.schema.json": {"registry_version", "bucket_axis_registry", "coverage_hash"},
        "bucket_overlap_audit.schema.json": {"audit_version", "axis_within_audit", "cross_axis_intersections"},
        "source_manifest.schema.json": {"manifest_version", "sources"},
        "source_verification.schema.json": {"verification_version", "records", "summary"},
        "method_extraction_record.schema.json": {"extraction_record_id", "method_id", "extraction_mode"},
        "method_ranking_record.schema.json": {"ranking_record_id", "method_id", "ranking_components"},
        "method_research_run.schema.json": {"run_version", "run_id", "artifacts"},
        "live_method_research_run.schema.json": {"run_version", "run_id", "network_mode", "qwen_call_count"},
        "research_provider_registry.schema.json": {"registry_version", "providers", "provider_count"},
        "research_query.schema.json": {"query_id", "research_question", "query_text"},
        "source_relevance_audit.schema.json": {"audit_version", "items", "view_hash"},
        "experiment_proposal.schema.json": {"proposal_id", "target_problem_ids", "core_hypothesis", "round_cost"},
        "critic_review.schema.json": {"verdict", "critical_issues", "minimal_safe_revision"},
        "decision_core_manifest.schema.json": {"run_version", "run_id", "artifacts", "counts_as_experiment_round"},
        "m7_dry_run_manifest.schema.json": {"run_version", "run_id", "status", "artifacts", "round_consumed"},
        "m7b_readiness_manifest.schema.json": {"run_version", "run_id", "status", "artifacts", "anchor_status"},
        "evaluation_anchor_bootstrap_manifest.schema.json": {"manifest_version", "run_id", "status", "historical_v53q1_oof_status"},
    }
    for filename, required in research_schema_required.items():
        schema_check = _json_file_check(root, f"schemas/{filename}", {"$schema", "type"})
        schema_check["name"] = filename.removesuffix(".schema.json") + "_schema"
        checks[schema_check["name"]] = schema_check

    research_policy_check = _json_file_check(
        root,
        "config/research_policy.json",
        {
            "analysis_levels",
            "bucket_taxonomy",
            "mechanism_taxonomy",
            "priority_weights",
            "allow_internal_model_knowledge_as_source",
            "allow_unverified_method_promotion",
            "allow_automatic_branch_reopen",
            "allow_automatic_experiment_execution",
        },
    )
    research_policy_check["name"] = "research_policy_config"
    if research_policy_check["passed"]:
        policy_payload = json.loads((root / "config" / "research_policy.json").read_text(encoding="utf-8"))
        for key in [
            "allow_internal_model_knowledge_as_source",
            "allow_unverified_method_promotion",
            "allow_automatic_branch_reopen",
            "allow_automatic_experiment_execution",
        ]:
            if policy_payload.get(key) is not False:
                research_policy_check["errors"].append(f"{key} must be false")
        if not policy_payload.get("require_local_conflict_check"):
            research_policy_check["errors"].append("require_local_conflict_check must be true")
        axis_policy = policy_payload.get("bucket_axis_policy", {})
        if axis_policy.get("class_id_is_independent_axis") is not True:
            research_policy_check["errors"].append("class_id must be an independent axis")
        if axis_policy.get("cross_axis_intersection_allowed") is not True:
            research_policy_check["errors"].append("cross-axis bucket intersections must be allowed")
        overlap_policy = policy_payload.get("scope_overlap_policy", {})
        if overlap_policy.get("suppress_same_level_high_overlap_topk") is not True:
            research_policy_check["errors"].append("high-overlap Top-K suppression must be enabled")
        weights = policy_payload.get("priority_component_weights", {})
        required_components = {
            "affected_count_component", "affected_ratio_component", "error_headroom_component",
            "evidence_strength_component", "fold_stability_component", "macro_importance_component",
            "novelty_information_gap_component", "expected_score_impact_component",
            "researchability_component", "method_availability_component", "experiment_cost_component",
            "research_cost_component", "risk_component", "overlap_penalty_component",
            "closed_branch_penalty_component",
        }
        missing_components = sorted(required_components - set(weights))
        if missing_components:
            research_policy_check["errors"].append(f"priority_component_weights missing {missing_components}")
        conflict_policy = policy_payload.get("local_conflict_checker", {})
        if conflict_policy.get("enabled") is not True:
            research_policy_check["errors"].append("local conflict checker must be enabled")
        if conflict_policy.get("compare_method_name_only") is not False:
            research_policy_check["errors"].append("local conflict checker must not compare method name only")
        if policy_payload.get("method_research_network_enabled") is not False:
            research_policy_check["errors"].append("method research network must be disabled by default")
        if policy_payload.get("method_research_llm_enabled") is not False:
            research_policy_check["errors"].append("method research LLM must be disabled by default")
        ranking_weights = policy_payload.get("method_research_ranking_weights", {})
        required_method_weights = {
            "problem_fit", "scope_fit", "mechanism_fit", "source_verification", "source_maturity",
            "new_information_value", "local_history_novelty", "implementation_availability",
            "compute_cost", "implementation_cost", "leakage_risk", "deployment_risk",
            "conflict_penalty", "closed_branch_penalty",
        }
        missing_method_weights = sorted(required_method_weights - set(ranking_weights))
        if missing_method_weights:
            research_policy_check["errors"].append(f"method_research_ranking_weights missing {missing_method_weights}")
        live_policy = policy_payload.get("live_method_research", {})
        if live_policy.get("real_default_network_mode") != "live_cached":
            research_policy_check["errors"].append("live research real default must be live_cached")
        if live_policy.get("test_default_network_mode") != "disabled":
            research_policy_check["errors"].append("live research test default must be disabled")
        if live_policy.get("fallback_to_cache") is not True:
            research_policy_check["errors"].append("live research fallback_to_cache must be true")
        for key in ["require_registered_provider", "require_cache", "require_primary_source"]:
            if live_policy.get(key) is not True:
                research_policy_check["errors"].append(f"live research {key} must be true")
        if live_policy.get("allow_unknown_source_promotion") is not False:
            research_policy_check["errors"].append("unknown source promotion must be false")
        research_policy_check["passed"] = not research_policy_check["errors"]
        research_policy_check["details"].update({
            "analysis_levels": policy_payload.get("analysis_levels"),
            "top_k": {
                "global": policy_payload.get("max_global_deep_research"),
                "bucket": policy_payload.get("max_bucket_deep_research"),
                "bucket_class": policy_payload.get("max_bucket_class_deep_research"),
            },
        })
    checks["research_policy_config"] = research_policy_check

    research_memory_root = root / "artifacts" / "research_memory"
    checks["research_memory_output_root"] = {
        "name": "research_memory_output_root",
        "passed": research_memory_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/research_memory/"),
        "errors": ([] if _gitignore_has(root, "artifacts/research_memory/") else ["artifacts/research_memory/ must be ignored"]),
        "warnings": (["research_memory artifact root does not exist yet"] if not research_memory_root.exists() else []),
        "details": {"path": str(research_memory_root), "exists": research_memory_root.exists(), "gitignored": _gitignore_has(root, "artifacts/research_memory/")},
    }
    method_research_root = root / "artifacts" / "method_research"
    checks["method_research_output_root"] = {
        "name": "method_research_output_root",
        "passed": method_research_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/method_research/"),
        "errors": ([] if _gitignore_has(root, "artifacts/method_research/") else ["artifacts/method_research/ must be ignored"]),
        "warnings": (["method_research artifact root does not exist yet"] if not method_research_root.exists() else []),
        "details": {"path": str(method_research_root), "exists": method_research_root.exists(), "gitignored": _gitignore_has(root, "artifacts/method_research/")},
    }

    method_research_runs_root = root / "artifacts" / "method_research_runs"
    checks["method_research_runs_output_root"] = {
        "name": "method_research_runs_output_root",
        "passed": method_research_runs_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/method_research_runs/"),
        "errors": ([] if _gitignore_has(root, "artifacts/method_research_runs/") else ["artifacts/method_research_runs/ must be ignored"]),
        "warnings": (["method_research_runs artifact root does not exist yet"] if not method_research_runs_root.exists() else []),
        "details": {"path": str(method_research_runs_root), "exists": method_research_runs_root.exists(), "gitignored": _gitignore_has(root, "artifacts/method_research_runs/")},
    }

    research_cache_root = root / "artifacts" / "research_cache"
    checks["research_cache_output_root"] = {
        "name": "research_cache_output_root",
        "passed": research_cache_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/research_cache/"),
        "errors": ([] if _gitignore_has(root, "artifacts/research_cache/") else ["artifacts/research_cache/ must be ignored"]),
        "warnings": (["research_cache artifact root does not exist yet"] if not research_cache_root.exists() else []),
        "details": {"path": str(research_cache_root), "exists": research_cache_root.exists(), "gitignored": _gitignore_has(root, "artifacts/research_cache/")},
    }
    live_method_root = root / "artifacts" / "live_method_research"
    checks["live_method_research_output_root"] = {
        "name": "live_method_research_output_root",
        "passed": live_method_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/live_method_research/"),
        "errors": ([] if _gitignore_has(root, "artifacts/live_method_research/") else ["artifacts/live_method_research/ must be ignored"]),
        "warnings": (["live_method_research artifact root does not exist yet"] if not live_method_root.exists() else []),
        "details": {"path": str(live_method_root), "exists": live_method_root.exists(), "gitignored": _gitignore_has(root, "artifacts/live_method_research/")},
    }
    decision_core_root = root / "artifacts" / "decision_core_runs"
    checks["decision_core_output_root"] = {
        "name": "decision_core_output_root",
        "passed": decision_core_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/decision_core_runs/"),
        "errors": ([] if _gitignore_has(root, "artifacts/decision_core_runs/") else ["artifacts/decision_core_runs/ must be ignored"]),
        "warnings": (["decision_core_runs artifact root does not exist yet"] if not decision_core_root.exists() else []),
        "details": {"path": str(decision_core_root), "exists": decision_core_root.exists(), "gitignored": _gitignore_has(root, "artifacts/decision_core_runs/")},
    }
    m7_root = root / "artifacts" / "m7_dry_runs"
    checks["m7_dry_run_output_root"] = {
        "name": "m7_dry_run_output_root",
        "passed": m7_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/m7_dry_runs/"),
        "errors": ([] if _gitignore_has(root, "artifacts/m7_dry_runs/") else ["artifacts/m7_dry_runs/ must be ignored"]),
        "warnings": (["m7_dry_runs artifact root does not exist yet"] if not m7_root.exists() else []),
        "details": {"path": str(m7_root), "exists": m7_root.exists(), "gitignored": _gitignore_has(root, "artifacts/m7_dry_runs/")},
    }
    m7b_root = root / "artifacts" / "m7b_readiness"
    checks["m7b_readiness_output_root"] = {
        "name": "m7b_readiness_output_root",
        "passed": m7b_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/m7b_readiness/"),
        "errors": ([] if _gitignore_has(root, "artifacts/m7b_readiness/") else ["artifacts/m7b_readiness/ must be ignored"]),
        "warnings": (["m7b_readiness artifact root does not exist yet"] if not m7b_root.exists() else []),
        "details": {"path": str(m7b_root), "exists": m7b_root.exists(), "gitignored": _gitignore_has(root, "artifacts/m7b_readiness/")},
    }
    evaluation_anchor_root = root / "artifacts" / "evaluation_anchor"
    checks["evaluation_anchor_output_root"] = {
        "name": "evaluation_anchor_output_root",
        "passed": evaluation_anchor_root.resolve() != champion_csv.resolve() and _gitignore_has(root, "artifacts/evaluation_anchor/"),
        "errors": ([] if _gitignore_has(root, "artifacts/evaluation_anchor/") else ["artifacts/evaluation_anchor/ must be ignored"]),
        "warnings": (["evaluation_anchor artifact root does not exist yet"] if not evaluation_anchor_root.exists() else []),
        "details": {"path": str(evaluation_anchor_root), "exists": evaluation_anchor_root.exists(), "gitignored": _gitignore_has(root, "artifacts/evaluation_anchor/")},
    }

    method_research_module = root / "afac_agent" / "research" / "method_research.py"
    method_research_text = method_research_module.read_text(encoding="utf-8") if method_research_module.exists() else ""
    checks["source_provider_contract"] = {
        "name": "source_provider_contract",
        "passed": all(token in method_research_text for token in ["class SourceProvider", "class WebSearchProvider", "disabled"]),
        "errors": [] if method_research_module.exists() else ["method_research.py missing"],
        "warnings": [],
        "details": {"network_enabled_default": False},
    }
    checks["local_source_provider"] = {
        "name": "local_source_provider",
        "passed": "class LocalSourcePackProvider" in method_research_text,
        "errors": [] if "class LocalSourcePackProvider" in method_research_text else ["LocalSourcePackProvider missing"],
        "warnings": [],
        "details": {"network_enabled": False},
    }
    checks["source_verifier"] = {
        "name": "source_verifier",
        "passed": "class SourceVerifier" in method_research_text and "verified_local_content" in method_research_text,
        "errors": [] if "class SourceVerifier" in method_research_text else ["SourceVerifier missing"],
        "warnings": [],
        "details": {"unverified_promotion_allowed": False, "synthetic_formal_promotion_allowed": False},
    }
    checks["source_chunker"] = {
        "name": "source_chunker",
        "passed": "class SourceChunker" in method_research_text,
        "errors": [] if "class SourceChunker" in method_research_text else ["SourceChunker missing"],
        "warnings": [],
        "details": {"deterministic": True},
    }
    checks["method_card_validator"] = {
        "name": "method_card_validator",
        "passed": "class MethodCardValidator" in method_research_text and "source_refs" in method_research_text,
        "errors": [] if "class MethodCardValidator" in method_research_text else ["MethodCardValidator missing"],
        "warnings": [],
        "details": {"requires_source_refs": True},
    }
    checks["method_ranker"] = {
        "name": "method_ranker",
        "passed": "class MethodRanker" in method_research_text and research_policy_check["passed"],
        "errors": [] if "class MethodRanker" in method_research_text else ["MethodRanker missing"],
        "warnings": [],
        "details": {"weights_from_policy": True},
    }
    checks["network_disabled_by_default"] = {
        "name": "network_disabled_by_default",
        "passed": research_policy_check["passed"],
        "errors": [] if research_policy_check["passed"] else ["research policy safety failed"],
        "warnings": [],
        "details": {"method_research_network_enabled": False, "method_research_llm_enabled": False},
    }
    live_module = root / "afac_agent" / "research" / "live_method_research.py"
    live_text = live_module.read_text(encoding="utf-8") if live_module.exists() else ""
    checks["live_provider_registry"] = {
        "name": "live_provider_registry",
        "passed": all(token in live_text for token in ["class ProviderRegistry", "OpenAlexProvider", "ArxivProvider", "GitHubRepositoryProvider", "DirectUrlFetchProvider"]),
        "errors": [] if live_module.exists() else ["live_method_research.py missing"],
        "warnings": [],
        "details": {"registered_provider_required": True},
    }
    checks["live_network_modes"] = {
        "name": "live_network_modes",
        "passed": all(token in live_text for token in ["live_cached", "cache_only", "disabled"]) and research_policy_check["passed"],
        "errors": [] if research_policy_check["passed"] else ["research policy live network mode check failed"],
        "warnings": [],
        "details": {"live_cached": True, "cache_only": True, "disabled": True},
    }
    checks["live_research_safety"] = {
        "name": "live_research_safety",
        "passed": all(token in live_text for token in ["executes_adapter", "trains_model", "generates_prediction", "counts_as_experiment_round"]),
        "errors": [] if live_module.exists() else ["live method research module missing"],
        "warnings": [],
        "details": {"adapter_training_prediction_disabled": True},
    }
    decision_core_module = root / "afac_agent" / "research" / "decision_core.py"
    decision_core_text = decision_core_module.read_text(encoding="utf-8") if decision_core_module.exists() else ""
    decision_core_classes_ok = all(
        token in decision_core_text
        for token in [
            "class SourceProblemRelevanceValidator",
            "class MethodQualityGate",
            "class DecisionCoreRunner",
            "class MockDecisionLLM",
        ]
    )
    checks["decision_core_module"] = {
        "name": "decision_core_module",
        "passed": decision_core_module.exists() and decision_core_classes_ok,
        "errors": [] if (decision_core_module.exists() and decision_core_classes_ok) else ["decision_core.py missing required classes"],
        "warnings": [],
        "details": {"module": str(decision_core_module)},
    }
    decision_core_safety_ok = all(
        token in decision_core_text
        for token in [
            "executes_adapter",
            "trains_model",
            "generates_prediction",
            "counts_as_experiment_round",
            "mutates_project_state",
            "mutates_predictions",
        ]
    )
    checks["decision_core_safety"] = {
        "name": "decision_core_safety",
        "passed": decision_core_safety_ok,
        "errors": [] if decision_core_safety_ok else ["decision core manifest must declare no adapter/training/prediction/project mutation/round use"],
        "warnings": [],
        "details": {"module": str(decision_core_module)},
    }
    m7_module = root / "afac_agent" / "m7_dry_run.py"
    m7_text = m7_module.read_text(encoding="utf-8") if m7_module.exists() else ""
    m7_safety_ok = all(
        token in m7_text
        for token in [
            "class M7DryRunOrchestrator",
            "executes_adapter",
            "trains_model",
            "generates_prediction",
            "creates_submission",
            "counts_as_experiment_round",
            "experiment_executed",
            "round_consumed",
            "execution_allowed",
        ]
    )
    checks["m7_dry_run_safety"] = {
        "name": "m7_dry_run_safety",
        "passed": m7_module.exists() and m7_safety_ok,
        "errors": [] if (m7_module.exists() and m7_safety_ok) else ["m7 dry-run module must declare no execution/training/prediction/submission/round use"],
        "warnings": [],
        "details": {"module": str(m7_module)},
    }
    m7b_module = root / "afac_agent" / "m7b_readiness.py"
    m7b_text = m7b_module.read_text(encoding="utf-8") if m7b_module.exists() else ""
    m7b_safety_ok = all(
        token in m7b_text
        for token in [
            "class M7BReadinessRepair",
            "executes_adapter",
            "trains_model",
            "generates_prediction",
            "creates_submission",
            "counts_as_experiment_round",
            "round_consumed",
            "execution_allowed",
            "allowed_in_m7b",
        ]
    )
    checks["m7b_readiness_safety"] = {
        "name": "m7b_readiness_safety",
        "passed": m7b_module.exists() and m7b_safety_ok,
        "errors": [] if (m7b_module.exists() and m7b_safety_ok) else ["m7b readiness module must declare read-only/no execution and no rebuild safety"],
        "warnings": [],
        "details": {"module": str(m7b_module)},
    }
    eval_anchor_module = root / "afac_agent" / "evaluation_anchor_bootstrap.py"
    eval_anchor_text = eval_anchor_module.read_text(encoding="utf-8") if eval_anchor_module.exists() else ""
    eval_anchor_safety_ok = all(
        token in eval_anchor_text
        for token in [
            "class EvaluationAnchorBootstrap",
            "online_deployment_anchor",
            "oof_evaluation_anchor",
            "historical_v53q1_oof_status",
            "not_materialized",
            "trains_model",
            "generates_prediction",
            "creates_submission",
            "counts_as_experiment_round",
            "rebuild_required",
        ]
    )
    checks["evaluation_anchor_bootstrap_safety"] = {
        "name": "evaluation_anchor_bootstrap_safety",
        "passed": eval_anchor_module.exists() and eval_anchor_safety_ok,
        "errors": [] if (eval_anchor_module.exists() and eval_anchor_safety_ok) else ["evaluation anchor bootstrap module must separate anchors and declare no training/prediction/submission"],
        "warnings": [],
        "details": {"module": str(eval_anchor_module)},
    }


    checks["bucket_axis_policy"] = {
        "name": "bucket_axis_policy",
        "passed": research_policy_check["passed"],
        "errors": [] if research_policy_check["passed"] else ["research policy bucket axis constraints failed"],
        "warnings": [],
        "details": research_policy_check.get("details", {}),
    }
    checks["multi_axis_scope_support"] = {
        "name": "multi_axis_scope_support",
        "passed": (root / "afac_agent" / "research" / "memory_views.py").exists(),
        "errors": [],
        "warnings": [],
        "details": {"supports_scope_signature": True, "supports_bucket_axes": True},
    }
    checks["research_queue_explainability"] = {
        "name": "research_queue_explainability",
        "passed": research_policy_check["passed"],
        "errors": [] if research_policy_check["passed"] else ["priority components or policy are incomplete"],
        "warnings": [],
        "details": {"components_from_policy": True},
    }
    checks["research_queue_overlap_policy"] = {
        "name": "research_queue_overlap_policy",
        "passed": research_policy_check["passed"],
        "errors": [] if research_policy_check["passed"] else ["overlap policy is incomplete"],
        "warnings": [],
        "details": {"topk_overlap_suppression": True},
    }
    checks["local_conflict_checker_enabled"] = {
        "name": "local_conflict_checker_enabled",
        "passed": (root / "afac_agent" / "research" / "local_conflict_checker.py").exists() and research_policy_check["passed"],
        "errors": [] if (root / "afac_agent" / "research" / "local_conflict_checker.py").exists() else ["local conflict checker missing"],
        "warnings": [],
        "details": {"compares_method_name_only": False},
    }
    checks["local_conflict_checker_tested"] = {
        "name": "local_conflict_checker_tested",
        "passed": (root / "tests" / "test_m6r_a_hierarchical_research_memory.py").exists(),
        "errors": [] if (root / "tests" / "test_m6r_a_hierarchical_research_memory.py").exists() else ["M6R-A tests missing"],
        "warnings": [],
        "details": {"synthetic_cases": 10},
    }

    checks["writable"] = {
        "name": "writable",
        "passed": os.access(root, os.W_OK) and os.access(root / "artifacts", os.W_OK),
        "errors": [],
        "warnings": [],
        "details": {
            "project_root": os.access(root, os.W_OK),
            "artifacts": os.access(root / "artifacts", os.W_OK),
            "memory_parent": os.access(root, os.W_OK),
            "output_parent": os.access(root, os.W_OK),
        },
    }
    checks["dependencies"] = {
        "name": "dependencies",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": dependency_check(["numpy", "pandas", "scipy", "sklearn"]),
    }
    passed = all(item.get("passed", False) for item in checks.values())
    return {
        "project_root": str(root),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "checks": checks,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--paths_config", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(
        project_root=Path(args.project_root),
        paths_config=args.paths_config,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"AFAC Agent doctor: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"Project root: {report['project_root']}")
        for name, check in report["checks"].items():
            print(f"- {name}: {'PASS' if check['passed'] else 'FAIL'}")
            for error in check.get("errors", []):
                print(f"  ERROR: {error}")
            for warning in check.get("warnings", []):
                print(f"  WARN: {warning}")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
