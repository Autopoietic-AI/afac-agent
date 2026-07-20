# -*- coding: utf-8 -*-
"""M0/M1 contract validation helpers.

These checks are intentionally lightweight and dependency-free.  They validate
the current engineering contracts without changing competition definitions,
folds, gates, predictions, or champion artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import ValidationReport


PROJECT_STATE_REQUIRED = {
    "project": str,
    "task": str,
    "mode": str,
    "dataset_id": str,
    "online_version": str,
    "history_imported": bool,
    "data_profile_ready": bool,
    "anchor_registered": bool,
    "anchor_oof_analyzed": bool,
    "main_contradiction": str,
    "active_layer": str,
    "active_experts": list,
    "closed_branches": list,
    "retained_signals": list,
    "current_error_buckets": dict,
    "next_required_capability": str,
    "budget": dict,
}

BUDGET_REQUIRED = {
    "total_seconds": (int, float),
    "used_seconds": (int, float),
    "safety_margin_seconds": (int, float),
    "rounds_used": int,
    "max_rounds": int,
}

TOOL_REQUIRED = {
    "name": str,
    "task": str,
    "layer": str,
    "description": str,
    "action_type": str,
    "expected_runtime_seconds": int,
    "prediction_changing": bool,
    "submission_creating": bool,
    "read_only": bool,
    "counts_as_experiment_round": bool,
    "mutates_predictions": bool,
    "mutates_project_state": bool,
    "requires_gpu": bool,
    "required_state": dict,
    "forbidden_closed_branches": list,
    "command_template": list,
}

MEMORY_REQUIRED = {
    "version": str,
    "layer": str,
    "status": str,
    "metrics": dict,
    "diagnosis": str,
    "lesson": str,
    "decision": str,
}

ANCHOR_MANIFEST_REQUIRED = {
    "version": str,
    "online_score": (int, float),
    "a1_csv": str,
    "sha256": str,
    "rows": int,
    "num_classes": int,
    "checks": dict,
    "passed": bool,
    "registered_as_anchor": bool,
}


def _load_json(path: Path) -> tuple[Any | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception as exc:  # pragma: no cover - exact parser text varies
        return None, [f"{path}: cannot read JSON: {exc}"]


def _check_required(
    payload: dict[str, Any],
    required: dict[str, Any],
    *,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    for key, expected_type in required.items():
        if key not in payload:
            errors.append(f"{prefix}.{key}: missing")
            continue
        if not isinstance(payload[key], expected_type):
            errors.append(
                f"{prefix}.{key}: expected {expected_type}, "
                f"got {type(payload[key]).__name__}"
            )
    return errors


def validate_project_state_file(path: str | Path) -> ValidationReport:
    path = Path(path)
    payload, errors = _load_json(path)
    warnings: list[str] = []
    details: dict[str, Any] = {"path": str(path)}
    if isinstance(payload, dict):
        errors.extend(
            _check_required(payload, PROJECT_STATE_REQUIRED, prefix="project_state")
        )
        if isinstance(payload.get("budget"), dict):
            errors.extend(
                _check_required(payload["budget"], BUDGET_REQUIRED, prefix="budget")
            )
        if payload.get("task") not in {"A1", "A2"}:
            errors.append("project_state.task: must be A1 or A2")
        if payload.get("online_version") != "v53Q-1":
            warnings.append("online_version differs from confirmed A1 champion")
        if payload.get("online_score") != 0.78:
            warnings.append("online_score differs from confirmed 0.7800")
        details["online_version"] = payload.get("online_version")
        details["online_score"] = payload.get("online_score")
    return ValidationReport(
        name="project_state",
        passed=not errors,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def validate_tool_registry_file(path: str | Path) -> ValidationReport:
    path = Path(path)
    payload, errors = _load_json(path)
    warnings: list[str] = []
    details: dict[str, Any] = {"path": str(path), "tool_count": 0}
    names: set[str] = set()
    if isinstance(payload, dict):
        tools = payload.get("tools")
        if not isinstance(tools, list):
            errors.append("tool_registry.tools: missing or not list")
            tools = []
        details["tool_count"] = len(tools)
        for index, tool in enumerate(tools):
            prefix = f"tools[{index}]"
            if not isinstance(tool, dict):
                errors.append(f"{prefix}: not object")
                continue
            errors.extend(_check_required(tool, TOOL_REQUIRED, prefix=prefix))
            name = str(tool.get("name", ""))
            if name in names:
                errors.append(f"{prefix}.name: duplicate {name}")
            names.add(name)
            if tool.get("task") not in {"A1", "A2"}:
                errors.append(f"{prefix}.task: must be A1 or A2")
            if tool.get("expected_runtime_seconds", 0) <= 0:
                errors.append(f"{prefix}.expected_runtime_seconds: must be positive")
            if not tool.get("command_template"):
                warnings.append(f"{name}: command_template is not yet bound")
            if "required_inputs" in tool and not isinstance(
                tool["required_inputs"], dict
            ):
                errors.append(f"{prefix}.required_inputs: must be object")
    return ValidationReport(
        name="tool_registry",
        passed=not errors,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def _records_from_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        if not path.exists():
            return records, []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                errors.append(f"line {line_number}: record is not object")
        return records, errors

    payload, errors = _load_json(path)
    if not errors and isinstance(payload, dict):
        experiments = payload.get("experiments", [])
        if isinstance(experiments, list):
            return experiments, []
        return [], ["experiments: missing or not list"]
    return [], errors


def validate_memory_records_file(path: str | Path) -> ValidationReport:
    path = Path(path)
    records, errors = _records_from_json_or_jsonl(path)
    versions: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"records[{index}]: not object")
            continue
        errors.extend(_check_required(record, MEMORY_REQUIRED, prefix=f"records[{index}]"))
        version = str(record.get("version", ""))
        if version in versions:
            errors.append(f"records[{index}].version: duplicate {version}")
        versions.add(version)
    return ValidationReport(
        name="memory_records",
        passed=not errors,
        errors=errors,
        details={"path": str(path), "record_count": len(records)},
    )


def validate_trajectory_file(path: str | Path) -> ValidationReport:
    path = Path(path)
    if not path.exists():
        return ValidationReport(
            name="trajectory",
            passed=True,
            warnings=["trajectory file does not exist yet"],
            details={"path": str(path), "entry_count": 0},
        )
    payload, errors = _load_json(path)
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                errors.append(f"entries[{index}]: not object")
                continue
            for key in [
                "round",
                "timestamp",
                "state_before",
                "feedback",
                "agent_decision",
                "state_after",
            ]:
                if key not in item:
                    errors.append(f"entries[{index}].{key}: missing")
    else:
        errors.append("trajectory: must be a JSON list")
    return ValidationReport(
        name="trajectory",
        passed=not errors,
        errors=errors,
        details={
            "path": str(path),
            "entry_count": len(payload) if isinstance(payload, list) else 0,
        },
    )


def validate_anchor_manifest_file(path: str | Path) -> ValidationReport:
    path = Path(path)
    payload, errors = _load_json(path)
    if isinstance(payload, dict):
        errors.extend(
            _check_required(payload, ANCHOR_MANIFEST_REQUIRED, prefix="anchor_manifest")
        )
    return ValidationReport(
        name="anchor_manifest",
        passed=not errors,
        errors=errors,
        details={"path": str(path)},
    )


def validate_a1_data_profile_dir(path: str | Path) -> ValidationReport:
    path = Path(path)
    profile_path = path / "a1_data_profile.json"
    manifest_path = path / "a1_profile_manifest.json"
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"path": str(path)}
    if not profile_path.exists() or not manifest_path.exists():
        return ValidationReport(
            name="a1_data_profile",
            passed=True,
            warnings=["M2 data profile artifact is not present"],
            details={**details, "artifact_valid": False},
        )

    profile, profile_errors = _load_json(profile_path)
    manifest, manifest_errors = _load_json(manifest_path)
    errors.extend(profile_errors)
    errors.extend(manifest_errors)
    if isinstance(profile, dict):
        errors.extend(
            _check_required(
                profile,
                {
                    "metadata": dict,
                    "analysis_tier": str,
                    "dataset": dict,
                    "graph": dict,
                    "exact_hop_policy": dict,
                    "test_policy": dict,
                    "test_profile": dict,
                },
                prefix="a1_data_profile",
            )
        )
        if profile.get("analysis_tier") not in {
            "dataset_only",
            "fold_aware_structure",
            "full_anchor_oof",
        }:
            errors.append("a1_data_profile.analysis_tier: invalid")
        test_policy = profile.get("test_policy", {})
        if isinstance(test_policy, dict) and test_policy.get(
            "truth_dependent_metrics_emitted"
        ):
            errors.append("test truth-dependent metrics must not be emitted")
        exact_hop_policy = profile.get("exact_hop_policy", {})
        if isinstance(exact_hop_policy, dict):
            if exact_hop_policy.get("primary_exact_hop_view") != "either_direction":
                errors.append("primary exact-hop view must be either_direction")
            breakdown = exact_hop_policy.get("directed_exact_hop_breakdown", {})
            if isinstance(breakdown, dict) and breakdown.get("status") != "not_generated":
                errors.append("directed exact-hop breakdown must be explicitly not_generated")
        test_profile = profile.get("test_profile", {})
        if isinstance(test_profile, dict):
            distribution = test_profile.get("champion_predicted_label_distribution", {})
            if isinstance(distribution, dict) and distribution.get("status") == "observed":
                counts = distribution.get("predicted_class_counts", {})
                ratios = distribution.get("predicted_class_ratios", {})
                total = distribution.get("total_test_nodes")
                if isinstance(counts, dict) and sum(counts.values()) != total:
                    errors.append("champion predicted counts must sum to total_test_nodes")
                if isinstance(ratios, dict) and abs(sum(ratios.values()) - 1.0) > 1e-9:
                    errors.append("champion predicted ratios must sum to 1")
    if isinstance(manifest, dict):
        errors.extend(
            _check_required(
                manifest,
                {
                    "profile_version": str,
                    "analysis_tier": str,
                    "core_result_hash": str,
                    "generated_files": list,
                    "read_only": bool,
                    "counts_as_experiment_round": bool,
                    "mutates_project_state": bool,
                    "mutates_predictions": bool,
                    "requires_gpu": bool,
                },
                prefix="a1_profile_manifest",
            )
        )
        if manifest.get("read_only") is not True:
            errors.append("a1_profile_manifest.read_only must be true")
        if manifest.get("counts_as_experiment_round") is not False:
            errors.append("a1_profile_manifest must not consume experiment rounds")
        if manifest.get("mutates_project_state") is not False:
            errors.append("a1_profile_manifest must not mutate project state")
        details["analysis_tier"] = manifest.get("analysis_tier")
        details["core_result_hash"] = manifest.get("core_result_hash")
    details["artifact_valid"] = not errors
    return ValidationReport(
        name="a1_data_profile",
        passed=not errors,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def validate_a1_champion_csv(
    path: str | Path,
    *,
    expected_rows: int,
    num_classes: int,
) -> ValidationReport:
    path = Path(path)
    errors: list[str] = []
    details: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        return ValidationReport(
            name="champion_csv",
            passed=False,
            errors=[f"{path}: missing"],
            details=details,
        )
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    details["rows"] = len(rows)
    if reader.fieldnames != ["test_idx", "label"]:
        errors.append(f"columns must be ['test_idx', 'label'], got {reader.fieldnames}")
    if len(rows) != expected_rows:
        errors.append(f"rows: expected {expected_rows}, got {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        test_idx = row.get("test_idx", "")
        label = row.get("label", "")
        if test_idx in seen:
            errors.append(f"row {index}: duplicated test_idx {test_idx}")
        seen.add(test_idx)
        try:
            value = int(label)
        except ValueError:
            errors.append(f"row {index}: label is not int")
            continue
        if not 0 <= value < num_classes:
            errors.append(f"row {index}: label {value} out of range")
    return ValidationReport(
        name="champion_csv",
        passed=not errors,
        errors=errors,
        details=details,
    )


def reports_to_checks(
    reports: Iterable[ValidationReport],
) -> dict[str, dict[str, Any]]:
    return {report.name: report.to_dict() for report in reports}
