# -*- coding: utf-8 -*-
"""Read-only audit adapter for the v53Q-1 transition-stable Edge-H2 patch."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from afac_agent.adapters.base import (
    AdapterContext,
    AdapterDescription,
    OutputManifest,
    RawExecutionResult,
)
from afac_agent.adapters.runner import sha256_file
from afac_agent.schemas import ValidationReport


N_CLASSES = 10
EXPECTED_ROWS = 2751
MODEL_NAME = "confidence_plus_edge"


def _selected(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_prediction_csv(
    path: Path,
    *,
    name: str,
) -> tuple[list[dict[str, int]], list[str]]:
    errors: list[str] = []
    if not path.exists():
        return [], [f"{name}: missing {path}"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as exc:
        return [], [f"{name}: cannot read csv: {exc}"]
    if fieldnames != ["test_idx", "label"]:
        errors.append(f"{name}: columns must be ['test_idx', 'label'], got {fieldnames}")
        return [], errors
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"{name}: expected {EXPECTED_ROWS} rows, got {len(rows)}")
    seen: set[int] = set()
    parsed: list[dict[str, int]] = []
    for index, row in enumerate(rows):
        if row.get("test_idx") in {None, ""} or row.get("label") in {None, ""}:
            errors.append(f"{name}: row {index} has null/empty field")
            continue
        try:
            test_idx = int(str(row["test_idx"]).strip())
            label = int(str(row["label"]).strip())
        except ValueError:
            errors.append(f"{name}: row {index} has non-integer values")
            continue
        if test_idx in seen:
            errors.append(f"{name}: duplicate test_idx {test_idx}")
        seen.add(test_idx)
        if not 0 <= label < N_CLASSES:
            errors.append(f"{name}: row {index} label {label} out of range")
        parsed.append({"test_idx": test_idx, "label": label})
    return parsed, errors


def _load_meta_csv(
    path: Path,
    *,
    name: str,
    required_columns: set[str],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not path.exists():
        return {}, [f"{name}: missing {path}"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or [])
    except Exception as exc:
        return {}, [f"{name}: cannot read csv: {exc}"]
    missing = sorted(required_columns - fieldnames)
    if missing:
        errors.append(f"{name}: missing columns {missing}")
    selected_rows = [
        row
        for row in rows
        if row.get("model_name") == MODEL_NAME and _selected(row.get("selected"))
    ]
    return {
        "rows": len(rows),
        "columns": sorted(fieldnames),
        "selected_count": len(selected_rows),
    }, errors


def _diff_rows(
    base_rows: list[dict[str, int]],
    champion_rows: list[dict[str, int]],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    base_idx = [row["test_idx"] for row in base_rows]
    champion_idx = [row["test_idx"] for row in champion_rows]
    if base_idx != champion_idx:
        errors.append("base_csv: test_idx sequence must exactly match champion_csv")
        return [], errors
    diffs: list[dict[str, Any]] = []
    for base, champion in zip(base_rows, champion_rows):
        if base["label"] == champion["label"]:
            continue
        diffs.append(
            {
                "test_idx": int(champion["test_idx"]),
                "old_label": int(base["label"]),
                "new_label": int(champion["label"]),
                "transition": f"{int(base['label'])}->{int(champion['label'])}",
            }
        )
    return diffs, errors


def _audit_document_consistency(
    path: Path,
    diffs: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"audit markdown could not be parsed: {exc}")
        return False, warnings
    consistent = True
    if "Changes: `4`" in text and len(diffs) != 4:
        consistent = False
    for diff in diffs:
        old_new = f"{diff['old_label']} -> {diff['new_label']}"
        if str(diff["test_idx"]) not in text or old_new not in text:
            consistent = False
    return consistent, warnings


class Adapter:
    def __init__(self) -> None:
        self._details: dict[str, Any] = {}
        self._warnings: list[str] = []

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            tool_name="A1_V53Q1_PATCH_AUDIT",
            adapter_id="A1_V53Q1_PATCH_AUDIT",
            adapter_version="m3a_v1",
            target_problem="A1_v53Q1_champion_patch_audit",
            execution_mode="audit",
            read_only=True,
            counts_as_experiment_round=False,
            mutates_predictions=False,
            mutates_project_state=False,
            requires_gpu=False,
        )

    def _build_audit(self, context: AdapterContext) -> tuple[dict[str, Any], list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        champion = context.path("anchor_csv")
        base = context.path("v53q1_base_csv")
        oof_meta = context.path("v49a_oof_meta_csv")
        test_meta = context.path("v49a_test_meta_csv")
        audit_md = context.path("v53q1_audit_md")
        patch_py = context.path("v53q1_patch_py")
        assert champion is not None
        assert base is not None
        assert oof_meta is not None
        assert test_meta is not None
        assert audit_md is not None

        champion_rows, champion_errors = _load_prediction_csv(champion, name="champion_csv")
        base_rows, base_errors = _load_prediction_csv(base, name="base_csv")
        errors.extend(champion_errors)
        errors.extend(base_errors)

        diffs: list[dict[str, Any]] = []
        if not champion_errors and not base_errors:
            diffs, diff_errors = _diff_rows(base_rows, champion_rows)
            errors.extend(diff_errors)

        oof_info, oof_errors = _load_meta_csv(
            oof_meta,
            name="v49a_oof_meta_csv",
            required_columns={
                "fold",
                "global_idx",
                "rescue",
                "damage",
                "neutral",
                "base_pred",
                "h2_pred",
                "model_name",
                "rescue_score",
                "selected",
            },
        )
        test_info, test_errors = _load_meta_csv(
            test_meta,
            name="v49a_test_meta_csv",
            required_columns={
                "global_idx",
                "base_pred",
                "h2_pred",
                "model_name",
                "rescue_score",
                "selected",
            },
        )
        errors.extend(oof_errors)
        errors.extend(test_errors)

        audit_consistent = False
        if audit_md.exists():
            audit_consistent, audit_warnings = _audit_document_consistency(audit_md, diffs)
            warnings.extend(audit_warnings)
        else:
            errors.append(f"v53q1_audit_md: missing {audit_md}")

        patch_hash = ""
        if patch_py:
            if patch_py.exists():
                patch_hash = sha256_file(patch_py)
            else:
                warnings.append(f"optional patch source missing: {patch_py}")

        champion_hash = sha256_file(champion) if champion.exists() else ""
        base_hash = sha256_file(base) if base.exists() else ""
        details = {
            "champion_hash": champion_hash,
            "champion_rows": len(champion_rows),
            "base_hash": base_hash,
            "base_rows": len(base_rows),
            "patch_diff_count": len(diffs),
            "patch_transitions": diffs,
            "oof_meta_hash": sha256_file(oof_meta) if oof_meta.exists() else "",
            "oof_meta_rows": oof_info.get("rows", 0),
            "oof_meta_selected_count": oof_info.get("selected_count", 0),
            "test_meta_hash": sha256_file(test_meta) if test_meta.exists() else "",
            "test_meta_rows": test_info.get("rows", 0),
            "test_meta_selected_count": test_info.get("selected_count", 0),
            "audit_md_hash": sha256_file(audit_md) if audit_md.exists() else "",
            "patch_source_hash": patch_hash,
            "audit_document_consistent": audit_consistent,
            "champion_integrity_pass": not champion_errors,
        }
        return details, errors, warnings

    def validate_inputs(self, context: AdapterContext) -> ValidationReport:
        details, errors, warnings = self._build_audit(context)
        self._details = details
        self._warnings = warnings
        return ValidationReport(
            name="A1_V53Q1_PATCH_AUDIT",
            passed=not errors,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def execute(self, context: AdapterContext) -> RawExecutionResult:
        details = self._details or self._build_audit(context)[0]
        stdout = json.dumps(
            {
                "adapter": "A1_V53Q1_PATCH_AUDIT",
                "status": "completed",
                "patch_diff_count": details.get("patch_diff_count"),
                "oof_meta_selected_count": details.get("oof_meta_selected_count"),
                "test_meta_selected_count": details.get("test_meta_selected_count"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return RawExecutionResult(
            status="completed",
            returncode=0,
            stdout=stdout + "\n",
            stderr="",
            metrics=details,
            warnings=self._warnings,
        )

    def collect_outputs(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
    ) -> OutputManifest:
        assert context.run_dir is not None
        audit_details = context.run_dir / "audit_details.json"
        audit_details.write_text(
            json.dumps(
                raw_result.metrics,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return OutputManifest(
            artifacts={
                "audit_details": str(audit_details),
            }
        )

    def normalize_result(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
        outputs: OutputManifest,
    ) -> Dict[str, Any]:
        return {
            "status": raw_result.status,
            "metrics": raw_result.metrics,
            "bucket_metrics": raw_result.bucket_metrics,
            "class_metrics": raw_result.class_metrics,
            "warnings": [],
            "failure_reason": raw_result.failure_reason,
        }
