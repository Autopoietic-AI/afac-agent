# -*- coding: utf-8 -*-
"""Build deterministic M4A feedback artifacts from Adapter results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import (
    DEFAULT_FEEDBACK_ROOT,
    FEEDBACK_VERSION,
    NORMALIZER_VERSION,
    pretty_json,
    sha256_file,
    stable_hash,
    validate_adapter_execution_result,
    validate_experiment_feedback,
)
from .normalizers import NORMALIZER_REGISTRY


class FeedbackBuilder:
    def __init__(self, *, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def _resolve(self, value: str | Path | None) -> Path:
        if value is None or str(value).strip() == "":
            return (self.project_root / DEFAULT_FEEDBACK_ROOT).resolve()
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _standard_result(
        self,
        *,
        status: str,
        feedback_id: str = "",
        core_feedback_hash: str = "",
        feedback_kind: str = "",
        evaluation_tier: str = "",
        recommendation: str = "",
        artifacts: dict[str, str] | None = None,
        failure_reason: str = "",
        warnings: list[str] | None = None,
        missing_inputs: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "feedback_id": feedback_id,
            "core_feedback_hash": core_feedback_hash,
            "feedback_kind": feedback_kind,
            "evaluation_tier": evaluation_tier,
            "recommendation": recommendation,
            "artifacts": artifacts or {},
            "failure_reason": failure_reason,
            "warnings": warnings or [],
            "missing_inputs": missing_inputs or [],
            "details": details or {},
        }

    def build(
        self,
        *,
        execution_result_path: str | Path,
        out_root: str | Path | None = None,
        dry_run: bool = False,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        execution_path = self._resolve(execution_result_path)
        output_root = self._resolve(out_root)
        if not execution_path.exists():
            return self._standard_result(
                status="unavailable",
                failure_reason="execution_result_missing",
                missing_inputs=["execution_result"],
            )
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._standard_result(
                status="failed",
                failure_reason="execution_result_unreadable",
                warnings=[str(exc)],
            )
        if not isinstance(execution, dict):
            return self._standard_result(
                status="failed",
                failure_reason="execution_result_schema_invalid",
            )
        schema_errors = validate_adapter_execution_result(execution)
        if schema_errors:
            return self._standard_result(
                status="failed",
                failure_reason="execution_result_schema_invalid",
                warnings=schema_errors,
            )
        tool_name = str(execution.get("tool_name", ""))
        normalizer = NORMALIZER_REGISTRY.get(tool_name)
        if normalizer is None:
            return self._standard_result(
                status="blocked",
                failure_reason="normalizer_not_registered",
                details={"tool_name": tool_name},
            )
        execution_hash = sha256_file(execution_path)
        feedback = normalizer(
            execution,
            execution_hash,
            execution_path.name,
        )
        fingerprint_payload = {
            "feedback_version": FEEDBACK_VERSION,
            "tool_name": feedback["tool_name"],
            "execution_identity_hash": feedback["execution_identity_hash"],
            "execution_result_hash": feedback["execution_result_hash"],
            "normalizer_version": NORMALIZER_VERSION,
            "normalized_evidence": feedback["evidence"],
            "normalized_metrics": feedback["metrics"],
            "candidate_changes": feedback["candidate_changes"],
            "recommendation": feedback["recommendation"],
        }
        feedback_id = stable_hash(fingerprint_payload)
        feedback["feedback_id"] = feedback_id
        validation_errors = validate_experiment_feedback(feedback)
        if validation_errors:
            return self._standard_result(
                status="failed",
                failure_reason="feedback_schema_invalid",
                warnings=validation_errors,
            )
        core_feedback_hash = stable_hash(feedback)
        if dry_run:
            return self._standard_result(
                status="dry_run",
                feedback_id=feedback_id,
                core_feedback_hash=core_feedback_hash,
                feedback_kind=feedback["feedback_kind"],
                evaluation_tier=feedback["evaluation_tier"],
                recommendation=feedback["recommendation"],
                details={
                    "tool_name": tool_name,
                    "normalizer_version": NORMALIZER_VERSION,
                    "would_write_root": str(output_root),
                },
            )
        run_dir = output_root / tool_name / feedback_id
        feedback_path = run_dir / "experiment_feedback.json"
        manifest_path = run_dir / "feedback_manifest.json"
        report_path = run_dir / "FEEDBACK_REPORT.md"
        if feedback_path.exists() and not force_rebuild:
            try:
                previous = json.loads(feedback_path.read_text(encoding="utf-8"))
            except Exception:
                previous = {}
            if previous == feedback:
                return self._standard_result(
                    status="duplicate",
                    feedback_id=feedback_id,
                    core_feedback_hash=core_feedback_hash,
                    feedback_kind=feedback["feedback_kind"],
                    evaluation_tier=feedback["evaluation_tier"],
                    recommendation=feedback["recommendation"],
                    artifacts={
                        "run_dir": str(run_dir),
                        "experiment_feedback": str(feedback_path),
                        "feedback_manifest": str(manifest_path),
                        "feedback_report": str(report_path),
                    },
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(pretty_json(feedback) + "\n", encoding="utf-8")
        manifest = {
            "feedback_version": FEEDBACK_VERSION,
            "feedback_id": feedback_id,
            "core_feedback_hash": core_feedback_hash,
            "tool_name": tool_name,
            "normalizer_version": NORMALIZER_VERSION,
            "execution_result_path": str(execution_path),
            "output_dir": str(run_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deterministic_feedback_json": True,
        }
        manifest_path.write_text(pretty_json(manifest) + "\n", encoding="utf-8")
        report_path.write_text(self._render_report(feedback), encoding="utf-8")
        return self._standard_result(
            status="completed",
            feedback_id=feedback_id,
            core_feedback_hash=core_feedback_hash,
            feedback_kind=feedback["feedback_kind"],
            evaluation_tier=feedback["evaluation_tier"],
            recommendation=feedback["recommendation"],
            artifacts={
                "run_dir": str(run_dir),
                "experiment_feedback": str(feedback_path),
                "feedback_manifest": str(manifest_path),
                "feedback_report": str(report_path),
            },
        )

    def _render_report(self, feedback: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"# Feedback Report: {feedback['tool_name']}",
                "",
                f"- feedback_id: `{feedback['feedback_id']}`",
                f"- kind: `{feedback['feedback_kind']}`",
                f"- tier: `{feedback['evaluation_tier']}`",
                f"- status: `{feedback['status']}`",
                f"- recommendation: `{feedback['recommendation']}`",
                "",
                "## Evidence",
                "",
                "```json",
                pretty_json(feedback["evidence"]),
                "```",
                "",
                "## Limitations",
                "",
                "```json",
                pretty_json(feedback["limitations"]),
                "```",
                "",
            ]
        )
