# -*- coding: utf-8 -*-
"""Adapter binding for the read-only M4B A1 OOF candidate evaluator."""

from __future__ import annotations

import json
from typing import Any

from afac_agent.adapters.base import (
    AdapterContext,
    AdapterDescription,
    OutputManifest,
    RawExecutionResult,
)
from afac_agent.evaluation.a1_oof_candidate_evaluator import (
    evaluate_a1_oof_candidate,
)
from afac_agent.schemas import ValidationReport


class Adapter:
    def __init__(self) -> None:
        self._evaluation: dict[str, Any] = {}
        self._warnings: list[str] = []

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            tool_name="A1_OOF_CANDIDATE_EVALUATOR",
            adapter_id="A1_OOF_CANDIDATE_EVALUATOR",
            adapter_version="m4b_v1",
            target_problem="A1_oof_candidate_evaluation",
            execution_mode="evaluate",
            read_only=True,
            counts_as_experiment_round=False,
            mutates_predictions=False,
            mutates_project_state=False,
            requires_gpu=False,
        )

    def _evaluate(
        self,
        context: AdapterContext,
        *,
        write_outputs: bool,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        a1_npz = context.path("a1_npz")
        candidate_oof_npz = context.path("candidate_oof_npz")
        if a1_npz is None or candidate_oof_npz is None:
            missing = []
            if a1_npz is None:
                missing.append("a1_npz")
            if candidate_oof_npz is None:
                missing.append("candidate_oof_npz")
            return {}, [f"missing required inputs: {missing}"], []
        return evaluate_a1_oof_candidate(
            a1_npz=a1_npz,
            candidate_oof_npz=candidate_oof_npz,
            parent_oof_npz=context.path("parent_oof_npz"),
            canonical_fold_csv=context.path("canonical_fold_csv"),
            anchor_manifest_json=context.path("anchor_manifest_json"),
            node_bucket_csv=context.path("node_bucket_csv"),
            m2_profile_json=context.path("m2_profile_json"),
            evaluation_policy_json=context.path("evaluation_policy_json"),
            output_dir=context.run_dir,
            evaluation_id=context.identity_hash,
            write_outputs=write_outputs,
        )

    def validate_inputs(self, context: AdapterContext) -> ValidationReport:
        evaluation, errors, warnings = self._evaluate(context, write_outputs=False)
        self._evaluation = evaluation
        self._warnings = warnings
        return ValidationReport(
            name="A1_OOF_CANDIDATE_EVALUATOR",
            passed=not errors,
            errors=errors,
            warnings=warnings,
            details={
                "analysis_tier": evaluation.get("analysis_tier", ""),
                "comparison_scope": evaluation.get("comparison_scope", ""),
                "test_truth_used": evaluation.get("test_truth_used", False),
            },
        )

    def execute(self, context: AdapterContext) -> RawExecutionResult:
        evaluation, errors, warnings = self._evaluate(context, write_outputs=True)
        if errors:
            return RawExecutionResult(
                status="failed",
                returncode=2,
                stderr="\n".join(errors) + "\n",
                failure_reason="evaluation_failed",
                warnings=warnings,
            )
        stdout = json.dumps(
            {
                "adapter": "A1_OOF_CANDIDATE_EVALUATOR",
                "status": "completed",
                "analysis_tier": evaluation.get("analysis_tier"),
                "comparison_scope": evaluation.get("comparison_scope"),
                "candidate_accuracy": evaluation.get("overall_metrics", {}).get(
                    "candidate_accuracy"
                ),
                "parent_identity_status": evaluation.get("parent_identity_status"),
                "test_truth_used": evaluation.get("test_truth_used"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return RawExecutionResult(
            status="completed",
            returncode=0,
            stdout=stdout + "\n",
            stderr="",
            metrics=evaluation,
            bucket_metrics=evaluation.get("bucket_metrics", []),
            class_metrics=evaluation.get("class_metrics", []),
            warnings=warnings,
        )

    def collect_outputs(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
    ) -> OutputManifest:
        return OutputManifest(
            artifacts={
                key: str(value)
                for key, value in raw_result.metrics.get("artifacts", {}).items()
            }
        )

    def normalize_result(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
        outputs: OutputManifest,
    ) -> dict[str, Any]:
        return {
            "status": raw_result.status,
            "metrics": raw_result.metrics,
            "bucket_metrics": raw_result.bucket_metrics,
            "class_metrics": raw_result.class_metrics,
            "warnings": [],
            "failure_reason": raw_result.failure_reason,
        }
