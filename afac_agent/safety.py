# -*- coding: utf-8 -*-
"""比赛与科研安全门。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .schemas import ProjectState, ToolSpec


class SafetyGate:
    def precheck(
        self,
        state: ProjectState,
        tool: ToolSpec,
    ) -> Dict[str, Any]:
        checks = {
            "serial_only": True,
            "budget_available": state.budget.can_run(
                tool.expected_runtime_seconds
            ),
            "tool_registered": True,
            "closed_branch_not_reentered": not any(
                branch in state.closed_branches
                for branch in tool.forbidden_closed_branches
            ),
            "prediction_change_declared":
                isinstance(tool.prediction_changing, bool),
            "submission_change_declared":
                isinstance(tool.submission_creating, bool),
            "read_only_declared":
                isinstance(tool.read_only, bool),
            "round_consumption_declared":
                isinstance(tool.counts_as_experiment_round, bool),
            "project_state_mutation_declared":
                isinstance(tool.mutates_project_state, bool),
            "prediction_mutation_declared":
                isinstance(tool.mutates_predictions, bool),
            "gpu_requirement_declared":
                isinstance(tool.requires_gpu, bool),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
        }

    @staticmethod
    def validate_a1_csv(
        path: str | Path,
        *,
        expected_rows: int,
        num_classes: int,
    ) -> Dict[str, Any]:
        import pandas as pd

        frame = pd.read_csv(path)
        checks = {
            "columns_exact":
                list(frame.columns) == ["test_idx", "label"],
            "row_count":
                len(frame) == expected_rows,
            "test_idx_unique":
                not frame["test_idx"].duplicated().any(),
            "no_null":
                not frame.isna().any().any(),
            "label_integer":
                bool(
                    (frame["label"].astype(int) == frame["label"])
                    .all()
                ),
            "label_range":
                bool(
                    frame["label"].between(
                        0, num_classes - 1
                    ).all()
                ),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
        }
