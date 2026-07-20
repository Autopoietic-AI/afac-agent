# -*- coding: utf-8 -*-
"""统一Schema：状态、工具、动作、结果、预算与轨迹。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


TaskName = Literal["A1", "A2"]
DecisionName = Literal[
    "RUN",
    "STOP_BRANCH",
    "RETURN_TO_ANCHOR",
    "WAIT_FOR_INPUT",
    "FINALIZE",
]


@dataclass
class BudgetState:
    total_seconds: int = 7200
    used_seconds: float = 0.0
    safety_margin_seconds: int = 300
    rounds_used: int = 0
    max_rounds: int = 12

    @property
    def remaining_seconds(self) -> float:
        return max(
            0.0,
            self.total_seconds
            - self.used_seconds
            - self.safety_margin_seconds,
        )

    def can_run(self, expected_seconds: int) -> bool:
        return (
            self.rounds_used < self.max_rounds
            and expected_seconds <= self.remaining_seconds
        )

    def consume(self, elapsed_seconds: float) -> None:
        self.used_seconds += float(elapsed_seconds)
        self.rounds_used += 1


@dataclass
class ProjectState:
    project: str
    task: TaskName
    mode: str
    dataset_id: str
    online_score: Optional[float]
    online_version: str
    history_imported: bool
    data_profile_ready: bool
    anchor_registered: bool
    anchor_oof_analyzed: bool
    main_contradiction: str
    active_layer: str
    active_experts: List[str]
    closed_branches: List[str]
    retained_signals: List[str]
    current_error_buckets: Dict[str, Any]
    next_required_capability: str
    budget: BudgetState

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolSpec:
    name: str
    task: TaskName
    layer: str
    description: str
    action_type: str
    expected_runtime_seconds: int
    prediction_changing: bool
    submission_creating: bool
    read_only: bool = False
    counts_as_experiment_round: bool = True
    mutates_predictions: bool = False
    mutates_project_state: bool = True
    requires_gpu: bool = False
    required_state: Dict[str, Any] = field(default_factory=dict)
    required_inputs: Dict[str, Any] = field(default_factory=dict)
    forbidden_closed_branches: List[str] = field(default_factory=list)
    command_template: List[str] = field(default_factory=list)


@dataclass
class AgentDecision:
    decision: DecisionName
    action: str
    reason: str
    contradiction_target: str
    expected_information_gain: str
    expected_model_gain: str
    risks: List[str]
    frozen_contract: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StandardExperimentResult:
    version: str
    task: TaskName
    status: str
    metric_name: str
    overall_metric: Optional[float]
    elapsed_seconds: float
    config: Dict[str, Any]
    folds: List[Dict[str, Any]] = field(default_factory=list)
    buckets: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    rescue_damage: Dict[str, Any] = field(default_factory=dict)
    oracle: Dict[str, Any] = field(default_factory=dict)
    distribution_shift: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    diagnosis: str = ""
    lesson: str = ""
    decision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
