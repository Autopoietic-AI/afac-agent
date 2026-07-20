# -*- coding: utf-8 -*-
"""Minimal Adapter protocol for AFAC Agent M3A."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol

from afac_agent.schemas import ToolSpec, ValidationReport


@dataclass
class AdapterDescription:
    tool_name: str
    adapter_id: str
    adapter_version: str
    target_problem: str
    execution_mode: str
    read_only: bool
    counts_as_experiment_round: bool
    mutates_predictions: bool
    mutates_project_state: bool
    requires_gpu: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterContext:
    project_root: Path
    tool: ToolSpec
    variables: Dict[str, str]
    run_dir: Path | None = None
    identity_hash: str = ""

    def path(self, key: str) -> Path | None:
        value = str(self.variables.get(key, "")).strip()
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


@dataclass
class RawExecutionResult:
    status: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    bucket_metrics: List[Dict[str, Any]] = field(default_factory=list)
    class_metrics: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    failure_reason: str = ""
    missing_inputs: List[str] = field(default_factory=list)


@dataclass
class OutputManifest:
    artifacts: Dict[str, str] = field(default_factory=dict)


class ToolAdapter(Protocol):
    def describe(self) -> AdapterDescription:
        ...

    def validate_inputs(self, context: AdapterContext) -> ValidationReport:
        ...

    def execute(self, context: AdapterContext) -> RawExecutionResult:
        ...

    def collect_outputs(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
    ) -> OutputManifest:
        ...

    def normalize_result(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
        outputs: OutputManifest,
    ) -> Dict[str, Any]:
        ...
