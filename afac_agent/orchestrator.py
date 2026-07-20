# -*- coding: utf-8 -*-
"""Agent v1主控：Planner → Safety → Tool → State → Trajectory。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .planner import HierarchicalPlanner
from .registry import ToolRegistry
from .safety import SafetyGate
from .state_store import StateStore
from .trajectory import TrajectoryLogger


class AgentOrchestrator:
    def __init__(
        self,
        *,
        project_root: Path,
        state_path: Path,
        registry_path: Path,
        trajectory_path: Path,
    ):
        self.project_root = project_root
        self.state_store = StateStore(state_path)
        self.registry = ToolRegistry(registry_path)
        self.planner = HierarchicalPlanner(self.registry)
        self.safety = SafetyGate()
        self.trajectory = TrajectoryLogger(
            trajectory_path
        )

    def _resolve_command(
        self,
        template: List[str],
        variables: Dict[str, str],
    ) -> List[str]:
        command = []
        for token in template:
            resolved = token
            for key, value in variables.items():
                resolved = resolved.replace(
                    "{" + key + "}",
                    value,
                )
            command.append(resolved)
        return command

    def _missing_required_inputs(
        self,
        tool,
        variables: Dict[str, str],
    ) -> tuple[List[str], Dict[str, str]]:
        missing_inputs: List[str] = []
        missing_files: Dict[str, str] = {}
        required_inputs = getattr(tool, "required_inputs", {}) or {}

        for key, spec in required_inputs.items():
            value = str(variables.get(key, "")).strip()
            if not value:
                missing_inputs.append(key)
                continue
            kind = ""
            if isinstance(spec, dict):
                kind = str(spec.get("kind", ""))
            elif isinstance(spec, str):
                kind = spec
            if kind == "file" and not Path(value).exists():
                missing_inputs.append(key)
                missing_files[key] = value

        return missing_inputs, missing_files

    def run_once(
        self,
        *,
        variables: Dict[str, str],
        execute: bool,
    ) -> Dict[str, Any]:
        state = self.state_store.load()
        before = state.to_dict()
        decision = self.planner.select(state)
        if decision.action == "NONE":
            result = {
                "status": "waiting_for_input",
                "reason": decision.reason,
                "elapsed_seconds": 0.0,
            }
            self.trajectory.append(
                state_before=before,
                decision=decision.to_dict(),
                result=result,
                state_after=state.to_dict(),
            )
            return {
                "decision": decision.to_dict(),
                "result": result,
                "state": state.to_dict(),
            }

        tool = self.registry.get(decision.action)
        safety = self.safety.precheck(state, tool)
        if not safety["passed"]:
            result = {
                "status": "blocked",
                "safety": safety,
                "elapsed_seconds": 0.0,
            }
            self.trajectory.append(
                state_before=before,
                decision=decision.to_dict(),
                result=result,
                state_after=state.to_dict(),
            )
            return {
                "decision": decision.to_dict(),
                "result": result,
                "state": state.to_dict(),
            }

        command = self._resolve_command(
            tool.command_template,
            variables,
        )
        if not tool.command_template:
            result = {
                "status": "waiting_for_input",
                "reason": "unbound_tool",
                "tool": tool.name,
                "missing_inputs": [],
                "command": command,
                "safety": safety,
                "elapsed_seconds": 0.0,
            }
            self.trajectory.append(
                state_before=before,
                decision=decision.to_dict(),
                result=result,
                state_after=state.to_dict(),
            )
            return {
                "decision": decision.to_dict(),
                "result": result,
                "state": state.to_dict(),
            }

        missing_inputs, missing_files = self._missing_required_inputs(
            tool,
            variables,
        )
        unresolved_placeholders = [
            token
            for token in command
            if "{" in token and "}" in token
        ]
        started = time.time()
        if missing_inputs or unresolved_placeholders:
            result = {
                "status": "waiting_for_input",
                "missing_inputs": missing_inputs,
                "missing_files": missing_files,
                "unresolved_placeholders": unresolved_placeholders,
                "command": command,
                "safety": safety,
                "elapsed_seconds": 0.0,
            }
        elif not execute:
            result = {
                "status": "dry_run",
                "command": command,
                "safety": safety,
                "elapsed_seconds": 0.0,
            }
        else:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.project_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=tool.expected_runtime_seconds,
                )
                result = {
                    "status": (
                        "success"
                        if completed.returncode == 0
                        else "failed"
                    ),
                    "returncode": completed.returncode,
                    "command": command,
                    "stdout_tail": completed.stdout[-8000:],
                    "stderr_tail": completed.stderr[-8000:],
                    "safety": safety,
                    "elapsed_seconds":
                        time.time() - started,
                }
            except FileNotFoundError as exc:
                result = {
                    "status": "waiting_for_input",
                    "reason": "executable_or_input_missing",
                    "error": str(exc),
                    "command": command,
                    "safety": safety,
                    "elapsed_seconds": time.time() - started,
                }
            except subprocess.TimeoutExpired as exc:
                result = {
                    "status": "failed",
                    "reason": "timeout",
                    "error": str(exc),
                    "command": command,
                    "safety": safety,
                    "elapsed_seconds": time.time() - started,
                }

        if result["status"] == "success":
            if tool.counts_as_experiment_round:
                state.budget.consume(
                    result["elapsed_seconds"]
                )
            if tool.mutates_project_state:
                if tool.name == "IMPORT_CONFIRMED_HISTORY":
                    state.history_imported = True
                    state.next_required_capability = (
                        "register_anchor"
                    )
                elif tool.name == "REGISTER_ONLINE_ANCHOR":
                    state.anchor_registered = True
                    state.next_required_capability = (
                        "profile_dataset"
                        if not state.data_profile_ready
                        else "analyze_anchor_oof"
                    )
                elif tool.name == "PROFILE_A1_DATASET":
                    state.data_profile_ready = True
                    state.next_required_capability = (
                        "register_anchor"
                        if not state.anchor_registered
                        else "analyze_anchor_oof"
                    )
                elif tool.name == "ANALYZE_ANCHOR_OOF":
                    state.anchor_oof_analyzed = True
                    state.next_required_capability = (
                        "new_isolated_signal"
                    )

                self.state_store.save(state)

        self.trajectory.append(
            state_before=before,
            decision=decision.to_dict(),
            result=result,
            state_after=state.to_dict(),
        )
        return {
            "decision": decision.to_dict(),
            "result": result,
            "state": state.to_dict(),
        }
