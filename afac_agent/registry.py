# -*- coding: utf-8 -*-
"""Agent受限工具注册表。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .schemas import ProjectState, ToolSpec


class ToolRegistry:
    def __init__(self, path: str | Path):
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        self.tools: Dict[str, ToolSpec] = {
            item["name"]: ToolSpec(**item)
            for item in payload["tools"]
        }

    def get(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise KeyError(f"未注册工具: {name}")
        return self.tools[name]

    def eligible(
        self,
        state: ProjectState,
    ) -> List[ToolSpec]:
        result = []
        for tool in self.tools.values():
            if tool.task != state.task:
                continue
            if any(
                branch in state.closed_branches
                for branch in tool.forbidden_closed_branches
            ):
                continue
            passed = True
            for key, expected in tool.required_state.items():
                if getattr(state, key, None) != expected:
                    passed = False
                    break
            if not passed:
                continue
            if not state.budget.can_run(
                tool.expected_runtime_seconds
            ):
                continue
            result.append(tool)
        return result
