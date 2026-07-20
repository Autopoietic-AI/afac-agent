# -*- coding: utf-8 -*-
"""项目状态持久化。"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import BudgetState, ProjectState


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> ProjectState:
        payload = json.loads(
            self.path.read_text(encoding="utf-8")
        )
        payload["budget"] = BudgetState(**payload["budget"])
        return ProjectState(**payload)

    def save(self, state: ProjectState) -> None:
        self.path.write_text(
            json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
