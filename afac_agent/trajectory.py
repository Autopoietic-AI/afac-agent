# -*- coding: utf-8 -*-
"""实时、不可补写语义的Trajectory记录。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class TrajectoryLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        if self.path.exists():
            self.entries = json.loads(
                self.path.read_text(encoding="utf-8")
            )

    def append(
        self,
        *,
        state_before: Dict[str, Any],
        decision: Dict[str, Any],
        result: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> None:
        self.entries.append(
            {
                "round": len(self.entries),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "state_before": state_before,
                "current_experiment_config":
                    decision.get("frozen_contract", {}),
                "feedback": result,
                "agent_decision": decision,
                "state_after": state_after,
            }
        )
        self.path.write_text(
            json.dumps(
                self.entries,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
