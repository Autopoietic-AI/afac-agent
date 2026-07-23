# -*- coding: utf-8 -*-
"""AFAC Agent v1 B2 sequence recommendation package (AFAC2026 B榜推荐).

B2 is isolated from A1/A2/B1:
- task_family: sequence_recommendation
- data_root: C:/Users/李天皓/agent比赛/B推荐
- fold_identity: AFAC_B2_FOLD_V1
- evaluation_anchor: B2_EVAL_ANCHOR_V1
- online_anchor: B2_ONLINE_ANCHOR
- memory_namespace: B2
"""
from __future__ import annotations

from .anchors import B2_EVAL_ANCHOR_ID, B2_ONLINE_ANCHOR_ID
from .closed_loop import B2ClosedLoopRunner, run_b2_closed_loop
from .fold import AFAC_B2_FOLD_V1
from .task_adapter import B2Dataset, B2TaskAdapter

__all__ = [
    "AFAC_B2_FOLD_V1",
    "B2ClosedLoopRunner",
    "B2Dataset",
    "B2_EVAL_ANCHOR_ID",
    "B2_ONLINE_ANCHOR_ID",
    "B2TaskAdapter",
    "run_b2_closed_loop",
]
