# -*- coding: utf-8 -*-
"""B1 node classification package (AFAC2026 B-list).

B1 establishes an entirely separate namespace from A1/A2:
- task_family: node_classification
- data_root: C:/Users/李天皓/agent比赛/B分类
- fold_identity: AFAC_B1_FOLD_V1
- evaluation_anchor: B1_EVAL_ANCHOR_V1
- memory_namespace: B1

The B1 closed loop is data-first, autonomous, single-process, single-GPU/CPU,
bounded to 3 scientific rounds and 2 hours wall-clock.
"""
from __future__ import annotations

from .closed_loop import B1ClosedLoopRunner, run_b1_closed_loop
from .task_adapter import NodeClassificationTaskAdapter

__all__ = ["B1ClosedLoopRunner", "NodeClassificationTaskAdapter", "run_b1_closed_loop"]
