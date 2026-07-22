# -*- coding: utf-8 -*-
"""A2 task integration package (recommendation task).

Modules:
- task_adapter: read-only A2 data loading and validation.
- evaluator: ranking metrics (NDCG@10 / HitRate@10 / MRR@10 / candidate recall).
- fold: AFAC_A2_FOLD_V1 validation.
- profiler: A2 data profiler and bucket registry.
- anchors: A2 online/evaluation anchor manifests.
- portfolio: A2 asset registration and verification.
- fusion: recommendation fusion operators with cross-fit.
- complementarity: retriever/ranker complementarity audit.
- integration: A2 integration dry-run orchestrator.
"""
from __future__ import annotations

from .anchors import (
    A2_EVAL_ANCHOR_ID,
    A2_ONLINE_ANCHOR_ID,
    materialize_evaluation_anchor,
    materialize_online_anchor,
)
from .evaluator import A2Evaluator, A2ScoreEvaluation
from .fold import AFAC_A2_FOLD_V1, validate_fold_csv
from .integration import A2IntegrationRunner, run_a2_integration
from .task_adapter import A2Dataset, A2TaskAdapter

__all__ = [
    "A2_EVAL_ANCHOR_ID",
    "A2_ONLINE_ANCHOR_ID",
    "A2Dataset",
    "A2Evaluator",
    "A2IntegrationRunner",
    "A2ScoreEvaluation",
    "A2TaskAdapter",
    "AFAC_A2_FOLD_V1",
    "materialize_evaluation_anchor",
    "materialize_online_anchor",
    "run_a2_integration",
    "validate_fold_csv",
]
