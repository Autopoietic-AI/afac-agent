# -*- coding: utf-8 -*-
"""AFAC v2.0 problem hierarchy.

Tree levels: Task -> Pipeline Stage -> Bucket -> Class/Item Type ->
Error Mechanism.  Each node tracks evidence, headroom, hypotheses and
closed routes so the research loop can target the highest-value open
problems instead of re-treading finished ground.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

from afac_agent.research.event_store import stable_hash

EPSILON = 1e-9


class ProblemLevel(IntEnum):
    TASK = 0
    PIPELINE_STAGE = 1
    BUCKET = 2
    CLASS_OR_ITEM_TYPE = 3
    ERROR_MECHANISM = 4


LEVEL_NAMES = [level.name.lower() for level in ProblemLevel]


@dataclass
class ProblemNode:
    """One node in the problem hierarchy tree."""

    node_id: str = ""
    level: int = 0
    task: str = ""
    pipeline_stage: str = ""
    bucket: str = ""
    class_or_item_type: str = ""
    error_mechanism: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)  # {evidence_id, metric, value}
    headroom: float = 0.0
    uncertainty: float = 0.0
    expected_information_gain: float = 0.0
    validation_risk: float = 0.0
    compute_cost: float = 1.0
    current_best: float = 0.0
    open_hypotheses: list[str] = field(default_factory=list)
    closed_routes: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    def path(self) -> dict[str, str]:
        """Identity path from the root down to this node's level."""
        parts = [
            ("task", self.task),
            ("pipeline_stage", self.pipeline_stage),
            ("bucket", self.bucket),
            ("class_or_item_type", self.class_or_item_type),
            ("error_mechanism", self.error_mechanism),
        ]
        return {name: value for name, value in parts[: self.level + 1]}

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = stable_hash({"problem_path": self.path()})


class ProblemHierarchy:
    """Level-validated tree of :class:`ProblemNode` objects."""

    def __init__(self) -> None:
        self.nodes: dict[str, ProblemNode] = {}
        self.root_ids: list[str] = []

    def add_node(self, node: ProblemNode, parent_id: str | None = None) -> str:
        """Add a node, validating that it sits exactly one level below its parent."""
        if node.node_id in self.nodes:
            return node.node_id
        if parent_id is None:
            if node.level != int(ProblemLevel.TASK):
                raise ValueError("only task-level nodes may be roots")
            self.root_ids.append(node.node_id)
        else:
            parent = self.nodes.get(parent_id)
            if parent is None:
                raise ValueError(f"unknown parent node: {parent_id}")
            if node.level != parent.level + 1:
                raise ValueError(
                    f"level order violation: parent level {parent.level}, child level {node.level}"
                )
            parent_path = parent.path()
            child_path = node.path()
            for key, value in parent_path.items():
                if child_path.get(key) != value:
                    raise ValueError(f"path mismatch at {key}: {child_path.get(key)!r} != {value!r}")
            parent.children.append(node.node_id)
        self.nodes[node.node_id] = node
        return node.node_id

    def close_route(self, node_id: str, route: str) -> None:
        node = self.nodes[node_id]
        if route not in node.closed_routes:
            node.closed_routes.append(route)
        if route in node.open_hypotheses:
            node.open_hypotheses.remove(route)

    def open_hypothesis(self, node_id: str, hypothesis: str) -> None:
        node = self.nodes[node_id]
        if hypothesis not in node.open_hypotheses:
            node.open_hypotheses.append(hypothesis)

    @staticmethod
    def target_score(node: ProblemNode) -> float:
        """Rank score: expected value per unit of risk-adjusted compute."""
        return (node.headroom * node.expected_information_gain) / max(
            EPSILON, node.compute_cost * (1.0 + node.validation_risk)
        )

    def select_targets(self, strategy: str = "roi", top_k: int | None = None) -> list[ProblemNode]:
        """Rank nodes with open headroom by the target score (descending)."""
        if strategy != "roi":
            raise ValueError(f"unknown selection strategy: {strategy}")
        candidates = [n for n in self.nodes.values() if n.headroom > 0 and n.open_hypotheses]
        ranked = sorted(candidates, key=lambda n: (-self.target_score(n), n.node_id))
        return ranked if top_k is None else ranked[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_ids": list(self.root_ids),
            "nodes": {node_id: asdict(node) for node_id, node in sorted(self.nodes.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemHierarchy":
        hierarchy = cls()
        hierarchy.root_ids = list(payload.get("root_ids", []))
        for node_id, node_payload in payload.get("nodes", {}).items():
            node = ProblemNode(**node_payload)
            node.node_id = node_id
            hierarchy.nodes[node_id] = node
        return hierarchy
