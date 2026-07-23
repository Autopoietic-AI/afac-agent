# -*- coding: utf-8 -*-
"""AFAC v2.0 hierarchical model constructor (model genomes).

A model genome describes a candidate solution across layers L0..L9.
Every child genome must follow the rule:

    Parent + One Primary Change + Optional Safety Adjustment

i.e. exactly one primary layer (L1..L7, L9) may differ from the parent,
plus optional safety adjustments restricted to {L0 validation, L8
decision_postprocessing}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afac_agent.research.event_store import stable_hash

MODEL_LAYERS: dict[str, str] = {
    "L0": "validation",
    "L1": "data_view",
    "L2": "information_source",
    "L3": "feature",
    "L4": "representation",
    "L5": "model",
    "L6": "objective",
    "L7": "fusion_routing",
    "L8": "decision_postprocessing",
    "L9": "deployment",
}

# Layers allowed to carry optional safety adjustments alongside the one
# primary change.
SAFETY_ADJUSTMENT_LAYERS = frozenset({"L0", "L8"})


@dataclass
class ModelGenome:
    """A layered description of one candidate solution."""

    genome_id: str = ""
    layers: dict[str, Any] = field(default_factory=dict)  # layer id -> descriptor
    parent_genome_id: str = ""
    changed_layers: list[str] = field(default_factory=list)
    fixed_layers: list[str] = field(default_factory=list)
    new_information_source: str = ""
    target_problem_id: str = ""
    adapter_id: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    ablation: dict[str, Any] = field(default_factory=dict)
    success_condition: str = ""
    failure_condition: str = ""
    stop_condition: str = ""

    def __post_init__(self) -> None:
        if not self.genome_id:
            self.genome_id = stable_hash({"model_layers": self.layers})


def validate_genome_delta(parent: ModelGenome, child: ModelGenome) -> dict[str, Any]:
    """Validate that child = parent + one primary change + optional safety fix.

    Returns ``{"status": "ok"|"violation", "violations": [...]}``.
    """
    violations: list[str] = []
    layer_ids = set(parent.layers) | set(child.layers)

    unknown = sorted(layer_ids - set(MODEL_LAYERS))
    if unknown:
        violations.append(f"unknown_layers:{','.join(unknown)}")

    changed = sorted(
        layer for layer in layer_ids if parent.layers.get(layer) != child.layers.get(layer)
    )
    primary = [layer for layer in changed if layer not in SAFETY_ADJUSTMENT_LAYERS]

    if len(primary) == 0:
        violations.append("no_primary_change")
    elif len(primary) > 1:
        violations.append(f"multiple_primary_changes:{','.join(primary)}")

    safety_changed = [layer for layer in changed if layer in SAFETY_ADJUSTMENT_LAYERS]
    if len(safety_changed) > len(SAFETY_ADJUSTMENT_LAYERS):
        violations.append("too_many_safety_adjustments")

    return {
        "status": "violation" if violations else "ok",
        "violations": violations,
        "changed_layers": changed,
        "primary_changed_layers": primary,
        "safety_adjusted_layers": safety_changed,
    }
