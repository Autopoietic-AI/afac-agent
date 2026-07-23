# -*- coding: utf-8 -*-
"""AFAC v2.0 competition intelligence + MLE-STAR loop.

Card store for distilled competition knowledge (solutions, methods,
failure modes, validation schemes) plus a state machine implementing the
MLE-STAR refinement loop:

    strong_initial_solution -> solution_decomposition -> component_ablation
    -> bottleneck_selection -> targeted_refinement -> controlled_evaluation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from afac_agent.research.event_store import stable_hash


@dataclass
class CardBase:
    card_id: str = ""
    task_regime: str = ""
    data_regime: str = ""
    error_mechanism: str = ""
    validation_mechanism: str = ""
    compute_budget: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""

    def __post_init__(self) -> None:
        if not self.card_id:
            self.card_id = stable_hash(
                {
                    "card_type": type(self).__name__,
                    "task_regime": self.task_regime,
                    "data_regime": self.data_regime,
                    "error_mechanism": self.error_mechanism,
                    "validation_mechanism": self.validation_mechanism,
                    "compute_budget": self.compute_budget,
                    "content": self.content,
                    "source": self.source,
                }
            )


@dataclass
class CompetitionSolutionCard(CardBase):
    """A full competition solution distilled into reusable form."""


@dataclass
class MethodCard(CardBase):
    """A single method / technique with its regime of applicability."""


@dataclass
class FailureModeCard(CardBase):
    """A known failure mode and how it was diagnosed / mitigated."""


@dataclass
class ValidationCard(CardBase):
    """A validation scheme and the regime where it is trustworthy."""


_CARD_TYPES = {
    "CompetitionSolutionCard": CompetitionSolutionCard,
    "MethodCard": MethodCard,
    "FailureModeCard": FailureModeCard,
    "ValidationCard": ValidationCard,
}

_FILTER_FIELDS = (
    "task_regime",
    "data_regime",
    "error_mechanism",
    "validation_mechanism",
    "compute_budget",
)


class CardStore:
    def __init__(self) -> None:
        self.cards: dict[str, CardBase] = {}

    def add(self, card: CardBase) -> str:
        self.cards[card.card_id] = card
        return card.card_id

    def retrieve(
        self,
        task_regime: str | None = None,
        data_regime: str | None = None,
        error_mechanism: str | None = None,
        validation_mechanism: str | None = None,
        compute_budget: str | None = None,
    ) -> list[CardBase]:
        """Return cards ranked by specificity of match (most specific first).

        A card must match at least one provided criterion to be returned.
        Ranking: number of matched criteria descending, then number of
        mismatched criteria ascending, then card_id for determinism.
        """
        filters = {
            "task_regime": task_regime,
            "data_regime": data_regime,
            "error_mechanism": error_mechanism,
            "validation_mechanism": validation_mechanism,
            "compute_budget": compute_budget,
        }
        active = {key: value for key, value in filters.items() if value is not None}
        if not active:
            return sorted(self.cards.values(), key=lambda c: c.card_id)
        scored: list[tuple[int, int, str, CardBase]] = []
        for card in self.cards.values():
            matched = sum(1 for key, value in active.items() if getattr(card, key) == value)
            mismatched = sum(1 for key, value in active.items() if getattr(card, key) != value)
            if matched == 0:
                continue
            scored.append((-matched, mismatched, card.card_id, card))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [card for _, _, _, card in scored]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": [
                {"card_type": type(card).__name__, "payload": asdict(card)}
                for card in sorted(self.cards.values(), key=lambda c: c.card_id)
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CardStore":
        store = cls()
        for entry in payload.get("cards", []):
            card_type = _CARD_TYPES[entry["card_type"]]
            store.add(card_type(**entry["payload"]))
        return store


MLESTAR_STAGES = [
    "strong_initial_solution",
    "solution_decomposition",
    "component_ablation",
    "bottleneck_selection",
    "targeted_refinement",
    "controlled_evaluation",
]

_STAGE_REQUIRED_INPUTS: dict[str, list[str]] = {
    "strong_initial_solution": ["train_data", "baseline_config", "validation_split"],
    "solution_decomposition": ["initial_solution", "component_inventory"],
    "component_ablation": ["components", "evaluation_protocol"],
    "bottleneck_selection": ["ablation_results"],
    "targeted_refinement": ["bottleneck_component", "refinement_hypotheses"],
    "controlled_evaluation": ["refined_solution", "held_out_validation"],
}


class MLESTARLoop:
    """Ordered state machine for the MLE-STAR refinement loop."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def current_stage(self) -> str | None:
        """Current stage name, or None once the loop is complete."""
        if len(self.records) >= len(MLESTAR_STAGES):
            return None
        return MLESTAR_STAGES[len(self.records)]

    def _validate_ablation_record(self, record: dict[str, Any]) -> None:
        """Ablation records must name each component and its measured delta."""
        ablations = record.get("ablations")
        if ablations is None:
            ablations = [record] if "component" in record or "delta" in record else []
        if not ablations:
            raise ValueError("component_ablation record must name component and measured delta")
        for entry in ablations:
            if "component" not in entry or "delta" not in entry:
                raise ValueError("ablation entries must include 'component' and 'delta'")

    def advance(self, record: dict[str, Any]) -> str:
        """Validate and store the record for the current stage, then advance.

        The record's "stage" (if present) must equal the current stage;
        out-of-order advancement raises ValueError.
        """
        stage = self.current_stage()
        if stage is None:
            raise ValueError("MLE-STAR loop already complete")
        declared = record.get("stage", stage)
        if declared != stage:
            raise ValueError(f"stage order violation: expected {stage!r}, got {declared!r}")
        if stage == "component_ablation":
            self._validate_ablation_record(record)
        stored = {**record, "stage": stage}
        self.records.append(stored)
        return stage

    def plan_next(self) -> dict[str, Any]:
        """Describe the next stage and the inputs it requires."""
        stage = self.current_stage()
        if stage is None:
            return {"stage": None, "required_inputs": []}
        return {"stage": stage, "required_inputs": list(_STAGE_REQUIRED_INPUTS[stage])}
