# -*- coding: utf-8 -*-
"""AFAC v2.1 Proposal-to-Operator Compiler.

Compiles an LLM M6B proposal into a structured operator plan.  It selects
operators from the Capability Registry and refuses to silently downgrade a
proposal to a diagnostic.  If the requested operator is unavailable the
compiler reports ``blocked_missing_adapter``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..research.event_store import stable_hash
from .capability_registry import CapabilityRegistry, default_registry
from .experiment_kind import ExperimentKind, kind_from_operator_and_folds


# Mapping from legacy / LLM diagnostic_type names to canonical operator_ids.
# Only names that truly exist as implemented operators are listed.
DIAGNOSTIC_TYPE_TO_OPERATOR: dict[str, str] = {
    # B2 recommendation
    "candidate_recall_diagnostic": "candidate_recall_diagnostic",
    "small_retrieval_compare": "retrieval_union_experiment",
    "cached_replay": "retrieval_union_experiment",
    "candidate_ranker_experiment": "candidate_ranker_experiment",
    "bucket_specialist_experiment": "bucket_specialist_experiment",
    "protected_rerank_experiment": "protected_rerank_experiment",
    "retrieval_union_experiment": "retrieval_union_experiment",
    # B1 node classification
    "classification_recall_diagnostic": "classification_recall_diagnostic",
    "feature_lr": "feature_baseline_experiment",
    "feature_mlp": "feature_baseline_experiment",
    "feature_baseline_screen": "feature_baseline_experiment",
    "feature_baseline_experiment": "feature_baseline_experiment",
    "graph_lp": "graph_propagation_experiment",
    "graph_appnp": "graph_propagation_experiment",
    "graph_propagation_experiment": "graph_propagation_experiment",
    "hop_reliability": "hop_reliability_experiment",
    "hop_reliability_experiment": "hop_reliability_experiment",
    "feature_graph_residual": "feature_graph_residual_experiment",
    "feature_graph_residual_experiment": "feature_graph_residual_experiment",
    "bucket_specialist_b1": "bucket_specialist_experiment_b1",
    "bucket_specialist_experiment_b1": "bucket_specialist_experiment_b1",
}

DEFAULT_RETRIEVAL_SOURCES = ["popularity", "history", "pair_transition"]
RETRIEVAL_SOURCE_WHITELIST = {
    "popularity",
    "history",
    "repeat",
    "last_transition",
    "pair_transition",
    "itemcf_1hop",
    "itemcf_2hop",
    "attribute_recall",
    "sequence_recall",
    "novel_recall",
}


@dataclass
class CompiledOperator:
    """Structured output of the ProposalOperatorCompiler."""

    operator_id: str
    operator_family: str
    parent_candidate_id: str
    data_view: str
    retrieval_sources: list[str]
    candidate_union_rule: str
    feature_set: list[str]
    model_family: str
    objective: str
    target_panel_id: str
    target_metric_name: str
    primary_change: str
    safety_adjustment: str
    proposed_fidelity: str
    changed_layers: list[str]
    fixed_layers: list[str]
    hyperparameters: dict[str, Any]
    runtime_estimate_seconds: float
    status: str
    reason: str = ""
    available: bool = False
    experiment_kind: ExperimentKind = ExperimentKind.DETERMINISTIC_DIAGNOSTIC

    def semantic_genome_inputs(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "parent_candidate_id": self.parent_candidate_id,
            "retrieval_sources": sorted(self.retrieval_sources),
            "feature_set": sorted(self.feature_set),
            "model_family": self.model_family,
            "objective": self.objective,
            "target_panel_id": self.target_panel_id,
            "primary_change": self.primary_change,
            "hyperparameters": self.hyperparameters,
        }

    def semantic_genome_hash(self) -> str:
        return stable_hash(self.semantic_genome_inputs())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(value)]


def compile_proposal(
    proposal: dict[str, Any],
    *,
    parent_candidate_id: str = "",
    registry: CapabilityRegistry | None = None,
    formal_mode: bool = False,
    task: str = "B2",
    default_target_panel: str = "B2_STANDARD_PANEL",
    default_target_metric: str = "hit_rate@10",
) -> CompiledOperator:
    """Compile an M6B proposal into a real operator plan.

    In ``formal_mode`` (round > 1 of a real run), diagnostic-only operators are
    rejected; the compiler demands a real scientific operator.
    """
    registry = registry or default_registry()
    diagnostic_type = str(proposal.get("diagnostic_type") or "").strip()
    operator_id = DIAGNOSTIC_TYPE_TO_OPERATOR.get(diagnostic_type, diagnostic_type)

    record = registry.records.get(operator_id)
    if record is None:
        return CompiledOperator(
            operator_id=operator_id,
            operator_family="unknown",
            parent_candidate_id=parent_candidate_id,
            data_view="",
            retrieval_sources=[],
            candidate_union_rule="",
            feature_set=[],
            model_family="",
            objective="",
            target_panel_id=default_target_panel,
            target_metric_name=default_target_metric,
            primary_change="",
            safety_adjustment="",
            proposed_fidelity="F0_DETERMINISTIC",
            changed_layers=[],
            fixed_layers=[],
            hyperparameters={},
            runtime_estimate_seconds=float(proposal.get("budget_seconds", 60.0)),
            status="blocked_missing_adapter",
            reason=f"operator_id {operator_id!r} is not registered in the capability registry",
        )

    available = record.implemented and record.available and (not task or not record.tasks or task in record.tasks)

    is_diagnostic = record.supports_diagnostic and not (record.supports_screen or record.supports_confirm)
    if formal_mode and is_diagnostic:
        available = False
        reason = f"diagnostic operator {operator_id!r} is not admissible in formal mode"
    elif not available:
        reason = (
            f"operator {operator_id!r} unavailable: implemented={record.implemented}, "
            f"available={record.available}, missing={record.missing_dependency}"
        )
    else:
        reason = "compiled"

    sources = [s for s in _as_list(proposal.get("information_sources")) if s in RETRIEVAL_SOURCE_WHITELIST]
    if not sources:
        sources = list(DEFAULT_RETRIEVAL_SOURCES)

    feature_set = _as_list(proposal.get("feature_set")) or [
        "source_score",
        "source_rank",
        "source_count",
        "rrf",
        "popularity",
        "recency",
        "transition",
        "repeat",
        "history_novel",
        "seq_len_raw",
        "seq_len_dedup",
    ]

    model_family = {
        "candidate_ranker_experiment": "gbdt_binary",
        "bucket_specialist_experiment": "bucket_rules",
        "protected_rerank_experiment": "protected_rules",
        "retrieval_union_experiment": "retrieval_union",
        "candidate_recall_diagnostic": "retrieval_union",
    }.get(operator_id, record.family)

    objective = str(proposal.get("objective") or "hit_rate_maximization")
    target_panel_id = str(proposal.get("target_panel_id") or default_target_panel)
    target_metric_name = str(proposal.get("target_metric_name") or default_target_metric)

    runtime = float(proposal.get("budget_seconds", record.expected_runtime_seconds))
    # Clamp cheap diagnostics to their hard cost contract.
    if is_diagnostic:
        runtime = min(runtime, 120.0)

    kind = kind_from_operator_and_folds(operator_id, 0 if is_diagnostic else 2)

    return CompiledOperator(
        operator_id=operator_id,
        operator_family=record.family,
        parent_candidate_id=parent_candidate_id,
        data_view=str(proposal.get("data_view") or "b2_train_view"),
        retrieval_sources=sources,
        candidate_union_rule=str(proposal.get("candidate_union_rule") or "rrf_union"),
        feature_set=feature_set,
        model_family=model_family,
        objective=objective,
        target_panel_id=target_panel_id,
        target_metric_name=target_metric_name,
        primary_change=str(proposal.get("primary_change") or f"operator={operator_id}"),
        safety_adjustment=str(proposal.get("safety_adjustment") or "none"),
        proposed_fidelity="F0_DETERMINISTIC" if is_diagnostic else "F1_SCREEN",
        changed_layers=["L5"],
        fixed_layers=["L0", "L1", "L2", "L3", "L4", "L6", "L7", "L8", "L9"],
        hyperparameters={"budget_seconds": runtime, "fold_plan": str(proposal.get("fold_plan") or "F1_SCREEN")},
        runtime_estimate_seconds=runtime,
        status="compiled" if available else "blocked_missing_adapter",
        reason=reason,
        available=available,
        experiment_kind=kind,
    )


def semantic_revision_delta(previous: CompiledOperator, current: CompiledOperator) -> dict[str, Any]:
    """Compare two compiled operators and report whether the revision is real."""
    prev_hash = previous.semantic_genome_hash()
    curr_hash = current.semantic_genome_hash()
    fields = {
        "operator_id": previous.operator_id != current.operator_id,
        "parent_candidate_id": previous.parent_candidate_id != current.parent_candidate_id,
        "retrieval_sources": sorted(previous.retrieval_sources) != sorted(current.retrieval_sources),
        "feature_set": sorted(previous.feature_set) != sorted(current.feature_set),
        "model_family": previous.model_family != current.model_family,
        "objective": previous.objective != current.objective,
        "target_panel_id": previous.target_panel_id != current.target_panel_id,
        "primary_change": previous.primary_change != current.primary_change,
        "hyperparameters": previous.hyperparameters != current.hyperparameters,
    }
    return {
        "previous_genome_hash": prev_hash,
        "new_genome_hash": curr_hash,
        "semantic_delta": {k: v for k, v in fields.items() if v},
        "duplicate": prev_hash == curr_hash,
    }
