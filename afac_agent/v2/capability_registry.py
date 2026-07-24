# -*- coding: utf-8 -*-
"""AFAC v2.0 capability registry.

Tracks which operators are implemented and which are actually *available*
in the current environment (e.g. lightgbm / torch are not installed here).
Selection reports expose the gap between the scientifically best option and
the available fallback so the controller can detect availability bias; a
fallback is never reported as scientifically optimal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from afac_agent.research.event_store import stable_hash


@dataclass
class OperatorRecord:
    family: str
    operator_id: str = ""
    implemented: bool = True
    available: bool = True
    required_inputs: list[str] = field(default_factory=list)
    adapter_path: str = ""
    expected_runtime_seconds: float = 60.0
    expected_memory_mb: float = 256.0
    supports_diagnostic: bool = False
    supports_screen: bool = False
    supports_confirm: bool = False
    supports_full_cv: bool = False
    supports_full_train: bool = False
    supports_oof: bool = True
    supports_test: bool = True
    supports_checkpoint: bool = False
    supports_warm_start: bool = False
    deployment_ready: bool = True
    scientific_priority: float = 0.0  # higher = more valuable scientifically
    missing_dependency: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)  # empty = applies to every task

    def __post_init__(self) -> None:
        if not self.operator_id:
            self.operator_id = stable_hash({"family": self.family, "inputs": self.required_inputs})[:16]


@dataclass
class SelectionReport:
    family: str
    best_scientific_option: str | None
    best_available_option: str | None
    selected_option: str | None
    selection_gap: float  # scientific_priority(best) - scientific_priority(selected)
    availability_bias: bool  # True when best_scientific != selected
    status: str  # "selected" | "waiting_for_adapter"

    def to_dict(self) -> dict:
        return asdict(self)


class CapabilityRegistry:
    def __init__(self) -> None:
        self.records: dict[str, OperatorRecord] = {}

    def register(self, record: OperatorRecord) -> str:
        self.records[record.operator_id] = record
        return record.operator_id

    def query(self, family: str | None = None, task: str | None = None) -> list[OperatorRecord]:
        results = []
        for record in self.records.values():
            if family is not None and record.family != family:
                continue
            if task is not None and record.tasks and task not in record.tasks:
                continue
            results.append(record)
        return sorted(results, key=lambda r: (-r.scientific_priority, r.operator_id))

    def best_scientific_option(self, family: str) -> OperatorRecord | None:
        """Scientifically best registered option, regardless of implementation or availability.

        The scientific optimum is a property of the method, not of the local
        environment: an unimplemented or unavailable operator (e.g. missing
        lightgbm/torch) still counts as the scientific best so the selection
        gap and availability bias are reported honestly.
        """
        candidates = self.query(family=family)
        return candidates[0] if candidates else None

    def best_available_option(self, family: str) -> OperatorRecord | None:
        """Best option that is both implemented and available right now."""
        candidates = [r for r in self.query(family=family) if r.implemented and r.available]
        return candidates[0] if candidates else None

    def select(self, family: str) -> SelectionReport:
        """Select the best available option and report the scientific gap.

        When the scientifically best option is unavailable the report either
        selects a fallback (status "selected", availability_bias=True) or
        waits for an adapter (status "waiting_for_adapter") when no fallback
        exists.  It never equates the fallback with the scientific optimum:
        ``best_scientific_option`` and ``selection_gap`` always reflect the
        true best.
        """
        best_scientific = self.best_scientific_option(family)
        best_available = self.best_available_option(family)
        selected = best_available
        if selected is None:
            status = "waiting_for_adapter"
        else:
            status = "selected"
        gap = 0.0
        if best_scientific is not None and selected is not None:
            gap = best_scientific.scientific_priority - selected.scientific_priority
        bias = (
            best_scientific is not None
            and selected is not None
            and best_scientific.operator_id != selected.operator_id
        )
        return SelectionReport(
            family=family,
            best_scientific_option=best_scientific.operator_id if best_scientific else None,
            best_available_option=best_available.operator_id if best_available else None,
            selected_option=selected.operator_id if selected else None,
            selection_gap=gap,
            availability_bias=bias,
            status=status,
        )


def _op(
    family: str,
    operator_id: str,
    *,
    priority: float,
    available: bool = True,
    missing: list[str] | None = None,
    runtime: float = 60.0,
    tasks: list[str] | None = None,
    adapter_path: str = "afac_agent.v2.operators.recommendation",
    supports_diagnostic: bool = False,
    supports_screen: bool = False,
    supports_confirm: bool = False,
    supports_full_cv: bool = False,
    supports_full_train: bool = False,
    supports_checkpoint: bool = False,
    supports_warm_start: bool = False,
    deployment_ready: bool | None = None,
) -> OperatorRecord:
    implemented = not missing
    if deployment_ready is None:
        deployment_ready = implemented and not missing
    return OperatorRecord(
        family=family,
        operator_id=operator_id,
        implemented=implemented,
        available=available and implemented,
        adapter_path=adapter_path,
        expected_runtime_seconds=runtime,
        scientific_priority=priority,
        missing_dependency=list(missing or []),
        deployment_ready=deployment_ready,
        tasks=list(tasks or []),
        supports_diagnostic=supports_diagnostic,
        supports_screen=supports_screen,
        supports_confirm=supports_confirm,
        supports_full_cv=supports_full_cv,
        supports_full_train=supports_full_train,
        supports_checkpoint=supports_checkpoint,
        supports_warm_start=supports_warm_start,
    )


def default_registry() -> CapabilityRegistry:
    """Registry preloaded for AFAC classification and recommendation tasks.

    Operators depending on lightgbm / xgboost / torch are marked
    implemented=False, available=False with the missing dependency recorded;
    sklearn/numpy-based operators are available.
    """
    registry = CapabilityRegistry()
    records = [
        # --- classification families ---
        _op("linear", "linear_logistic", priority=0.4, runtime=30.0),
        _op("gbdt", "gbdt_lightgbm", priority=0.9, available=False, missing=["lightgbm"], runtime=120.0),
        _op("gbdt", "gbdt_hist_sklearn", priority=0.7, runtime=180.0),
        _op("residual_mlp", "residual_mlp_torch", priority=0.8, available=False, missing=["torch"], runtime=600.0),
        _op("pca_lowrank", "pca_lowrank_sklearn", priority=0.5, runtime=60.0),
        _op("prototype", "prototype_nearest_centroid", priority=0.5, runtime=20.0),
        _op("label_propagation", "label_propagation_sklearn", priority=0.6, runtime=90.0),
        _op("sgc", "sgc_numpy", priority=0.6, runtime=90.0),
        _op("appnp", "appnp_torch", priority=0.75, available=False, missing=["torch"], runtime=300.0),
        _op("graphsage", "graphsage_torch", priority=0.8, available=False, missing=["torch"], runtime=420.0),
        _op("gcn", "gcn_torch", priority=0.7, available=False, missing=["torch"], runtime=360.0),
        _op("heterophily_gnn", "heterophily_gnn_torch", priority=0.85, available=False, missing=["torch"], runtime=480.0),
        # --- recommendation families ---
        _op("retrieval_popularity", "retrieval_popularity_topn", priority=0.3, runtime=10.0),
        _op("retrieval_history", "retrieval_history_repeat", priority=0.4, runtime=10.0),
        _op("retrieval_itemcf", "retrieval_itemcf_cosine", priority=0.6, runtime=120.0),
        _op("retrieval_item2vec", "retrieval_item2vec_torch", priority=0.7, available=False, missing=["torch"], runtime=600.0),
        _op("retrieval_random_walk", "retrieval_random_walk_ppr", priority=0.55, runtime=180.0),
        _op("ranker_lightgbm_lambdarank", "ranker_lightgbm_lambdarank", priority=0.9, available=False, missing=["lightgbm"], runtime=300.0),
        _op("ranker_gbdt_binary", "ranker_gbdt_binary_sklearn", priority=0.65, runtime=240.0),
        _op("ranker_logistic", "ranker_logistic_sklearn", priority=0.45, runtime=60.0),
        _op("rerank_protected", "rerank_protected_rules", priority=0.5, runtime=15.0),
        # --- B2 v2.1 scientific operators ---
        _op(
            "retrieval_union",
            "retrieval_union_experiment",
            priority=0.5,
            runtime=120.0,
            tasks=["B2"],
            supports_diagnostic=True,
            supports_screen=True,
            supports_confirm=True,
            supports_full_cv=True,
            supports_full_train=True,
            supports_checkpoint=False,
        ),
        _op(
            "candidate_ranker",
            "candidate_ranker_experiment",
            priority=0.85,
            runtime=300.0,
            tasks=["B2"],
            supports_diagnostic=False,
            supports_screen=True,
            supports_confirm=True,
            supports_full_cv=True,
            supports_full_train=True,
            supports_checkpoint=True,
            supports_warm_start=True,
        ),
        _op(
            "bucket_specialist",
            "bucket_specialist_experiment",
            priority=0.6,
            runtime=180.0,
            tasks=["B2"],
            supports_diagnostic=False,
            supports_screen=True,
            supports_confirm=True,
            supports_full_cv=True,
            supports_full_train=True,
            supports_checkpoint=True,
        ),
        _op(
            "protected_rerank",
            "protected_rerank_experiment",
            priority=0.55,
            runtime=120.0,
            tasks=["B2"],
            supports_diagnostic=False,
            supports_screen=True,
            supports_confirm=True,
            supports_full_cv=True,
            supports_full_train=True,
            supports_checkpoint=True,
        ),
        _op(
            "retrieval_diagnostic",
            "candidate_recall_diagnostic",
            priority=0.1,
            runtime=90.0,
            tasks=["B2"],
            supports_diagnostic=True,
            supports_screen=False,
            supports_confirm=False,
            supports_full_cv=False,
            supports_full_train=False,
            supports_checkpoint=False,
            deployment_ready=False,
        ),
    ]
    for record in records:
        registry.register(record)
    return registry
