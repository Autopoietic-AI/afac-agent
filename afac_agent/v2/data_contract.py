# -*- coding: utf-8 -*-
"""AFAC v2.1 canonical data contract for recommendation (B2).

Reconciles item, user, and interaction counts across Input Discovery,
Data Intelligence, the Experiment Executor, and Deployment.  Every scale
value keeps a provenance record so that `n_items=14065` and `n_items=40011`
can never be silently mixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..research.event_store import stable_hash


@dataclass
class ScaleValue:
    """One measured scale with full provenance."""

    value: int
    source_file: str
    source_column: str
    counting_rule: str
    deduplicated: bool
    sampled: bool
    membership_hash: str


@dataclass
class B2CanonicalDataContract:
    """Canonical B2 data contract."""

    n_train_users_total: ScaleValue | None = None
    n_test_users_total: ScaleValue | None = None
    n_items_total: ScaleValue | None = None
    n_interactions_total: ScaleValue | None = None
    n_candidate_items_total: ScaleValue | None = None
    profiler_sample_train_users: ScaleValue | None = None
    profiler_sample_test_users: ScaleValue | None = None
    profile_scope: str = "full"
    item_universe_source: str = ""
    user_universe_source: str = ""
    interaction_item_column: str = ""
    target_item_column: str = ""
    candidate_item_column: str = ""
    status: str = "unknown"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def sv(v: ScaleValue | None) -> dict[str, Any] | None:
            if v is None:
                return None
            return {
                "value": v.value,
                "source_file": v.source_file,
                "source_column": v.source_column,
                "counting_rule": v.counting_rule,
                "deduplicated": v.deduplicated,
                "sampled": v.sampled,
                "membership_hash": v.membership_hash,
            }

        return {
            "n_train_users_total": sv(self.n_train_users_total),
            "n_test_users_total": sv(self.n_test_users_total),
            "n_items_total": sv(self.n_items_total),
            "n_interactions_total": sv(self.n_interactions_total),
            "n_candidate_items_total": sv(self.n_candidate_items_total),
            "profiler_sample_train_users": sv(self.profiler_sample_train_users),
            "profiler_sample_test_users": sv(self.profiler_sample_test_users),
            "profile_scope": self.profile_scope,
            "item_universe_source": self.item_universe_source,
            "user_universe_source": self.user_universe_source,
            "interaction_item_column": self.interaction_item_column,
            "target_item_column": self.target_item_column,
            "candidate_item_column": self.candidate_item_column,
            "status": self.status,
            "errors": self.errors,
        }

    def data_hash(self) -> str:
        return stable_hash(self.to_dict())


def _hash_items(item_ids: list[str]) -> str:
    return stable_hash({"sorted_unique_item_ids": sorted(set(item_ids))})


def _hash_users(user_ids: list[str]) -> str:
    return stable_hash({"sorted_unique_user_ids": sorted(set(user_ids))})


def build_b2_data_contract(
    *,
    data_root: Path,
    item_df: Any,
    train_df: Any,
    test_df: Any,
    train_seq: dict[str, list[str]],
    test_seq: dict[str, list[str]],
    train_targets: dict[str, str],
    item_universe: set[str],
    sampled_train_users: list[str] | None = None,
    sampled_test_users: list[str] | None = None,
    profile_scope: str = "full",
) -> B2CanonicalDataContract:
    """Build the canonical contract from already-loaded tables.

    ``item_universe`` must come from the official item table (`item.csv`).
    ``train_targets`` maps train uid -> target iid.
    """
    item_ids = sorted(item_universe)
    train_uids = sorted(train_df["uid"].astype(str).unique())
    test_uids = sorted(test_df["uid"].astype(str).unique())

    # Interaction count: raw sequence items + train targets.
    n_interactions = sum(len(s) for s in train_seq.values()) + len(train_targets)

    contract = B2CanonicalDataContract(
        n_train_users_total=ScaleValue(
            value=len(train_uids),
            source_file=str(data_root / "train.csv"),
            source_column="uid",
            counting_rule="unique_users",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_users(train_uids),
        ),
        n_test_users_total=ScaleValue(
            value=len(test_uids),
            source_file=str(data_root / "test.csv"),
            source_column="uid",
            counting_rule="unique_users",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_users(test_uids),
        ),
        n_items_total=ScaleValue(
            value=len(item_ids),
            source_file=str(data_root / "item.csv"),
            source_column="iid",
            counting_rule="unique_items",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_items(item_ids),
        ),
        n_interactions_total=ScaleValue(
            value=n_interactions,
            source_file=str(data_root / "train.csv"),
            source_column="item_seq_raw,target_iid",
            counting_rule="raw_sequence_items_plus_targets",
            deduplicated=False,
            sampled=False,
            membership_hash=stable_hash({"n_interactions": n_interactions}),
        ),
        n_candidate_items_total=ScaleValue(
            value=len(item_ids),
            source_file=str(data_root / "item.csv"),
            source_column="iid",
            counting_rule="unique_items_legal_for_candidates",
            deduplicated=True,
            sampled=False,
            membership_hash=_hash_items(item_ids),
        ),
        profile_scope=profile_scope,
        item_universe_source="item.csv::iid",
        user_universe_source="train.csv::uid + test.csv::uid",
        interaction_item_column="item_seq_raw",
        target_item_column="target_iid",
        candidate_item_column="iid",
    )

    if sampled_train_users is not None:
        contract.profiler_sample_train_users = ScaleValue(
            value=len(sampled_train_users),
            source_file=str(data_root / "train.csv"),
            source_column="uid",
            counting_rule="unique_users_sampled_for_profiler",
            deduplicated=True,
            sampled=True,
            membership_hash=_hash_users(sampled_train_users),
        )
    if sampled_test_users is not None:
        contract.profiler_sample_test_users = ScaleValue(
            value=len(sampled_test_users),
            source_file=str(data_root / "test.csv"),
            source_column="uid",
            counting_rule="unique_users_sampled_for_profiler",
            deduplicated=True,
            sampled=True,
            membership_hash=_hash_users(sampled_test_users),
        )

    errors: list[str] = []
    # User count cannot impersonate item count
    if contract.n_train_users_total.value == contract.n_items_total.value:
        errors.append("n_train_users_total equals n_items_total; likely counting users as items")

    # Cross-check item universe against train/test sequences and targets
    observed_items: set[str] = set()
    for seq in train_seq.values():
        observed_items.update(seq)
    for seq in test_seq.values():
        observed_items.update(seq)
    observed_items.update(train_targets.values())
    illegal_observed = observed_items - item_universe
    if illegal_observed:
        errors.append(f"observed items not in item.csv universe: {len(illegal_observed)} (sample {sorted(illegal_observed)[:5]})")

    # Train/test user disjointness
    overlap = set(train_uids) & set(test_uids)
    if overlap:
        errors.append(f"train/test uid overlap: {len(overlap)}")

    contract.status = "passed" if not errors else "blocked_data_contract_mismatch"
    contract.errors = errors
    return contract


def reconcile_data_contract(
    input_discovery_n_items: int,
    data_intelligence_n_items: int,
    experiment_executor_item_universe: set[str],
    deployment_candidate_legality_universe: set[str],
) -> dict[str, Any]:
    """Cross-source consistency check used before Problem Selection."""
    errors: list[str] = []
    if input_discovery_n_items != data_intelligence_n_items:
        errors.append(
            f"input_discovery n_items={input_discovery_n_items} != "
            f"data_intelligence n_items={data_intelligence_n_items}"
        )
    if len(experiment_executor_item_universe) != input_discovery_n_items:
        errors.append(
            f"experiment_executor item universe size={len(experiment_executor_item_universe)} != "
            f"input_discovery n_items={input_discovery_n_items}"
        )
    if deployment_candidate_legality_universe != experiment_executor_item_universe:
        errors.append("deployment candidate legality universe differs from experiment executor item universe")
    return {
        "status": "passed" if not errors else "blocked_data_contract_mismatch",
        "errors": errors,
    }
