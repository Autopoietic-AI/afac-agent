# -*- coding: utf-8 -*-
"""Research taxonomy constants for M6R-A."""

from __future__ import annotations

SCOPE_LEVELS = ["global", "bucket", "bucket_class", "cross_bucket_class", "error_mechanism"]
BUCKETS = [
    "isolated",
    "one_hop_available",
    "exact2_only",
    "exact3_4_only",
    "no_visible_train_within_4_hops",
    "graph_visible",
]
MECHANISMS = [
    "attribute_insufficiency",
    "class_imbalance",
    "neighbor_unreliability",
    "heterophily",
    "over_smoothing",
    "under_propagation",
    "directionality_loss",
    "multi_hop_signal_missing",
    "teacher_signal_not_transferable",
    "calibration_error",
    "confidence_routing_error",
    "candidate_scope_mismatch",
    "representation_only_change",
    "information_source_missing",
    "uniform_smoothing_damage",
    "multi_hop_signal_opportunity",
    "other",
]
INFORMATION_SOURCE_TYPES = [
    "existing_node_attributes",
    "graph_topology",
    "one_hop_topology",
    "exact_two_hop_topology",
    "higher_order_topology",
    "directed_path_signal",
    "teacher_soft_targets",
    "self_supervised_signal",
    "community_signal",
    "prototype_signal",
    "external_pretraining",
    "cross_task_signal",
    "other",
]
