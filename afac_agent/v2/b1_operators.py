# -*- coding: utf-8 -*-
"""AFAC v2.2 B1 scientific operators for the v2 orchestrator.

Each operator implements the v2 experiment execution contract:
fit on fold complement, predict on fold, return OOF predictions and metrics.
"""
from __future__ import annotations

from typing import Any

import time

import numpy as np
from scipy.sparse import csr_matrix

from ..b1.evaluator import B1Evaluator, _accuracy, _macro, _delta
from ..b1.fold import B1Folds, build_panels
from ..b1.models import (
    APPNPFeatureModel,
    FeatureLogistic,
    FeatureMLP,
    LabelPropagationModel,
    NeighborFeatureModel,
)


def _fold_oof(
    *,
    model_family: str,
    n_classes: int,
    features: csr_matrix,
    adj: csr_matrix,
    labels: np.ndarray,
    train_idx: np.ndarray,
    folds: np.ndarray,
    fold_ids: list[int],
    deadline_monotonic: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a model family fold-by-fold, returning OOF predictions and metrics.

    When ``deadline_monotonic`` is set (``time.monotonic()`` hard deadline),
    the loop stops before starting a fold that would begin past the deadline
    and reports ``budget_aborted=True`` with the completed folds' partial OOF,
    so a single slow experiment can never blow the hard wall-clock contract.
    """
    n = features.shape[0]
    oof_proba = np.zeros((n, n_classes), dtype=np.float64)
    _X_all_dense = features.toarray() if hasattr(features, "toarray") else np.asarray(features)

    # Filter kwargs to model-specific params
    _model_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("graph_view", "model_family", "base_model")}

    folds_completed: list[int] = []
    budget_aborted = False
    for f in fold_ids:
        if deadline_monotonic is not None and folds_completed and time.monotonic() > deadline_monotonic:
            budget_aborted = True
            break
        fit_mask = folds != f
        eval_mask = folds == f
        fit_idx = train_idx[fit_mask]
        eval_idx = train_idx[eval_mask]

        if model_family == "feature_logistic":
            model = FeatureLogistic(model_id=f"fold_{f}", n_classes=n_classes, **_model_kwargs)
            model.fit(_X_all_dense[fit_idx], labels[fit_idx])
            oof_proba[eval_idx] = model.predict_proba(_X_all_dense[eval_idx])
        elif model_family == "feature_mlp":
            model = FeatureMLP(model_id=f"fold_{f}", n_classes=n_classes, **_model_kwargs)
            model.fit(_X_all_dense[fit_idx], labels[fit_idx])
            oof_proba[eval_idx] = model.predict_proba(_X_all_dense[eval_idx])
        elif model_family == "label_propagation":
            view = kwargs.get("graph_view", "undirected_union")
            A = _resolve_view(adj, view)
            model = LabelPropagationModel(model_id=f"fold_{f}", n_classes=n_classes, adj=A, **_model_kwargs)
            model.fit(_X_all_dense[fit_idx], labels[fit_idx], train_idx=fit_idx)
            oof_proba[eval_idx] = model.predict_proba(_X_all_dense)[eval_idx]  # LP full matrix
        elif model_family == "appnp_logistic":
            view = kwargs.get("graph_view", "undirected_union")
            A = _resolve_view(adj, view)
            model = APPNPFeatureModel(model_id=f"fold_{f}", n_classes=n_classes, adj=A, **_model_kwargs)
            model.fit(_X_all_dense[fit_idx], labels[fit_idx], X_all=_X_all_dense, train_idx=fit_idx)
            oof_proba[eval_idx] = model.predict_proba(_X_all_dense[eval_idx])
        elif model_family == "neighbor_logistic":
            view = kwargs.get("graph_view", "undirected_union")
            A = _resolve_view(adj, view)
            model = NeighborFeatureModel(model_id=f"fold_{f}", n_classes=n_classes, adj=A, **_model_kwargs)
            model.fit(_X_all_dense[fit_idx], labels[fit_idx], X_all=_X_all_dense, train_idx=fit_idx)
            oof_proba[eval_idx] = model.predict_proba(_X_all_dense[eval_idx])
        else:
            raise ValueError(f"unknown model_family: {model_family}")
        folds_completed.append(int(f))

    oof_pred = oof_proba.argmax(axis=1)
    # Only evaluate on train nodes (where we have labels)
    eval_mask = np.isin(np.arange(n), train_idx)
    parent_correct = np.zeros(n, dtype=bool)  # parent = majority class
    majority = np.bincount(labels[train_idx]).argmax()
    parent_pred = np.full(n, majority)
    parent_correct[train_idx] = parent_pred[train_idx] == labels[train_idx]
    cand_correct = oof_pred == labels

    rd = _delta(parent_correct, cand_correct, eval_mask)
    return {
        "oof_proba": oof_proba,
        "oof_pred": oof_pred.tolist(),
        "parent_pred": parent_pred.tolist(),
        "overall_accuracy": float(_accuracy(labels, oof_pred, eval_mask)),
        "macro_accuracy": float(_macro(labels, oof_pred, n_classes, eval_mask)),
        "rescue_damage": {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v for k, v in rd.items()},
        "parent_accuracy": float(_accuracy(labels, parent_pred, eval_mask)),
        "folds_completed": folds_completed,
        "budget_aborted": budget_aborted,
    }


def _resolve_view(adj: csr_matrix, view: str) -> csr_matrix:
    if view == "directed_out":
        return adj
    if view == "directed_in":
        return adj.T.tocsr()
    if view == "undirected_union":
        A = adj + adj.T
        A.data = np.ones_like(A.data)
        return A.tocsr()
    raise ValueError(f"unknown view: {view}")


# ---------------------------------------------------------------------------
# Public operator entry-points (called by the v2 orchestrator)
# ---------------------------------------------------------------------------


def run_feature_baseline_experiment(
    *,
    n_classes: int,
    features: csr_matrix,
    adj: csr_matrix,
    labels: np.ndarray,
    train_idx: np.ndarray,
    folds: np.ndarray,
    fold_ids: list[int],
    model_family: str = "feature_logistic",
    **kwargs: Any,
) -> dict[str, Any]:
    """Feature-only baseline: LR or MLP."""
    return _fold_oof(
        model_family=model_family,
        n_classes=n_classes,
        features=features,
        adj=adj,
        labels=labels,
        train_idx=train_idx,
        folds=folds,
        fold_ids=fold_ids,
        **kwargs,
    )


def run_graph_propagation_experiment(
    *,
    n_classes: int,
    features: csr_matrix,
    adj: csr_matrix,
    labels: np.ndarray,
    train_idx: np.ndarray,
    folds: np.ndarray,
    fold_ids: list[int],
    graph_view: str = "undirected_union",
    model_family: str = "label_propagation",
    alpha: float = 0.9,
    **kwargs: Any,
) -> dict[str, Any]:
    """Graph propagation: LP or APPNP."""
    return _fold_oof(
        model_family=model_family,
        n_classes=n_classes,
        features=features,
        adj=adj,
        labels=labels,
        train_idx=train_idx,
        folds=folds,
        fold_ids=fold_ids,
        graph_view=graph_view,
        alpha=alpha,
        **kwargs,
    )


def run_feature_graph_residual_experiment(
    *,
    feature_result: dict[str, Any],
    graph_result: dict[str, Any],
    n_classes: int,
    labels: np.ndarray,
    train_idx: np.ndarray,
    max_delta: float = 0.3,
    max_changed_fraction: float = 0.3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Combine feature baseline with graph residual under damage guard."""
    feat_proba = feature_result["oof_proba"]
    graph_proba = graph_result["oof_proba"]
    n = feat_proba.shape[0]
    eval_mask = np.isin(np.arange(n), train_idx)

    # Simple blend: graph provides residual delta to feature
    delta = graph_proba - feat_proba
    delta = np.clip(delta, -max_delta, max_delta)
    combined_proba = feat_proba + delta
    combined_proba = np.clip(combined_proba, 0, 1)
    row_sums = combined_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    combined_proba /= row_sums

    combined_pred = combined_proba.argmax(axis=1)
    feat_pred = feat_proba.argmax(axis=1)
    parent_correct = feat_pred == labels
    cand_correct = combined_pred == labels

    changed = (combined_pred != feat_pred) & eval_mask
    changed_frac = float(np.mean(changed)) if eval_mask.any() else 0.0
    if changed_frac > max_changed_fraction:
        # Fall back to feature baseline
        return {
            "oof_proba": feat_proba,
            "oof_pred": feat_pred,
            "parent_pred": feat_pred,
            "overall_accuracy": feature_result["overall_accuracy"],
            "macro_accuracy": feature_result["macro_accuracy"],
            "rescue_damage": {"rescue": 0, "damage": 0, "net": 0, "changed_count": 0, "change_precision": None},
            "parent_accuracy": feature_result["overall_accuracy"],
            "fallback_parent": True,
        }

    return {
        "oof_proba": combined_proba,
        "oof_pred": combined_pred,
        "parent_pred": feat_pred,
        "overall_accuracy": _accuracy(labels, combined_pred, eval_mask),
        "macro_accuracy": _macro(labels, combined_pred, n_classes, eval_mask),
        "rescue_damage": _delta(parent_correct, cand_correct, eval_mask),
        "parent_accuracy": _accuracy(labels, feat_pred, eval_mask),
        "fallback_parent": False,
        "changed_fraction": changed_frac,
    }


def run_bucket_specialist_experiment(
    *,
    feature_result: dict[str, Any],
    graph_result: dict[str, Any],
    n_classes: int,
    labels: np.ndarray,
    train_idx: np.ndarray,
    features: csr_matrix,
    adj: csr_matrix,
    **kwargs: Any,
) -> dict[str, Any]:
    """Route nodes to feature or graph expert based on degree and connectivity."""
    n = features.shape[0]
    eval_mask = np.isin(np.arange(n), train_idx)

    feat_proba = feature_result["oof_proba"]
    graph_proba = graph_result["oof_proba"]

    # Simple routing: low-degree → graph expert, high-degree → feature expert
    degrees = np.diff(adj.indptr)
    median_deg = np.median(degrees[train_idx]) if len(train_idx) else 5

    specialist_proba = feat_proba.copy()
    low_deg_mask = degrees <= median_deg
    specialist_proba[low_deg_mask] = graph_proba[low_deg_mask]

    specialist_pred = specialist_proba.argmax(axis=1)
    feat_pred = feat_proba.argmax(axis=1)

    parent_correct = feat_pred == labels
    cand_correct = specialist_pred == labels

    route_frac = float(np.mean(low_deg_mask[eval_mask])) if eval_mask.any() else 0.0

    return {
        "oof_proba": specialist_proba,
        "oof_pred": specialist_pred,
        "parent_pred": feat_pred,
        "overall_accuracy": _accuracy(labels, specialist_pred, eval_mask),
        "macro_accuracy": _macro(labels, specialist_pred, n_classes, eval_mask),
        "rescue_damage": _delta(parent_correct, cand_correct, eval_mask),
        "parent_accuracy": _accuracy(labels, feat_pred, eval_mask),
        "route_fraction": route_frac,
        "median_degree_threshold": float(median_deg),
    }
