# -*- coding: utf-8 -*-
"""B1 multi-panel evaluation and fusion operators.

Evaluation covers Standard, Degree-matched, Propensity-matched, Low-degree and
Test-like panels.  Fusion operators support probability/logit blend, bucket
route, class-weighted blend, and cross-fit node-level gate.  All learned
parameters are selected via strict outer-fold cross-fit on AFAC_B1_FOLD_V1.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import softmax

from .fold import B1Folds


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(y_true.shape[0], dtype=bool)
    return float(np.mean(y_true[mask] == y_pred[mask])) if mask.any() else 0.0


def _macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(y_true.shape[0], dtype=bool)
    vals = []
    for c in range(n_classes):
        m = mask & (y_true == c)
        if m.any():
            vals.append(float(np.mean(y_pred[m] == c)))
    return float(np.mean(vals)) if vals else 0.0


def _delta(parent_correct: np.ndarray, candidate_correct: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(parent_correct.shape[0], dtype=bool)
    rescue = int(np.sum(~parent_correct[mask] & candidate_correct[mask]))
    damage = int(np.sum(parent_correct[mask] & ~candidate_correct[mask]))
    changed = rescue + damage
    return {"rescue": rescue, "damage": damage, "net": rescue - damage, "changed_count": changed, "change_precision": rescue / changed if changed else None}


class B1Evaluator:
    def __init__(self, labels: np.ndarray, n_classes: int, folds: B1Folds, panels: dict[str, Any]) -> None:
        self.labels = labels
        self.n_classes = n_classes
        self.folds = folds
        self.panels = panels

    def evaluate(self, proba: np.ndarray, *, asset_id: str, panel_ids: list[str] | None = None) -> dict[str, Any]:
        pred = proba.argmax(axis=1)
        y = self.labels[self.folds.train_idx]
        panel_ids = panel_ids or ["B1_STANDARD_PANEL"]
        panel_results = {}
        for pid in panel_ids:
            _, val_mask = self._panel_masks(pid)
            val_idx_global = self.folds.train_idx[val_mask]
            y_true = self.labels[val_idx_global]
            y_pred = pred[val_idx_global]
            panel_results[pid] = {
                "accuracy": _accuracy(y_true, y_pred),
                "macro": _macro(y_true, y_pred, self.n_classes),
                "per_class": {c: (float(np.mean(y_pred[y_true == c] == c)) if (y_true == c).any() else None) for c in range(self.n_classes)},
                "n_val": int(val_mask.sum()),
            }
        # Standard 5-fold per-fold accuracies
        fold_accs = []
        for f in range(5):
            m = self.folds.folds == f
            y_true = y[m]
            y_pred = pred[self.folds.train_idx[m]]
            fold_accs.append(float(np.mean(y_true == y_pred)))
        worst_fold = min(fold_accs)
        positive_fold_count = sum(1 for a in fold_accs if a >= worst_fold - 1e-9)
        return {
            "asset_id": asset_id,
            "overall_accuracy": _accuracy(y, pred[self.folds.train_idx]),
            "macro_accuracy": _macro(y, pred[self.folds.train_idx], self.n_classes),
            "fold_accuracies": fold_accs,
            "worst_fold_accuracy": worst_fold,
            "positive_fold_count": positive_fold_count,
            "panels": panel_results,
        }

    def compare(self, baseline_pred: np.ndarray, candidate_pred: np.ndarray, panel_id: str = "B1_STANDARD_PANEL") -> dict[str, Any]:
        _, val_mask = self._panel_masks(panel_id)
        val_idx = self.folds.train_idx[val_mask]
        y = self.labels[val_idx]
        base_correct = baseline_pred[val_idx] == y
        cand_correct = candidate_pred[val_idx] == y
        return _delta(base_correct, cand_correct)

    def _panel_masks(self, panel_id: str) -> tuple[np.ndarray, np.ndarray]:
        from .fold import validation_mask
        return validation_mask(panel_id, self.folds, held_fold=None)


# ---------------------------------------------------------------- fusion operators


def _logit(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    log = np.log(p)
    return log - log.mean(axis=1, keepdims=True)


def op_probability_blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def op_logit_blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return softmax((1.0 - alpha) * _logit(a) + alpha * _logit(b), axis=1)


def op_bucket_route(a: np.ndarray, b: np.ndarray, route_mask: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[route_mask] = b[route_mask]
    return out


def op_class_weighted_blend(a: np.ndarray, b: np.ndarray, alpha: float, y: np.ndarray) -> np.ndarray:
    out = a.copy()
    for c in range(a.shape[1]):
        mask = y == c
        if mask.any():
            out[mask] = (1.0 - alpha) * a[mask] + alpha * b[mask]
    return out


def cross_fit_fusion(
    *,
    operator: str,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    route_mask: np.ndarray | None = None,
    alpha_grid: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strict outer-fold cross-fit for fusion operators with learned parameters."""
    n = proba_a.shape[0]
    out = np.zeros_like(proba_a)
    assignments = []
    for held in sorted(set(folds.tolist())):
        fit = folds != held
        valid = folds == held
        best_alpha, best_score = None, -1.0
        for alpha in alpha_grid:
            if operator == "probability_blend":
                p = op_probability_blend(proba_a[fit], proba_b[fit], alpha)
            elif operator == "logit_blend":
                p = op_logit_blend(proba_a[fit], proba_b[fit], alpha)
            elif operator == "class_weighted_blend":
                p = op_class_weighted_blend(proba_a[fit], proba_b[fit], alpha, y[fit])
            else:
                raise ValueError(operator)
            score = float(np.mean(p.argmax(axis=1) == y[fit]))
            if score > best_score + 1e-12:
                best_score = score
                best_alpha = alpha
        if operator == "probability_blend":
            out[valid] = op_probability_blend(proba_a[valid], proba_b[valid], best_alpha)
        elif operator == "logit_blend":
            out[valid] = op_logit_blend(proba_a[valid], proba_b[valid], best_alpha)
        elif operator == "class_weighted_blend":
            out[valid] = op_class_weighted_blend(proba_a[valid], proba_b[valid], best_alpha, y[valid])
        assignments.append({"held_fold": int(held), "alpha": best_alpha})
    return out, {"mode": "outer_fold_cross_fit", "assignments": assignments}


def cross_fit_node_gate(
    *,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    meta: np.ndarray,  # (n_nodes, k)
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-fit logistic gate choosing between two probability vectors."""
    from sklearn.linear_model import LogisticRegression
    n = proba_a.shape[0]
    out = np.zeros_like(proba_a)
    assignments = []
    for held in sorted(set(folds.tolist())):
        fit = folds != held
        valid = folds == held
        # target: which model is correct on fit set
        correct_a = proba_a[fit].argmax(axis=1) == y[fit]
        correct_b = proba_b[fit].argmax(axis=1) == y[fit]
        # prefer b if b correct and a wrong; prefer a if a correct and b wrong;
        # tie-break to model with higher fit accuracy
        acc_a = float(correct_a.mean())
        acc_b = float(correct_b.mean())
        target = np.where(correct_b & ~correct_a, 1, np.where(correct_a & ~correct_b, 0, 1 if acc_b >= acc_a else 0))
        if len(set(target.tolist())) < 2:
            choose_b = np.full(valid.sum(), bool(target[0]))
        else:
            clf = LogisticRegression(max_iter=200, solver="lbfgs")
            clf.fit(meta[fit], target)
            choose_b = clf.predict(meta[valid]).astype(bool)
        out[valid] = np.where(choose_b[:, None], proba_b[valid], proba_a[valid])
        assignments.append({"held_fold": int(held), "choose_b_fraction": float(choose_b.mean())})
    return out, {"mode": "outer_fold_node_gate", "assignments": assignments}
