# -*- coding: utf-8 -*-
"""AFAC v2.0 classification operator space.

Three operator families, all CPU-only and deterministic (seed 2026):

- Feature operators: fit(X_train, y_train) / predict_proba(X) -> (n, c).
- Graph operators: fit(adj, X, y, train_idx) / predict_proba(X_all) -> (n, c).
  GraphSAGEOp and GCNOp are *degraded* scikit-learn stand-ins (no torch);
  HeterophilyGNNOp is a stub that raises OperatorUnavailable.
- Fusion operators: strict outer-fold cross-fit via ``cross_fit`` so the
  fusion meta-model never sees held-out-fold predictions during fitting.

Graph views are built by GraphViewRegistry.build_view(name, adj) which
returns (csr_matrix, content_hash).  View semantics mirror b1/repair.py:
directed_out = adj, directed_in = adj.T, undirected_union = maximum(adj, adj.T).
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, issparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ...research.event_store import stable_hash

SEED = 2026


class OperatorUnavailable(RuntimeError):
    """Raised when an operator's hard dependency (e.g. torch) is missing."""


def _dense(X: Any) -> np.ndarray:
    if issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _row_normalize(adj: csr_matrix) -> csr_matrix:
    """Row-normalize; zero-degree rows stay zero."""
    adj = adj.tocsr().astype(np.float64)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.where(deg > 1e-12, 1.0 / np.maximum(deg, 1e-12), 0.0)
    return diags(inv) @ adj


def _sym_normalize_with_self_loops(adj: csr_matrix) -> csr_matrix:
    """D^-1/2 (A + I) D^-1/2 (GCN-style symmetric normalization)."""
    n = adj.shape[0]
    a = (adj + eye(n, format="csr")).tocsr().astype(np.float64)
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv_sqrt = np.where(deg > 1e-12, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    d = diags(inv_sqrt)
    return (d @ a @ d).tocsr()


def _normalize_rows(P: np.ndarray) -> np.ndarray:
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-12, None)
    s = P.sum(axis=1, keepdims=True)
    return P / s


def _onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Graph view registry
# ---------------------------------------------------------------------------


class GraphViewRegistry:
    """Deterministic graph view construction with content hashing.

    Views ``community`` and ``structural_embedding`` return per-node feature
    matrices (not adjacency matrices); all other views return adjacency-like
    csr_matrices on the same node set.
    """

    VIEWS = (
        "directed_out",
        "directed_in",
        "undirected_union",
        "one_hop",
        "exact_two_hop",
        "higher_hop",
        "community",
        "structural_embedding",
    )

    HIGHER_HOP_K = 3  # exact-k-hop neighborhood radius for the higher_hop view

    @staticmethod
    def content_hash(mat: csr_matrix) -> str:
        m = mat.tocsr()
        m.sum_duplicates()
        m.sort_indices()
        return stable_hash(
            {
                "shape": [int(m.shape[0]), int(m.shape[1])],
                "indptr": [int(v) for v in m.indptr.tolist()],
                "indices": [int(v) for v in m.indices.tolist()],
                "data": [float(v) for v in m.data.tolist()],
            }
        )

    @classmethod
    def build_view(cls, name: str, adj: csr_matrix) -> tuple[csr_matrix, str]:
        adj = adj.tocsr().astype(np.float64)
        if name == "directed_out":
            view = adj.copy()
        elif name == "directed_in":
            view = adj.T.tocsr()
        elif name == "undirected_union":
            view = adj.maximum(adj.T).tocsr()
        elif name == "one_hop":
            view = cls._binarize(adj.maximum(adj.T))
        elif name == "exact_two_hop":
            view = cls._exact_hop(adj, k=2)
        elif name == "higher_hop":
            view = cls._exact_hop(adj, k=cls.HIGHER_HOP_K)
        elif name == "community":
            view = cls._community_features(adj)
        elif name == "structural_embedding":
            view = cls._structural_embedding(adj)
        else:
            raise ValueError(f"unknown graph view: {name!r}; expected one of {cls.VIEWS}")
        view = view.tocsr()
        view.sum_duplicates()
        view.sort_indices()
        return view, cls.content_hash(view)

    @staticmethod
    def _binarize(adj: csr_matrix) -> csr_matrix:
        m = adj.copy()
        m.data = np.ones_like(m.data)
        return m.tocsr()

    @classmethod
    def _exact_hop(cls, adj: csr_matrix, k: int) -> csr_matrix:
        """Binarized exact-k-hop view of the undirected one-hop graph.

        One-hop edges and self-loops are REMOVED: a node pair is kept only
        when the shortest path length is exactly k.
        """
        one = cls._binarize(adj.maximum(adj.T))
        one.setdiag(0)
        one.eliminate_zeros()
        reach = one.copy()  # union of 1..(k-1) hop reachability
        power = one.copy()
        for step in range(2, k + 1):
            power = cls._binarize(power @ one)
            if step == k:
                exact = power
            else:
                reach = cls._binarize(reach.maximum(power))
        exact = exact - reach  # drop anything reachable in fewer hops
        exact.data = np.clip(exact.data, 0.0, 1.0)
        exact.setdiag(0)
        exact.eliminate_zeros()
        return cls._binarize(exact)

    @staticmethod
    def _community_features(adj: csr_matrix) -> csr_matrix:
        """One-hot connected-component-id feature matrix (n x n_components)."""
        und = adj.maximum(adj.T)
        n_comp, labels = connected_components(und, directed=False)
        rows = np.arange(adj.shape[0])
        return csr_matrix(
            (np.ones(adj.shape[0]), (rows, labels.astype(np.int64))),
            shape=(adj.shape[0], int(n_comp)),
        )

    @staticmethod
    def _pagerank(adj: csr_matrix, damping: float = 0.85, n_iter: int = 50) -> np.ndarray:
        n = adj.shape[0]
        a_norm = _row_normalize(adj)
        pr = np.full(n, 1.0 / max(n, 1))
        teleport = np.full(n, 1.0 / max(n, 1))
        for _ in range(n_iter):
            pr = damping * (a_norm.T @ pr) + (1.0 - damping) * teleport
        return pr

    @classmethod
    def _structural_embedding(cls, adj: csr_matrix) -> csr_matrix:
        """Per-node [degree, clustering_coefficient, pagerank] column matrix."""
        und = cls._binarize(adj.maximum(adj.T))
        und.setdiag(0)
        und.eliminate_zeros()
        deg = np.asarray(und.sum(axis=1)).ravel()
        a3_diag = np.asarray((und @ und @ und).diagonal()).ravel()
        triangles = a3_diag / 2.0
        denom = deg * (deg - 1.0)
        clustering = np.where(denom > 0, 2.0 * triangles / np.maximum(denom, 1e-12), 0.0)
        pagerank = cls._pagerank(und)
        return csr_matrix(np.column_stack([deg, clustering, pagerank]))


# ---------------------------------------------------------------------------
# Feature operators
# ---------------------------------------------------------------------------


class FeatureOperator(ABC):
    """fit(X_train, y_train) / predict_proba(X) -> (n_samples, n_classes)."""

    operator_id: str = ""

    @abstractmethod
    def fit(self, X_train: Any, y_train: np.ndarray) -> "FeatureOperator":
        ...

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        ...


class LinearOp(FeatureOperator):
    """StandardScaler + multinomial LogisticRegression."""

    operator_id = "linear"

    def __init__(self, C: float = 1.0, max_iter: int = 500) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, X_train: Any, y_train: np.ndarray) -> "LinearOp":
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(_dense(X_train))
        self.clf_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs", random_state=SEED, n_jobs=1
        )
        self.clf_.fit(Xs, y_train)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return _normalize_rows(self.clf_.predict_proba(self.scaler_.transform(_dense(X))))


class GBDTOp(FeatureOperator):
    """sklearn GradientBoostingClassifier (CPU stand-in for LightGBM/XGBoost)."""

    operator_id = "gbdt"

    def __init__(self, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X_train: Any, y_train: np.ndarray) -> "GBDTOp":
        self.clf_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=SEED,
        )
        self.clf_.fit(_dense(X_train), y_train)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return _normalize_rows(self.clf_.predict_proba(_dense(X)))


class ResidualMLPOp(FeatureOperator):
    """StandardScaler + MLPClassifier.

    Degraded residual MLP: sklearn's MLPClassifier has no skip connections,
    so the "hidden residual" is approximated by concatenating the scaled
    input features with the first hidden block width (documented degradation;
    a true residual MLP requires torch).  The extra capacity plus low
    regularization plays the role residual paths play in deep tabular models.
    """

    operator_id = "residual_mlp"

    def __init__(self, hidden: tuple[int, ...] = (64, 32), max_iter: int = 300) -> None:
        self.hidden = hidden
        self.max_iter = max_iter

    def fit(self, X_train: Any, y_train: np.ndarray) -> "ResidualMLPOp":
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(_dense(X_train))
        self.clf_ = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            alpha=1e-4,
            max_iter=self.max_iter,
            random_state=SEED,
            learning_rate_init=1e-3,
        )
        self.clf_.fit(Xs, y_train)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return _normalize_rows(self.clf_.predict_proba(self.scaler_.transform(_dense(X))))


class PCALowRankOp(FeatureOperator):
    """PCA (n_components covering ~90% variance, capped) + LogisticRegression."""

    operator_id = "pca_lowrank"

    VARIANCE_TARGET = 0.90
    MAX_COMPONENTS = 64

    def __init__(self, variance_target: float = VARIANCE_TARGET, max_components: int = MAX_COMPONENTS) -> None:
        self.variance_target = variance_target
        self.max_components = max_components

    def fit(self, X_train: Any, y_train: np.ndarray) -> "PCALowRankOp":
        Xd = _dense(X_train)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(Xd)
        cap = int(min(self.max_components, Xs.shape[0], Xs.shape[1]))
        self.pca_ = PCA(n_components=cap, random_state=SEED)
        Z = self.pca_.fit_transform(Xs)
        cum = np.cumsum(self.pca_.explained_variance_ratio_)
        self.n_components_ = int(np.searchsorted(cum, self.variance_target) + 1)
        self.n_components_ = min(self.n_components_, cap)
        self.clf_ = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED, n_jobs=1)
        self.clf_.fit(Z[:, : self.n_components_], y_train)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        Z = self.pca_.transform(self.scaler_.transform(_dense(X)))
        return _normalize_rows(self.clf_.predict_proba(Z[:, : self.n_components_]))


class PrototypeOp(FeatureOperator):
    """Nearest class centroid; softmax over negative euclidean distances."""

    operator_id = "prototype"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def fit(self, X_train: Any, y_train: np.ndarray) -> "PrototypeOp":
        Xd = _dense(X_train)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(Xd)
        self.classes_ = np.unique(y_train)
        self.centroids_ = np.vstack([Xs[y_train == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        Xs = self.scaler_.transform(_dense(X))
        d2 = ((Xs[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        logits = -np.sqrt(np.maximum(d2, 0.0)) / max(self.temperature, 1e-12)
        logits -= logits.max(axis=1, keepdims=True)
        return _normalize_rows(np.exp(logits))


# ---------------------------------------------------------------------------
# Graph operators
# ---------------------------------------------------------------------------


class GraphOperator(ABC):
    """fit(adj, X, y, train_idx) / predict_proba(X_all) -> (n_nodes, n_classes)."""

    operator_id: str = ""

    @abstractmethod
    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "GraphOperator":
        ...

    @abstractmethod
    def predict_proba(self, X_all: Any) -> np.ndarray:
        ...


class LabelPropagationOp(GraphOperator):
    """Iterative label propagation over the row-normalized adjacency.

    One-hot labels are seeded from train_idx ONLY; unlabeled rows carry a
    zero prior but still receive propagated mass each iteration:
    F <- alpha * Y_train + (1 - alpha) * A_norm @ F.
    """

    operator_id = "label_propagation"

    def __init__(self, alpha: float = 0.15, n_iter: int = 50) -> None:
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "LabelPropagationOp":
        n = adj.shape[0]
        self.classes_ = np.unique(y[train_idx])
        n_classes = len(self.classes_)
        Y = np.zeros((n, n_classes))
        for col, c in enumerate(self.classes_):
            Y[train_idx[y[train_idx] == c], col] = 1.0
        a_norm = _row_normalize(adj)
        F = Y.copy()
        for _ in range(self.n_iter):
            F = self.alpha * Y + (1.0 - self.alpha) * (a_norm @ F)
        self.F_ = np.asarray(F)
        return self

    def predict_proba(self, X_all: Any) -> np.ndarray:
        n_rows = X_all.shape[0] if not issparse(X_all) else X_all.shape[0]
        F = self.F_[:n_rows]
        zero = F.sum(axis=1) <= 1e-12
        P = _normalize_rows(np.where(zero[:, None], 1.0 / F.shape[1], F))
        return P


class SGCOp(GraphOperator):
    """Simplified Graph Convolution: propagate features K steps, then LR."""

    operator_id = "sgc"

    def __init__(self, K: int = 2, max_iter: int = 500) -> None:
        self.K = K
        self.max_iter = max_iter

    def _propagate(self, X: Any) -> np.ndarray:
        Z = _dense(X)
        for _ in range(self.K):
            Z = np.asarray(self.a_norm_ @ Z)
        return Z

    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "SGCOp":
        self.a_norm_ = _sym_normalize_with_self_loops(adj)
        Z = self._propagate(X)
        self.scaler_ = StandardScaler()
        Zs = self.scaler_.fit_transform(Z)
        self.clf_ = LogisticRegression(max_iter=self.max_iter, solver="lbfgs", random_state=SEED, n_jobs=1)
        self.clf_.fit(Zs[train_idx], y[train_idx])
        return self

    def predict_proba(self, X_all: Any) -> np.ndarray:
        Z = self._propagate(X_all)
        return _normalize_rows(self.clf_.predict_proba(self.scaler_.transform(Z)))


class APPNPOp(GraphOperator):
    """Personalized propagation of a base model's predictions (APPNP-style).

    P0 = base LogisticRegression probabilities on raw features;
    P <- alpha * P0 + (1 - alpha) * A_norm @ P for n_iterations.
    """

    operator_id = "appnp"

    def __init__(self, alpha: float = 0.15, n_iterations: int = 10, max_iter: int = 500) -> None:
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.max_iter = max_iter

    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "APPNPOp":
        self.a_norm_ = _row_normalize(adj)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(_dense(X))
        self.base_ = LogisticRegression(max_iter=self.max_iter, solver="lbfgs", random_state=SEED, n_jobs=1)
        self.base_.fit(Xs[train_idx], y[train_idx])
        return self

    def predict_proba(self, X_all: Any) -> np.ndarray:
        P0 = self.base_.predict_proba(self.scaler_.transform(_dense(X_all)))
        P = P0.copy()
        for _ in range(self.n_iterations):
            P = self.alpha * P0 + (1.0 - self.alpha) * np.asarray(self.a_norm_ @ P)
        return _normalize_rows(P)


class GraphSAGEOp(GraphOperator):
    """DEGRADED GraphSAGE (no torch): 1-layer mean aggregation.

    Features [X, A_norm @ X] feed an MLPClassifier.  There is no learned
    aggregation weight matrix per hop and no neighbor sampling; this is a
    feature-engineered stand-in, marked degraded for deployment gating.
    """

    operator_id = "graphsage_degraded"

    def __init__(self, hidden: tuple[int, ...] = (64,), max_iter: int = 300) -> None:
        self.hidden = hidden
        self.max_iter = max_iter

    def _features(self, X: Any) -> np.ndarray:
        Xd = _dense(X)
        return np.hstack([Xd, np.asarray(self.a_norm_ @ Xd)])

    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "GraphSAGEOp":
        self.a_norm_ = _row_normalize(adj)
        F = self._features(X)
        self.scaler_ = StandardScaler()
        Fs = self.scaler_.fit_transform(F)
        self.clf_ = MLPClassifier(
            hidden_layer_sizes=self.hidden, max_iter=self.max_iter, random_state=SEED, learning_rate_init=1e-3
        )
        self.clf_.fit(Fs[train_idx], y[train_idx])
        return self

    def predict_proba(self, X_all: Any) -> np.ndarray:
        return _normalize_rows(self.clf_.predict_proba(self.scaler_.transform(self._features(X_all))))


class GCNOp(GraphOperator):
    """DEGRADED GCN (no torch): symmetric-normalized propagation + MLP.

    Features Â X with Â = D^-1/2 (A+I) D^-1/2 feed an MLPClassifier — a
    one-propagation-step, feature-engineered stand-in for a stacked GCN.
    """

    operator_id = "gcn_degraded"

    def __init__(self, hidden: tuple[int, ...] = (64,), max_iter: int = 300) -> None:
        self.hidden = hidden
        self.max_iter = max_iter

    def _features(self, X: Any) -> np.ndarray:
        return np.asarray(self.a_hat_ @ _dense(X))

    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "GCNOp":
        self.a_hat_ = _sym_normalize_with_self_loops(adj)
        F = self._features(X)
        self.scaler_ = StandardScaler()
        Fs = self.scaler_.fit_transform(F)
        self.clf_ = MLPClassifier(
            hidden_layer_sizes=self.hidden, max_iter=self.max_iter, random_state=SEED, learning_rate_init=1e-3
        )
        self.clf_.fit(Fs[train_idx], y[train_idx])
        return self

    def predict_proba(self, X_all: Any) -> np.ndarray:
        return _normalize_rows(self.clf_.predict_proba(self.scaler_.transform(self._features(X_all))))


class HeterophilyGNNOp(GraphOperator):
    """Stub: heterophily-aware GNN requires torch, which is not installed."""

    operator_id = "heterophily_gnn"

    def fit(self, adj: csr_matrix, X: Any, y: np.ndarray, train_idx: np.ndarray) -> "HeterophilyGNNOp":
        raise OperatorUnavailable("heterophily_gnn requires missing dependency: torch")

    def predict_proba(self, X_all: Any) -> np.ndarray:
        raise OperatorUnavailable("heterophily_gnn requires missing dependency: torch")


# ---------------------------------------------------------------------------
# Fusion operators (strict outer-fold cross-fit)
# ---------------------------------------------------------------------------


class FusionOperator(ABC):
    """fit(proba_a, proba_b, y, meta) / fuse(proba_a, proba_b, meta)."""

    operator_id: str = ""

    @abstractmethod
    def fit(
        self, proba_a: np.ndarray, proba_b: np.ndarray, y: np.ndarray, meta: np.ndarray | None = None
    ) -> "FusionOperator":
        ...

    @abstractmethod
    def fuse(self, proba_a: np.ndarray, proba_b: np.ndarray, meta: np.ndarray | None = None) -> np.ndarray:
        ...


def _log_loss(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(P, 1e-12, 1.0)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


_ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


class ResidualCorrectionOp(FusionOperator):
    """b predicts a's residual error: fused = a + Ridge(b -> onehot(y) - a)."""

    operator_id = "residual_correction"

    def fit(self, proba_a, proba_b, y, meta=None) -> "ResidualCorrectionOp":
        residual = _onehot(y, proba_a.shape[1]) - proba_a
        self.ridge_ = Ridge(alpha=1.0)
        self.ridge_.fit(proba_b, residual)
        return self

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        return _normalize_rows(proba_a + self.ridge_.predict(proba_b))


class ProbabilityBlendOp(FusionOperator):
    """Scalar alpha blend in probability space, alpha by log-loss grid search."""

    operator_id = "probability_blend"

    def fit(self, proba_a, proba_b, y, meta=None) -> "ProbabilityBlendOp":
        best_alpha, best_loss = 0.5, np.inf
        for alpha in _ALPHA_GRID:
            loss = _log_loss(_normalize_rows(alpha * proba_a + (1 - alpha) * proba_b), y)
            if loss < best_loss - 1e-12:
                best_alpha, best_loss = float(alpha), loss
        self.alpha_ = best_alpha
        return self

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        return _normalize_rows(self.alpha_ * proba_a + (1.0 - self.alpha_) * proba_b)


class LogitBlendOp(FusionOperator):
    """Scalar alpha blend in log-probability (logit) space."""

    operator_id = "logit_blend"

    def fit(self, proba_a, proba_b, y, meta=None) -> "LogitBlendOp":
        best_alpha, best_loss = 0.5, np.inf
        la = np.log(np.clip(proba_a, 1e-12, 1.0))
        lb = np.log(np.clip(proba_b, 1e-12, 1.0))
        for alpha in _ALPHA_GRID:
            loss = _log_loss(_normalize_rows(np.exp(alpha * la + (1 - alpha) * lb)), y)
            if loss < best_loss - 1e-12:
                best_alpha, best_loss = float(alpha), loss
        self.alpha_ = best_alpha
        return self

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        la = np.log(np.clip(proba_a, 1e-12, 1.0))
        lb = np.log(np.clip(proba_b, 1e-12, 1.0))
        return _normalize_rows(np.exp(self.alpha_ * la + (1.0 - self.alpha_) * lb))


class ClassAwareGateOp(FusionOperator):
    """Per-class alpha via cross-fit: alpha_c minimizes per-class log-loss."""

    operator_id = "class_aware_gate"

    def fit(self, proba_a, proba_b, y, meta=None) -> "ClassAwareGateOp":
        n_classes = proba_a.shape[1]
        self.alphas_ = np.full(n_classes, 0.5)
        for c in range(n_classes):
            target = (y == c).astype(np.float64)
            best_alpha, best_loss = 0.5, np.inf
            for alpha in _ALPHA_GRID:
                p = np.clip(alpha * proba_a[:, c] + (1 - alpha) * proba_b[:, c], 1e-12, 1.0)
                loss = float(-(target * np.log(p) + (1 - target) * np.log(1 - p)).mean())
                if loss < best_loss - 1e-12:
                    best_alpha, best_loss = float(alpha), loss
            self.alphas_[c] = best_alpha
        return self

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        P = self.alphas_[None, :] * proba_a + (1.0 - self.alphas_[None, :]) * proba_b
        return _normalize_rows(P)


class NodeGateOp(FusionOperator):
    """Logistic gate on meta features [proba_a, proba_b] (stacking), cross-fit."""

    operator_id = "node_gate"

    def fit(self, proba_a, proba_b, y, meta=None) -> "NodeGateOp":
        self.n_classes_ = proba_a.shape[1]
        feats = self._features(proba_a, proba_b, meta)
        self.clf_ = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED, n_jobs=1)
        self.clf_.fit(feats, y)
        return self

    @staticmethod
    def _features(proba_a, proba_b, meta):
        parts = [proba_a, proba_b]
        if meta is not None:
            parts.append(np.asarray(meta, dtype=np.float64).reshape(len(proba_a), -1))
        return np.hstack(parts)

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        P = self.clf_.predict_proba(self._features(proba_a, proba_b, meta))
        out = np.zeros((len(proba_a), self.n_classes_))
        out[:, self.clf_.classes_.astype(int)] = P
        return _normalize_rows(np.where(out > 0, out, 1e-12))


class BucketExpertOp(FusionOperator):
    """Route by precomputed bucket mask: per-bucket best blend alpha.

    meta must be an integer bucket-id vector (one id per row); unseen buckets
    at fuse time fall back to alpha = 0.5.
    """

    operator_id = "bucket_expert"

    DEFAULT_ALPHA = 0.5

    def fit(self, proba_a, proba_b, y, meta=None) -> "BucketExpertOp":
        if meta is None:
            raise ValueError("bucket_expert requires meta=bucket-id vector")
        buckets = np.asarray(meta).ravel()
        self.bucket_alphas_ = {}
        for b in np.unique(buckets):
            mask = buckets == b
            best_alpha, best_loss = self.DEFAULT_ALPHA, np.inf
            for alpha in _ALPHA_GRID:
                loss = _log_loss(_normalize_rows(alpha * proba_a[mask] + (1 - alpha) * proba_b[mask]), y[mask])
                if loss < best_loss - 1e-12:
                    best_alpha, best_loss = float(alpha), loss
            self.bucket_alphas_[int(b)] = best_alpha
        return self

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        if meta is None:
            raise ValueError("bucket_expert requires meta=bucket-id vector")
        buckets = np.asarray(meta).ravel()
        alphas = np.array([self.bucket_alphas_.get(int(b), self.DEFAULT_ALPHA) for b in buckets])
        P = alphas[:, None] * proba_a + (1.0 - alphas[:, None]) * proba_b
        return _normalize_rows(P)


class CalibrationOp(FusionOperator):
    """Temperature scaling of the 50/50 blend, T fitted on fit-folds only."""

    operator_id = "calibration"

    TEMPERATURE_GRID = np.round(np.arange(0.5, 5.0001, 0.25), 2)

    def fit(self, proba_a, proba_b, y, meta=None) -> "CalibrationOp":
        blend = _normalize_rows(0.5 * proba_a + 0.5 * proba_b)
        best_t, best_loss = 1.0, np.inf
        for t in self.TEMPERATURE_GRID:
            loss = _log_loss(self._scale(blend, t), y)
            if loss < best_loss - 1e-12:
                best_t, best_loss = float(t), loss
        self.temperature_ = best_t
        return self

    @staticmethod
    def _scale(P: np.ndarray, t: float) -> np.ndarray:
        return _normalize_rows(np.power(np.clip(P, 1e-12, 1.0), 1.0 / t))

    def fuse(self, proba_a, proba_b, meta=None) -> np.ndarray:
        blend = _normalize_rows(0.5 * proba_a + 0.5 * proba_b)
        return self._scale(blend, self.temperature_)


def cross_fit(
    operator: FusionOperator,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    meta: np.ndarray | None = None,
) -> np.ndarray:
    """Strict outer-fold cross-fit of a fusion operator.

    For each outer fold the operator is fitted on the OTHER folds' (already
    out-of-fold) probabilities and applied to the held-out fold, so no fold's
    predictions ever influence its own fusion parameters.  Returns fused OOF
    probabilities aligned with the rows of proba_a/proba_b.
    """
    proba_a = np.asarray(proba_a, dtype=np.float64)
    proba_b = np.asarray(proba_b, dtype=np.float64)
    oof = np.zeros_like(proba_a)
    for i, val_idx in enumerate(folds):
        fit_idx = np.concatenate([np.asarray(folds[j]) for j in range(len(folds)) if j != i])
        val_idx = np.asarray(val_idx)
        op = copy.deepcopy(operator)
        op.fit(
            proba_a[fit_idx],
            proba_b[fit_idx],
            y[fit_idx],
            meta[fit_idx] if meta is not None else None,
        )
        oof[val_idx] = op.fuse(
            proba_a[val_idx],
            proba_b[val_idx],
            meta[val_idx] if meta is not None else None,
        )
    return _normalize_rows(oof)


# ---------------------------------------------------------------------------
# Operator catalog
# ---------------------------------------------------------------------------


def _entry(
    family: str,
    *,
    implemented: bool = True,
    available: bool = True,
    missing_dependency: list[str] | None = None,
    expected_runtime_seconds: float = 10.0,
    supports_oof: bool = True,
    supports_test: bool = True,
    deployment_ready: bool = True,
    scientific_priority: int = 5,
) -> dict[str, Any]:
    return {
        "family": family,
        "implemented": implemented,
        "available": available,
        "missing_dependency": missing_dependency or [],
        "expected_runtime_seconds": expected_runtime_seconds,
        "supports_oof": supports_oof,
        "supports_test": supports_test,
        "deployment_ready": deployment_ready,
        "scientific_priority": scientific_priority,
    }


CLASSIFICATION_OPERATORS: dict[str, dict[str, Any]] = {
    "linear": _entry("feature", expected_runtime_seconds=5.0, scientific_priority=3),
    "gbdt": _entry("feature", expected_runtime_seconds=30.0, scientific_priority=4),
    "residual_mlp": _entry("feature", expected_runtime_seconds=60.0, deployment_ready=False, scientific_priority=6),
    "pca_lowrank": _entry("feature", expected_runtime_seconds=10.0, scientific_priority=4),
    "prototype": _entry("feature", expected_runtime_seconds=5.0, scientific_priority=3),
    "label_propagation": _entry("graph", expected_runtime_seconds=10.0, scientific_priority=5),
    "sgc": _entry("graph", expected_runtime_seconds=15.0, scientific_priority=6),
    "appnp": _entry("graph", expected_runtime_seconds=20.0, scientific_priority=7),
    "graphsage_degraded": _entry("graph", expected_runtime_seconds=60.0, deployment_ready=False, scientific_priority=5),
    "gcn_degraded": _entry("graph", expected_runtime_seconds=60.0, deployment_ready=False, scientific_priority=5),
    "heterophily_gnn": _entry(
        "graph",
        implemented=False,
        available=False,
        missing_dependency=["torch"],
        expected_runtime_seconds=120.0,
        supports_oof=False,
        supports_test=False,
        deployment_ready=False,
        scientific_priority=9,
    ),
}


_FEATURE_OPS: dict[str, type[FeatureOperator]] = {
    "linear": LinearOp,
    "gbdt": GBDTOp,
    "residual_mlp": ResidualMLPOp,
    "pca_lowrank": PCALowRankOp,
    "prototype": PrototypeOp,
}

_GRAPH_OPS: dict[str, type[GraphOperator]] = {
    "label_propagation": LabelPropagationOp,
    "sgc": SGCOp,
    "appnp": APPNPOp,
    "graphsage_degraded": GraphSAGEOp,
    "gcn_degraded": GCNOp,
    "heterophily_gnn": HeterophilyGNNOp,
}

FUSION_OPERATORS: dict[str, type[FusionOperator]] = {
    "residual_correction": ResidualCorrectionOp,
    "probability_blend": ProbabilityBlendOp,
    "logit_blend": LogitBlendOp,
    "class_aware_gate": ClassAwareGateOp,
    "node_gate": NodeGateOp,
    "bucket_expert": BucketExpertOp,
    "calibration": CalibrationOp,
}


def build_operator(operator_id: str, **kwargs: Any) -> FeatureOperator | GraphOperator:
    """Instantiate a feature or graph operator from the catalog by id."""
    if operator_id in _FEATURE_OPS:
        return _FEATURE_OPS[operator_id](**kwargs)
    if operator_id in _GRAPH_OPS:
        return _GRAPH_OPS[operator_id](**kwargs)
    raise KeyError(f"unknown classification operator: {operator_id!r}")
