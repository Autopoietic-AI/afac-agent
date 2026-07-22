# -*- coding: utf-8 -*-
"""B1 model adapters (CPU-only scikit-learn + sparse propagation).

All models expose:
  fit(X_train, y_train, **kwargs)
  predict_proba(X_all) -> (n_nodes, n_classes)

OOF assembly is handled by the runner, which calls fit/predict per fold.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class B1Model(ABC):
    def __init__(self, *, model_id: str, n_classes: int) -> None:
        self.model_id = model_id
        self.n_classes = n_classes
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> "B1Model":
        ...

    @abstractmethod
    def predict_proba(self, X_all: np.ndarray) -> np.ndarray:
        ...


def _dense(X: Any) -> np.ndarray:
    if issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float32)


class FeatureLogistic(B1Model):
    """Feature-only Logistic Regression."""

    def __init__(self, *, model_id: str, n_classes: int, C: float = 1.0, max_iter: int = 300) -> None:
        super().__init__(model_id=model_id, n_classes=n_classes)
        self.C = C
        self.max_iter = max_iter
        self.clf: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> "FeatureLogistic":
        Xd = _dense(X_train)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(Xd)
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=2026,
            n_jobs=1,
        )
        self.clf.fit(Xs, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X_all: np.ndarray) -> np.ndarray:
        if self.clf is None or self.scaler is None:
            raise RuntimeError("not fitted")
        Xs = self.scaler.transform(_dense(X_all))
        return self.clf.predict_proba(Xs).astype(np.float64)


class FeatureMLP(B1Model):
    """Feature-only small MLP with early stopping."""

    def __init__(
        self,
        *,
        model_id: str,
        n_classes: int,
        hidden: tuple[int, ...] = (256, 128),
        alpha: float = 1e-4,
        max_iter: int = 200,
    ) -> None:
        super().__init__(model_id=model_id, n_classes=n_classes)
        self.hidden = hidden
        self.alpha = alpha
        self.max_iter = max_iter
        self.clf: MLPClassifier | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> "FeatureMLP":
        self.clf = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=2026,
            learning_rate_init=1e-3,
        )
        self.clf.fit(_dense(X_train), y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X_all: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("not fitted")
        return self.clf.predict_proba(_dense(X_all)).astype(np.float64)


class LabelPropagationModel(B1Model):
    """Iterative label propagation over a normalized adjacency matrix."""

    def __init__(
        self,
        *,
        model_id: str,
        n_classes: int,
        adj: csr_matrix,
        alpha: float = 0.9,
        n_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(model_id=model_id, n_classes=n_classes)
        self.adj = adj
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.train_idx: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> "LabelPropagationModel":
        # X_train ignored; y_train corresponds to train_idx passed via kwargs
        self.train_idx = np.asarray(kwargs["train_idx"], dtype=np.int64)
        n = self.adj.shape[0]
        Y = np.zeros((n, self.n_classes), dtype=np.float64)
        for i, yi in zip(self.train_idx, y_train):
            Y[i, yi] = 1.0
        # Row-normalized transition matrix
        deg = np.diff(self.adj.indptr).astype(np.float64)
        deg[deg == 0] = 1.0
        P = self.adj.T.astype(np.float64)
        for i in range(n):
            P.data[P.indptr[i]:P.indptr[i + 1]] /= deg[i]
        for _ in range(self.n_iter):
            Y_new = self.alpha * (P @ Y)
            Y_new[self.train_idx] = 0.0
            Y_new[self.train_idx] += np.eye(self.n_classes)[y_train]  # clamp
            if np.linalg.norm(Y_new - Y, ord="fro") < self.tol:
                break
            Y = Y_new
        self.Y_ = Y
        self.is_fitted = True
        return self

    def predict_proba(self, X_all: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("not fitted")
        return self.Y_ / (self.Y_.sum(axis=1, keepdims=True) + 1e-12)


class APPNPFeatureModel(B1Model):
    """APPNP-style feature smoothing + logistic regression."""

    def __init__(
        self,
        *,
        model_id: str,
        n_classes: int,
        adj: csr_matrix,
        alpha: float = 0.2,
        K: int = 10,
        base: str = "logistic",
    ) -> None:
        super().__init__(model_id=model_id, n_classes=n_classes)
        self.adj = adj
        self.alpha = alpha
        self.K = K
        self.base = base
        self.clf: LogisticRegression | MLPClassifier | None = None

    def _smooth_features(self, X: np.ndarray) -> np.ndarray:
        n = self.adj.shape[0]
        deg = np.diff(self.adj.indptr).astype(np.float64)
        deg[deg == 0] = 1.0
        P = self.adj.T.astype(np.float64)
        for i in range(n):
            P.data[P.indptr[i]:P.indptr[i + 1]] /= deg[i]
        Z = (1.0 - self.alpha) * X
        H = Z.copy()
        for _ in range(self.K):
            H = self.alpha * (P @ H) + Z
        return H

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> "APPNPFeatureModel":
        X_all = _dense(kwargs["X_all"])
        H = self._smooth_features(X_all)
        train_idx = np.asarray(kwargs["train_idx"], dtype=np.int64)
        if self.base == "mlp":
            self.clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=150, early_stopping=True, random_state=2026)
        else:
            self.clf = LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=1, random_state=2026)
        self.clf.fit(H[train_idx], y_train)
        self.H_ = H
        self.is_fitted = True
        return self

    def predict_proba(self, X_all: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("not fitted")
        return self.clf.predict_proba(self.H_).astype(np.float64)


class NeighborFeatureModel(B1Model):
    """Concatenate node features with neighbor-mean features, then LR."""

    def __init__(
        self,
        *,
        model_id: str,
        n_classes: int,
        adj: csr_matrix,
        base: str = "logistic",
    ) -> None:
        super().__init__(model_id=model_id, n_classes=n_classes)
        self.adj = adj
        self.base = base
        self.clf: LogisticRegression | MLPClassifier | None = None
        self.H_: np.ndarray | None = None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        n = self.adj.shape[0]
        deg = np.diff(self.adj.indptr).astype(np.float64)
        deg[deg == 0] = 1.0
        # mean = A @ X / deg
        mean = (self.adj.astype(np.float64) @ X) / deg[:, None]
        return np.hstack([X, mean])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> "NeighborFeatureModel":
        X_all = _dense(kwargs["X_all"])
        H = self._augment(X_all)
        train_idx = np.asarray(kwargs["train_idx"], dtype=np.int64)
        if self.base == "mlp":
            self.clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=150, early_stopping=True, random_state=2026)
        else:
            self.clf = LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=1, random_state=2026)
        self.clf.fit(H[train_idx], y_train)
        self.H_ = H
        self.is_fitted = True
        return self

    def predict_proba(self, X_all: np.ndarray) -> np.ndarray:
        if self.clf is None or self.H_ is None:
            raise RuntimeError("not fitted")
        return self.clf.predict_proba(self.H_).astype(np.float64)


def instantiate_model(
    *,
    model_id: str,
    model_family: str,
    n_classes: int,
    adj: csr_matrix,
    X_all: np.ndarray,
    view: str = "undirected_union",
    **kwargs: Any,
) -> B1Model:
    A = adj if view == "undirected_union" else (
        adj + adj.T if view == "directed_out" else adj + adj.T
    )
    A = A.tocsr()
    if model_family == "feature_logistic":
        return FeatureLogistic(model_id=model_id, n_classes=n_classes, C=kwargs.get("C", 1.0))
    if model_family == "feature_mlp":
        return FeatureMLP(model_id=model_id, n_classes=n_classes, hidden=kwargs.get("hidden", (256, 128)))
    if model_family == "label_propagation":
        return LabelPropagationModel(model_id=model_id, n_classes=n_classes, adj=A, alpha=kwargs.get("alpha", 0.9), n_iter=kwargs.get("n_iter", 50))
    if model_family == "appnp_logistic":
        return APPNPFeatureModel(model_id=model_id, n_classes=n_classes, adj=A, alpha=kwargs.get("alpha", 0.2), K=kwargs.get("K", 10), base="logistic")
    if model_family == "neighbor_logistic":
        return NeighborFeatureModel(model_id=model_id, n_classes=n_classes, adj=A, base="logistic")
    raise ValueError(f"unknown model_family: {model_family}")
