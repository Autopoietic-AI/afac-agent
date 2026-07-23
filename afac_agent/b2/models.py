# -*- coding: utf-8 -*-
"""B2 recommendation baselines.

Retrieval models:
  - HistoryRecallModel
  - PopularityModel
  - ItemItemCooccurrenceModel
  - LastItemTransitionModel
  - ScoreBlendModel

Ranking model:
  - CandidateRankerModel (LogisticRegression or MLP over user/item/history feats)

All models expose ``fit(user_ids, item_list, ...)`` and ``topk(user_ids, k=10)``.
OOF assembly is performed by the closed-loop runner, which trains on fit users
and predicts on held-out users for each fold.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _cold_start_score(pop: dict[str, float], item_list: list[str]) -> np.ndarray:
    return np.array([pop.get(iid, 0.0) for iid in item_list], dtype=np.float64)


class B2Model(ABC):
    def __init__(self, *, model_id: str) -> None:
        self.model_id = model_id
        self.is_fitted = False

    @abstractmethod
    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "B2Model":
        ...

    @abstractmethod
    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        """Return (n_users, n_items) score array aligned to ``item_list``."""
        ...

    def topk(self, user_ids: list[str], k: int = 10) -> np.ndarray:
        scores = self.predict_scores(user_ids)
        idx = np.argsort(-scores, axis=1)[:, :k]
        item_arr = np.array(self.item_list(), dtype=object)
        return item_arr[idx]

    @abstractmethod
    def item_list(self) -> list[str]:
        ...


class PopularityModel(B2Model):
    """Global item popularity."""

    def __init__(self, *, model_id: str = "B2_POPULARITY", count_col: str = "item_seq_raw") -> None:
        super().__init__(model_id=model_id)
        self.count_col = count_col
        self.pop: dict[str, float] = {}
        self.items: list[str] = []

    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "PopularityModel":
        self.items = list(item_list)
        counts: Counter[str] = Counter()
        for uid in user_ids:
            counts.update(train_seq.get(uid, []))
        # Add target counts if available
        if train_targets:
            counts.update(train_targets.values())
        self.pop = {iid: float(counts.get(iid, 0)) for iid in self.items}
        self.is_fitted = True
        return self

    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        base = _cold_start_score(self.pop, self.items).astype(np.float32)
        return np.tile(base, (len(user_ids), 1))

    def item_list(self) -> list[str]:
        return self.items


class HistoryRecallModel(B2Model):
    """Score items by their frequency in the user's own history."""

    def __init__(self, *, model_id: str = "B2_HISTORY_RECALL") -> None:
        super().__init__(model_id=model_id)
        self.items: list[str] = []
        self.train_seq: dict[str, list[str]] = {}
        self.pop_model: PopularityModel | None = None

    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "HistoryRecallModel":
        self.items = list(item_list)
        self.train_seq = train_seq
        self.pop_model = PopularityModel(model_id="_pop_fallback").fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
        self.is_fitted = True
        return self

    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        # Build result in batches to limit peak memory for large user x item matrices.
        batch_size = 4096
        n_users = len(user_ids)
        n_items = len(self.items)
        scores = np.zeros((n_users, n_items), dtype=np.float32)
        item2idx = {iid: i for i, iid in enumerate(self.items)}
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_uids = user_ids[start:end]
            for i, uid in enumerate(batch_uids):
                seq = self.train_seq.get(uid, [])
                if not seq:
                    scores[start + i] = self.pop_model.predict_scores([uid])[0].astype(np.float32)
                    continue
                counts = Counter(seq)
                for iid, cnt in counts.items():
                    if iid in item2idx:
                        scores[start + i, item2idx[iid]] = float(cnt)
        return scores

    def item_list(self) -> list[str]:
        return self.items


class ItemItemCooccurrenceModel(B2Model):
    """Aggregate co-occurrence scores between history items and candidate items."""

    def __init__(self, *, model_id: str = "B2_ITEM_ITEM_COOCCURRENCE", window: int = 5) -> None:
        super().__init__(model_id=model_id)
        self.window = window
        self.items: list[str] = []
        self.item2idx: dict[str, int] = {}
        self.cooc: csr_matrix | None = None
        self.train_seq: dict[str, list[str]] = {}
        self.pop_model: PopularityModel | None = None

    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "ItemItemCooccurrenceModel":
        self.items = list(item_list)
        self.item2idx = {iid: i for i, iid in enumerate(self.items)}
        self.train_seq = train_seq

        n = len(self.items)
        row, col, data = [], [], []
        for uid in user_ids:
            seq = train_seq.get(uid, [])
            dedup = list(dict.fromkeys(seq))
            for i, a in enumerate(dedup):
                if a not in self.item2idx:
                    continue
                for b in dedup[i + 1: min(len(dedup), i + 1 + self.window)]:
                    if b not in self.item2idx:
                        continue
                    ai, bi = self.item2idx[a], self.item2idx[b]
                    row.extend([ai, bi])
                    col.extend([bi, ai])
                    data.extend([1.0, 1.0])
        if row:
            self.cooc = csr_matrix((np.asarray(data, dtype=np.float32), (np.asarray(row), np.asarray(col))), shape=(n, n))
            # Normalize by row sum
            row_sums = np.asarray(self.cooc.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            self.cooc = self.cooc.multiply(1.0 / row_sums[:, None]).tocsr()
        else:
            self.cooc = csr_matrix((n, n))
        self.pop_model = PopularityModel(model_id="_pop_fallback").fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
        self.is_fitted = True
        return self

    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        batch_size = 4096
        n_users = len(user_ids)
        n_items = len(self.items)
        scores = np.zeros((n_users, n_items), dtype=np.float32)
        if self.cooc is None:
            return scores
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_uids = user_ids[start:end]
            for i, uid in enumerate(batch_uids):
                seq = self.train_seq.get(uid, [])
                if not seq:
                    scores[start + i] = self.pop_model.predict_scores([uid])[0].astype(np.float32)
                    continue
                hist_idx = [self.item2idx[iid] for iid in dict.fromkeys(seq) if iid in self.item2idx]
                if not hist_idx:
                    scores[start + i] = self.pop_model.predict_scores([uid])[0].astype(np.float32)
                    continue
                agg = np.asarray(self.cooc[hist_idx].sum(axis=0)).ravel().astype(np.float32)
                scores[start + i] = agg
        return scores

    def item_list(self) -> list[str]:
        return self.items


class LastItemTransitionModel(B2Model):
    """Recommend items that tend to follow the user's last history item."""

    def __init__(self, *, model_id: str = "B2_LAST_ITEM_TRANSITION") -> None:
        super().__init__(model_id=model_id)
        self.items: list[str] = []
        self.item2idx: dict[str, int] = {}
        self.trans: csr_matrix | None = None
        self.train_seq: dict[str, list[str]] = {}
        self.pop_model: PopularityModel | None = None

    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "LastItemTransitionModel":
        self.items = list(item_list)
        self.item2idx = {iid: i for i, iid in enumerate(self.items)}
        self.train_seq = train_seq
        n = len(self.items)
        row, col, data = [], [], []
        for uid in user_ids:
            seq = train_seq.get(uid, [])
            if not seq:
                continue
            for prev_iid, next_iid in zip(seq[:-1], seq[1:]):
                if prev_iid in self.item2idx and next_iid in self.item2idx:
                    row.append(self.item2idx[prev_iid])
                    col.append(self.item2idx[next_iid])
                    data.append(1.0)
        if row:
            self.trans = csr_matrix((np.asarray(data, dtype=np.float32), (np.asarray(row), np.asarray(col))), shape=(n, n))
            row_sums = np.asarray(self.trans.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            self.trans = self.trans.multiply(1.0 / row_sums[:, None]).tocsr()
        else:
            self.trans = csr_matrix((n, n))
        self.pop_model = PopularityModel(model_id="_pop_fallback").fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
        self.is_fitted = True
        return self

    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        batch_size = 4096
        n_users = len(user_ids)
        n_items = len(self.items)
        scores = np.zeros((n_users, n_items), dtype=np.float32)
        if self.trans is None:
            return scores
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_uids = user_ids[start:end]
            for i, uid in enumerate(batch_uids):
                seq = self.train_seq.get(uid, [])
                if not seq:
                    scores[start + i] = self.pop_model.predict_scores([uid])[0].astype(np.float32)
                    continue
                last = seq[-1]
                if last not in self.item2idx:
                    scores[start + i] = self.pop_model.predict_scores([uid])[0].astype(np.float32)
                    continue
                scores[start + i] = np.asarray(self.trans[self.item2idx[last]].toarray()).ravel().astype(np.float32)
        return scores

    def item_list(self) -> list[str]:
        return self.items


class ScoreBlendModel(B2Model):
    """Weighted blend of multiple retrieval model scores."""

    def __init__(
        self,
        *,
        model_id: str = "B2_SCORE_BLEND",
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        self.weights = weights or {}
        self.models: dict[str, B2Model] = {}
        self.items: list[str] = []

    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "ScoreBlendModel":
        self.items = list(item_list)
        self.models = {}
        for key in self.weights:
            if key == "history":
                self.models[key] = HistoryRecallModel().fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
            elif key == "popularity":
                self.models[key] = PopularityModel().fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
            elif key == "cooccurrence":
                self.models[key] = ItemItemCooccurrenceModel().fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
            elif key == "last_transition":
                self.models[key] = LastItemTransitionModel().fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)
        self.is_fitted = True
        return self

    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        if not self.models:
            return np.zeros((len(user_ids), len(self.items)), dtype=np.float32)
        total = np.zeros((len(user_ids), len(self.items)), dtype=np.float32)
        for key, model in self.models.items():
            total += self.weights.get(key, 0.0) * model.predict_scores(user_ids)
        return total

    def item_list(self) -> list[str]:
        return self.items


class CandidateRankerModel(B2Model):
    """Retrieve candidates then re-rank with a small classifier using user/item/history features."""

    def __init__(
        self,
        *,
        model_id: str = "B2_CANDIDATE_RANKER",
        retriever_weights: dict[str, float] | None = None,
        n_candidates: int = 100,
        n_negatives: int = 25,
        base: str = "logistic",
        use_item_features: bool = True,
        use_user_features: bool = True,
    ) -> None:
        super().__init__(model_id=model_id)
        self.retriever_weights = retriever_weights or {"history": 1.0, "cooccurrence": 1.0, "popularity": 0.5}
        self.n_candidates = n_candidates
        self.n_negatives = n_negatives
        self.base = base
        self.use_item_features = use_item_features
        self.use_user_features = use_user_features
        self.items: list[str] = []
        self.item2idx: dict[str, int] = {}
        self.user2idx: dict[str, int] = {}
        self.retriever: ScoreBlendModel | None = None
        self.scaler: StandardScaler | None = None
        self.clf: LogisticRegression | MLPClassifier | None = None
        self.user_df: pd.DataFrame | None = None
        self.item_df: pd.DataFrame | None = None
        self.user_features: np.ndarray | None = None
        self.item_features: np.ndarray | None = None
        self.train_seq: dict[str, list[str]] = {}
        self._cooc_model: ItemItemCooccurrenceModel | None = None

    def _build_feature_matrix(self, user_ids: list[str], item_ids: list[str], histories: list[list[str]]) -> np.ndarray:
        feats = []
        if self.use_user_features and self.user_features is not None:
            uidx = [self.user2idx.get(u, -1) for u in user_ids]
            uf = np.array([self.user_features[i] if i >= 0 else np.zeros(self.user_features.shape[1]) for i in uidx])
            feats.append(uf)
        if self.use_item_features and self.item_features is not None:
            iidx = [self.item2idx.get(i, -1) for i in item_ids]
            iff = np.array([self.item_features[i] if i >= 0 else np.zeros(self.item_features.shape[1]) for i in iidx])
            feats.append(iff)
        # History match features (cached cooc model, batched computation)
        if self._cooc_model is None:
            self._cooc_model = ItemItemCooccurrenceModel(model_id="_cooc").fit(list(self.train_seq.keys()), self.items, self.train_seq)
        unique_uids = list(dict.fromkeys(user_ids))
        uid_to_pos = {uid: i for i, uid in enumerate(unique_uids)}
        cooc_all = self._cooc_model.predict_scores(unique_uids)
        hist_scores = []
        cooc_scores = []
        for uid, iid in zip(user_ids, item_ids):
            seq = self.train_seq.get(uid, [])
            counts = Counter(seq)
            hist_scores.append(float(counts.get(iid, 0)))
            cooc_scores.append(float(cooc_all[uid_to_pos[uid], self.item2idx.get(iid, 0)]))
        feats.append(np.column_stack([hist_scores, cooc_scores]))
        return np.hstack(feats)

    def fit(self, user_ids: list[str], item_list: list[str], train_seq: dict[str, list[str]], train_targets: dict[str, str] | None = None, user_df: pd.DataFrame | None = None, item_df: pd.DataFrame | None = None) -> "CandidateRankerModel":
        self.items = list(item_list)
        self.item2idx = {iid: i for i, iid in enumerate(self.items)}
        self.train_seq = train_seq
        self.user_df = user_df
        self.item_df = item_df

        if user_df is not None and "uid" in user_df.columns:
            self.user2idx = {str(u): i for i, u in enumerate(user_df["uid"].astype(str))}
            feat_cols = [c for c in user_df.columns if c != "uid"]
            self.user_features = user_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        if item_df is not None and "iid" in item_df.columns:
            feat_cols = [c for c in item_df.columns if c != "iid"]
            self.item_features = item_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

        # Build retriever
        self.retriever = ScoreBlendModel(model_id="_retriever", weights=self.retriever_weights).fit(user_ids, item_list, train_seq, train_targets, user_df, item_df)

        # Build training pairs (positive target + sampled negatives from candidates)
        if train_targets is None:
            # Cannot train ranker without targets; fall back to retriever
            self.is_fitted = True
            return self

        train_uids_with_target = [u for u in user_ids if u in train_targets]
        candidates_per_user: dict[str, list[str]] = {}
        all_scores = self.retriever.predict_scores(train_uids_with_target)
        for uid, scores in zip(train_uids_with_target, all_scores):
            top_idx = np.argsort(-scores)[:self.n_candidates]
            candidates_per_user[uid] = [self.items[i] for i in top_idx]

        X_rows, y_rows, pair_users, pair_items = [], [], [], []
        for uid in train_uids_with_target:
            target = train_targets[uid]
            cands = candidates_per_user.get(uid, [])
            if target not in cands:
                cands = [target] + cands[:self.n_candidates - 1]
            positives = [target]
            negs = [c for c in cands if c != target][:self.n_negatives]
            for iid in positives + negs:
                pair_users.append(uid)
                pair_items.append(iid)
                y_rows.append(1 if iid in positives else 0)

        if len(set(y_rows)) < 2:
            self.is_fitted = True
            return self

        X = self._build_feature_matrix(pair_users, pair_items, [self.train_seq.get(u, []) for u in pair_users])
        y = np.array(y_rows, dtype=np.int64)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        if self.base == "mlp":
            self.clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=150, early_stopping=True, random_state=2026)
        else:
            self.clf = LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=1, random_state=2026)
        self.clf.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict_scores(self, user_ids: list[str]) -> np.ndarray:
        if self.retriever is None:
            return np.zeros((len(user_ids), len(self.items)), dtype=np.float32)
        retriever_scores = self.retriever.predict_scores(user_ids)
        if self.clf is None or self.scaler is None:
            return retriever_scores
        scores = np.zeros((len(user_ids), len(self.items)), dtype=np.float32)
        pair_users: list[str] = []
        pair_items: list[str] = []
        pair_positions: list[tuple[int, int]] = []
        for i, uid in enumerate(user_ids):
            top_idx = np.argsort(-retriever_scores[i])[:self.n_candidates]
            for j in top_idx:
                pair_users.append(uid)
                pair_items.append(self.items[j])
                pair_positions.append((i, j))
        if pair_users:
            X = self._build_feature_matrix(pair_users, pair_items, [self.train_seq.get(u, []) for u in pair_users])
            Xs = self.scaler.transform(X)
            prob = self.clf.predict_proba(Xs)[:, 1]
            for (i, j), p in zip(pair_positions, prob):
                scores[i, j] = p
        return scores

    def item_list(self) -> list[str]:
        return self.items


def instantiate_model(
    *,
    model_id: str,
    model_family: str,
    **kwargs: Any,
) -> B2Model:
    if model_family == "popularity":
        return PopularityModel(model_id=model_id)
    if model_family == "history_recall":
        return HistoryRecallModel(model_id=model_id)
    if model_family == "item_item_cooccurrence":
        return ItemItemCooccurrenceModel(model_id=model_id, window=kwargs.get("window", 5))
    if model_family == "last_item_transition":
        return LastItemTransitionModel(model_id=model_id)
    if model_family == "score_blend":
        return ScoreBlendModel(model_id=model_id, weights=kwargs.get("weights", {"history": 1.0, "popularity": 0.5}))
    if model_family == "candidate_ranker":
        return CandidateRankerModel(
            model_id=model_id,
            retriever_weights=kwargs.get("retriever_weights", {"history": 1.0, "cooccurrence": 1.0, "popularity": 0.5}),
            n_candidates=kwargs.get("n_candidates", 100),
            n_negatives=kwargs.get("n_negatives", 25),
            base=kwargs.get("base", "logistic"),
        )
    raise ValueError(f"unknown model_family: {model_family}")
