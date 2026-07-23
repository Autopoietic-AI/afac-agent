# -*- coding: utf-8 -*-
"""AFAC v2.0 recommendation operator space.

Pipeline stages R0-R8 with two build modes.  Every retriever emits bounded
per-user candidate dicts into :class:`SparseCandidateTable`; no operator ever
materializes a dense user×item matrix (see ``afac_agent.v2.memory_safe``).

R3 ranking note: the requested backend is a true candidate-level LambdaRank
(lightgbm).  lightgbm is NOT installed in this environment, so the actual
backend is a sklearn GradientBoostingClassifier trained with binary logistic
loss ONLY on candidate-table rows (never the full item catalog).  The gap is
recorded honestly in ``CandidateTableRanker.BACKEND_INFO``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import GradientBoostingClassifier

from afac_agent.research.event_store import stable_hash
from afac_agent.v2.memory_safe import SEED, SparseCandidateTable

# -- pipeline stages -----------------------------------------------------------
R0_DATA_VIEWS = "R0_DATA_VIEWS"
R1_MULTI_SOURCE_RETRIEVAL = "R1_MULTI_SOURCE_RETRIEVAL"
R2_CANDIDATE_UNION = "R2_CANDIDATE_UNION"
R3_BASE_LAMBDARANK = "R3_BASE_LAMBDARANK"
R4_STABLE_TOP10_ANCHOR = "R4_STABLE_TOP10_ANCHOR"
R5_BUCKET_SPECIALIST = "R5_BUCKET_SPECIALIST"
R6_PROTECTED_RESIDUAL_RERANK = "R6_PROTECTED_RESIDUAL_RERANK"
R7_BOUNDARY_ADMISSION = "R7_BOUNDARY_ADMISSION"
R8_MULTI_EXPERT_ENSEMBLE = "R8_MULTI_EXPERT_ENSEMBLE"

PIPELINE_STAGES = (
    R0_DATA_VIEWS,
    R1_MULTI_SOURCE_RETRIEVAL,
    R2_CANDIDATE_UNION,
    R3_BASE_LAMBDARANK,
    R4_STABLE_TOP10_ANCHOR,
    R5_BUCKET_SPECIALIST,
    R6_PROTECTED_RESIDUAL_RERANK,
    R7_BOUNDARY_ADMISSION,
    R8_MULTI_EXPERT_ENSEMBLE,
)

# -- modes -----------------------------------------------------------------------
GREENFIELD_BUILD_MODE = "greenfield_build"
CHAMPION_PRESERVING_REFINEMENT_MODE = "champion_preserving_refinement"


def _audit(
    *,
    changed_count: int = 0,
    kept_slots: list[int] | None = None,
    admissions: list[dict[str, Any]] | None = None,
    violations: list[str] | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "changed_count": changed_count,
        "kept_slots": kept_slots or [],
        "admissions": admissions or [],
        "violations": violations or [],
        "fallback_used": fallback_used,
    }


# =================================================================================
# R1 — multi-source retrieval
# =================================================================================


def _build_bigram(train_seq: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for seq in train_seq.values():
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a][b] += 1
    return {a: {b: float(c) for b, c in nxt.items()} for a, nxt in counts.items()}


def _build_cooc(train_seq: dict[str, list[str]], window: int = 5) -> dict[str, dict[str, float]]:
    """Symmetric windowed item-item co-occurrence over deduped sequences."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for seq in train_seq.values():
        dedup = list(dict.fromkeys(seq))
        for i, a in enumerate(dedup):
            for b in dedup[i + 1 : i + 1 + window]:
                counts[a][b] += 1
                counts[b][a] += 1
    return {a: {b: float(c) for b, c in nbr.items()} for a, nbr in counts.items()}


def _row_normalize(graph: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for a, nbr in graph.items():
        total = sum(nbr.values()) or 1.0
        out[a] = {b: w / total for b, w in nbr.items()}
    return out


def _truncate_neighbors(graph: dict[str, dict[str, float]], top_m: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for a, nbr in graph.items():
        best = sorted(nbr.items(), key=lambda kv: (-kv[1], kv[0]))[:top_m]
        out[a] = dict(best)
    return out


class BaseRetriever:
    """Common retriever interface (R1).

    ``fit`` stores shared statistics; ``retrieve`` emits a bounded
    :class:`SparseCandidateTable` — at most ``max_per_user`` items per user,
    never a dense full-catalog score row.
    """

    source_id = "base"
    stage = R1_MULTI_SOURCE_RETRIEVAL
    degraded = False

    def __init__(self) -> None:
        self.train_seq: dict[str, list[str]] = {}
        self.targets: dict[str, str] = {}
        self.pop: dict[str, float] = {}
        self.pop_sorted: list[str] = []
        self.user_df: pd.DataFrame | None = None
        self.item_df: pd.DataFrame | None = None
        self.is_fitted = False

    def fit(
        self,
        train_seq: dict[str, list[str]],
        train_targets: dict[str, str] | None = None,
        user_df: pd.DataFrame | None = None,
        item_df: pd.DataFrame | None = None,
    ) -> "BaseRetriever":
        self.train_seq = {str(u): [str(i) for i in s] for u, s in (train_seq or {}).items()}
        self.targets = {str(u): str(t) for u, t in (train_targets or {}).items()}
        self.user_df = user_df
        self.item_df = item_df
        counts: Counter = Counter()
        for seq in self.train_seq.values():
            counts.update(seq)
        counts.update(self.targets.values())
        self.pop = {iid: float(c) for iid, c in counts.items()}
        self.pop_sorted = [iid for iid, _ in sorted(self.pop.items(), key=lambda kv: (-kv[1], kv[0]))]
        self._fit()
        self.is_fitted = True
        return self

    def _fit(self) -> None:
        return None

    def _scores_for(self, user_id: str) -> dict[str, float]:
        raise NotImplementedError

    def retrieve(self, user_ids: Iterable[str], max_per_user: int = 200) -> SparseCandidateTable:
        table = SparseCandidateTable(
            source_id=self.source_id,
            max_per_user=max_per_user,
            user_ids=[str(u) for u in user_ids],
        )
        for uid in table.user_ids:
            scores = self._scores_for(uid)
            if scores:
                table.add_scores(uid, scores, self.source_id)
        return table


class PopularityRetriever(BaseRetriever):
    """Global item popularity (history + observed targets)."""

    source_id = "popularity"

    def _scores_for(self, user_id: str) -> dict[str, float]:
        return {iid: self.pop[iid] for iid in self.pop_sorted}


class HistoryRetriever(BaseRetriever):
    """Score items by their frequency in the user's own history."""

    source_id = "history"

    def _scores_for(self, user_id: str) -> dict[str, float]:
        counts = Counter(self.train_seq.get(user_id, []))
        return {iid: float(c) for iid, c in counts.items()}


class RepeatRetriever(BaseRetriever):
    """History items with count >= 2 boosted over one-off items."""

    source_id = "repeat"
    repeat_boost = 2.0

    def _scores_for(self, user_id: str) -> dict[str, float]:
        counts = Counter(self.train_seq.get(user_id, []))
        return {
            iid: float(c) * (self.repeat_boost if c >= 2 else 0.5)
            for iid, c in counts.items()
        }


class LastTransitionRetriever(BaseRetriever):
    """Bigram transitions from the user's last history item."""

    source_id = "last_transition"

    def _fit(self) -> None:
        self._bigram = _row_normalize(_build_bigram(self.train_seq))

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        if not seq:
            return {}
        return dict(self._bigram.get(seq[-1], {}))


class PairTransitionRetriever(BaseRetriever):
    """Sum of bigram transition scores from ALL history items."""

    source_id = "pair_transition"

    def _fit(self) -> None:
        self._bigram = _row_normalize(_build_bigram(self.train_seq))

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        scores: dict[str, float] = defaultdict(float)
        for iid in dict.fromkeys(seq):
            for nxt, w in self._bigram.get(iid, {}).items():
                scores[nxt] += w
        return dict(scores)


class HistToTargetRetriever(BaseRetriever):
    """History item -> observed train targets co-occurrence."""

    source_id = "hist_to_target"

    def _fit(self) -> None:
        counts: dict[str, Counter] = defaultdict(Counter)
        for uid, target in self.targets.items():
            for hist in dict.fromkeys(self.train_seq.get(uid, [])):
                counts[hist][target] += 1
        self._h2t = {h: {t: float(c) for t, c in nxt.items()} for h, nxt in counts.items()}

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        scores: dict[str, float] = defaultdict(float)
        for iid in dict.fromkeys(seq):
            for tgt, w in self._h2t.get(iid, {}).items():
                scores[tgt] += w
        return dict(scores)


class ItemCF1HopRetriever(BaseRetriever):
    """Item-item co-occurrence (row-normalized), aggregated over history."""

    source_id = "itemcf_1hop"

    def __init__(self, window: int = 5) -> None:
        super().__init__()
        self.window = window

    def _fit(self) -> None:
        self._cooc = _row_normalize(_build_cooc(self.train_seq, self.window))

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        scores: dict[str, float] = defaultdict(float)
        for iid in dict.fromkeys(seq):
            for nbr, w in self._cooc.get(iid, {}).items():
                scores[nbr] += w
        return dict(scores)


class ItemCF2HopRetriever(BaseRetriever):
    """Two-hop over the sparse item-item graph, each node truncated to top-M."""

    source_id = "itemcf_2hop"

    def __init__(self, window: int = 5, top_m: int = 64) -> None:
        super().__init__()
        self.window = window
        self.top_m = top_m

    def _fit(self) -> None:
        one_hop = _truncate_neighbors(_build_cooc(self.train_seq, self.window), self.top_m)
        self._one_hop = _row_normalize(one_hop)

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        scores: dict[str, float] = defaultdict(float)
        for iid in dict.fromkeys(seq):
            for mid, w1 in self._one_hop.get(iid, {}).items():
                for nbr, w2 in self._one_hop.get(mid, {}).items():
                    scores[nbr] += w1 * w2
        return dict(scores)


class Item2VecRetriever(BaseRetriever):
    """PPMI + scipy svds low-rank (k=32) item embeddings.

    degraded=True: this is a small-k truncated SVD approximation of item2vec,
    not a trained skip-gram; and when the catalog is too small for svds it
    degrades further to row-normalized co-occurrence rows as embeddings.
    Candidate scoring is bounded to the co-occurrence neighborhood pool of the
    user's history items — never a dense full-catalog pass.
    """

    source_id = "item2vec"
    degraded = True

    def __init__(self, k: int = 32, window: int = 3, pool_per_hist: int = 32) -> None:
        super().__init__()
        self.k = k
        self.window = window
        self.pool_per_hist = pool_per_hist

    def _fit(self) -> None:
        cooc = _build_cooc(self.train_seq, self.window)
        self._cooc = cooc
        items = sorted(cooc.keys() | {b for nbr in cooc.values() for b in nbr})
        n = len(items)
        self._items = items
        self._item2idx = {iid: i for i, iid in enumerate(items)}
        self._vectors: dict[str, np.ndarray] = {}
        if n < 4:
            for iid in items:
                row = cooc.get(iid, {})
                total = sum(row.values()) or 1.0
                self._vectors[iid] = np.array([row.get(b, 0.0) / total for b in items], dtype=np.float64)
            return
        rows, cols, data = [], [], []
        for a, nbr in cooc.items():
            for b, w in nbr.items():
                rows.append(self._item2idx[a])
                cols.append(self._item2idx[b])
                data.append(w)
        mat = csr_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(n, n),
        )
        total = mat.sum() or 1.0
        row_mass = np.asarray(mat.sum(axis=1)).ravel() / total
        col_mass = np.asarray(mat.sum(axis=0)).ravel() / total
        coo = mat.tocoo()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((coo.data / total) / np.maximum(row_mass[coo.row] * col_mass[coo.col], 1e-12))
        ppmi = csr_matrix(
            (np.maximum(pmi, 0.0), (coo.row, coo.col)), shape=(n, n)
        )
        k = min(self.k, n - 3)
        if k >= 2:
            v0 = np.full(n, 1.0 / np.sqrt(n))  # deterministic ARPACK start
            u, s, _vt = svds(ppmi, k=k, v0=v0)
            emb = u * s
        else:
            emb = ppmi.toarray()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        for i, iid in enumerate(items):
            self._vectors[iid] = emb[i]

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        hist = [iid for iid in dict.fromkeys(seq) if iid in self._vectors]
        if not hist:
            return {}
        # bounded candidate pool: top co-occurrence neighbors of history items
        pool: set[str] = set()
        for iid in hist:
            nbr = sorted(
                self._cooc.get(iid, {}).items(), key=lambda kv: (-kv[1], kv[0])
            )[: self.pool_per_hist]
            pool.update(b for b, _ in nbr)
        pool.difference_update(hist)
        if not pool:
            return {}
        hist_mat = np.stack([self._vectors[i] for i in hist])
        centroid = hist_mat.mean(axis=0)
        norm = np.linalg.norm(centroid) or 1.0
        centroid = centroid / norm
        scores: dict[str, float] = {}
        for iid in sorted(pool):
            vec = self._vectors.get(iid)
            if vec is None:
                continue
            scores[iid] = float(np.dot(centroid, vec))
        return scores


class RandomWalkRetriever(BaseRetriever):
    """S-walk: few-step random walk with restart from history items.

    Deterministic power iteration over the sparse item graph (no sampling):
    ``v <- alpha * P^T v + (1 - alpha) * restart``.
    """

    source_id = "swalk"

    def __init__(self, steps: int = 3, alpha: float = 0.8, window: int = 5) -> None:
        super().__init__()
        self.steps = steps
        self.alpha = alpha
        self.window = window

    def _fit(self) -> None:
        self._graph = _row_normalize(_build_cooc(self.train_seq, self.window))

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        hist = [iid for iid in dict.fromkeys(seq) if iid in self._graph]
        if not hist:
            return {}
        restart = {iid: 1.0 / len(hist) for iid in hist}
        v = dict(restart)
        for _ in range(self.steps):
            nxt: dict[str, float] = defaultdict(float)
            for a, prob in v.items():
                for b, w in self._graph.get(a, {}).items():
                    nxt[b] += prob * w
            v = {
                iid: self.alpha * nxt.get(iid, 0.0) + (1.0 - self.alpha) * restart.get(iid, 0.0)
                for iid in set(nxt) | set(restart)
            }
        return v


class UserAttributeRetriever(BaseRetriever):
    """Items/targets consumed by users sharing the same attribute profile."""

    source_id = "user_attr"

    def _fit(self) -> None:
        self._user_key: dict[str, tuple[str, ...]] = {}
        if self.user_df is not None and "uid" in self.user_df.columns:
            cols = [c for c in self.user_df.columns if c != "uid"]
            for row in self.user_df.itertuples(index=False):
                uid = str(getattr(row, "uid"))
                self._user_key[uid] = tuple(str(getattr(row, c)) for c in cols)
        self._group_items: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self._group_targets: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for uid, seq in self.train_seq.items():
            key = self._user_key.get(uid)
            if key is None:
                continue
            self._group_items[key].update(seq)
            if uid in self.targets:
                self._group_targets[key][self.targets[uid]] += 1

    def _scores_for(self, user_id: str) -> dict[str, float]:
        key = self._user_key.get(user_id)
        if key is None:
            return {}
        scores: dict[str, float] = defaultdict(float)
        for iid, c in self._group_items.get(key, {}).items():
            scores[iid] += float(c)
        for iid, c in self._group_targets.get(key, {}).items():
            scores[iid] += 2.0 * float(c)
        return dict(scores)


class ItemAttributeRetriever(BaseRetriever):
    """Items sharing attribute values with the user's history items."""

    source_id = "item_attr"

    def _fit(self) -> None:
        # inverted index over individual (column, value) attribute pairs
        self._attr_items: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._item_attrs: dict[str, list[tuple[str, str]]] = {}
        if self.item_df is not None and "iid" in self.item_df.columns:
            cols = [c for c in self.item_df.columns if c != "iid"]
            for row in self.item_df.itertuples(index=False):
                iid = str(getattr(row, "iid"))
                pairs = [(c, str(getattr(row, c))) for c in cols]
                self._item_attrs[iid] = pairs
                for pair in pairs:
                    self._attr_items[pair].append(iid)

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        scores: dict[str, float] = defaultdict(float)
        hist_counts = Counter(seq)
        for iid, cnt in hist_counts.items():
            for pair in self._item_attrs.get(iid, []):
                for other in self._attr_items.get(pair, []):
                    if other != iid:
                        scores[other] += float(cnt)
        return dict(scores)


class SequenceRecallRetriever(BaseRetriever):
    """Suffix-match sequence similarity: items that follow matching suffixes."""

    source_id = "sequence_recall"

    def _fit(self) -> None:
        self._suffix_next: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self._suffix_target: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for uid, seq in self.train_seq.items():
            for i in range(len(seq) - 1):
                for width in (1, 2):
                    if i + 1 - width >= 0:
                        suffix = tuple(seq[i + 1 - width : i + 1])
                        self._suffix_next[suffix][seq[i + 1]] += 1
            if uid in self.targets and seq:
                for width in (1, 2):
                    if len(seq) >= width:
                        self._suffix_target[tuple(seq[-width:])][self.targets[uid]] += 1

    def _scores_for(self, user_id: str) -> dict[str, float]:
        seq = self.train_seq.get(user_id, [])
        if not seq:
            return {}
        scores: dict[str, float] = defaultdict(float)
        for width in (1, 2):
            if len(seq) >= width:
                suffix = tuple(seq[-width:])
                for iid, c in self._suffix_next.get(suffix, {}).items():
                    scores[iid] += float(c)
                for iid, c in self._suffix_target.get(suffix, {}).items():
                    scores[iid] += 2.0 * float(c)
        return dict(scores)


class NovelRecallRetriever(BaseRetriever):
    """Popular items NOT present in the user's history."""

    source_id = "novel_recall"

    def _scores_for(self, user_id: str) -> dict[str, float]:
        hist = set(self.train_seq.get(user_id, []))
        return {iid: self.pop[iid] for iid in self.pop_sorted if iid not in hist}


class ColdStartRetriever(BaseRetriever):
    """Popularity + attribute fallback; only emits for empty-history users."""

    source_id = "cold_start"

    def _fit(self) -> None:
        self._attr = UserAttributeRetriever()
        self._attr.fit(self.train_seq, self.targets, self.user_df, self.item_df)

    def _scores_for(self, user_id: str) -> dict[str, float]:
        if self.train_seq.get(user_id):
            return {}
        scores: dict[str, float] = {iid: self.pop[iid] for iid in self.pop_sorted}
        for iid, w in self._attr._scores_for(user_id).items():
            scores[iid] = scores.get(iid, 0.0) + w
        return scores


ALL_RETRIEVER_CLASSES = (
    PopularityRetriever,
    HistoryRetriever,
    RepeatRetriever,
    LastTransitionRetriever,
    PairTransitionRetriever,
    HistToTargetRetriever,
    ItemCF1HopRetriever,
    ItemCF2HopRetriever,
    Item2VecRetriever,
    RandomWalkRetriever,
    UserAttributeRetriever,
    ItemAttributeRetriever,
    SequenceRecallRetriever,
    NovelRecallRetriever,
    ColdStartRetriever,
)


# =================================================================================
# R2 — candidate union
# =================================================================================


class CandidateUnion:
    """Merge retriever tables into one sparse table with union features.

    Per candidate: per-source score/rank, source_count, RRF (k=60).
    """

    stage = R2_CANDIDATE_UNION

    def __init__(self, rrf_k: int = 60, max_per_user: int = 200) -> None:
        self.rrf_k = rrf_k
        self.max_per_user = max_per_user

    def merge(self, tables: Iterable[SparseCandidateTable]) -> SparseCandidateTable:
        return SparseCandidateTable.union(
            *list(tables), rrf_k=self.rrf_k, max_per_user=self.max_per_user
        )

    def feature_rows(self, table: SparseCandidateTable) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for uid in table.user_ids:
            for entry in table.entries_for(uid):
                rows.append(
                    {
                        "user_id": uid,
                        "item_id": entry.item_id,
                        "sources": {k: dict(v) for k, v in entry.sources.items()},
                        "source_count": entry.source_count,
                        "rrf": entry.rrf,
                        "rank": entry.rank,
                    }
                )
        return rows


# =================================================================================
# R3 — candidate-level LambdaRank (honest sklearn backend)
# =================================================================================

BASE_FEATURE_NAMES = (
    "rrf",
    "source_count",
    "best_source_rank",
    "mean_source_score",
    "max_source_score",
    "popularity",
    "history_count",
    "repeat_flag",
    "in_history",
    "novel_flag",
    "recency",
    "transition_score",
    "seq_len_raw",
    "seq_len_dedup",
    "long_tail_flag",
    "seq_bucket_id",
    "propensity",
    "rank_band_id",
    "hist_attr_overlap",
    "attr_match",
)

RANK_BANDS = (3, 10, 30, 100)


@dataclass
class RankerContext:
    train_seq: dict[str, list[str]] = field(default_factory=dict)
    train_targets: dict[str, str] = field(default_factory=dict)
    user_df: pd.DataFrame | None = None
    item_df: pd.DataFrame | None = None
    popularity: dict[str, float] = field(default_factory=dict)
    propensity: dict[str, float] = field(default_factory=dict)


class CandidateTableRanker:
    """R3 — true candidate-level ranking: trains ONLY on candidate-table rows.

    The requested backend (lightgbm LambdaRank) is unavailable in this
    environment; the actual backend is sklearn GradientBoostingClassifier
    with binary logistic loss over candidate rows — positives are targets
    present in the candidates, hard negatives are non-target candidates
    (union ranks 8-30, high-popularity wrong items, same-attribute wrong
    items), capped per user.  ``BACKEND_INFO`` records the gap and the
    resulting availability bias.
    """

    stage = R3_BASE_LAMBDARANK
    BACKEND_INFO = {
        "requested": "lightgbm_lambdarank",
        "actual": "sklearn_gbdt_binary",
        "missing_dependency": ["lightgbm"],
        "availability_bias": True,
    }

    def __init__(
        self,
        n_negatives: int = 25,
        hard_neg_rank_lo: int = 8,
        hard_neg_rank_hi: int = 30,
        n_estimators: int = 64,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        seed: int = SEED,
    ) -> None:
        self.n_negatives = n_negatives
        self.hard_neg_rank_lo = hard_neg_rank_lo
        self.hard_neg_rank_hi = hard_neg_rank_hi
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.ctx = RankerContext()
        self.table_: SparseCandidateTable | None = None
        self.clf: GradientBoostingClassifier | None = None
        self.feature_names_: list[str] = list(BASE_FEATURE_NAMES)
        self.is_fitted = False
        self.fallback_used = False
        # derived context
        self._bigram: dict[str, dict[str, float]] = {}
        self._long_tail: set[str] = set()
        self._user_num: dict[str, list[float]] = {}
        self._item_num: dict[str, list[float]] = {}
        self._user_cat: dict[str, dict[str, str]] = {}
        self._item_cat: dict[str, dict[str, str]] = {}
        self._hist_attr_values: dict[str, set[str]] = {}
        self._source_max: dict[str, float] = {}

    # -- context -------------------------------------------------------------
    def set_context(
        self,
        *,
        train_seq: dict[str, list[str]],
        train_targets: dict[str, str] | None = None,
        user_df: pd.DataFrame | None = None,
        item_df: pd.DataFrame | None = None,
        popularity: dict[str, float] | None = None,
        propensity: dict[str, float] | None = None,
    ) -> "CandidateTableRanker":
        self.ctx = RankerContext(
            train_seq={str(u): [str(i) for i in s] for u, s in (train_seq or {}).items()},
            train_targets={str(u): str(t) for u, t in (train_targets or {}).items()},
            user_df=user_df,
            item_df=item_df,
            popularity=dict(popularity or {}),
            propensity={str(k): float(v) for k, v in (propensity or {}).items()},
        )
        if not self.ctx.popularity:
            counts: Counter = Counter()
            for seq in self.ctx.train_seq.values():
                counts.update(seq)
            counts.update(self.ctx.train_targets.values())
            self.ctx.popularity = {iid: float(c) for iid, c in counts.items()}
        self._bigram = _build_bigram(self.ctx.train_seq)
        ordered = sorted(self.ctx.popularity.items(), key=lambda kv: (-kv[1], kv[0]))
        half = len(ordered) // 2
        self._long_tail = {iid for iid, _ in ordered[half:]}
        self._user_num, self._user_cat = self._split_frame(user_df, "uid")
        self._item_num, self._item_cat = self._split_frame(item_df, "iid")
        self._hist_attr_values = {}
        for uid, seq in self.ctx.train_seq.items():
            values: set[str] = set()
            for iid in dict.fromkeys(seq):
                values.update(self._item_cat.get(iid, {}).values())
            self._hist_attr_values[uid] = values
        return self

    @staticmethod
    def _split_frame(
        df: pd.DataFrame | None, id_col: str
    ) -> tuple[dict[str, list[float]], dict[str, dict[str, str]]]:
        num: dict[str, list[float]] = {}
        cat: dict[str, dict[str, str]] = {}
        if df is None or id_col not in df.columns:
            return num, cat
        cols = [c for c in df.columns if c != id_col]
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        is_num_col = {c: bool(numeric[c].notna().any()) for c in cols}
        numeric = numeric.fillna(0.0)
        for pos, row in enumerate(df.itertuples(index=False)):
            key = str(getattr(row, id_col))
            num[key] = [
                float(numeric.iloc[pos][c]) if is_num_col[c] else 0.0 for c in cols
            ]
            cat[key] = {c: str(getattr(row, c)) for c in cols if not is_num_col[c]}
        return num, cat

    # -- featurization ----------------------------------------------------------
    def _rank_band(self, rank: int) -> float:
        for band, edge in enumerate(RANK_BANDS):
            if rank <= edge:
                return float(band)
        return float(len(RANK_BANDS))

    def _featurize(self, user_id: str, item_id: str, row: dict[str, Any]) -> dict[str, float]:
        seq = self.ctx.train_seq.get(user_id, [])
        sources: dict[str, dict[str, float]] = row.get("sources", {})
        norm_scores = []
        for src, info in sources.items():
            smax = self._source_max.get(src) or 1.0
            norm_scores.append(float(info.get("score", 0.0)) / smax)
        best_rank = min((int(info.get("rank", 10**6)) for info in sources.values()), default=10**6)
        cnt = seq.count(item_id)
        recency = 0.0
        if cnt:
            pos_from_end = len(seq) - 1 - max(i for i, x in enumerate(seq) if x == item_id)
            recency = 1.0 / (1.0 + pos_from_end)
        transition = 0.0
        for hist in dict.fromkeys(seq):
            transition += self._bigram.get(hist, {}).get(item_id, 0.0)
        item_attr = set(self._item_cat.get(item_id, {}).values())
        hist_attr = self._hist_attr_values.get(user_id, set())
        overlap = (len(item_attr & hist_attr) / len(item_attr)) if item_attr else 0.0
        shared_cols = set(self._user_cat.get(user_id, {})) & set(self._item_cat.get(item_id, {}))
        attr_match = float(
            sum(
                1
                for c in shared_cols
                if self._user_cat[user_id][c] == self._item_cat[item_id][c]
            )
        )
        feats: dict[str, float] = {
            "rrf": float(row.get("rrf", 0.0)),
            "source_count": float(row.get("source_count", len(sources))),
            "best_source_rank": 1.0 / max(best_rank, 1),
            "mean_source_score": float(np.mean(norm_scores)) if norm_scores else 0.0,
            "max_source_score": float(max(norm_scores)) if norm_scores else 0.0,
            "popularity": float(np.log1p(self.ctx.popularity.get(item_id, 0.0))),
            "history_count": float(cnt),
            "repeat_flag": 1.0 if cnt >= 2 else 0.0,
            "in_history": 1.0 if cnt > 0 else 0.0,
            "novel_flag": 0.0 if cnt > 0 else 1.0,
            "recency": recency,
            "transition_score": float(np.log1p(transition)),
            "seq_len_raw": float(len(seq)),
            "seq_len_dedup": float(len(set(seq))),
            "long_tail_flag": 1.0 if item_id in self._long_tail else 0.0,
            "seq_bucket_id": float(min(len(seq), 4)),
            "propensity": float(row.get("propensity", self.ctx.propensity.get(item_id, 1.0))),
            "rank_band_id": self._rank_band(int(row.get("rank", 10**6))),
            "hist_attr_overlap": float(overlap),
            "attr_match": attr_match,
        }
        for i, v in enumerate(self._user_num.get(user_id, [])):
            feats[f"ufeat_{i}"] = float(v)
        for i, v in enumerate(self._item_num.get(item_id, [])):
            feats[f"ifeat_{i}"] = float(v)
        return feats

    def _make_row(self, user_id: str, entry: Any) -> dict[str, Any]:
        row = {
            "user_id": user_id,
            "item_id": entry.item_id,
            "sources": {k: dict(v) for k, v in entry.sources.items()},
            "source_count": entry.source_count,
            "rrf": entry.rrf,
            "rank": entry.rank,
        }
        row["features"] = self._featurize(user_id, entry.item_id, row)
        return row

    def _prepare_table(self, table: SparseCandidateTable) -> None:
        self.table_ = table
        self._source_max = {}
        for uid in table.user_ids:
            for entry in table.entries_for(uid):
                for src, info in entry.sources.items():
                    cur = self._source_max.get(src, 0.0)
                    self._source_max[src] = max(cur, float(info.get("score", 0.0)))

    # -- training rows -------------------------------------------------------------
    def build_training_rows(
        self, table: SparseCandidateTable, user_ids: Iterable[str] | None = None
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Positives = targets present in candidates; hard negatives capped."""
        self._prepare_table(table)
        rows: list[dict[str, Any]] = []
        labels: list[int] = []
        uids = [str(u) for u in (user_ids or table.user_ids)]
        for uid in uids:
            target = self.ctx.train_targets.get(uid)
            if target is None:
                continue
            entries = table.entries_for(uid)
            by_item = {e.item_id: e for e in entries}
            if target not in by_item:
                continue
            rows.append(self._make_row(uid, by_item[target]))
            labels.append(1)
            # hard negatives: union ranks 8-30, then high-popularity wrong
            # items, then same-attribute wrong items
            picked: list[str] = []
            for e in entries:
                if e.item_id == target:
                    continue
                if self.hard_neg_rank_lo <= e.rank <= self.hard_neg_rank_hi:
                    picked.append(e.item_id)
            by_pop = sorted(
                (e for e in entries if e.item_id != target and e.item_id not in picked),
                key=lambda e: (-self.ctx.popularity.get(e.item_id, 0.0), e.item_id),
            )
            picked.extend(e.item_id for e in by_pop[:5])
            target_attrs = set(self._item_cat.get(target, {}).values())
            same_attr = [
                e.item_id
                for e in entries
                if e.item_id != target
                and e.item_id not in picked
                and target_attrs & set(self._item_cat.get(e.item_id, {}).values())
            ]
            picked.extend(same_attr[:5])
            for iid in picked[: self.n_negatives]:
                rows.append(self._make_row(uid, by_item[iid]))
                labels.append(0)
        return rows, labels

    def build_scoring_rows(
        self, table: SparseCandidateTable | None = None, user_ids: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        table = table or self.table_
        if table is None:
            return []
        if table is not self.table_:
            self._prepare_table(table)
        rows: list[dict[str, Any]] = []
        for uid in [str(u) for u in (user_ids or table.user_ids)]:
            for entry in table.entries_for(uid):
                rows.append(self._make_row(uid, entry))
        return rows

    # -- fit / score / rerank ---------------------------------------------------------
    def _vectorize(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.array(
            [[float(r["features"].get(name, 0.0)) for name in self.feature_names_] for r in rows],
            dtype=np.float64,
        )

    def fit(self, candidate_rows: list[dict[str, Any]], labels: list[int]) -> "CandidateTableRanker":
        names: set[str] = set()
        for r in candidate_rows:
            names.update(r.get("features", {}).keys())
        self.feature_names_ = [n for n in BASE_FEATURE_NAMES] + sorted(
            n for n in names if n not in BASE_FEATURE_NAMES
        )
        y = np.asarray(labels, dtype=np.int64)
        if len(candidate_rows) == 0 or len(set(y.tolist())) < 2:
            self.fallback_used = True
            self.is_fitted = True
            return self
        X = self._vectorize(candidate_rows)
        self.clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def score(self, candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fill blended score (model prob + rrf normalized per user)."""
        if not candidate_rows:
            return candidate_rows
        rrf_max: dict[str, float] = defaultdict(float)
        for r in candidate_rows:
            rrf_max[r["user_id"]] = max(rrf_max[r["user_id"]], float(r.get("rrf", 0.0)))
        probs: np.ndarray
        if self.clf is not None:
            probs = self.clf.predict_proba(self._vectorize(candidate_rows))[:, 1]
        else:
            probs = np.zeros(len(candidate_rows), dtype=np.float64)
        for r, p in zip(candidate_rows, probs):
            rrf_norm = float(r.get("rrf", 0.0)) / (rrf_max[r["user_id"]] or 1.0)
            r["model_score"] = float(p)
            r["score"] = 0.5 * float(p) + 0.5 * rrf_norm
        return candidate_rows

    def rerank(self, user_ids: Iterable[str], k: int = 10) -> list[list[str]]:
        rows = self.score(self.build_scoring_rows(user_ids=[str(u) for u in user_ids]))
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_user[r["user_id"]].append(r)
        out: list[list[str]] = []
        for uid in [str(u) for u in user_ids]:
            ranked = sorted(by_user.get(uid, []), key=lambda r: (-r["score"], r["item_id"]))
            out.append([r["item_id"] for r in ranked[:k]])
        return out


# =================================================================================
# R4 — stable top-10 anchor
# =================================================================================


@dataclass(frozen=True)
class StableTop10Anchor:
    """Immutable snapshot of a ranked top-10 list with a content hash."""

    user_id: str
    top10: tuple[str, ...]
    scores: tuple[float, ...] = ()
    stage: str = R4_STABLE_TOP10_ANCHOR
    anchor_hash: str = ""

    def __post_init__(self) -> None:
        if not self.anchor_hash:
            object.__setattr__(
                self,
                "anchor_hash",
                stable_hash(
                    {
                        "stage": self.stage,
                        "user_id": self.user_id,
                        "top10": list(self.top10),
                        "scores": [round(float(s), 12) for s in self.scores],
                    }
                ),
            )

    @classmethod
    def snapshot(
        cls, user_id: str, ranked_items: Iterable[str], scores: Iterable[float] | None = None
    ) -> "StableTop10Anchor":
        top10 = tuple(str(i) for i in list(ranked_items)[:10])
        score_tuple = tuple(float(s) for s in (scores or []))[:10]
        return cls(user_id=str(user_id), top10=top10, scores=score_tuple)


# =================================================================================
# R5 — bucket specialists
# =================================================================================

EXPERT_IDS = (
    "Len0",
    "Len1",
    "Len2",
    "ExactLen3",
    "Len4Plus",
    "Repeat",
    "Explore",
    "Novel",
    "History",
    "LongTail",
    "ColdUser",
)

DEFAULT_EXPERT_WEIGHTS: dict[str, dict[str, float]] = {
    "Len0": {"popularity": 2.0, "rrf": 1.0},
    "Len1": {"recency": 2.0, "transition_score": 2.0, "rrf": 1.0},
    "Len2": {"transition_score": 2.0, "recency": 1.5, "rrf": 1.0},
    "ExactLen3": {"transition_score": 2.0, "hist_attr_overlap": 1.5, "rrf": 1.0},
    "Len4Plus": {"source_count": 1.5, "rrf": 2.0, "hist_attr_overlap": 1.0},
    "Repeat": {"history_count": 2.0, "repeat_flag": 3.0, "rrf": 1.0},
    "Explore": {"novel_flag": 1.5, "source_count": 1.5, "rrf": 1.0},
    "Novel": {"novel_flag": 3.0, "popularity": 1.5, "rrf": 1.0},
    "History": {"in_history": 2.0, "history_count": 1.5, "rrf": 1.0},
    "LongTail": {"long_tail_flag": 2.0, "popularity": -0.5, "rrf": 1.0},
    "ColdUser": {"popularity": 2.0, "attr_match": 1.5, "rrf": 1.0},
}


class ExpertRouter:
    """R5 — assign a user to bucket-specialist experts from sequence features."""

    stage = R5_BUCKET_SPECIALIST
    EXPERT_IDS = EXPERT_IDS

    def __init__(self, expert_weights: dict[str, dict[str, float]] | None = None) -> None:
        self.expert_weights = expert_weights or {k: dict(v) for k, v in DEFAULT_EXPERT_WEIGHTS.items()}
        self.train_seq: dict[str, list[str]] = {}
        self.popularity: dict[str, float] = {}
        self._median_pop = 0.0

    def fit(
        self,
        train_seq: dict[str, list[str]],
        popularity: dict[str, float] | None = None,
    ) -> "ExpertRouter":
        self.train_seq = {str(u): [str(i) for i in s] for u, s in (train_seq or {}).items()}
        if popularity is None:
            counts: Counter = Counter()
            for seq in self.train_seq.values():
                counts.update(seq)
            popularity = {iid: float(c) for iid, c in counts.items()}
        self.popularity = dict(popularity)
        vals = sorted(self.popularity.values())
        self._median_pop = vals[len(vals) // 2] if vals else 0.0
        return self

    def route(self, user_id: str) -> list[str]:
        seq = self.train_seq.get(str(user_id), [])
        n_raw = len(seq)
        n_dedup = len(set(seq))
        experts: list[str] = []
        if n_raw == 0:
            experts.extend(["Len0", "ColdUser"])
            return experts
        if n_raw == 1:
            experts.append("Len1")
        elif n_raw == 2:
            experts.append("Len2")
        elif n_raw == 3:
            experts.append("ExactLen3")
        else:
            experts.append("Len4Plus")
        if n_dedup < n_raw:
            experts.append("Repeat")
        if n_dedup >= 4:
            experts.append("Explore")
        if n_dedup == n_raw and n_raw >= 2:
            experts.append("Novel")
        experts.append("History")
        if seq:
            mean_pop = sum(self.popularity.get(i, 0.0) for i in set(seq)) / n_dedup
            if mean_pop <= self._median_pop:
                experts.append("LongTail")
        return experts

    def reweight_rows(
        self, rows: list[dict[str, Any]], expert_ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        """Re-weight candidate features with the routed experts' weight dicts."""
        weights: dict[str, float] = defaultdict(float)
        for eid in expert_ids:
            for fname, w in self.expert_weights.get(eid, {}).items():
                weights[fname] += float(w)
        for row in rows:
            feats = row.get("features", {})
            row["expert_score"] = float(
                sum(w * float(feats.get(fname, 0.0)) for fname, w in weights.items())
            )
        return rows


# =================================================================================
# R6 — protected residual rerank
# =================================================================================


def topk_protection(
    parent_top10: list[str], proposed_top10: list[str], k: int
) -> tuple[list[str], dict[str, Any]]:
    """First ``k`` slots of the parent are immutable; the rest may rerank."""
    try:
        parent = [str(i) for i in parent_top10]
        k = max(0, min(int(k), len(parent)))
        protected = parent[:k]
        used = set(protected)
        fill = [str(i) for i in proposed_top10 if str(i) not in used]
        fill += [i for i in parent[k:] if i not in used and i not in fill]
        new = protected + fill[: len(parent) - k]
        changed = sum(1 for a, b in zip(new, parent) if a != b)
        return new, _audit(changed_count=changed, kept_slots=list(range(k)))
    except Exception as exc:  # pragma: no cover - defensive
        return fallback_keep_parent(parent_top10, f"topk_protection: {exc}")


def history_slot_protection(
    parent_top10: list[str], proposed_top10: list[str], history_items: Iterable[str]
) -> tuple[list[str], dict[str, Any]]:
    """History items keep their parent slots; other slots may rerank."""
    try:
        parent = [str(i) for i in parent_top10]
        history = {str(i) for i in history_items}
        protected_slots = {i: item for i, item in enumerate(parent) if item in history}
        new = list(parent)
        used = set(protected_slots.values())
        fill = [str(i) for i in proposed_top10 if str(i) not in used]
        free_slots = [i for i in range(len(parent)) if i not in protected_slots]
        for slot, item in zip(free_slots, fill):
            new[slot] = item
            used.add(item)
        changed = sum(1 for a, b in zip(new, parent) if a != b)
        return new, _audit(changed_count=changed, kept_slots=sorted(protected_slots))
    except Exception as exc:  # pragma: no cover - defensive
        return fallback_keep_parent(parent_top10, f"history_slot_protection: {exc}")


def novel_only_rerank(
    parent_top10: list[str], proposed_top10: list[str], history_items: Iterable[str]
) -> tuple[list[str], dict[str, Any]]:
    """Only novel items may move, and only into slots not pinned by history items."""
    try:
        parent = [str(i) for i in parent_top10]
        history = {str(i) for i in history_items}
        new = list(parent)
        pinned = {i for i, item in enumerate(parent) if item in history}
        free_slots = [i for i in range(len(parent)) if i not in pinned]
        kept = {parent[i] for i in pinned}
        fill = [
            str(i)
            for i in proposed_top10
            if str(i) not in history and str(i) not in kept
        ]
        seen: set[str] = set()
        for slot in free_slots:
            while fill and fill[0] in seen:
                fill.pop(0)
            if fill:
                item = fill.pop(0)
                new[slot] = item
                seen.add(item)
        changed = sum(1 for a, b in zip(new, parent) if a != b)
        return new, _audit(changed_count=changed, kept_slots=sorted(pinned))
    except Exception as exc:  # pragma: no cover - defensive
        return fallback_keep_parent(parent_top10, f"novel_only_rerank: {exc}")


# =================================================================================
# R7 — boundary admission
# =================================================================================


def fallback_keep_parent(
    parent_top10: list[str], reason: str
) -> tuple[list[str], dict[str, Any]]:
    """Any constraint violation or exception: return the parent unchanged."""
    return list(parent_top10), _audit(
        changed_count=0,
        kept_slots=list(range(len(parent_top10))),
        violations=[reason],
        fallback_used=True,
    )


def position10_admission(
    parent_top10: list[str],
    external_candidate: str,
    margin: float,
    incumbent_score: float,
    external_score: float,
) -> tuple[list[str], dict[str, Any]]:
    """At most ONE external candidate may enter, only at position 10 (last slot),
    and only if ``external_score > incumbent_score + margin``."""
    try:
        parent = [str(i) for i in parent_top10]
        candidate = str(external_candidate)
        if not parent:
            return fallback_keep_parent(parent, "empty_parent")
        if candidate in parent:
            return fallback_keep_parent(parent, "candidate_already_present")
        if not np.isfinite(external_score) or not np.isfinite(incumbent_score):
            return fallback_keep_parent(parent, "non_finite_score")
        if external_score > incumbent_score + margin:
            new = parent[:-1] + [candidate]
            return new, _audit(
                changed_count=1,
                kept_slots=list(range(len(parent) - 1)),
                admissions=[
                    {
                        "candidate": candidate,
                        "position": len(parent),
                        "external_score": float(external_score),
                        "incumbent_score": float(incumbent_score),
                        "margin": float(margin),
                    }
                ],
            )
        return list(parent), _audit(
            changed_count=0, kept_slots=list(range(len(parent)))
        )
    except Exception as exc:  # pragma: no cover - defensive
        return fallback_keep_parent(parent_top10, f"position10_admission: {exc}")


# =================================================================================
# R8 — multi-expert ensemble
# =================================================================================


class MultiExpertEnsemble:
    """R8 — combine expert outputs under the protection constraints."""

    stage = R8_MULTI_EXPERT_ENSEMBLE

    def __init__(self, protect_k: int = 3, rrf_k: int = 60) -> None:
        self.protect_k = protect_k
        self.rrf_k = rrf_k

    def combine(
        self,
        parent_top10: list[str],
        expert_lists: dict[str, list[str]],
    ) -> tuple[list[str], dict[str, Any]]:
        try:
            parent = [str(i) for i in parent_top10]
            rrf: dict[str, float] = defaultdict(float)
            for items in expert_lists.values():
                for rank, iid in enumerate(items, start=1):
                    rrf[str(iid)] += 1.0 / (self.rrf_k + rank)
            merged = [iid for iid, _ in sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))]
            k = max(0, min(self.protect_k, len(parent)))
            protected = parent[:k]
            used = set(protected)
            fill = [i for i in merged if i not in used]
            fill += [i for i in parent[k:] if i not in used and i not in fill]
            new = protected + fill[: len(parent) - k]
            changed = sum(1 for a, b in zip(new, parent) if a != b)
            return new, _audit(changed_count=changed, kept_slots=list(range(k)))
        except Exception as exc:
            return fallback_keep_parent(parent_top10, f"multi_expert_ensemble: {exc}")
