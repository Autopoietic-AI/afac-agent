# -*- coding: utf-8 -*-
"""AFAC v2.0 memory-safe executor primitives.

The v1.6 retrievers in ``afac_agent.b2.models`` materialize dense
``(n_users, n_items)`` float arrays, which does not scale to the full B2
catalog.  The v2 stack must never build a full user-by-item dense matrix:
candidates live in :class:`SparseCandidateTable` (sparse per-user dicts),
scoring is chunked, and top-k extraction uses per-user heaps.

``FORBIDDEN_PATTERN`` records the banned default so reviews and the
preflight check can point at a single canonical statement.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np
import psutil
from scipy.sparse import csr_matrix

SEED = 2026

FORBIDDEN_PATTERN = (
    "default dense user×item float64 matrix is forbidden; use "
    "SparseCandidateTable, chunked scoring, float32, TopK streaming, "
    "memmap, batch candidate features"
)


def chunked(iterable: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield ``iterable`` in list batches of at most ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def topk_streaming(
    user_item_scores: Iterable[tuple[str, str, float]],
    k: int,
) -> dict[str, list[str]]:
    """Per-user top-k via heaps over a stream of (user, item, score).

    Never sorts the full item catalog: each user keeps a size-k min-heap.
    Ties break deterministically on the item id.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    heaps: dict[str, list[tuple[float, str]]] = {}
    for uid, iid, score in user_item_scores:
        entry = (float(score), str(iid))
        heap = heaps.setdefault(str(uid), [])
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return {
        uid: [iid for _, iid in sorted(heap, key=lambda t: (-t[0], t[1]))]
        for uid, heap in heaps.items()
    }


@dataclass
class CandidateEntry:
    """One (user, item) candidate inside a :class:`SparseCandidateTable`."""

    item_id: str
    score: float
    rank: int
    source: str
    sources: dict[str, dict[str, float]] = field(default_factory=dict)
    source_count: int = 1
    rrf: float = 0.0


class SparseCandidateTable:
    """Per-user sparse candidate storage without dense user×item materialization.

    Backed by ``dict[int, dict[int, CandidateEntry]]`` keyed by user/item
    vocab indices; a CSR view is assembled on demand and only ever covers
    the bounded candidate set actually stored.
    """

    def __init__(
        self,
        *,
        source_id: str = "unknown",
        max_per_user: int = 200,
        user_ids: Iterable[str] | None = None,
        item_ids: Iterable[str] | None = None,
    ) -> None:
        self.source_id = source_id
        self.max_per_user = max_per_user
        self.user_ids: list[str] = [str(u) for u in (user_ids or [])]
        self.item_ids: list[str] = [str(i) for i in (item_ids or [])]
        self._user_idx: dict[str, int] = {u: i for i, u in enumerate(self.user_ids)}
        self._item_idx: dict[str, int] = {i: n for n, i in enumerate(self.item_ids)}
        self._data: dict[int, dict[int, CandidateEntry]] = {}

    # -- vocab helpers -----------------------------------------------------
    def _uidx(self, user_id: str) -> int:
        user_id = str(user_id)
        idx = self._user_idx.get(user_id)
        if idx is None:
            idx = len(self.user_ids)
            self._user_idx[user_id] = idx
            self.user_ids.append(user_id)
        return idx

    def _iidx(self, item_id: str) -> int:
        item_id = str(item_id)
        idx = self._item_idx.get(item_id)
        if idx is None:
            idx = len(self.item_ids)
            self._item_idx[item_id] = idx
            self.item_ids.append(item_id)
        return idx

    # -- mutation -----------------------------------------------------------
    def add_scores(self, user_id: str, scores: dict[str, float], source: str) -> None:
        """Merge ``scores`` for one user; keep only top ``max_per_user``.

        Repeated calls for the same user accumulate (scores sum), then the
        per-user set is re-ranked (score desc, item id asc) and trimmed.
        """
        uidx = self._uidx(user_id)
        row = self._data.setdefault(uidx, {})
        for iid, score in (scores or {}).items():
            iidx = self._iidx(iid)
            entry = row.get(iidx)
            if entry is None:
                row[iidx] = CandidateEntry(
                    item_id=str(iid),
                    score=float(score),
                    rank=0,
                    source=source,
                    sources={source: {"score": float(score), "rank": 0}},
                    source_count=1,
                    rrf=0.0,
                )
            else:
                entry.score += float(score)
                src = entry.sources.setdefault(source, {"score": 0.0, "rank": 0})
                src["score"] += float(score)
        self._rerank_user(row, source)
        self._trim(uidx)

    @staticmethod
    def _rerank_user(row: dict[int, CandidateEntry], source: str) -> None:
        ordered = sorted(row.values(), key=lambda e: (-e.score, e.item_id))
        for rank, entry in enumerate(ordered, start=1):
            entry.rank = rank
            src = entry.sources.get(source)
            if src is not None:
                src["rank"] = float(rank)

    def _trim(self, uidx: int) -> None:
        row = self._data.get(uidx, {})
        if len(row) <= self.max_per_user:
            return
        ordered = sorted(row.values(), key=lambda e: (-e.score, e.item_id))
        keep = {e.item_id for e in ordered[: self.max_per_user]}
        self._data[uidx] = {
            iidx: e for iidx, e in row.items() if e.item_id in keep
        }

    # -- union ---------------------------------------------------------------
    @classmethod
    def union(
        cls,
        *tables: "SparseCandidateTable",
        rrf_k: int = 60,
        max_per_user: int = 200,
        source_id: str = "union",
    ) -> "SparseCandidateTable":
        """Merge tables; per-candidate source_count and RRF (k=``rrf_k``)."""
        merged = cls(source_id=source_id, max_per_user=max_per_user)
        acc: dict[str, dict[str, CandidateEntry]] = {}
        for t_pos, table in enumerate(tables):
            for uidx, row in table._data.items():
                uid = table.user_ids[uidx]
                user_acc = acc.setdefault(uid, {})
                for entry in row.values():
                    src_key = table.source_id
                    if any(src_key in e.sources for e in user_acc.values()):
                        src_key = f"{table.source_id}#{t_pos}"
                    tgt = user_acc.get(entry.item_id)
                    if tgt is None:
                        tgt = CandidateEntry(
                            item_id=entry.item_id,
                            score=0.0,
                            rank=0,
                            source=source_id,
                            sources={},
                            source_count=0,
                            rrf=0.0,
                        )
                        user_acc[entry.item_id] = tgt
                    src_rank = entry.sources.get(table.source_id, {}).get("rank") or entry.rank
                    src_rank = max(int(src_rank), 1)
                    src_info = entry.sources.get(table.source_id)
                    src_score = float(src_info["score"]) if isinstance(src_info, dict) else float(entry.score)
                    tgt.sources[src_key] = {"score": src_score, "rank": float(src_rank)}
                    tgt.rrf += 1.0 / (rrf_k + src_rank)
        for uid, user_acc in acc.items():
            uidx = merged._uidx(uid)
            ordered = sorted(user_acc.values(), key=lambda e: (-e.rrf, e.item_id))
            row: dict[int, CandidateEntry] = {}
            for rank, entry in enumerate(ordered[:max_per_user], start=1):
                entry.rank = rank
                entry.score = entry.rrf
                entry.source_count = len(entry.sources)
                iidx = merged._iidx(entry.item_id)
                row[iidx] = entry
            merged._data[uidx] = row
        return merged

    # -- access ---------------------------------------------------------------
    def get(self, user_id: str, item_id: str) -> CandidateEntry | None:
        uidx = self._user_idx.get(str(user_id))
        if uidx is None:
            return None
        iidx = self._item_idx.get(str(item_id))
        if iidx is None:
            return None
        return self._data.get(uidx, {}).get(iidx)

    def entries_for(self, user_id: str) -> list[CandidateEntry]:
        uidx = self._user_idx.get(str(user_id))
        if uidx is None:
            return []
        return sorted(self._data.get(uidx, {}).values(), key=lambda e: e.rank)

    @property
    def n_entries(self) -> int:
        return sum(len(row) for row in self._data.values())

    def topk(self, user_id: str, k: int) -> list[str]:
        entries = self.entries_for(user_id)
        return [e.item_id for e in entries[:k]]

    def to_topk_lists(self, k: int) -> list[list[str]]:
        """Top-k lists aligned to ``self.user_ids`` order."""
        return [self.topk(uid, k) for uid in self.user_ids]

    def to_csr(self, user_ids: Iterable[str] | None = None) -> csr_matrix:
        """CSR view of the BOUNDED stored candidate set (never the catalog)."""
        uids = [str(u) for u in (user_ids or self.user_ids)]
        rows, cols, data = [], [], []
        for r, uid in enumerate(uids):
            for entry in self._data.get(self._user_idx.get(uid, -1), {}).values():
                rows.append(r)
                cols.append(self._item_idx[entry.item_id])
                data.append(entry.score)
        return csr_matrix(
            (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(len(uids), len(self.item_ids)),
        )

    def coverage(self, targets: dict[str, str]) -> dict[str, Any]:
        """Hit statistics of stored candidates against ``targets`` (uid->iid)."""
        n_users = 0
        n_hit = 0
        ranks: list[int] = []
        for uid, iid in (targets or {}).items():
            n_users += 1
            entry = self.get(uid, iid)
            if entry is not None:
                n_hit += 1
                ranks.append(entry.rank)
        return {
            "n_users": n_users,
            "n_hit": n_hit,
            "hit_rate": (n_hit / n_users) if n_users else 0.0,
            "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
        }


class MemoryPreflight:
    """Budget gate for any dense allocation the pipeline might attempt."""

    def __init__(self, budget_mb: float = 4096.0) -> None:
        self.budget_mb = float(budget_mb)
        self.budget_bytes = int(self.budget_mb * 1024 * 1024)

    @staticmethod
    def estimate_dense_bytes(n_rows: int, n_cols: int, dtype: Any = np.float64) -> int:
        itemsize = np.dtype(dtype).itemsize
        return int(n_rows) * int(n_cols) * int(itemsize)

    def check_dense(
        self,
        n_rows: int,
        n_cols: int,
        dtype: Any = np.float64,
    ) -> dict[str, Any]:
        est_bytes = self.estimate_dense_bytes(n_rows, n_cols, dtype)
        est_mb = est_bytes / (1024 * 1024)
        itemsize = np.dtype(dtype).itemsize
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if est_bytes <= self.budget_bytes:
            return {
                "allowed": True,
                "estimated_mb": est_mb,
                "available_mb": available_mb,
                "suggested": "ok",
                "reason": f"estimated {est_mb:.1f}MB fits budget {self.budget_mb:.1f}MB",
            }
        if itemsize >= 8:
            # A full user×item float64 matrix above budget is NEVER allowed.
            suggested = "use_sparse"
            reason = (
                f"estimated {est_mb:.1f}MB exceeds budget {self.budget_mb:.1f}MB; "
                f"{FORBIDDEN_PATTERN}"
            )
        elif self.estimate_dense_bytes(1, n_cols, dtype) <= self.budget_bytes:
            suggested = "reduce_batch"
            reason = (
                f"estimated {est_mb:.1f}MB exceeds budget {self.budget_mb:.1f}MB; "
                f"one row fits, so chunk with safe_batch_size"
            )
        else:
            suggested = "use_sparse"
            reason = (
                f"estimated {est_mb:.1f}MB exceeds budget {self.budget_mb:.1f}MB "
                f"and even a single row does not fit; {FORBIDDEN_PATTERN}"
            )
        return {
            "allowed": False,
            "estimated_mb": est_mb,
            "available_mb": available_mb,
            "suggested": suggested,
            "reason": reason,
        }

    def safe_batch_size(self, n_rows: int, n_cols: int, dtype: Any = np.float64) -> int:
        """Largest batch row count fitting the budget (minimum 1)."""
        per_row = self.estimate_dense_bytes(1, n_cols, dtype)
        if per_row <= 0:
            return max(int(n_rows), 1)
        rows_fit = self.budget_bytes // per_row
        return max(1, min(int(n_rows), int(rows_fit)))
