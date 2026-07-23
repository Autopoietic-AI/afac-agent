# -*- coding: utf-8 -*-
"""B2 Data Intelligence orchestrator.

Runs deterministic audits on the B2 sequence-recommendation data, writes
required artifacts under ``artifacts/b2_runs/<run_id>/data_intelligence/``,
and emits ``DATA_INTELLIGENCE_REPORT.md``.  No model training, no Test truth.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..research.event_store import json_dumps, rel_ref, sha256_file, stable_hash
from .task_adapter import B2Dataset, B2TaskAdapter

DATA_INTELLIGENCE_VERSION = "b2_data_intelligence_v1"

OUTPUT_FILES = {
    "dataset_fingerprint.json": "fingerprint",
    "integrity_audit.json": "integrity_audit",
    "user_sequence_audit.json": "user_sequence_audit",
    "item_audit.json": "item_audit",
    "train_test_shift_audit.json": "shift_audit",
    "retrieval_conclusions.json": "retrieval_conclusions",
    "ranking_conclusions.json": "ranking_conclusions",
    "regime_identification.json": "regime_identification",
    "model_search_prior.json": "model_search_prior",
    "validation_protocol.json": "validation_protocol",
}


def _sha256_of_files(paths: list[Path]) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in paths if p.is_file()}


def _item_popularity(dataset: B2Dataset) -> pd.Series:
    counts: Counter[str] = Counter()
    for seq in dataset.train_seq.values():
        counts.update(seq)
    return pd.Series(counts).sort_values(ascending=False)


def _build_co_occurrence(dataset: B2Dataset, max_items: int = 20000) -> csr_matrix:
    item_counts = _item_popularity(dataset)
    frequent_items = set(item_counts.head(max_items).index)
    item2idx = {iid: idx for idx, iid in enumerate(sorted(frequent_items))}
    n = len(item2idx)
    row, col, data = [], [], []
    for seq in dataset.train_seq.values():
        dedup = [iid for iid in dict.fromkeys(seq) if iid in item2idx]
        for i, a in enumerate(dedup):
            for b in dedup[i + 1: i + 6]:
                ai, bi = item2idx[a], item2idx[b]
                row.extend([ai, bi])
                col.extend([bi, ai])
                data.extend([1.0, 1.0])
    if not row:
        return csr_matrix((n, n))
    return csr_matrix((np.asarray(data, dtype=np.float32), (np.asarray(row), np.asarray(col))), shape=(n, n))


def _integrity_audit(dataset: B2Dataset, file_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "status": dataset.validation["status"],
        "errors": dataset.validation["errors"],
        "warnings": dataset.validation["warnings"],
        "file_hashes": file_hashes,
        "n_train": dataset.validation["n_train"],
        "n_test": dataset.validation["n_test"],
        "n_users": dataset.validation["n_users"],
        "n_items": dataset.validation["n_items"],
        "top_k": dataset.top_k,
        "test_truth_hidden": dataset.validation["test_truth_hidden"],
        "train_test_uid_overlap": dataset.validation["train_test_uid_overlap"],
    }


def _user_sequence_audit(dataset: B2Dataset) -> dict[str, Any]:
    train_lens = np.array([len(s) for s in dataset.train_seq.values()], dtype=np.float64)
    test_lens = np.array([len(s) for s in dataset.test_seq.values()], dtype=np.float64)
    empty_test = int(np.sum(test_lens == 0))

    # Repeat-ratio within raw sequences (dedup vs raw)
    if "item_seq_raw" in dataset.train_df.columns and "item_seq_dedup" in dataset.train_df.columns:
        raw_lens = dataset.train_df["item_seq_raw"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0).values
        dedup_lens = dataset.train_df["item_seq_dedup"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0).values
        repeat_ratio = float(np.mean((raw_lens - dedup_lens) / (raw_lens + 1e-12)))
    else:
        repeat_ratio = None

    # Target position distribution: where in raw sequence does the target appear?
    target_positions: list[int] = []
    if "target_iid" in dataset.train_df.columns:
        for uid, target in zip(dataset.train_df["uid"], dataset.train_df["target_iid"]):
            seq = dataset.train_seq.get(str(uid), [])
            if str(target) in seq:
                # last occurrence
                target_positions.append(len(seq) - 1 - seq[::-1].index(str(target)))
            else:
                target_positions.append(-1)
    in_history_rate = float(np.mean([p >= 0 for p in target_positions])) if target_positions else None

    return {
        "train_sequence_length": {"mean": float(train_lens.mean()), "median": float(np.median(train_lens)), "std": float(train_lens.std()), "min": int(train_lens.min()), "max": int(train_lens.max())},
        "test_sequence_length": {"mean": float(test_lens.mean()), "median": float(np.median(test_lens)), "std": float(test_lens.std()), "min": int(test_lens.min()), "max": int(test_lens.max())},
        "empty_test_sequences": empty_test,
        "repeat_ratio_mean": repeat_ratio,
        "target_in_history_rate": in_history_rate,
        "target_position_in_history_mean": (float(np.mean([p for p in target_positions if p >= 0])) if target_positions and any(p >= 0 for p in target_positions) else None),
    }


def _item_audit(dataset: B2Dataset) -> dict[str, Any]:
    pop = _item_popularity(dataset)
    total_interactions = int(pop.sum())
    coverage = float(pop.size / dataset.n_items)
    gini = float(_gini(pop.values)) if len(pop) > 0 else 0.0
    head_items = set(pop.head(int(0.05 * dataset.n_items)).index)
    tail_items = set(pop.tail(int(0.5 * dataset.n_items)).index)
    # Item features
    item_cols = [c for c in dataset.item_df.columns if c != "iid"]
    return {
        "n_items": dataset.n_items,
        "n_items_observed_train": int(pop.size),
        "train_item_coverage": coverage,
        "total_interactions": total_interactions,
        "popularity_gini": gini,
        "head_items_top5pct": len(head_items),
        "tail_items_bottom50pct": len(tail_items),
        "item_feature_columns": item_cols,
        "most_popular_item": str(pop.index[0]) if len(pop) > 0 else None,
        "most_popular_count": int(pop.iloc[0]) if len(pop) > 0 else 0,
    }


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = values.size
    if n == 0 or values.sum() == 0:
        return 0.0
    cum = np.cumsum(values)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _shift_audit(dataset: B2Dataset) -> dict[str, Any]:
    # Build a small propensity model: train vs test based on sequence length, user features, item popularity
    train_meta = []
    test_meta = []
    pop = _item_popularity(dataset).to_dict()
    for uid, seq in dataset.train_seq.items():
        train_meta.append({"uid": uid, "len": len(seq), "pop_mean": np.mean([pop.get(iid, 0) for iid in seq]) if seq else 0})
    for uid, seq in dataset.test_seq.items():
        test_meta.append({"uid": uid, "len": len(seq), "pop_mean": np.mean([pop.get(iid, 0) for iid in seq]) if seq else 0})
    train_meta_df = pd.DataFrame(train_meta)
    test_meta_df = pd.DataFrame(test_meta)
    X = pd.concat([train_meta_df[["len", "pop_mean"]], test_meta_df[["len", "pop_mean"]]], ignore_index=True).fillna(0)
    y = np.array([0] * len(train_meta_df) + [1] * len(test_meta_df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    aucs = []
    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.iloc[tr])
        clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=2026)
        clf.fit(Xs, y[tr])
        prob = clf.predict_proba(scaler.transform(X.iloc[va]))[:, 1]
        aucs.append(roc_auc_score(y[va], prob))
    auc = float(np.mean(aucs))
    len_diff = float(test_meta_df["len"].mean() - train_meta_df["len"].mean())
    pop_diff = float(test_meta_df["pop_mean"].mean() - train_meta_df["pop_mean"].mean())
    return {
        "propensity_auc_len_pop": auc,
        "test_train_mean_length_diff": len_diff,
        "test_train_mean_popularity_diff": pop_diff,
        "shift_interpretation": "strong_shift" if auc > 0.65 else ("moderate_shift" if auc > 0.55 else "weak_shift"),
    }


def _retrieval_conclusions(dataset: B2Dataset) -> dict[str, Any]:
    pop = _item_popularity(dataset)
    co = _build_co_occurrence(dataset)
    n_co_items = co.shape[0]
    co_nnz = int(co.nnz)
    # Estimate how often a target can be retrieved from history of same user
    if "target_iid" in dataset.train_df.columns:
        hits = 0
        total = 0
        for uid, target in zip(dataset.train_df["uid"], dataset.train_df["target_iid"]):
            seq = dataset.train_seq.get(str(uid), [])
            if seq:
                total += 1
                if str(target) in seq:
                    hits += 1
        history_recall = hits / total if total else None
    else:
        history_recall = None
    return {
        "history_recall_ceiling": history_recall,
        "co_occurrence_matrix_shape": [n_co_items, n_co_items],
        "co_occurrence_nnz": co_nnz,
        "popularity_head_coverage": float(pop.head(1000).sum() / pop.sum()) if pop.sum() > 0 else None,
        "retrieval_recommendation": "combine_history_cooccurrence_popularity",
    }


def _ranking_conclusions(dataset: B2Dataset) -> dict[str, Any]:
    # How often target is the most recent item? (repeat-consumption signal)
    if "target_iid" in dataset.train_df.columns:
        last_hit = 0
        any_hit = 0
        total = 0
        for uid, target in zip(dataset.train_df["uid"], dataset.train_df["target_iid"]):
            seq = dataset.train_seq.get(str(uid), [])
            if seq:
                total += 1
                if str(target) in seq:
                    any_hit += 1
                    if seq[-1] == str(target):
                        last_hit += 1
        repeat_rate = any_hit / total if total else None
        last_repeat_rate = last_hit / any_hit if any_hit else None
    else:
        repeat_rate = None
        last_repeat_rate = None
    return {
        "target_repeat_consumption_rate": repeat_rate,
        "target_is_last_item_rate": last_repeat_rate,
        "ranking_recommendation": "rank_by_recency_and_frequency_then_blend",
    }


def _regime_identification(dataset: B2Dataset, audits: dict[str, Any]) -> dict[str, Any]:
    shift = audits["shift_audit"]["shift_interpretation"]
    item = audits["item_audit"]
    seq = audits["user_sequence_audit"]
    retrieval = audits["retrieval_conclusions"]
    long_tail = item["popularity_gini"] > 0.7
    cold_start = seq["empty_test_sequences"] > 0
    history_ceiling = retrieval.get("history_recall_ceiling")
    primary = "sparse_sequence_recommendation"
    secondary = []
    if long_tail:
        secondary.append("long_tail_dominance")
    if cold_start:
        secondary.append("cold_start_users")
    if shift in ("moderate_shift", "strong_shift"):
        secondary.append("train_test_distribution_shift")
    if history_ceiling is not None and history_ceiling < 0.5:
        secondary.append("low_history_recall_ceiling")
    return {
        "primary_problem": primary,
        "secondary_problems": secondary,
        "long_tail": long_tail,
        "cold_start_users": cold_start,
        "validation_risk": "medium" if (long_tail or cold_start or shift == "strong_shift") else "low",
        "initial_hypotheses": [
            "history_recall_provides_floor",
            "co_occurrence_improves_history_missing_cases",
            "popularity_recovers_cold_start",
            "rank_fusion_outperforms_single_signal",
        ],
        "model_search_prior": ["history_recall", "item_cooccurrence", "popularity", "score_blend", "rank_fusion"],
        "avoid_list": ["heavy_deep_model_without_regime_evidence"],
    }


def _validation_protocol(dataset: B2Dataset) -> dict[str, Any]:
    return {
        "fold_identity": "AFAC_B2_FOLD_V1",
        "n_splits": 5,
        "stratify_on": "user_sequence_length_bucket",
        "seed": 2026,
        "leakage_check": "no_history_target_overlap_between_train_and_val",
        "test_truth_used": False,
        "evaluation_metrics": ["ndcg@10", "hit_rate@10", "mrr@10", "candidate_recall@10"],
    }


def run_data_intelligence(
    *,
    data_root: str | Path,
    out_root: str | Path = "artifacts/b2_runs",
    project_root: str | Path = ".",
    force_rebuild: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    out_root = project_root / out_root
    data_root = Path(data_root)
    adapter = B2TaskAdapter(data_root, task_id="B2")
    missing = adapter.missing_files()
    if missing:
        return {"status": "waiting_for_input", "missing_files": missing, "artifacts": {}}
    dataset = adapter.load()
    if dataset.validation["status"] != "passed":
        return {"status": "validation_failed", "validation": dataset.validation, "artifacts": {}}

    root = adapter.actual_root()
    file_hashes = _sha256_of_files([root / n for n in ("train.csv", "test.csv", "user.csv", "item.csv", "sample_submission.csv", "metadata.json")])
    run_id = stable_hash({"version": DATA_INTELLIGENCE_VERSION, "input_hashes": file_hashes})[:24]
    out_dir = out_root / run_id / "data_intelligence"
    if out_dir.exists() and not force_rebuild:
        manifest_path = out_dir / "data_intelligence_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {"status": manifest.get("status", "duplicate"), "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    integrity = _integrity_audit(dataset, file_hashes)
    user_seq = _user_sequence_audit(dataset)
    item = _item_audit(dataset)
    shift = _shift_audit(dataset)
    retrieval = _retrieval_conclusions(dataset)
    ranking = _ranking_conclusions(dataset)
    regime = _regime_identification(dataset, {
        "user_sequence_audit": user_seq,
        "item_audit": item,
        "shift_audit": shift,
        "retrieval_conclusions": retrieval,
        "ranking_conclusions": ranking,
    })
    val_protocol = _validation_protocol(dataset)

    artifacts: dict[str, str] = {}

    def write(name: str, payload: dict[str, Any]) -> None:
        path = out_dir / name
        path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
        artifacts[Path(name).stem] = rel_ref(path, project_root)

    write("dataset_fingerprint.json", {
        "task_id": dataset.task_id,
        "data_root": str(dataset.data_root),
        "n_train": len(dataset.train_df),
        "n_test": len(dataset.test_df),
        "n_users": len(dataset.user_df),
        "n_items": dataset.n_items,
        "top_k": dataset.top_k,
        "file_hashes": file_hashes,
    })
    write("integrity_audit.json", integrity)
    write("user_sequence_audit.json", user_seq)
    write("item_audit.json", item)
    write("train_test_shift_audit.json", shift)
    write("retrieval_conclusions.json", retrieval)
    write("ranking_conclusions.json", ranking)
    write("regime_identification.json", regime)
    write("model_search_prior.json", {"model_search_prior": regime["model_search_prior"], "avoid_list": regime["avoid_list"]})
    write("validation_protocol.json", val_protocol)

    report = out_dir / "DATA_INTELLIGENCE_REPORT.md"
    report.write_text(_report(dataset, integrity, user_seq, item, shift, retrieval, ranking, regime), encoding="utf-8")
    artifacts["DATA_INTELLIGENCE_REPORT"] = rel_ref(report, project_root)

    manifest = {
        "manifest_version": DATA_INTELLIGENCE_VERSION,
        "run_id": run_id,
        "status": "verified",
        "created_at_epoch_seconds": time.time(),
        "duration_seconds": round(time.time() - started, 6),
        "input_hashes": file_hashes,
        "data_intelligence_status": "verified",
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "data_intelligence_manifest.json"
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    artifacts["data_intelligence_manifest"] = rel_ref(manifest_path, project_root)
    return {"status": "verified", "run_id": run_id, "artifacts": artifacts}


def _report(dataset: B2Dataset, integrity: dict, user_seq: dict, item: dict, shift: dict, retrieval: dict, ranking: dict, regime: dict) -> str:
    return "\n".join([
        "# B2 Data Intelligence Report",
        "",
        f"task: B2 sequence recommendation; train={len(dataset.train_df)}; test={len(dataset.test_df)}; items={dataset.n_items}; top_k={dataset.top_k}",
        "",
        "## Integrity",
        f"- status: {integrity['status']}",
        f"- test_truth_hidden: {integrity['test_truth_hidden']}",
        f"- train/test uid overlap: {integrity['train_test_uid_overlap']}",
        "",
        "## User / Sequence",
        f"- train mean sequence length: {user_seq['train_sequence_length']['mean']:.2f}",
        f"- test mean sequence length: {user_seq['test_sequence_length']['mean']:.2f}",
        f"- empty test sequences: {user_seq['empty_test_sequences']}",
        f"- target in history rate: {user_seq['target_in_history_rate']}",
        f"- target position in history mean: {user_seq['target_position_in_history_mean']}",
        "",
        "## Item",
        f"- observed item coverage: {item['train_item_coverage']:.4f}",
        f"- popularity gini: {item['popularity_gini']:.4f}",
        f"- most popular item: {item['most_popular_item']} ({item['most_popular_count']} interactions)",
        "",
        "## Train/Test Shift",
        f"- propensity_auc: {shift['propensity_auc_len_pop']:.4f}",
        f"- shift_interpretation: {shift['shift_interpretation']}",
        "",
        "## Retrieval",
        f"- history_recall_ceiling: {retrieval['history_recall_ceiling']}",
        f"- retrieval_recommendation: {retrieval['retrieval_recommendation']}",
        "",
        "## Ranking",
        f"- target_repeat_consumption_rate: {ranking['target_repeat_consumption_rate']}",
        f"- ranking_recommendation: {ranking['ranking_recommendation']}",
        "",
        "## Regime",
        f"- primary_problem: {regime['primary_problem']}",
        f"- secondary_problems: {regime['secondary_problems']}",
        f"- validation_risk: {regime['validation_risk']}",
        "",
    ])
