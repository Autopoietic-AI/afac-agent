# -*- coding: utf-8 -*-
"""B2 candidate_ranker real-scale benchmark (v2.1 known-gap closure).

Runs the *real* orchestrator code path (``_run_folded_experiment`` with a
compiled ``candidate_ranker_experiment``) on the full B2 dataset — 40k train
users, 14,065 items — for one F1 screen (2 folds), instrumenting per-stage
timings and peak RSS so the 2-hour formal run has an honest runtime estimate
for the ranker operator.  This is a build-time benchmark: no LLM calls, no
deployment, no Test truth, no frozen-asset mutation.

Usage:

    python -m tools.b2_ranker_benchmark \
        --data-root "C:/Users/.../B推荐/B推荐" \
        --out-dir artifacts/b2_science_repair/ranker_benchmark
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from afac_agent.b2.task_adapter import B2TaskAdapter
from afac_agent.research.event_store import json_dumps
from afac_agent.v2.adaptive_fold import CanonicalFolds
from afac_agent.v2.orchestrator import V2AutonomousResearchOrchestrator
from afac_agent.v2.proposal_compiler import compile_proposal

RANKER_SOURCES = ["popularity", "history", "pair_transition"]
BENCH_FOLDS = [0, 1]  # F1_SCREEN prefix on B2


def _instrument() -> dict[str, float]:
    """Monkeypatch timing counters onto the operator classes (benchmark-local)."""
    timings: dict[str, float] = {
        "retriever_fit_seconds": 0.0,
        "retrieve_seconds": 0.0,
        "union_merge_seconds": 0.0,
        "build_training_rows_seconds": 0.0,
        "ranker_fit_seconds": 0.0,
        "build_scoring_rows_seconds": 0.0,
        "rerank_seconds": 0.0,
    }
    from afac_agent.v2.operators import recommendation as rec

    def _wrap(cls: Any, name: str, key: str) -> None:
        original = getattr(cls, name)

        def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            out = original(self, *args, **kwargs)
            timings[key] += time.monotonic() - t0
            return out

        setattr(cls, name, timed)

    for cls_name in (
        "PopularityRetriever",
        "HistoryRetriever",
        "PairTransitionRetriever",
        "LastTransitionRetriever",
        "RepeatRetriever",
    ):
        cls = getattr(rec, cls_name, None)
        if cls is None:
            continue
        _wrap(cls, "fit", "retriever_fit_seconds")
        _wrap(cls, "retrieve", "retrieve_seconds")
    _wrap(rec.CandidateUnion, "merge", "union_merge_seconds")
    _wrap(rec.CandidateTableRanker, "build_training_rows", "build_training_rows_seconds")
    _wrap(rec.CandidateTableRanker, "fit", "ranker_fit_seconds")
    _wrap(rec.CandidateTableRanker, "build_scoring_rows", "build_scoring_rows_seconds")
    _wrap(rec.CandidateTableRanker, "rerank", "rerank_seconds")
    return timings


def run_benchmark(*, project_root: Path, data_root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    report: dict[str, Any] = {"benchmark": "b2_candidate_ranker_real_scale_v1"}

    t_load = time.monotonic()
    adapter = B2TaskAdapter(data_root, task_id="B2")
    dataset = adapter.load()
    report["data_load_seconds"] = round(time.monotonic() - t_load, 3)
    report["dataset"] = {
        "n_train_users": len(dataset.train_seq),
        "n_test_users": len(dataset.test_seq),
        "n_items": int(dataset.n_items),
        "validation_status": dataset.validation.get("status"),
    }

    targets = {str(u): str(t) for u, t in zip(dataset.train_df["uid"], dataset.train_df["target_iid"])}
    uids = sorted(targets)
    seq_bins = np.array([min(5, len(dataset.train_seq.get(u, []))) for u in uids], dtype=np.int64)
    canonical = CanonicalFolds.build(uids, stratify_bins=seq_bins)
    report["canonical_fold_hash"] = canonical.fold_hash

    orchestrator = V2AutonomousResearchOrchestrator(
        project_root=project_root,
        task="B2",
        data_root=data_root,
        out_root=out_dir,
        require_llm=False,
        smoke=False,
        no_deployment=True,
        force_new_execution=True,
    )
    orchestrator._canonical = canonical
    orchestrator._full_train_seq = dataset.train_seq

    compiled = compile_proposal(
        {
            "diagnostic_type": "candidate_ranker_experiment",
            "information_sources": RANKER_SOURCES,
            "budget_seconds": 600,
        },
        parent_candidate_id="anchor_popularity_b2",
        formal_mode=False,
        task="B2",
    )
    assert compiled.status == "compiled", compiled.reason
    report["operator_id"] = compiled.operator_id
    report["retrieval_sources"] = list(compiled.retrieval_sources)

    timings = _instrument()
    rss_before = proc.memory_info().rss
    peak_rss = rss_before

    fold_results: dict[int, dict[str, Any]] = {}
    t_all = time.monotonic()
    for fold_id in BENCH_FOLDS:
        t_fold = time.monotonic()
        outcome = orchestrator._run_folded_experiment(
            compiled, dataset, targets, [fold_id], parent_kind="popularity"
        )
        fold_results[fold_id] = outcome
        peak_rss = max(peak_rss, proc.memory_info().rss)
        n_eval = outcome["result"]["n_eval_users"]
        fold_results[fold_id] = {
            "seconds": round(time.monotonic() - t_fold, 3),
            "n_eval_users": n_eval,
            "candidate_hit_rate@10": outcome["candidate_fold_metrics"].get(fold_id),
            "parent_hit_rate@10": outcome["parent_fold_metrics"].get(fold_id),
        }
    total_seconds = time.monotonic() - t_all
    peak_rss = max(peak_rss, proc.memory_info().rss)

    # Candidate-table scale from the last fold's result.
    last = outcome["result"]
    report["candidate_scale"] = {
        "n_eval_users_last_fold": last["n_eval_users"],
        "pool_recall": last["pool_recall"],
        "top10_metrics": last["top10_metrics"],
    }
    report["folds"] = {str(k): v for k, v in fold_results.items()}
    report["stage_timings_seconds"] = {k: round(v, 3) for k, v in timings.items()}
    report["total_ranker_screen_seconds"] = round(total_seconds, 3)
    report["seconds_per_fold"] = round(total_seconds / len(BENCH_FOLDS), 3)
    report["estimated_3fold_confirm_seconds"] = round(total_seconds / len(BENCH_FOLDS) * 3, 3)
    report["estimated_5fold_seconds"] = round(total_seconds / len(BENCH_FOLDS) * 5, 3)
    report["memory"] = {
        "rss_before_mb": round(rss_before / 1e6, 1),
        "peak_rss_mb": round(peak_rss / 1e6, 1),
        "delta_mb": round((peak_rss - rss_before) / 1e6, 1),
    }
    report["fits_2h_formal_budget"] = bool(
        report["estimated_3fold_confirm_seconds"] + 300 + 120 < 7200
    )
    return report


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "b2_ranker_benchmark.json").write_text(json_dumps(report) + "\n", encoding="utf-8")
    lines = [
        "# B2 Operator Performance Report — candidate_ranker real scale",
        "",
        f"- data_load: {report['data_load_seconds']}s; dataset: {report['dataset']}",
        f"- operator: `{report['operator_id']}`; sources: {report['retrieval_sources']}",
        f"- folds: {json.dumps(report['folds'], ensure_ascii=False)}",
        f"- stage timings (s): {json.dumps(report['stage_timings_seconds'], ensure_ascii=False)}",
        f"- total 2-fold screen: {report['total_ranker_screen_seconds']}s "
        f"({report['seconds_per_fold']}s/fold)",
        f"- estimated 3-fold confirm: {report['estimated_3fold_confirm_seconds']}s",
        f"- estimated 5-fold: {report['estimated_5fold_seconds']}s",
        f"- memory: {json.dumps(report['memory'], ensure_ascii=False)}",
        f"- candidate scale: {json.dumps(report['candidate_scale'], ensure_ascii=False)}",
        f"- fits 2h formal budget (confirm + reserve + margin): "
        f"**{report['fits_2h_formal_budget']}**",
        "",
    ]
    (out_dir / "B2_OPERATOR_PERFORMANCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="artifacts/b2_science_repair/ranker_benchmark")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    report = run_benchmark(project_root=project_root, data_root=Path(args.data_root), out_dir=out_dir)
    _write_report(out_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
