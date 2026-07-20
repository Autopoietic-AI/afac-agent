from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from conftest import PROJECT_ROOT


CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
PROJECT_STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_a1(path: Path) -> dict[str, np.ndarray]:
    labels = np.arange(14, dtype=np.int64) % 10
    train_idx = np.arange(10, dtype=np.int64)
    test_idx = np.arange(10, 14, dtype=np.int64)
    edges = [
        (1, 2),
        (3, 4),
        (3, 5),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 10),
        (6, 11),
        (6, 12),
    ]
    rows = [src for src, _ in edges]
    cols = [dst for _, dst in edges]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(14, 14))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=np.asarray(adj.shape, dtype=np.int64),
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    return {"labels": labels, "train_idx": train_idx, "test_idx": test_idx}


def _proba_from_pred(pred: np.ndarray) -> np.ndarray:
    proba = np.full((len(pred), 10), 0.1 / 9.0, dtype=np.float64)
    proba[np.arange(len(pred)), pred] = 0.9
    return proba


def _write_oof(
    path: Path,
    *,
    labels: np.ndarray,
    train_idx: np.ndarray,
    candidate_pred: np.ndarray | None = None,
    parent_pred: np.ndarray | None = None,
    shuffle: bool = False,
    bucket: np.ndarray | None = None,
    omit_train_idx: bool = False,
) -> Path:
    if candidate_pred is None:
        candidate_pred = labels[train_idx]
    order = np.arange(len(train_idx))
    if shuffle:
        order = np.asarray([3, 1, 0, 2, 5, 4, 6, 8, 7, 9])
    payload = {
        "proba": _proba_from_pred(candidate_pred)[order],
        "labels": labels[train_idx][order],
    }
    if not omit_train_idx:
        payload["train_idx"] = train_idx[order]
    if parent_pred is not None:
        payload["base_proba"] = _proba_from_pred(parent_pred)[order]
    if bucket is not None:
        payload["bucket"] = bucket[order]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def _write_fold(path: Path, train_idx: np.ndarray, labels: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["global_idx", "fold", "label"])
        writer.writeheader()
        for idx in train_idx:
            writer.writerow(
                {
                    "global_idx": int(idx),
                    "fold": int(idx % 3),
                    "label": int(labels[idx]),
                }
            )
    return path


def _tool(*, output_root: Path):
    from afac_agent.registry import ToolRegistry

    tool = ToolRegistry(PROJECT_ROOT / "config" / "tool_registry.json").get(
        "A1_OOF_CANDIDATE_EVALUATOR"
    )
    tool.output_policy = dict(tool.output_policy or {})
    tool.output_policy["output_root"] = str(output_root)
    return tool


def _run_adapter(tmp_path: Path, variables: dict[str, str]):
    from afac_agent.adapters.runner import AdapterRunner

    return AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=_tool(output_root=tmp_path / "evaluation runs 中文 with space"),
        variables=variables,
        execute=True,
    )


def test_registry_binding_and_missing_candidate_waiting_for_input(tmp_path: Path) -> None:
    from afac_agent.adapters.runner import AdapterRunner

    data = _write_a1(tmp_path / "a1.npz")
    tool = _tool(output_root=tmp_path / "runs")
    assert tool.read_only is True
    assert tool.counts_as_experiment_round is False
    assert tool.mutates_predictions is False
    assert tool.adapter_entrypoint == "afac_agent.adapters.a1_oof_candidate_evaluator:Adapter"

    result = AdapterRunner(project_root=PROJECT_ROOT).run(
        tool=tool,
        variables={"a1_npz": str(tmp_path / "a1.npz"), "candidate_oof_npz": ""},
        execute=True,
    )
    assert len(data["train_idx"]) == 10
    assert result["status"] == "waiting_for_input"
    assert "candidate_oof_npz" in result["missing_inputs"]


def test_integrity_only_has_no_parent_gain_or_fake_anchor(tmp_path: Path) -> None:
    data = _write_a1(tmp_path / "a1.npz")
    candidate = _write_oof(
        tmp_path / "candidate.npz",
        labels=data["labels"],
        train_idx=data["train_idx"],
    )

    result = _run_adapter(
        tmp_path,
        {"a1_npz": str(tmp_path / "a1.npz"), "candidate_oof_npz": str(candidate)},
    )

    assert result["status"] == "completed"
    metrics = result["metrics"]
    assert metrics["analysis_tier"] == "integrity_only"
    assert metrics["comparison_scope"] == "candidate_integrity_only"
    assert metrics["parent_identity_status"] == "unavailable"
    assert metrics["test_truth_used"] is False
    assert metrics["rescue_damage"] == {"status": "unavailable", "reason": "missing_parent_oof"}
    assert "parent_accuracy" not in metrics["overall_metrics"]
    assert "gain" not in metrics["overall_metrics"]
    assert not list(Path(result["artifacts"]["run_dir"]).glob("A1*.csv"))
    assert not list(Path(result["artifacts"]["run_dir"]).glob("submission*.zip"))
    assert not list(Path(result["artifacts"]["run_dir"]).glob("*.npz"))


def test_embedded_parent_metrics_safe_reorder_buckets_duplicate_and_outputs(tmp_path: Path) -> None:
    data = _write_a1(tmp_path / "a1.npz")
    labels = data["labels"]
    train_idx = data["train_idx"]
    parent_pred = labels[train_idx].copy()
    parent_pred[[0, 9]] = [1, 0]
    candidate_pred = labels[train_idx].copy()
    candidate_pred[[1, 4, 7, 9]] = [2, 5, 8, 0]
    candidate = _write_oof(
        tmp_path / "candidate embedded.npz",
        labels=labels,
        train_idx=train_idx,
        candidate_pred=candidate_pred,
        parent_pred=parent_pred,
        shuffle=True,
        bucket=np.full(len(train_idx), 9, dtype=np.int64),
    )
    frozen_before = {p: _sha256(p) for p in [CHAMPION, PROJECT_STATE, HISTORY]}

    first = _run_adapter(
        tmp_path,
        {"a1_npz": str(tmp_path / "a1.npz"), "candidate_oof_npz": str(candidate)},
    )
    second = _run_adapter(
        tmp_path,
        {"a1_npz": str(tmp_path / "a1.npz"), "candidate_oof_npz": str(candidate)},
    )

    assert first["status"] == "completed"
    assert second["status"] == "duplicate"
    assert second["identity_hash"] == first["identity_hash"]
    metrics = first["metrics"]
    assert metrics["analysis_tier"] == "self_contained_oof_comparison"
    assert metrics["comparison_scope"] == "embedded_parent_unverified"
    assert metrics["parent_identity_status"] == "unverified"
    assert metrics["overall_metrics"]["parent_accuracy"] == pytest.approx(0.8)
    assert metrics["overall_metrics"]["candidate_accuracy"] == pytest.approx(0.6)
    assert metrics["overall_metrics"]["gain"] == pytest.approx(-0.2)
    assert metrics["rescue_damage"]["changed_count"] == 4
    assert metrics["rescue_damage"]["rescue"] == 1
    assert metrics["rescue_damage"]["damage"] == 3
    assert metrics["rescue_damage"]["net"] == -2
    assert metrics["rescue_damage"]["change_precision"] == pytest.approx(0.25)
    assert metrics["oracle_metrics"]["oracle_gain"] == pytest.approx(0.1)
    assert metrics["fold_metrics"] == {"status": "unavailable", "reason": "missing_canonical_fold"}
    assert {row["bucket"] for row in metrics["bucket_metrics"]} == {
        "Graph-visible",
        "Isolated",
        "degree_1",
        "degree_2_5",
        "degree_6p",
    }
    assert any("safely reordered" in warning for warning in first["warnings"])
    assert any("OOF bucket field" in warning for warning in first["warnings"])
    evaluation_path = Path(first["artifacts"]["a1_oof_evaluation"])
    assert evaluation_path.exists()
    assert Path(first["artifacts"]["class_metrics"]).exists()
    assert Path(first["artifacts"]["bucket_metrics"]).exists()
    assert Path(first["artifacts"]["changed_nodes_audit"]).exists()
    assert {p: _sha256(p) for p in frozen_before} == frozen_before
    assert json.loads(PROJECT_STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == 0


def test_valid_fold_stats_are_observed_without_rebuilding(tmp_path: Path) -> None:
    data = _write_a1(tmp_path / "a1.npz")
    labels = data["labels"]
    train_idx = data["train_idx"]
    parent_pred = labels[train_idx].copy()
    parent_pred[[0, 9]] = [1, 0]
    candidate_pred = labels[train_idx].copy()
    candidate_pred[[1, 4, 7, 9]] = [2, 5, 8, 0]
    candidate = _write_oof(
        tmp_path / "candidate.npz",
        labels=labels,
        train_idx=train_idx,
        candidate_pred=candidate_pred,
        parent_pred=parent_pred,
    )
    fold = _write_fold(tmp_path / "fold.csv", train_idx, labels)

    result = _run_adapter(
        tmp_path,
        {
            "a1_npz": str(tmp_path / "a1.npz"),
            "candidate_oof_npz": str(candidate),
            "canonical_fold_csv": str(fold),
        },
    )

    assert result["status"] == "completed"
    assert result["metrics"]["fold_metrics"]["status"] == "observed"
    assert result["metrics"]["fold_metrics"]["fold_count"] == 3
    assert Path(result["artifacts"]["fold_metrics"]).exists()


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda p: p.pop("train_idx"), "missing whitelisted train index"),
        (lambda p: p.__setitem__("proba", p["proba"][:, :9]), "proba shape"),
        (lambda p: p["proba"].__setitem__((0, 0), np.nan), "NaN/Inf"),
        (lambda p: p["proba"].__setitem__((0, 0), -0.1), "negative"),
        (lambda p: p["proba"].__setitem__((0, slice(None)), np.ones(10)), "rows must sum"),
        (lambda p: p["train_idx"].__setitem__(1, p["train_idx"][0]), "duplicates"),
        (lambda p: p["labels"].__setitem__(0, 9), "labels must match"),
    ],
)
def test_invalid_oof_inputs_fail(tmp_path: Path, mutator, expected: str) -> None:
    from afac_agent.evaluation.a1_oof_candidate_evaluator import evaluate_a1_oof_candidate

    data = _write_a1(tmp_path / "a1.npz")
    payload = {
        "proba": _proba_from_pred(data["labels"][data["train_idx"]]),
        "train_idx": data["train_idx"].copy(),
        "labels": data["labels"][data["train_idx"]].copy(),
    }
    mutator(payload)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **payload)

    evaluation, errors, _warnings = evaluate_a1_oof_candidate(
        a1_npz=tmp_path / "a1.npz",
        candidate_oof_npz=bad,
        write_outputs=False,
    )

    assert evaluation == {}
    assert any(expected in error for error in errors)


def test_feedback_normalizer_is_informational_without_policy(tmp_path: Path) -> None:
    from afac_agent.feedback.builder import FeedbackBuilder

    data = _write_a1(tmp_path / "a1.npz")
    labels = data["labels"]
    train_idx = data["train_idx"]
    candidate = _write_oof(
        tmp_path / "candidate.npz",
        labels=labels,
        train_idx=train_idx,
        parent_pred=labels[train_idx],
    )
    result = _run_adapter(
        tmp_path,
        {"a1_npz": str(tmp_path / "a1.npz"), "candidate_oof_npz": str(candidate)},
    )
    built = FeedbackBuilder(project_root=PROJECT_ROOT).build(
        execution_result_path=Path(result["artifacts"]["execution_result"]),
        out_root=tmp_path / "feedback 中文 with space",
    )
    feedback = json.loads(
        Path(built["artifacts"]["experiment_feedback"]).read_text(encoding="utf-8")
    )
    assert feedback["feedback_kind"] == "model_experiment"
    assert feedback["evaluation_tier"] == "oof_comparison"
    assert feedback["recommendation"] == "informational_only"


def test_cli_chinese_space_path_and_dry_run(tmp_path: Path) -> None:
    data = _write_a1(tmp_path / "输入 中文 with space" / "a1.npz")
    candidate = _write_oof(
        tmp_path / "输入 中文 with space" / "candidate.npz",
        labels=data["labels"],
        train_idx=data["train_idx"],
    )
    command = [
        sys.executable,
        "-m",
        "afac_agent.main",
        "run-adapter",
        "--project_root",
        str(PROJECT_ROOT),
        "--tool",
        "A1_OOF_CANDIDATE_EVALUATOR",
        "--a1_npz",
        str(tmp_path / "输入 中文 with space" / "a1.npz"),
        "--candidate_oof_npz",
        str(candidate),
        "--adapter_output_root",
        str(tmp_path / "输出 中文 with space"),
    ]
    dry = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["status"] == "dry_run"
    executed = subprocess.run([*command, "--execute"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert executed.returncode == 0, executed.stderr
    assert json.loads(executed.stdout)["status"] == "completed"


@pytest.mark.skipif(
    not Path(
        r"C:\Users\李天皓\agent比赛\model_pro\a1\new-a1\a1_openroute_cs_v1_v43c\correct_smooth_v1_oof_proba.npz"
    ).exists()
    or not Path(r"C:\Users\李天皓\agent比赛\A分类\A分类\A1.npz").exists(),
    reason="local historical v43C OOF smoke assets are not available",
)
def test_real_v43c_self_contained_oof_smoke(tmp_path: Path) -> None:
    result = _run_adapter(
        tmp_path,
        {
            "a1_npz": r"C:\Users\李天皓\agent比赛\A分类\A分类\A1.npz",
            "candidate_oof_npz": (
                r"C:\Users\李天皓\agent比赛\model_pro\a1\new-a1"
                r"\a1_openroute_cs_v1_v43c\correct_smooth_v1_oof_proba.npz"
            ),
        },
    )
    assert result["status"] == "completed"
    metrics = result["metrics"]
    assert metrics["analysis_tier"] == "self_contained_oof_comparison"
    assert metrics["comparison_scope"] == "embedded_parent_unverified"
    assert metrics["parent_identity_status"] == "unverified"
    assert metrics["candidate_identity"]["eligible_for_research_reopen"] is False
    assert metrics["candidate_identity"]["used_only_for_evaluator_validation"] is True
    assert metrics["input_validation"]["train_idx_count"] == 11001
    assert metrics["test_truth_used"] is False
