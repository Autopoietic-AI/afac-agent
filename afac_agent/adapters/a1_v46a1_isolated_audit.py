# -*- coding: utf-8 -*-
"""Read-only audit adapter for the v46A-1 isolated expert candidate."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from afac_agent.adapters.base import (
    AdapterContext,
    AdapterDescription,
    OutputManifest,
    RawExecutionResult,
)
from afac_agent.adapters.runner import sha256_file
from afac_agent.schemas import ValidationReport


N_CLASSES = 10


def _load_a1_npz(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    required = {
        "adj_data",
        "adj_indices",
        "adj_indptr",
        "adj_shape",
        "labels",
        "train_idx",
        "test_idx",
    }
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        return {}, [f"a1_npz: cannot read npz: {exc}"]
    with payload:
        missing = sorted(required - set(payload.files))
        if missing:
            return {}, [f"a1_npz: missing keys {missing}"]
        try:
            adjacency = sparse.csr_matrix(
                (
                    payload["adj_data"],
                    payload["adj_indices"],
                    payload["adj_indptr"],
                ),
                shape=tuple(payload["adj_shape"]),
            ).astype(np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
            train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
            test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        except Exception as exc:
            return {}, [f"a1_npz: invalid canonical graph arrays: {exc}"]
    if adjacency.shape[0] != adjacency.shape[1]:
        errors.append(f"a1_npz: adj must be square, got {adjacency.shape}")
    if labels.shape[0] != adjacency.shape[0]:
        errors.append("a1_npz: labels length must match adjacency node count")
    if len(set(map(int, train_idx))) != len(train_idx):
        errors.append("a1_npz: train_idx contains duplicates")
    if len(set(map(int, test_idx))) != len(test_idx):
        errors.append("a1_npz: test_idx contains duplicates")
    return {
        "adjacency": adjacency,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }, errors


def _either_direction_degrees(adjacency: sparse.csr_matrix) -> np.ndarray:
    graph = adjacency.tocsr(copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    either = graph + graph.T
    either.data = np.ones_like(either.data, dtype=np.int8)
    either.eliminate_zeros()
    return np.asarray(either.getnnz(axis=1), dtype=np.int64)


def _load_prediction_csv(
    path: Path,
    *,
    name: str,
    expected_test_idx: np.ndarray | None = None,
) -> tuple[list[dict[str, int]], list[str]]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as exc:
        return [], [f"{name}: cannot read csv: {exc}"]
    if fieldnames != ["test_idx", "label"]:
        return [], [f"{name}: columns must be ['test_idx', 'label'], got {fieldnames}"]
    if expected_test_idx is not None and len(rows) != len(expected_test_idx):
        errors.append(
            f"{name}: expected {len(expected_test_idx)} rows from A1 test_idx, got {len(rows)}"
        )
    seen: set[int] = set()
    parsed: list[dict[str, int]] = []
    for row_number, row in enumerate(rows):
        if row.get("test_idx") in {None, ""} or row.get("label") in {None, ""}:
            errors.append(f"{name}: row {row_number} has null/empty field")
            continue
        try:
            test_idx = int(str(row["test_idx"]).strip())
            label = int(str(row["label"]).strip())
        except ValueError:
            errors.append(f"{name}: row {row_number} has non-integer values")
            continue
        if test_idx in seen:
            errors.append(f"{name}: duplicate test_idx {test_idx}")
        seen.add(test_idx)
        if not 0 <= label < N_CLASSES:
            errors.append(f"{name}: row {row_number} label {label} out of range")
        parsed.append({"test_idx": test_idx, "label": label})
    if expected_test_idx is not None:
        observed = np.asarray([row["test_idx"] for row in parsed], dtype=np.int64)
        if not np.array_equal(observed, expected_test_idx):
            errors.append(f"{name}: test_idx sequence must exactly match A1 test_idx")
    return parsed, errors


def _prediction_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    counts = {str(label): 0 for label in range(N_CLASSES)}
    for row in rows:
        counts[str(row["label"])] += 1
    return counts


def _diff_rows(
    parent_rows: list[dict[str, int]],
    candidate_rows: list[dict[str, int]],
    isolated_by_test_idx: dict[int, bool],
) -> tuple[list[dict[str, Any]], list[str]]:
    parent_idx = [row["test_idx"] for row in parent_rows]
    candidate_idx = [row["test_idx"] for row in candidate_rows]
    if parent_idx != candidate_idx:
        return [], ["parent_csv: test_idx sequence must exactly match candidate_csv"]
    rows: list[dict[str, Any]] = []
    for parent, candidate in zip(parent_rows, candidate_rows):
        if parent["label"] == candidate["label"]:
            continue
        is_isolated = bool(isolated_by_test_idx[int(candidate["test_idx"])])
        rows.append(
            {
                "test_idx": int(candidate["test_idx"]),
                "parent_label": int(parent["label"]),
                "candidate_label": int(candidate["label"]),
                "transition": f"{int(parent['label'])}->{int(candidate['label'])}",
                "is_isolated": is_isolated,
                "is_graph_visible": not is_isolated,
            }
        )
    return rows, []


def _status_from_report(path: Path | None, diff_count: int | None) -> tuple[str, list[str]]:
    if path is None:
        return "unavailable", []
    warnings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return "unverified", [f"audit_report: cannot read report: {exc}"]
    lowered = text.lower()
    has_v43c = "v43c" in lowered
    has_rule = "balanced_seed_consensus" in lowered or "v46a-1" in lowered
    safe_match = False
    if diff_count is not None:
        for match in re.finditer(r"safe_changes[^0-9]*(\d+)", lowered):
            safe_match = safe_match or int(match.group(1)) == diff_count
    if has_v43c and has_rule and safe_match:
        return "verified", warnings
    if has_v43c and has_rule:
        return "evidence_backed", warnings
    warnings.append("parent identity could not be fully verified from audit report")
    return "unverified", warnings


def _load_oof_npz(
    path: Path,
    *,
    train_idx: np.ndarray,
    labels: np.ndarray,
    train_isolated: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        return {}, [f"candidate_oof_npz: cannot read npz: {exc}"]
    with payload:
        files = set(payload.files)
        if "train_idx" not in files or "proba" not in files:
            return {}, ["candidate_oof_npz: requires train_idx and proba arrays"]
        saved_train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        proba = np.asarray(payload["proba"], dtype=np.float64)
        saved_labels = np.asarray(
            payload["labels"] if "labels" in files else payload["y"],
            dtype=np.int64,
        ) if ("labels" in files or "y" in files) else None
    if not np.array_equal(saved_train_idx, train_idx):
        errors.append("candidate_oof_npz: train_idx must exactly match A1 train_idx")
    if proba.shape != (len(train_idx), N_CLASSES):
        errors.append(
            f"candidate_oof_npz: proba shape must be ({len(train_idx)}, {N_CLASSES}), got {proba.shape}"
        )
    if saved_labels is None:
        errors.append("candidate_oof_npz: missing labels/y")
    elif not np.array_equal(saved_labels, labels[train_idx]):
        errors.append("candidate_oof_npz: labels must match A1 train labels")
    if not np.isfinite(proba).all():
        errors.append("candidate_oof_npz: proba contains NaN/Inf")
    if (proba < -1e-8).any():
        errors.append("candidate_oof_npz: proba contains negative values")
    if proba.ndim == 2 and proba.shape[1] == N_CLASSES:
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            errors.append("candidate_oof_npz: probability rows must sum to 1")
    if errors:
        return {}, errors
    pred = proba.argmax(axis=1)
    truth = labels[train_idx]
    correct = pred == truth
    isolated_correct = correct[train_isolated]
    graph_correct = correct[~train_isolated]
    return {
        "oof_status": "observed",
        "overall_oof_accuracy": float(correct.mean()),
        "isolated_oof_accuracy": (
            float(isolated_correct.mean()) if len(isolated_correct) else None
        ),
        "graph_visible_oof_accuracy": (
            float(graph_correct.mean()) if len(graph_correct) else None
        ),
        "oof_prediction_distribution": {
            str(label): int((pred == label).sum())
            for label in range(N_CLASSES)
        },
    }, []


class Adapter:
    def __init__(self) -> None:
        self._details: dict[str, Any] = {}
        self._diff_rows: list[dict[str, Any]] = []
        self._warnings: list[str] = []

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            tool_name="A1_V46A1_ISOLATED_AUDIT",
            adapter_id="A1_V46A1_ISOLATED_AUDIT",
            adapter_version="m3b_v1",
            target_problem="A1_v46A1_isolated_expert_candidate_audit",
            execution_mode="audit",
            read_only=True,
            counts_as_experiment_round=False,
            mutates_predictions=False,
            mutates_project_state=False,
            requires_gpu=False,
        )

    def _build_audit(self, context: AdapterContext) -> tuple[dict[str, Any], list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        a1_npz = context.path("a1_npz")
        candidate_csv = context.path("candidate_csv")
        parent_csv = context.path("parent_csv")
        candidate_oof_npz = context.path("candidate_oof_npz")
        audit_report = context.path("audit_report")
        champion_csv = context.path("current_champion_csv")
        assert a1_npz is not None
        assert candidate_csv is not None

        bundle, npz_errors = _load_a1_npz(a1_npz)
        errors.extend(npz_errors)
        if errors:
            return {}, errors, warnings
        adjacency = bundle["adjacency"]
        labels = bundle["labels"]
        train_idx = bundle["train_idx"]
        test_idx = bundle["test_idx"]
        degree = _either_direction_degrees(adjacency)
        test_isolated_mask = degree[test_idx] == 0
        train_isolated_mask = degree[train_idx] == 0
        isolated_by_test_idx = {
            int(idx): bool(mask)
            for idx, mask in zip(test_idx, test_isolated_mask)
        }

        candidate_rows, candidate_errors = _load_prediction_csv(
            candidate_csv,
            name="candidate_csv",
            expected_test_idx=test_idx,
        )
        errors.extend(candidate_errors)
        candidate_integrity_pass = not candidate_errors

        parent_rows: list[dict[str, int]] = []
        diff_rows: list[dict[str, Any]] = []
        parent_identity_status = "unavailable"
        parent_hash = ""
        if parent_csv:
            parent_rows, parent_errors = _load_prediction_csv(
                parent_csv,
                name="parent_csv",
                expected_test_idx=test_idx,
            )
            errors.extend(parent_errors)
            if not parent_errors and not candidate_errors:
                parent_hash = sha256_file(parent_csv)
                diff_rows, diff_errors = _diff_rows(
                    parent_rows,
                    candidate_rows,
                    isolated_by_test_idx,
                )
                errors.extend(diff_errors)
                parent_identity_status, report_warnings = _status_from_report(
                    audit_report,
                    len(diff_rows),
                )
                warnings.extend(report_warnings)
        else:
            warnings.append("parent_csv unavailable; parent-to-candidate diff not generated")

        champion_diff_count: int | None = None
        champion_hash = ""
        if champion_csv:
            champion_rows, champion_errors = _load_prediction_csv(
                champion_csv,
                name="current_champion_csv",
                expected_test_idx=test_idx,
            )
            if champion_errors:
                warnings.extend(champion_errors)
            elif not candidate_errors:
                champion_hash = sha256_file(champion_csv)
                champion_diff_count = int(
                    sum(
                        int(a["label"] != b["label"])
                        for a, b in zip(candidate_rows, champion_rows)
                    )
                )

        if errors:
            return {}, errors, warnings

        isolated_diff_count = int(sum(row["is_isolated"] for row in diff_rows))
        graph_visible_diff_count = int(sum(row["is_graph_visible"] for row in diff_rows))
        isolated_only_pass = graph_visible_diff_count == 0
        warnings.append(f"graph_visible_diff_count={graph_visible_diff_count}")
        if graph_visible_diff_count:
            warnings.append("candidate modifies graph-visible test nodes")

        oof_metrics: dict[str, Any]
        if candidate_oof_npz:
            oof_metrics, oof_errors = _load_oof_npz(
                candidate_oof_npz,
                train_idx=train_idx,
                labels=labels,
                train_isolated=train_isolated_mask,
            )
            errors.extend(oof_errors)
            if errors:
                return {}, errors, warnings
        else:
            oof_metrics = {
                "oof_status": "unavailable",
                "oof_unavailable_reason": "missing_candidate_oof_npz",
            }

        details = {
            "candidate_hash": sha256_file(candidate_csv),
            "candidate_rows": len(candidate_rows),
            "candidate_integrity_pass": candidate_integrity_pass,
            "candidate_prediction_distribution": _prediction_counts(candidate_rows),
            "test_idx_strict_match": True,
            "parent_hash": parent_hash,
            "parent_identity_status": parent_identity_status,
            "diff_count": len(diff_rows),
            "isolated_diff_count": isolated_diff_count,
            "graph_visible_diff_count": graph_visible_diff_count,
            "isolated_only_pass": isolated_only_pass,
            "isolated_test_count": int(test_isolated_mask.sum()),
            "graph_visible_test_count": int((~test_isolated_mask).sum()),
            "candidate_champion_hash": champion_hash,
            "candidate_champion_diff_count": champion_diff_count,
            "canonical_graph_source": "A1.npz:adj CSR",
            "isolated_definition": "unique either-direction degree == 0 after self-loop removal",
            **oof_metrics,
        }
        self._diff_rows = diff_rows
        return details, [], warnings

    def validate_inputs(self, context: AdapterContext) -> ValidationReport:
        details, errors, warnings = self._build_audit(context)
        self._details = details
        self._warnings = warnings
        return ValidationReport(
            name="A1_V46A1_ISOLATED_AUDIT",
            passed=not errors,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def execute(self, context: AdapterContext) -> RawExecutionResult:
        details = self._details or self._build_audit(context)[0]
        stdout = json.dumps(
            {
                "adapter": "A1_V46A1_ISOLATED_AUDIT",
                "status": "completed",
                "diff_count": details.get("diff_count"),
                "isolated_diff_count": details.get("isolated_diff_count"),
                "graph_visible_diff_count": details.get("graph_visible_diff_count"),
                "oof_status": details.get("oof_status"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return RawExecutionResult(
            status="completed",
            returncode=0,
            stdout=stdout + "\n",
            stderr="",
            metrics=details,
            warnings=self._warnings,
        )

    def collect_outputs(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
    ) -> OutputManifest:
        assert context.run_dir is not None
        audit_details = context.run_dir / "audit_details.json"
        audit_details.write_text(
            json.dumps(
                raw_result.metrics,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        diff_path = context.run_dir / "candidate_diff_audit.csv"
        with diff_path.open("w", encoding="utf-8-sig", newline="") as file:
            fieldnames = [
                "test_idx",
                "parent_label",
                "candidate_label",
                "transition",
                "is_isolated",
                "is_graph_visible",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._diff_rows)
        return OutputManifest(
            artifacts={
                "audit_details": str(audit_details),
                "candidate_diff_audit": str(diff_path),
            }
        )

    def normalize_result(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
        outputs: OutputManifest,
    ) -> dict[str, Any]:
        return {
            "status": raw_result.status,
            "metrics": raw_result.metrics,
            "bucket_metrics": raw_result.bucket_metrics,
            "class_metrics": raw_result.class_metrics,
            "warnings": [],
            "failure_reason": raw_result.failure_reason,
        }
