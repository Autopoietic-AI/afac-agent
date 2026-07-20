# -*- coding: utf-8 -*-
"""Read-only audit adapter for v49A Edge Utility meta evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from afac_agent.adapters.base import (
    AdapterContext,
    AdapterDescription,
    OutputManifest,
    RawExecutionResult,
)
from afac_agent.adapters.runner import sha256_file
from afac_agent.schemas import ValidationReport


MODEL_NAME = "confidence_plus_edge"
N_CLASSES = 10
DEFAULT_GATE = {
    "minimum_support": 3,
    "minimum_precision": 2.0 / 3.0,
    "minimum_net": 1,
    "minimum_folds": 2,
}
OOF_REQUIRED_COLUMNS = {
    "fold",
    "global_idx",
    "true_label",
    "base_correct",
    "h2_correct",
    "rescue",
    "damage",
    "neutral",
    "target_rescue",
    "base_pred",
    "h2_pred",
    "model_name",
    "rescue_score",
    "selected",
}
TEST_REQUIRED_COLUMNS = {
    "global_idx",
    "base_pred",
    "h2_pred",
    "model_name",
    "rescue_score",
    "selected",
}
TEST_TRUTH_COLUMNS = {
    "true_label",
    "base_correct",
    "h2_correct",
    "rescue",
    "damage",
    "neutral",
    "target_rescue",
}


def _read_csv(path: Path, *, name: str) -> tuple[pd.DataFrame, list[str]]:
    try:
        return pd.read_csv(path), []
    except Exception as exc:
        return pd.DataFrame(), [f"{name}: cannot read csv: {exc}"]


def _load_a1_npz(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    required = {"labels", "train_idx", "test_idx"}
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        return {}, [f"a1_npz: cannot read npz: {exc}"]
    with payload:
        missing = sorted(required - set(payload.files))
        if missing:
            return {}, [f"a1_npz: missing keys {missing}"]
        labels = np.asarray(payload["labels"], dtype=np.int64)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
    errors: list[str] = []
    if len(set(map(int, train_idx))) != len(train_idx):
        errors.append("a1_npz: train_idx contains duplicates")
    if len(set(map(int, test_idx))) != len(test_idx):
        errors.append("a1_npz: test_idx contains duplicates")
    return {"labels": labels, "train_idx": train_idx, "test_idx": test_idx}, errors


def _coerce_bool_series(series: pd.Series, *, name: str) -> tuple[pd.Series, list[str]]:
    def parse(value: Any) -> bool | None:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    parsed = series.map(parse)
    invalid = parsed.isna()
    if invalid.any():
        safe = pd.Series(
            [False if value is None else bool(value) for value in parsed],
            index=series.index,
            dtype=bool,
        )
        return safe, [
            f"{name}: invalid boolean values in selected"
        ]
    return parsed.astype(bool), []


def _int_series(frame: pd.DataFrame, column: str, *, name: str) -> list[str]:
    errors: list[str] = []
    try:
        converted = pd.to_numeric(frame[column], errors="raise")
    except Exception:
        return [f"{name}: {column} must be integer-like"]
    if converted.isna().any() or not np.all(np.isclose(converted, np.round(converted))):
        errors.append(f"{name}: {column} must be integer-like")
    frame[column] = converted.astype(int)
    return errors


def _validate_oof(
    frame: pd.DataFrame,
    *,
    train_idx: np.ndarray,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    missing = sorted(OOF_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return frame, [f"oof_meta_csv: missing columns {missing}"]
    for column in [
        "fold",
        "global_idx",
        "true_label",
        "base_correct",
        "h2_correct",
        "rescue",
        "damage",
        "neutral",
        "target_rescue",
        "base_pred",
        "h2_pred",
    ]:
        errors.extend(_int_series(frame, column, name="oof_meta_csv"))
    selected, selected_errors = _coerce_bool_series(
        frame["selected"],
        name="oof_meta_csv",
    )
    frame["selected"] = selected
    errors.extend(selected_errors)
    if errors:
        return frame, errors

    train_set = set(map(int, train_idx))
    bad_idx = sorted(set(map(int, frame["global_idx"])) - train_set)
    if bad_idx:
        errors.append(f"oof_meta_csv: global_idx outside train_idx {bad_idx[:5]}")
    duplicated = frame.duplicated(["model_name", "global_idx"])
    if duplicated.any():
        errors.append("oof_meta_csv: duplicate (model_name, global_idx)")
    folds = sorted(set(map(int, frame["fold"])))
    if folds != list(range(len(folds))):
        errors.append(f"oof_meta_csv: fold ids must be contiguous from 0, got {folds}")
    for column in ["true_label", "base_pred", "h2_pred"]:
        if not frame[column].between(0, N_CLASSES - 1).all():
            errors.append(f"oof_meta_csv: {column} out of 0..{N_CLASSES - 1}")
    valid_idx = frame["global_idx"].to_numpy(dtype=int)
    if len(valid_idx) and not np.array_equal(
        frame["true_label"].to_numpy(dtype=int),
        labels[valid_idx],
    ):
        errors.append("oof_meta_csv: true_label must match A1 labels")

    base_correct = (frame["base_pred"] == frame["true_label"]).astype(int)
    h2_correct = (frame["h2_pred"] == frame["true_label"]).astype(int)
    rescue = ((base_correct == 0) & (h2_correct == 1)).astype(int)
    damage = ((base_correct == 1) & (h2_correct == 0)).astype(int)
    neutral = (
        (base_correct == 0)
        & (h2_correct == 0)
        & (frame["base_pred"] != frame["h2_pred"])
    ).astype(int)
    if not np.array_equal(frame["base_correct"].to_numpy(dtype=int), base_correct.to_numpy()):
        errors.append("oof_meta_csv: base_correct inconsistent")
    if not np.array_equal(frame["h2_correct"].to_numpy(dtype=int), h2_correct.to_numpy()):
        errors.append("oof_meta_csv: h2_correct inconsistent")
    if not np.array_equal(frame["rescue"].to_numpy(dtype=int), rescue.to_numpy()):
        errors.append("oof_meta_csv: rescue inconsistent")
    if not np.array_equal(frame["damage"].to_numpy(dtype=int), damage.to_numpy()):
        errors.append("oof_meta_csv: damage inconsistent")
    if not np.array_equal(frame["neutral"].to_numpy(dtype=int), neutral.to_numpy()):
        errors.append("oof_meta_csv: neutral inconsistent")
    if not np.array_equal(frame["target_rescue"].to_numpy(dtype=int), rescue.to_numpy()):
        errors.append("oof_meta_csv: target_rescue inconsistent")
    if ((frame["rescue"] == 1) & (frame["damage"] == 1)).any():
        errors.append("oof_meta_csv: row cannot be both rescue and damage")
    return frame, errors


def _validate_test(
    frame: pd.DataFrame,
    *,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str], bool]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(TEST_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return frame, [f"test_meta_csv: missing columns {missing}"], warnings, True
    for column in ["global_idx", "base_pred", "h2_pred"]:
        errors.extend(_int_series(frame, column, name="test_meta_csv"))
    selected, selected_errors = _coerce_bool_series(
        frame["selected"],
        name="test_meta_csv",
    )
    frame["selected"] = selected
    errors.extend(selected_errors)
    if errors:
        return frame, errors, warnings, True
    test_set = set(map(int, test_idx))
    bad_idx = sorted(set(map(int, frame["global_idx"])) - test_set)
    if bad_idx:
        errors.append(f"test_meta_csv: global_idx outside test_idx {bad_idx[:5]}")
    duplicated = frame.duplicated(["model_name", "global_idx"])
    if duplicated.any():
        errors.append("test_meta_csv: duplicate (model_name, global_idx)")
    for column in ["base_pred", "h2_pred"]:
        if not frame[column].between(0, N_CLASSES - 1).all():
            errors.append(f"test_meta_csv: {column} out of 0..{N_CLASSES - 1}")
    truth_columns = sorted(TEST_TRUTH_COLUMNS & set(frame.columns))
    test_truth_usage_pass = not truth_columns
    if truth_columns:
        warnings.append(
            f"test meta contains truth-dependent columns {truth_columns}; not used for metrics"
        )
    return frame, errors, warnings, test_truth_usage_pass


def _load_prediction_csv(
    path: Path,
    *,
    name: str,
    expected_test_idx: np.ndarray | None,
) -> tuple[pd.DataFrame, list[str]]:
    frame, errors = _read_csv(path, name=name)
    if errors:
        return frame, errors
    if list(frame.columns) != ["test_idx", "label"]:
        return frame, [f"{name}: columns must be ['test_idx', 'label']"]
    for column in ["test_idx", "label"]:
        errors.extend(_int_series(frame, column, name=name))
    if errors:
        return frame, errors
    if frame["test_idx"].duplicated().any():
        errors.append(f"{name}: duplicate test_idx")
    if not frame["label"].between(0, N_CLASSES - 1).all():
        errors.append(f"{name}: label out of range")
    if expected_test_idx is not None and not np.array_equal(
        frame["test_idx"].to_numpy(dtype=int),
        expected_test_idx,
    ):
        errors.append(f"{name}: test_idx sequence must exactly match A1 test_idx")
    return frame, errors


def _parse_gate_from_source(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None or not path.exists():
        return dict(DEFAULT_GATE), ["v53q1_patch_source unavailable; using frozen defaults"]
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return dict(DEFAULT_GATE), [f"v53q1_patch_source could not be read: {exc}"]

    def find_int(name: str, default: int) -> int:
        match = re.search(rf"{name}[^\\n]+default\\s*=\\s*([0-9]+)", text)
        return int(match.group(1)) if match else default

    def find_float(name: str, default: float) -> float:
        match = re.search(rf"{name}[^\\n]+default\\s*=\\s*([^,)]+)", text)
        if not match:
            return default
        expr = match.group(1).strip()
        if expr == "2.0 / 3.0":
            return 2.0 / 3.0
        try:
            return float(expr)
        except ValueError:
            return default

    return {
        "minimum_support": find_int("minimum_support", DEFAULT_GATE["minimum_support"]),
        "minimum_precision": find_float("minimum_precision", DEFAULT_GATE["minimum_precision"]),
        "minimum_net": find_int("minimum_net", DEFAULT_GATE["minimum_net"]),
        "minimum_folds": find_int("minimum_folds", DEFAULT_GATE["minimum_folds"]),
    }, []


def _transition(frame: pd.DataFrame) -> pd.Series:
    return frame["base_pred"].astype(int).astype(str) + "->" + frame["h2_pred"].astype(int).astype(str)


def _transition_summary(selected_oof: pd.DataFrame, gate: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if len(selected_oof) == 0:
        return pd.DataFrame(
            columns=[
                "transition",
                "support",
                "rescue",
                "damage",
                "neutral",
                "net",
                "decisive_precision",
                "fold_count",
                "allowed_by_gate",
            ]
        )
    selected_oof = selected_oof.copy()
    selected_oof["transition"] = _transition(selected_oof)
    for transition, group in selected_oof.groupby("transition"):
        rescue = int(group["rescue"].sum())
        damage = int(group["damage"].sum())
        neutral = int(group["neutral"].sum())
        decisive = rescue + damage
        support = int(len(group))
        net = rescue - damage
        precision = rescue / decisive if decisive else 0.0
        fold_count = int(group["fold"].nunique())
        allowed = (
            support >= int(gate["minimum_support"])
            and decisive > 0
            and precision >= float(gate["minimum_precision"])
            and net >= int(gate["minimum_net"])
            and fold_count >= int(gate["minimum_folds"])
        )
        rows.append(
            {
                "transition": str(transition),
                "support": support,
                "rescue": rescue,
                "damage": damage,
                "neutral": neutral,
                "net": net,
                "decisive_precision": precision,
                "fold_count": fold_count,
                "allowed_by_gate": allowed,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["allowed_by_gate", "net", "support", "transition"],
        ascending=[False, False, False, True],
    )


def _document_consistency(
    *,
    v53_audit: Path | None,
    v49_report: Path | None,
    champion_relations: list[dict[str, Any]],
    oof_selected_count: int,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    consistent = True
    if v53_audit and v53_audit.exists():
        try:
            text = v53_audit.read_text(encoding="utf-8")
            for row in champion_relations:
                old_new = f"{row['base_label']} -> {row['champion_label']}"
                if str(row["test_idx"]) not in text or old_new not in text:
                    consistent = False
                    warnings.append("v53q1 audit markdown does not mention every champion patch")
                    break
        except Exception as exc:
            consistent = False
            warnings.append(f"v53q1 audit markdown could not be parsed: {exc}")
    if v49_report and v49_report.exists():
        try:
            text = v49_report.read_text(encoding="utf-8")
            if str(oof_selected_count) not in text:
                consistent = False
                warnings.append("v49A report does not mention recomputed selected count")
        except Exception as exc:
            consistent = False
            warnings.append(f"v49A report could not be parsed: {exc}")
    return consistent, warnings


class Adapter:
    def __init__(self) -> None:
        self._details: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._transition_summary = pd.DataFrame()
        self._oof_selected = pd.DataFrame()
        self._test_selected = pd.DataFrame()
        self._champion_relation = pd.DataFrame()

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            tool_name="A1_V49A_EDGE_UTILITY_AUDIT",
            adapter_id="A1_V49A_EDGE_UTILITY_AUDIT",
            adapter_version="m3c_v1",
            target_problem="A1_v49A_edge_utility_transition_gate_audit",
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
        oof_meta = context.path("v49a_oof_meta_csv")
        test_meta = context.path("v49a_test_meta_csv")
        base_csv = context.path("v46a1_base_csv")
        champion_csv = context.path("current_champion_csv")
        v53_audit = context.path("v53q1_audit_md")
        patch_source = context.path("v53q1_patch_source")
        v49_report = context.path("v49a_report")
        v49_config = context.path("v49a_config")
        v49_fold_results = context.path("v49a_fold_results")
        assert a1_npz is not None
        assert oof_meta is not None
        assert test_meta is not None

        bundle, npz_errors = _load_a1_npz(a1_npz)
        errors.extend(npz_errors)
        if errors:
            return {}, errors, warnings
        labels = bundle["labels"]
        train_idx = bundle["train_idx"]
        test_idx = bundle["test_idx"]

        oof, oof_read_errors = _read_csv(oof_meta, name="oof_meta_csv")
        test, test_read_errors = _read_csv(test_meta, name="test_meta_csv")
        errors.extend(oof_read_errors)
        errors.extend(test_read_errors)
        if errors:
            return {}, errors, warnings
        oof, oof_errors = _validate_oof(oof, train_idx=train_idx, labels=labels)
        test, test_errors, test_warnings, test_truth_usage_pass = _validate_test(
            test,
            test_idx=test_idx,
        )
        errors.extend(oof_errors)
        errors.extend(test_errors)
        warnings.extend(test_warnings)
        if errors:
            return {}, errors, warnings

        gate, gate_warnings = _parse_gate_from_source(patch_source)
        warnings.extend(gate_warnings)
        selected_oof = oof[(oof["model_name"] == MODEL_NAME) & (oof["selected"])].copy()
        selected_test = test[(test["model_name"] == MODEL_NAME) & (test["selected"])].copy()
        selected_oof["transition"] = _transition(selected_oof) if len(selected_oof) else []
        selected_test["transition"] = _transition(selected_test) if len(selected_test) else []
        transition_summary = _transition_summary(selected_oof, gate)
        allowed_transitions = set(
            transition_summary.loc[
                transition_summary["allowed_by_gate"].astype(bool),
                "transition",
            ].astype(str)
        )
        transition_by_name = {
            str(row.transition): row
            for row in transition_summary.itertuples()
        }

        champion_relations: list[dict[str, Any]] = []
        champion_diff_count = 0
        if base_csv and champion_csv:
            base, base_errors = _load_prediction_csv(
                base_csv,
                name="v46a1_base_csv",
                expected_test_idx=test_idx,
            )
            champion, champion_errors = _load_prediction_csv(
                champion_csv,
                name="current_champion_csv",
                expected_test_idx=test_idx,
            )
            errors.extend(base_errors)
            errors.extend(champion_errors)
            if errors:
                return {}, errors, warnings
            diff = base.merge(champion, on="test_idx", suffixes=("_base", "_champion"))
            diff = diff[diff["label_base"] != diff["label_champion"]].copy()
            champion_diff_count = int(len(diff))
            selected_test_by_idx = {
                int(row.global_idx): row
                for row in selected_test.itertuples()
            }
            any_test_by_idx = {
                int(row.global_idx): row
                for row in test[test["model_name"] == MODEL_NAME].itertuples()
            }
            for row in diff.sort_values("test_idx").itertuples():
                transition = f"{int(row.label_base)}->{int(row.label_champion)}"
                selected_row = selected_test_by_idx.get(int(row.test_idx))
                any_row = any_test_by_idx.get(int(row.test_idx))
                gate_row = transition_by_name.get(transition)
                covered = any_row is not None
                selected = selected_row is not None
                gate_allowed = transition in allowed_transitions
                champion_relations.append(
                    {
                        "test_idx": int(row.test_idx),
                        "base_label": int(row.label_base),
                        "champion_label": int(row.label_champion),
                        "transition": transition,
                        "covered_by_test_meta": covered,
                        "selected": selected,
                        "gate_allowed": gate_allowed,
                        "oof_support": int(gate_row.support) if gate_row is not None else 0,
                        "oof_rescue": int(gate_row.rescue) if gate_row is not None else 0,
                        "oof_damage": int(gate_row.damage) if gate_row is not None else 0,
                        "oof_net": int(gate_row.net) if gate_row is not None else 0,
                        "oof_precision": (
                            float(gate_row.decisive_precision)
                            if gate_row is not None else 0.0
                        ),
                        "oof_fold_count": int(gate_row.fold_count) if gate_row is not None else 0,
                        "rescue_score": (
                            float(selected_row.rescue_score)
                            if selected_row is not None else None
                        ),
                        "audit_reason": "evidence_backed" if covered and selected and gate_allowed else "unverified",
                    }
                )

        champion_idx = {row["test_idx"] for row in champion_relations}
        non_champion_test_selected = selected_test[
            ~selected_test["global_idx"].astype(int).isin(champion_idx)
        ].copy()
        if len(non_champion_test_selected):
            non_champion_test_selected["observable_reason"] = "unverified"

        audit_consistent, doc_warnings = _document_consistency(
            v53_audit=v53_audit,
            v49_report=v49_report,
            champion_relations=champion_relations,
            oof_selected_count=len(selected_oof),
        )
        warnings.extend(doc_warnings)

        config_test_labels_used = None
        if v49_config and v49_config.exists():
            try:
                config_payload = json.loads(v49_config.read_text(encoding="utf-8"))
                config_test_labels_used = bool(config_payload.get("test_labels_used", False))
                if config_test_labels_used:
                    warnings.append("v49A config reports test_labels_used=true")
            except Exception as exc:
                warnings.append(f"v49a_config could not be parsed: {exc}")

        fold_result_rows = None
        if v49_fold_results and v49_fold_results.exists():
            try:
                fold_result_rows = int(len(pd.read_csv(v49_fold_results)))
            except Exception as exc:
                warnings.append(f"v49a_fold_results could not be parsed: {exc}")

        champion_patches_covered = int(
            sum(bool(row["covered_by_test_meta"]) for row in champion_relations)
        )
        champion_patches_selected = int(sum(bool(row["selected"]) for row in champion_relations))
        champion_patches_gate_allowed = int(
            sum(bool(row["gate_allowed"]) for row in champion_relations)
        )
        edge_integrity_pass = bool(test_truth_usage_pass and not config_test_labels_used)
        details = {
            "oof_meta_hash": sha256_file(oof_meta),
            "test_meta_hash": sha256_file(test_meta),
            "oof_row_count": int(len(oof)),
            "test_row_count": int(len(test)),
            "oof_selected_count": int(len(selected_oof)),
            "test_selected_count": int(len(selected_test)),
            "transition_count": int(len(transition_summary)),
            "allowed_transition_count": int(
                transition_summary["allowed_by_gate"].astype(bool).sum()
                if len(transition_summary) else 0
            ),
            "allowed_transitions": sorted(allowed_transitions),
            "gate_config": gate,
            "gate_recomputed_pass": True,
            "audit_document_consistent": audit_consistent,
            "champion_patch_diff_count": champion_diff_count,
            "champion_patches_covered_by_test_meta": champion_patches_covered,
            "champion_patches_selected": champion_patches_selected,
            "champion_patches_gate_allowed": champion_patches_gate_allowed,
            "non_champion_test_selected_count": int(len(non_champion_test_selected)),
            "non_champion_test_selected_idx": [
                int(value)
                for value in sorted(non_champion_test_selected["global_idx"].astype(int).tolist())
            ],
            "test_truth_usage_pass": bool(test_truth_usage_pass and not config_test_labels_used),
            "edge_utility_integrity_pass": edge_integrity_pass,
            "fold_result_rows": fold_result_rows,
            "config_test_labels_used": config_test_labels_used,
        }

        self._transition_summary = transition_summary
        self._oof_selected = selected_oof
        self._test_selected = selected_test
        self._champion_relation = pd.DataFrame(champion_relations)
        self._non_champion_test_selected = non_champion_test_selected
        return details, [], warnings

    def validate_inputs(self, context: AdapterContext) -> ValidationReport:
        details, errors, warnings = self._build_audit(context)
        self._details = details
        self._warnings = warnings
        return ValidationReport(
            name="A1_V49A_EDGE_UTILITY_AUDIT",
            passed=not errors,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def execute(self, context: AdapterContext) -> RawExecutionResult:
        details = self._details or self._build_audit(context)[0]
        stdout = json.dumps(
            {
                "adapter": "A1_V49A_EDGE_UTILITY_AUDIT",
                "status": "completed",
                "oof_selected_count": details.get("oof_selected_count"),
                "test_selected_count": details.get("test_selected_count"),
                "champion_patches_gate_allowed": details.get("champion_patches_gate_allowed"),
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
        transition_path = context.run_dir / "transition_gate_summary.csv"
        oof_selected_path = context.run_dir / "oof_selected_audit.csv"
        test_selected_path = context.run_dir / "test_selected_audit.csv"
        relation_path = context.run_dir / "champion_patch_relation.csv"
        self._transition_summary.to_csv(
            transition_path,
            index=False,
            encoding="utf-8-sig",
        )
        self._oof_selected.to_csv(
            oof_selected_path,
            index=False,
            encoding="utf-8-sig",
        )
        self._test_selected.to_csv(
            test_selected_path,
            index=False,
            encoding="utf-8-sig",
        )
        self._champion_relation.to_csv(
            relation_path,
            index=False,
            encoding="utf-8-sig",
        )
        return OutputManifest(
            artifacts={
                "audit_details": str(audit_details),
                "transition_gate_summary": str(transition_path),
                "oof_selected_audit": str(oof_selected_path),
                "test_selected_audit": str(test_selected_path),
                "champion_patch_relation": str(relation_path),
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
