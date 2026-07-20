# -*- coding: utf-8 -*-
"""Controlled replay adapter for the frozen v53Q-1 Edge-H2 patch.

This adapter is intentionally narrow: it executes the already-frozen patch
script inside an isolated adapter run directory, validates the resulting
candidate against the frozen Champion CSV, and never registers or submits it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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
FROZEN_GATE = {
    "minimum_support": 3,
    "minimum_precision": 2.0 / 3.0,
    "minimum_net": 1,
    "minimum_folds": 2,
}
PATCH_OUTPUTS = [
    "replay_candidate.csv",
    "v53q1_transition_stable_patch_diff.csv",
    "v53q1_transition_oof_summary.csv",
    "v53q1_crossfit_fold_results.csv",
    "v53q1_audit.json",
]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _semantic_hash(rows: list[dict[str, int]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row["test_idx"]))
    payload = {
        "columns": ["test_idx", "label"],
        "rows": [
            [int(row["test_idx"]), int(row["label"])]
            for row in ordered
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normal_transition(value: Any) -> str:
    text = str(value)
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return f"{int(numbers[0])}->{int(numbers[1])}"
    return text.strip()


def _parse_default_number(raw: str) -> float:
    text = raw.strip().rstrip(",")
    if "/" in text:
        left, right = text.split("/", 1)
        return float(left.strip()) / float(right.strip())
    return float(text)


def _parse_patch_gate(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    parsed: dict[str, float | int] = {}
    patterns = {
        "minimum_support": r"--minimum_support[^)\n]*default\s*=\s*([^,\)\n]+)",
        "minimum_precision": r"--minimum_precision[^)\n]*default\s*=\s*([^,\)\n]+(?:\s*/\s*[^,\)\n]+)?)",
        "minimum_net": r"--minimum_net[^)\n]*default\s*=\s*([^,\)\n]+)",
        "minimum_folds": r"--minimum_folds[^)\n]*default\s*=\s*([^,\)\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"missing frozen gate default: {key}")
        value = _parse_default_number(match.group(1))
        parsed[key] = float(value) if key == "minimum_precision" else int(value)
    return parsed


def _gate_from_registry(context: AdapterContext) -> dict[str, float | int]:
    policy = context.tool.output_policy or {}
    configured = policy.get("frozen_gate_config") or FROZEN_GATE
    return {
        "minimum_support": int(configured["minimum_support"]),
        "minimum_precision": float(configured["minimum_precision"]),
        "minimum_net": int(configured["minimum_net"]),
        "minimum_folds": int(configured["minimum_folds"]),
    }


def _gate_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left.get("minimum_support")) == int(right.get("minimum_support"))
        and abs(float(left.get("minimum_precision")) - float(right.get("minimum_precision"))) < 1e-12
        and int(left.get("minimum_net")) == int(right.get("minimum_net"))
        and int(left.get("minimum_folds")) == int(right.get("minimum_folds"))
    )


def _load_prediction_csv(path: Path, *, name: str) -> tuple[list[dict[str, int]], list[str]]:
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
    seen: set[int] = set()
    parsed: list[dict[str, int]] = []
    for index, row in enumerate(rows):
        if row.get("test_idx") in {None, ""} or row.get("label") in {None, ""}:
            errors.append(f"{name}: row {index} has null/empty field")
            continue
        try:
            test_idx = int(str(row["test_idx"]).strip())
            label = int(str(row["label"]).strip())
        except ValueError:
            errors.append(f"{name}: row {index} has non-integer values")
            continue
        if test_idx in seen:
            errors.append(f"{name}: duplicate test_idx {test_idx}")
        seen.add(test_idx)
        if not 0 <= label < N_CLASSES:
            errors.append(f"{name}: row {index} label {label} out of range")
        parsed.append({"test_idx": test_idx, "label": label})
    return parsed, errors


def _diff_predictions(
    before: list[dict[str, int]],
    after: list[dict[str, int]],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    before_idx = [row["test_idx"] for row in before]
    after_idx = [row["test_idx"] for row in after]
    if before_idx != after_idx:
        return [], ["test_idx sequence mismatch"]
    rows: list[dict[str, Any]] = []
    for left, right in zip(before, after):
        if int(left["label"]) == int(right["label"]):
            continue
        rows.append(
            {
                "test_idx": int(left["test_idx"]),
                "old_label": int(left["label"]),
                "new_label": int(right["label"]),
                "transition": f"{int(left['label'])}->{int(right['label'])}",
            }
        )
    return rows, errors


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"global_idx", "base_pred", "h2_pred", "model_name", "rescue_score", "selected"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"test meta missing columns: {missing}")
    selected = frame[
        (frame["model_name"] == MODEL_NAME)
        & (frame["selected"].astype(bool))
    ].copy()
    selected["transition"] = (
        selected["base_pred"].astype(int).astype(str)
        + "->"
        + selected["h2_pred"].astype(int).astype(str)
    )
    return selected


def _recompute_allowed_transitions(
    path: Path,
    gate: dict[str, float | int],
) -> tuple[set[str], pd.DataFrame]:
    frame = pd.read_csv(path)
    required = {
        "fold",
        "base_pred",
        "h2_pred",
        "rescue",
        "damage",
        "neutral",
        "model_name",
        "selected",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"oof meta missing columns: {missing}")
    selected = frame[
        (frame["model_name"] == MODEL_NAME)
        & (frame["selected"].astype(bool))
    ].copy()
    selected["transition"] = (
        selected["base_pred"].astype(int).astype(str)
        + "->"
        + selected["h2_pred"].astype(int).astype(str)
    )
    rows: list[dict[str, Any]] = []
    allowed: set[str] = set()
    for transition, group in selected.groupby("transition"):
        rescue = int(group["rescue"].sum())
        damage = int(group["damage"].sum())
        neutral = int(group["neutral"].sum())
        decisive = rescue + damage
        precision = rescue / decisive if decisive else 0.0
        net = rescue - damage
        support = int(len(group))
        folds = int(group["fold"].nunique())
        passed = (
            support >= int(gate["minimum_support"])
            and decisive > 0
            and precision >= float(gate["minimum_precision"])
            and net >= int(gate["minimum_net"])
            and folds >= int(gate["minimum_folds"])
        )
        if passed:
            allowed.add(str(transition))
        rows.append(
            {
                "transition": str(transition),
                "support": support,
                "rescue": rescue,
                "damage": damage,
                "neutral": neutral,
                "net": net,
                "decisive_precision": precision,
                "folds_observed": folds,
                "allowed_by_gate": passed,
            }
        )
    return allowed, pd.DataFrame(rows).sort_values(["allowed_by_gate", "net", "support"], ascending=[False, False, False])


class Adapter:
    def __init__(self) -> None:
        self._gate: dict[str, float | int] = {}
        self._warnings: list[str] = []

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            tool_name="A1_V53Q1_PATCH_REPLAY_SAFE",
            adapter_id="A1_V53Q1_PATCH_REPLAY_SAFE",
            adapter_version="m3d_v1",
            target_problem="A1_v53Q1_patch_replay_candidate",
            execution_mode="replay",
            read_only=False,
            counts_as_experiment_round=False,
            mutates_predictions=True,
            mutates_project_state=False,
            requires_gpu=False,
        )

    def validate_inputs(self, context: AdapterContext) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        patch_py = context.path("v53q1_patch_py")
        base = context.path("v53q1_base_csv")
        champion = context.path("current_champion_csv")
        oof = context.path("v49a_oof_meta_csv")
        test = context.path("v49a_test_meta_csv")
        assert patch_py is not None
        assert base is not None
        assert champion is not None
        assert oof is not None
        assert test is not None

        if not _truthy(context.variables.get("allow_prediction_artifact")):
            errors.append("prediction_artifact_permission_required")

        try:
            source_gate = _parse_patch_gate(patch_py)
            registry_gate = _gate_from_registry(context)
            if not _gate_equal(source_gate, registry_gate):
                errors.append("frozen_gate_config_mismatch")
            self._gate = registry_gate
        except Exception as exc:
            errors.append(f"gate_parse_failed: {exc}")
            self._gate = _gate_from_registry(context)

        for path, name in [
            (base, "base_csv"),
            (champion, "champion_csv"),
        ]:
            rows, row_errors = _load_prediction_csv(path, name=name)
            errors.extend(row_errors)
            if len(rows) == 2751:
                continue
            if name == "champion_csv":
                warnings.append(f"{name}: synthetic/non-A1 row count {len(rows)}")

        try:
            _recompute_allowed_transitions(oof, self._gate)
            _selected_frame(test)
        except Exception as exc:
            errors.append(str(exc))

        self._warnings = warnings
        return ValidationReport(
            name="A1_V53Q1_PATCH_REPLAY_SAFE",
            passed=not errors,
            errors=errors,
            warnings=warnings,
            details={"gate_config": self._gate},
        )

    def execute(self, context: AdapterContext) -> RawExecutionResult:
        assert context.run_dir is not None
        gate = self._gate or _gate_from_registry(context)
        base = context.path("v53q1_base_csv")
        oof = context.path("v49a_oof_meta_csv")
        test = context.path("v49a_test_meta_csv")
        patch_py = context.path("v53q1_patch_py")
        champion = context.path("current_champion_csv")
        assert base is not None
        assert oof is not None
        assert test is not None
        assert patch_py is not None
        assert champion is not None

        work_tmp = context.run_dir / "work_tmp"
        candidate_dir = context.run_dir / "candidate"
        candidate_csv = candidate_dir / "replay_candidate.csv"
        if candidate_dir.exists() or work_tmp.exists():
            return RawExecutionResult(
                status="blocked",
                returncode=0,
                failure_reason="output_path_already_exists",
                warnings=self._warnings,
            )
        work_tmp.mkdir(parents=True, exist_ok=False)
        command = [
            str(Path(context.variables.get("python", "")).resolve())
            if context.variables.get("python")
            else sys.executable,
            str(patch_py),
            "--base_csv",
            str(base),
            "--oof_meta_csv",
            str(oof),
            "--test_meta_csv",
            str(test),
            "--out_dir",
            str(work_tmp),
            "--output_name",
            "replay_candidate.csv",
            "--minimum_support",
            str(int(gate["minimum_support"])),
            "--minimum_precision",
            str(float(gate["minimum_precision"])),
            "--minimum_net",
            str(int(gate["minimum_net"])),
            "--minimum_folds",
            str(int(gate["minimum_folds"])),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=context.project_root,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=context.tool.expected_runtime_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return RawExecutionResult(
                status="failed",
                returncode=-1,
                stdout=exc.stdout or "",
                stderr=exc.stderr or str(exc),
                failure_reason="patch_execution_timeout",
                warnings=self._warnings,
                details={"command": command},
            )

        if completed.returncode != 0:
            return RawExecutionResult(
                status="failed",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                failure_reason="patch_execution_failed",
                warnings=self._warnings,
                details={"command": command},
            )

        generated_csv = work_tmp / "replay_candidate.csv"
        if not generated_csv.exists():
            return RawExecutionResult(
                status="failed",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                failure_reason="missing_replay_candidate",
                warnings=self._warnings,
                details={"command": command},
            )

        metrics, artifacts, errors = self._validate_replay_outputs(
            context=context,
            base=base,
            champion=champion,
            oof=oof,
            test=test,
            work_tmp=work_tmp,
        )
        if errors:
            return RawExecutionResult(
                status="failed",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                metrics=metrics,
                failure_reason=errors[0],
                warnings=self._warnings + errors[1:],
                details={"command": command, "artifacts": artifacts},
            )

        candidate_dir.mkdir(parents=True, exist_ok=False)
        moved_artifacts: dict[str, str] = {}
        for name in PATCH_OUTPUTS:
            source = work_tmp / name
            if source.exists():
                target = candidate_dir / name
                shutil.move(str(source), str(target))
                moved_artifacts[name] = str(target)
        try:
            work_tmp.rmdir()
        except OSError:
            pass

        metrics["artifact_type"] = "prediction_candidate"
        metrics["registered_as_champion"] = False
        metrics["submission_ready"] = False
        stdout = json.dumps(
            {
                "adapter": "A1_V53Q1_PATCH_REPLAY_SAFE",
                "status": "completed",
                "base_to_replay_diff_count": metrics["base_to_replay_diff_count"],
                "semantic_replay_pass": metrics["semantic_replay_pass"],
                "byte_replay_pass": metrics["byte_replay_pass"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return RawExecutionResult(
            status="completed",
            returncode=completed.returncode,
            stdout=completed.stdout + "\n" + stdout + "\n",
            stderr=completed.stderr,
            metrics=metrics,
            warnings=self._warnings,
            details={"command": command, "artifacts": {**artifacts, **moved_artifacts}},
        )

    def _validate_replay_outputs(
        self,
        *,
        context: AdapterContext,
        base: Path,
        champion: Path,
        oof: Path,
        test: Path,
        work_tmp: Path,
    ) -> tuple[dict[str, Any], dict[str, str], list[str]]:
        generated_csv = work_tmp / "replay_candidate.csv"
        base_rows, base_errors = _load_prediction_csv(base, name="base_csv")
        replay_rows, replay_errors = _load_prediction_csv(generated_csv, name="replay_candidate")
        champion_rows, champion_errors = _load_prediction_csv(champion, name="champion_csv")
        errors = base_errors + replay_errors + champion_errors
        if errors:
            return {}, {}, ["replay_candidate_schema_failed", *errors]
        if [row["test_idx"] for row in base_rows] != [row["test_idx"] for row in replay_rows]:
            errors.append("base_to_replay_test_idx_mismatch")
        if [row["test_idx"] for row in champion_rows] != [row["test_idx"] for row in replay_rows]:
            errors.append("replay_to_champion_test_idx_mismatch")
        if errors:
            return {}, {}, errors

        base_to_replay, diff_errors = _diff_predictions(base_rows, replay_rows)
        replay_to_champion, champion_diff_errors = _diff_predictions(replay_rows, champion_rows)
        errors.extend(diff_errors)
        errors.extend(champion_diff_errors)

        gate = self._gate or _gate_from_registry(context)
        allowed_transitions, transition_summary = _recompute_allowed_transitions(oof, gate)
        audit_json = work_tmp / "v53q1_audit.json"
        patch_audit: dict[str, Any] = {}
        patch_allowed: set[str] = set()
        if audit_json.exists():
            patch_audit = json.loads(audit_json.read_text(encoding="utf-8"))
            patch_allowed = {
                _normal_transition(item)
                for item in patch_audit.get("allowed_transitions", [])
            }
            if patch_allowed != allowed_transitions:
                errors.append("patch_gate_audit_mismatch")
            audit_gate = patch_audit.get("transition_gate", {})
            if audit_gate and not _gate_equal(audit_gate, gate):
                errors.append("patch_gate_config_mismatch")

        selected_test = _selected_frame(test)
        applied_idx = {int(row["test_idx"]) for row in base_to_replay}
        non_applied_rows: list[dict[str, Any]] = []
        for row in selected_test.sort_values("global_idx").itertuples():
            idx = int(row.global_idx)
            transition = str(row.transition)
            if idx in applied_idx:
                continue
            if transition not in allowed_transitions:
                reason = "transition_not_allowed_by_gate"
            else:
                reason = "unverified"
            non_applied_rows.append(
                {
                    "test_idx": idx,
                    "base_label": int(row.base_pred),
                    "candidate_label": int(row.h2_pred),
                    "transition": transition,
                    "selected_in_test_meta": True,
                    "applied_in_replay": False,
                    "reason": reason,
                }
            )

        semantic_replay = _semantic_hash(replay_rows)
        semantic_champion = _semantic_hash(champion_rows)
        candidate_hash = sha256_file(generated_csv)
        champion_hash = sha256_file(champion)
        semantic_pass = semantic_replay == semantic_champion
        byte_pass = candidate_hash == champion_hash
        if not semantic_pass:
            errors.append("semantic_replay_mismatch")
        elif not byte_pass:
            errors.append("byte_level_replay_mismatch")

        artifact_paths = {
            "base_to_replay_diff": str(context.run_dir / "base_to_replay_diff.csv"),
            "replay_to_champion_diff": str(context.run_dir / "replay_to_champion_diff.csv"),
            "non_applied_test_selected_audit": str(context.run_dir / "non_applied_test_selected_audit.csv"),
            "replay_audit": str(context.run_dir / "replay_audit.json"),
        }
        _write_csv(
            Path(artifact_paths["base_to_replay_diff"]),
            base_to_replay,
            ["test_idx", "old_label", "new_label", "transition"],
        )
        _write_csv(
            Path(artifact_paths["replay_to_champion_diff"]),
            replay_to_champion,
            ["test_idx", "old_label", "new_label", "transition"],
        )
        _write_csv(
            Path(artifact_paths["non_applied_test_selected_audit"]),
            non_applied_rows,
            [
                "test_idx",
                "base_label",
                "candidate_label",
                "transition",
                "selected_in_test_meta",
                "applied_in_replay",
                "reason",
            ],
        )
        metrics = {
            "gate_config": gate,
            "allowed_transitions": sorted(allowed_transitions),
            "patch_allowed_transitions": sorted(patch_allowed),
            "gate_recomputed_pass": patch_allowed == allowed_transitions,
            "base_rows": len(base_rows),
            "replay_rows": len(replay_rows),
            "champion_rows": len(champion_rows),
            "base_to_replay_diff_count": len(base_to_replay),
            "replay_to_champion_differing_row_count": len(replay_to_champion),
            "base_to_replay_diff": base_to_replay,
            "replay_to_champion_diff": replay_to_champion,
            "final_patch_test_idx": [int(row["test_idx"]) for row in base_to_replay],
            "non_applied_test_selected_count": len(non_applied_rows),
            "non_applied_test_selected": non_applied_rows,
            "semantic_hash": semantic_replay,
            "champion_semantic_hash": semantic_champion,
            "semantic_replay_pass": semantic_pass,
            "candidate_sha256": candidate_hash,
            "champion_sha256": champion_hash,
            "byte_replay_pass": byte_pass,
            "patch_returned_test_changes": patch_audit.get("test_changes"),
            "patch_test_labels_used": bool(patch_audit.get("test_labels_used", False)),
            "patch_submission_created": bool(patch_audit.get("submission_created", False)),
        }
        if metrics["patch_test_labels_used"]:
            errors.append("patch_used_test_labels")
        if metrics["patch_submission_created"]:
            errors.append("patch_created_submission")

        replay_audit = {
            "adapter": "A1_V53Q1_PATCH_REPLAY_SAFE",
            "identity_hash": context.identity_hash,
            "artifact_type": "prediction_candidate",
            "registered_as_champion": False,
            "submission_ready": False,
            "metrics": metrics,
            "patch_audit": patch_audit,
        }
        Path(artifact_paths["replay_audit"]).write_text(
            _json_dumps(replay_audit) + "\n",
            encoding="utf-8",
        )
        return metrics, artifact_paths, errors

    def collect_outputs(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
    ) -> OutputManifest:
        artifacts = dict(raw_result.details.get("artifacts", {}))
        candidate_csv = context.run_dir / "candidate" / "replay_candidate.csv"
        if candidate_csv.exists():
            artifacts["candidate_csv"] = str(candidate_csv)
            artifacts["candidate_dir"] = str(candidate_csv.parent)
        return OutputManifest(artifacts=artifacts)

    def normalize_result(
        self,
        context: AdapterContext,
        raw_result: RawExecutionResult,
        outputs: OutputManifest,
    ) -> dict[str, Any]:
        return {
            "status": raw_result.status,
            "command": raw_result.details.get("command", []),
            "metrics": raw_result.metrics,
            "bucket_metrics": raw_result.bucket_metrics,
            "class_metrics": raw_result.class_metrics,
            "warnings": [],
            "failure_reason": raw_result.failure_reason,
        }
