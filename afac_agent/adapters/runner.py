# -*- coding: utf-8 -*-
"""Safe execution wrapper for registered AFAC Agent adapters."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from afac_agent.adapters.base import AdapterContext
from afac_agent.schemas import ToolSpec


ALLOWED_ADAPTER_ENTRYPOINTS = {
    "afac_agent.adapters.a1_v53q1_patch_audit:Adapter",
    "afac_agent.adapters.a1_v46a1_isolated_audit:Adapter",
    "afac_agent.adapters.a1_v49a_edge_utility_audit:Adapter",
}
OPTIONAL_ADAPTER_INPUT_KEYS_BY_TOOL = {
    "A1_V53Q1_PATCH_AUDIT": {
        "v53q1_patch_py",
    },
    "A1_V46A1_ISOLATED_AUDIT": {
        "parent_csv",
        "candidate_oof_npz",
        "audit_report",
        "current_champion_csv",
    },
    "A1_V49A_EDGE_UTILITY_AUDIT": {
        "v46a1_base_csv",
        "current_champion_csv",
        "v53q1_audit_md",
        "v53q1_patch_source",
        "v49a_report",
        "v49a_config",
        "v49a_fold_results",
    },
}
ENTRYPOINT_PATTERN = re.compile(
    r"^afac_agent\.adapters\.[A-Za-z_][A-Za-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*$"
)
VOLATILE_IDENTITY_KEYS = {
    "started_at",
    "finished_at",
    "duration_seconds",
    "stdout_log",
    "stderr_log",
    "run_dir",
    "output_root",
    "absolute_output_root",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _stable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items())
            if str(key) not in VOLATILE_IDENTITY_KEYS
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not allowed in identity hash")
        return round(value, 12)
    return value


class AdapterRunner:
    def __init__(self, *, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def _standard_result(
        self,
        *,
        tool: ToolSpec,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_seconds: float = 0.0,
        identity_hash: str = "",
        command: List[str] | None = None,
        returncode: int | None = None,
        input_manifest: Dict[str, Any] | None = None,
        input_hashes: Dict[str, str] | None = None,
        metrics: Dict[str, Any] | None = None,
        bucket_metrics: List[Dict[str, Any]] | None = None,
        class_metrics: List[Dict[str, Any]] | None = None,
        artifacts: Dict[str, str] | None = None,
        stdout_log: str = "",
        stderr_log: str = "",
        warnings: List[str] | None = None,
        failure_reason: str = "",
        missing_inputs: List[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "tool_name": tool.name,
            "adapter_id": tool.adapter_id or tool.name,
            "adapter_version": tool.adapter_version or "",
            "status": status,
            "target_problem": tool.name,
            "execution_mode": tool.execution_mode or tool.action_type,
            "read_only": tool.read_only,
            "counts_as_experiment_round": tool.counts_as_experiment_round,
            "mutates_predictions": tool.mutates_predictions,
            "mutates_project_state": tool.mutates_project_state,
            "requires_gpu": tool.requires_gpu,
            "identity_hash": identity_hash,
            "started_at": started_at or "",
            "finished_at": finished_at or "",
            "duration_seconds": round(float(duration_seconds), 6),
            "command": command or [],
            "returncode": returncode,
            "input_manifest": input_manifest or {},
            "input_hashes": input_hashes or {},
            "metrics": metrics or {},
            "bucket_metrics": bucket_metrics or [],
            "class_metrics": class_metrics or [],
            "artifacts": artifacts or {},
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "warnings": warnings or [],
            "failure_reason": failure_reason,
            "missing_inputs": missing_inputs or [],
        }

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _resolve_output_root(self, tool: ToolSpec) -> Path:
        policy = tool.output_policy or {}
        root = str(policy.get("output_root", "artifacts/adapter_runs")).strip()
        return self._resolve_path(root)

    def _missing_inputs(
        self,
        tool: ToolSpec,
        variables: Dict[str, str],
    ) -> tuple[List[str], Dict[str, str]]:
        missing: List[str] = []
        missing_files: Dict[str, str] = {}
        for key, spec in (tool.required_inputs or {}).items():
            value = str(variables.get(key, "")).strip()
            if not value:
                missing.append(key)
                continue
            kind = spec.get("kind", "") if isinstance(spec, dict) else spec
            if kind == "file" and not self._resolve_path(value).exists():
                missing.append(key)
                missing_files[key] = value
        return missing, missing_files

    def _input_paths(
        self,
        tool: ToolSpec,
        variables: Dict[str, str],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        keys = set((tool.required_inputs or {}).keys())
        optional_keys = OPTIONAL_ADAPTER_INPUT_KEYS_BY_TOOL.get(tool.name, set())
        keys.update(
            key
            for key in optional_keys
            if variables.get(key)
        )
        for key in sorted(keys):
            value = str(variables.get(key, "")).strip()
            if value:
                paths[key] = self._resolve_path(value)
        return paths

    def hash_inputs(
        self,
        keys: Iterable[str],
        variables: Dict[str, str],
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for key in keys:
            value = str(variables.get(key, "")).strip()
            if not value:
                continue
            path = self._resolve_path(value)
            if path.exists() and path.is_file():
                result[key] = sha256_file(path)
        return result

    def compute_identity_hash(
        self,
        *,
        tool: ToolSpec,
        normalized_config: Dict[str, Any],
        input_hashes: Dict[str, str],
        output_policy: Dict[str, Any],
        frozen_gate_config: Dict[str, Any],
    ) -> str:
        payload = {
            "tool_name": tool.name,
            "adapter_id": tool.adapter_id or tool.name,
            "adapter_version": tool.adapter_version or "",
            "execution_mode": tool.execution_mode or tool.action_type,
            "normalized_config": _stable_value(normalized_config),
            "input_hashes": _stable_value(input_hashes),
            "output_policy": _stable_value(output_policy),
            "frozen_gate_config": _stable_value(frozen_gate_config),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_adapter(self, tool: ToolSpec):
        entrypoint = str(tool.adapter_entrypoint or "").strip()
        if not entrypoint:
            return None, self._standard_result(
                tool=tool,
                status="blocked",
                failure_reason="missing_adapter_entrypoint",
            )
        if (
            not ENTRYPOINT_PATTERN.match(entrypoint)
            or entrypoint not in ALLOWED_ADAPTER_ENTRYPOINTS
        ):
            return None, self._standard_result(
                tool=tool,
                status="blocked",
                failure_reason="unregistered_adapter_entrypoint",
            )
        module_name, class_name = entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
        return adapter_class(), None

    def _frozen_hashes(self) -> Dict[str, str]:
        relative_paths = [
            "artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv",
            "config/project_state.json",
            "history/confirmed_experiments_a1.json",
        ]
        hashes: Dict[str, str] = {}
        for relative in relative_paths:
            path = self.project_root / relative
            if path.exists():
                hashes[relative] = sha256_file(path)
        return hashes

    def _check_output_safety(
        self,
        *,
        tool: ToolSpec,
        output_root: Path,
        input_paths: Dict[str, Path],
    ) -> str:
        resolved_output = output_root.resolve()
        input_values = {path.resolve() for path in input_paths.values()}
        if resolved_output in input_values:
            return "unsafe_output_path"
        input_dirs = {path.resolve().parent for path in input_paths.values()}
        if resolved_output in input_dirs:
            return "unsafe_output_path"
        for forbidden in (tool.output_policy or {}).get("forbidden_paths", []):
            forbidden_path = self._resolve_path(str(forbidden)).resolve()
            if resolved_output == forbidden_path:
                return "unsafe_output_path"
        champion = (
            self.project_root
            / "artifacts"
            / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
        ).resolve()
        if resolved_output == champion:
            return "unsafe_output_path"
        return ""

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")

    def _duplicate_result(
        self,
        *,
        tool: ToolSpec,
        result_path: Path,
        identity_hash: str,
        input_hashes: Dict[str, str],
    ) -> Dict[str, Any] | None:
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if previous.get("identity_hash") != identity_hash:
            return None
        if previous.get("input_hashes") != input_hashes:
            return None
        if previous.get("status") not in {"completed", "duplicate"}:
            return None
        duplicate = dict(previous)
        duplicate["status"] = "duplicate"
        duplicate["artifacts"] = dict(previous.get("artifacts", {}))
        duplicate["artifacts"]["existing_execution_result"] = str(result_path)
        duplicate["tool_name"] = tool.name
        return duplicate

    def _forbidden_prediction_artifacts(self, run_dir: Path) -> List[str]:
        patterns = [
            "A1*.csv",
            "A2*.csv",
            "submission*.zip",
            "*.npz",
            "*.pt",
            "*.pth",
            "*.ckpt",
        ]
        paths: list[str] = []
        for pattern in patterns:
            paths.extend(str(path) for path in run_dir.rglob(pattern))
        return sorted(paths)

    def run(
        self,
        *,
        tool: ToolSpec,
        variables: Dict[str, str],
        execute: bool,
    ) -> Dict[str, Any]:
        adapter, blocked = self._load_adapter(tool)
        if blocked is not None:
            return blocked

        missing, missing_files = self._missing_inputs(tool, variables)
        if missing:
            return self._standard_result(
                tool=tool,
                status="waiting_for_input",
                missing_inputs=missing,
                failure_reason="missing_required_inputs",
                input_manifest={"missing_files": missing_files},
            )

        input_paths = self._input_paths(tool, variables)
        input_hashes = {
            key: sha256_file(path)
            for key, path in sorted(input_paths.items())
            if path.exists() and path.is_file()
        }
        if tool.name == "A1_V53Q1_PATCH_AUDIT":
            normalized_config = {
                "minimum_support": 3,
                "minimum_precision": round(2.0 / 3.0, 12),
                "minimum_net": 1,
                "minimum_folds": 2,
            }
            frozen_gate_config = normalized_config
        else:
            normalized_config = {
                "adapter_id": tool.adapter_id or tool.name,
                "adapter_version": tool.adapter_version or "",
                "execution_mode": tool.execution_mode or tool.action_type,
            }
            frozen_gate_config = {}
        output_policy = {
            "allow_overwrite": bool(
                (tool.output_policy or {}).get("allow_overwrite", False)
            ),
            "output_family": "adapter_runs",
        }
        identity_hash = self.compute_identity_hash(
            tool=tool,
            normalized_config=normalized_config,
            input_hashes=input_hashes,
            output_policy=output_policy,
            frozen_gate_config=frozen_gate_config,
        )
        output_root = self._resolve_output_root(tool)
        unsafe_reason = self._check_output_safety(
            tool=tool,
            output_root=output_root,
            input_paths=input_paths,
        )
        if unsafe_reason:
            return self._standard_result(
                tool=tool,
                status="blocked",
                identity_hash=identity_hash,
                input_hashes=input_hashes,
                failure_reason=unsafe_reason,
            )
        if not execute:
            return self._standard_result(
                tool=tool,
                status="dry_run",
                identity_hash=identity_hash,
                input_hashes=input_hashes,
                input_manifest={
                    key: str(path)
                    for key, path in sorted(input_paths.items())
                },
            )

        run_dir = output_root / tool.name / identity_hash
        result_path = run_dir / "execution_result.json"
        if result_path.exists():
            duplicate = self._duplicate_result(
                tool=tool,
                result_path=result_path,
                identity_hash=identity_hash,
                input_hashes=input_hashes,
            )
            if duplicate is not None:
                return duplicate

        run_dir.mkdir(parents=True, exist_ok=True)
        context = AdapterContext(
            project_root=self.project_root,
            tool=tool,
            variables=variables,
            run_dir=run_dir,
            identity_hash=identity_hash,
        )

        validation = adapter.validate_inputs(context)
        if not validation.passed:
            result = self._standard_result(
                tool=tool,
                status="failed",
                identity_hash=identity_hash,
                input_hashes=input_hashes,
                input_manifest={
                    key: str(path)
                    for key, path in sorted(input_paths.items())
                },
                warnings=validation.warnings,
                failure_reason="input_validation_failed",
            )
            self._write_json(run_dir / "execution_result.json", result)
            return result

        frozen_before = self._frozen_hashes()
        started_clock = time.time()
        started_at = _utc_now()
        raw = adapter.execute(context)
        finished_at = _utc_now()
        duration = time.time() - started_clock
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        stdout_path.write_text(raw.stdout, encoding="utf-8")
        stderr_path.write_text(raw.stderr, encoding="utf-8")
        outputs = adapter.collect_outputs(context, raw)
        normalized = adapter.normalize_result(context, raw, outputs)
        frozen_after = self._frozen_hashes()

        status = normalized.get("status", raw.status)
        failure_reason = normalized.get("failure_reason", raw.failure_reason)
        warnings = list(raw.warnings) + list(normalized.get("warnings", []))
        forbidden_artifacts = []
        if not tool.mutates_predictions:
            forbidden_artifacts = self._forbidden_prediction_artifacts(run_dir)
            if forbidden_artifacts:
                status = "failed"
                failure_reason = "prediction_artifact_forbidden"
                warnings.append("mutates_predictions=false but prediction artifact was generated")
        if tool.read_only and frozen_after != frozen_before:
            status = "failed"
            failure_reason = "read_only_contract_violated"
            warnings.append("frozen file hash changed during read-only adapter run")

        result = self._standard_result(
            tool=tool,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            identity_hash=identity_hash,
            command=[],
            returncode=raw.returncode,
            input_manifest={
                key: str(path)
                for key, path in sorted(input_paths.items())
            },
            input_hashes=input_hashes,
            metrics=normalized.get("metrics", raw.metrics),
            bucket_metrics=normalized.get("bucket_metrics", raw.bucket_metrics),
            class_metrics=normalized.get("class_metrics", raw.class_metrics),
            artifacts={
                **outputs.artifacts,
                "run_dir": str(run_dir),
                "execution_result": str(result_path),
                "forbidden_prediction_artifacts": forbidden_artifacts,
            },
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            warnings=warnings,
            failure_reason=failure_reason,
            missing_inputs=raw.missing_inputs,
        )
        self._write_json(run_dir / "input_manifest.json", result["input_manifest"])
        self._write_json(run_dir / "execution_result.json", result)
        return result
