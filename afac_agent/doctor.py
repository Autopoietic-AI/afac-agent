# -*- coding: utf-8 -*-
"""Environment and path preflight for AFAC Agent M0/M1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .paths import PathResolver
from .validation import (
    reports_to_checks,
    validate_a1_data_profile_dir,
    validate_a1_champion_csv,
    validate_memory_records_file,
    validate_project_state_file,
    validate_tool_registry_file,
    validate_trajectory_file,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_check(names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        result[name] = {
            "available": importlib.util.find_spec(name) is not None,
        }
    return result


def build_report(
    *,
    project_root: Path,
    paths_config: str = "",
) -> dict[str, Any]:
    resolver = PathResolver(project_root, paths_config or None)
    root = resolver.project_root
    champion_csv = resolver.a1_anchor_csv()
    reports = [
        validate_project_state_file(root / "config" / "project_state.json"),
        validate_tool_registry_file(root / "config" / "tool_registry.json"),
        validate_memory_records_file(root / "history" / "confirmed_experiments_a1.json"),
        validate_trajectory_file(root / "output" / "trajectory_A1.json"),
        validate_a1_champion_csv(champion_csv, expected_rows=2751, num_classes=10),
        validate_a1_data_profile_dir(resolver.a1_profile_out_dir()),
    ]
    checks = reports_to_checks(reports)
    try:
        project_state_payload = json.loads(
            (root / "config" / "project_state.json").read_text(encoding="utf-8")
        )
    except Exception:
        project_state_payload = {}
    profile_check = checks.get("a1_data_profile", {})
    profile_valid = bool(profile_check.get("details", {}).get("artifact_valid"))
    profile_input_hash_stale = False
    configured_npz = resolver.a1_npz()
    input_validation_path = resolver.a1_profile_out_dir() / "a1_input_validation.json"
    if profile_valid and configured_npz and configured_npz.exists() and input_validation_path.exists():
        try:
            input_validation = json.loads(
                input_validation_path.read_text(encoding="utf-8")
            )
            recorded_hash = input_validation.get("npz_path", {}).get("sha256")
            profile_input_hash_stale = recorded_hash != sha256(configured_npz)
        except Exception:
            profile_input_hash_stale = True
    if profile_input_hash_stale:
        profile_valid = False
        profile_check.setdefault("warnings", []).append(
            "legacy_data_profile_input_hash_stale"
        )
        profile_check.setdefault("details", {})[
            "legacy_data_profile_input_hash_stale"
        ] = True
    if project_state_payload.get("data_profile_ready") is True and not profile_valid:
        profile_check.setdefault("warnings", []).append(
            "legacy_data_profile_flag_stale"
        )
        profile_check.setdefault("details", {})[
            "legacy_data_profile_flag_stale"
        ] = True
        checks["a1_data_profile"] = profile_check
    if champion_csv.exists():
        checks["champion_csv"]["details"]["sha256"] = sha256(champion_csv)
    checks["paths"] = {
        "name": "paths",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": {
            "paths_config": str(resolver.paths_config),
            "paths_config_exists": resolver.paths_config.exists(),
            "a1_anchor_csv": str(champion_csv),
            "a1_dataset_npz": str(resolver.a1_npz() or ""),
            "a1_edges_csv": str(resolver.a1_edges_csv() or ""),
            "a1_fold_file": str(resolver.a1_fold_file() or ""),
            "a1_anchor_oof_npz": str(resolver.a1_anchor_oof_npz() or ""),
            "a1_reference_oof_npz": str(resolver.a1_reference_oof_npz() or ""),
            "a1_profile_out_dir": str(resolver.a1_profile_out_dir()),
        },
    }
    checks["writable"] = {
        "name": "writable",
        "passed": os.access(root, os.W_OK) and os.access(root / "artifacts", os.W_OK),
        "errors": [],
        "warnings": [],
        "details": {
            "project_root": os.access(root, os.W_OK),
            "artifacts": os.access(root / "artifacts", os.W_OK),
            "memory_parent": os.access(root, os.W_OK),
            "output_parent": os.access(root, os.W_OK),
        },
    }
    checks["dependencies"] = {
        "name": "dependencies",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": dependency_check(["numpy", "pandas", "scipy", "sklearn"]),
    }
    passed = all(item.get("passed", False) for item in checks.values())
    return {
        "project_root": str(root),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "checks": checks,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--paths_config", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(
        project_root=Path(args.project_root),
        paths_config=args.paths_config,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"AFAC Agent doctor: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"Project root: {report['project_root']}")
        for name, check in report["checks"].items():
            print(f"- {name}: {'PASS' if check['passed'] else 'FAIL'}")
            for error in check.get("errors", []):
                print(f"  ERROR: {error}")
            for warning in check.get("warnings", []):
                print(f"  WARN: {warning}")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
