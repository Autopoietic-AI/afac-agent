# -*- coding: utf-8 -*-
"""Project-root-relative path resolution for AFAC Agent M0."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_A1_ANCHOR_CSV = "artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv"
DEFAULT_A1_PROFILE_OUT_DIR = "artifacts/data_profile/a1_m2_v1"
DEFAULT_ADAPTER_OUTPUT_ROOT = "artifacts/adapter_runs"
DEFAULT_A1_V53Q1_AUDIT_MD = "artifacts/V53Q1_TRANSITION_STABLE_EDGE_H2_AUDIT.md"
DEFAULT_A1_V53Q1_PATCH_PY = "artifacts/a1_v53q1_transition_stable_edge_h2_patch.py"


def find_project_root(path: str | Path = ".") -> Path:
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    for candidate in [root, *root.parents]:
        if (
            (candidate / "config" / "tool_registry.json").exists()
            and (candidate / "afac_agent").is_dir()
        ):
            return candidate
    return root


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if (
        len(value) >= 2
        and value[0] in {"'", '"'}
        and value[-1] == value[0]
    ):
        return value[1:-1]
    return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Parse the small paths.local YAML shape without adding PyYAML."""

    path = Path(path)
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = _parse_scalar(value)
        if raw_line.startswith(" ") and current_section:
            result.setdefault(current_section, {})[key] = value
        else:
            result[key] = value
            current_section = None
    return result


class PathResolver:
    def __init__(
        self,
        project_root: str | Path = ".",
        paths_config: str | Path | None = None,
    ):
        self.project_root = find_project_root(project_root)
        default_config = self.project_root / "config" / "paths.local.yaml"
        self.paths_config = Path(paths_config).resolve() if paths_config else default_config
        self.config = load_simple_yaml(self.paths_config)

    def resolve(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        path = Path(text)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def get(self, section: str, key: str, default: str = "") -> Path | None:
        section_payload = self.config.get(section, {})
        if isinstance(section_payload, dict):
            value = section_payload.get(key, default)
        else:
            value = default
        return self.resolve(value)

    def a1_anchor_csv(self, override: str = "") -> Path:
        return (
            self.resolve(override)
            or self.get("a1", "anchor_csv", DEFAULT_A1_ANCHOR_CSV)
            or (self.project_root / DEFAULT_A1_ANCHOR_CSV)
        )

    def a1_npz(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get("a1", "dataset_npz", "")

    def a1_anchor_oof_npz(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get("a1", "anchor_oof_npz", "")

    def a1_reference_oof_npz(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get("a1", "reference_oof_npz", "")

    def a1_edges_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get("a1", "edges_csv", "")

    def a1_fold_file(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get("a1", "fold_file", "")

    def a1_profile_out_dir(self, override: str = "") -> Path:
        return (
            self.resolve(override)
            or self.get("runtime", "profile_out_dir", DEFAULT_A1_PROFILE_OUT_DIR)
            or (self.project_root / DEFAULT_A1_PROFILE_OUT_DIR)
        )

    def adapter_output_root(self, override: str = "") -> Path:
        return (
            self.resolve(override)
            or self.get("runtime", "adapter_output_root", DEFAULT_ADAPTER_OUTPUT_ROOT)
            or (self.project_root / DEFAULT_ADAPTER_OUTPUT_ROOT)
        )

    def a1_v53q1_base_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v53q1_patch_audit",
            "base_csv",
            "",
        )

    def a1_v49a_edge_oof_meta_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v53q1_patch_audit",
            "oof_meta_csv",
            "",
        )

    def a1_v49a_edge_test_meta_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v53q1_patch_audit",
            "test_meta_csv",
            "",
        )

    def a1_v53q1_audit_md(self, override: str = "") -> Path:
        return (
            self.resolve(override)
            or self.get("a1_v53q1_patch_audit", "audit_md", DEFAULT_A1_V53Q1_AUDIT_MD)
            or (self.project_root / DEFAULT_A1_V53Q1_AUDIT_MD)
        )

    def a1_v53q1_patch_py(self, override: str = "") -> Path:
        return (
            self.resolve(override)
            or self.get("a1_v53q1_patch_audit", "patch_py", DEFAULT_A1_V53Q1_PATCH_PY)
            or (self.project_root / DEFAULT_A1_V53Q1_PATCH_PY)
        )

    def a1_v46a1_candidate_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v46a1_isolated_audit",
            "candidate_csv",
            "",
        )

    def a1_v46a1_parent_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v46a1_isolated_audit",
            "parent_csv",
            "",
        )

    def a1_v46a1_candidate_oof_npz(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v46a1_isolated_audit",
            "candidate_oof_npz",
            "",
        )

    def a1_v46a1_audit_report(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v46a1_isolated_audit",
            "audit_report",
            "",
        )

    def a1_current_champion_csv(self, override: str = "") -> Path:
        return self.a1_anchor_csv(override)

    def a1_v49a_oof_meta_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v49a_edge_utility_audit",
            "oof_meta_csv",
            "",
        )

    def a1_v49a_test_meta_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v49a_edge_utility_audit",
            "test_meta_csv",
            "",
        )

    def a1_v49a_v46a1_base_csv(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v49a_edge_utility_audit",
            "v46a1_base_csv",
            "",
        )

    def a1_v49a_report(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v49a_edge_utility_audit",
            "report",
            "",
        )

    def a1_v49a_config(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v49a_edge_utility_audit",
            "config",
            "",
        )

    def a1_v49a_fold_results(self, override: str = "") -> Path | None:
        return self.resolve(override) or self.get(
            "a1_v49a_edge_utility_audit",
            "fold_results",
            "",
        )
