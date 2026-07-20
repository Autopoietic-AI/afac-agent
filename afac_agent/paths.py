# -*- coding: utf-8 -*-
"""Project-root-relative path resolution for AFAC Agent M0."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_A1_ANCHOR_CSV = "artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv"


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
