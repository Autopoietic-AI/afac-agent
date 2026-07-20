# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_m2_a1_data_profiler import _write_npz


def test_profile_cli_dataset_only_smoke(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "profile out"
    _write_npz(npz)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "afac_agent.profilers.a1_data_profiler",
            "--npz_path",
            str(npz),
            "--out_dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["analysis_tier"] == "dataset_only"
    assert (out_dir / "a1_data_profile.json").exists()


def test_legacy_tool_wrapper_delegates_to_profiler(tmp_path: Path) -> None:
    npz = tmp_path / "a1.npz"
    out_dir = tmp_path / "wrapper out"
    _write_npz(npz)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/profile_a1_dataset.py",
            "--npz_path",
            str(npz),
            "--out_dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["analysis_tier"] == "dataset_only"
    assert (out_dir / "a1_problem_map.json").exists()
