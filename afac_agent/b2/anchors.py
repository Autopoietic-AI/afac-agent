# -*- coding: utf-8 -*-
"""B2 evaluation and online anchor materialization.

- ``B2_EVAL_ANCHOR_V1`` is materialized by the closed loop from the best
  complete OOF retrieval candidate.
- ``B2_ONLINE_ANCHOR`` records the current online submission state and is
  updated to ``submitted`` only by an explicit human action.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..research.event_store import json_dumps, rel_ref, sha256_file, stable_hash
from .fold import AFAC_B2_FOLD_V1

B2_EVAL_ANCHOR_ID = "B2_EVAL_ANCHOR_V1"
B2_ONLINE_ANCHOR_ID = "B2_ONLINE_ANCHOR"


def materialize_evaluation_anchor(
    out_dir: Path,
    candidate: dict[str, Any],
    folds: Any,
    input_hashes: dict[str, str],
    project_root: Path,
) -> dict[str, Any]:
    """Write the B2 evaluation-anchor OOF artifact and manifest."""
    anchor_dir = out_dir / B2_EVAL_ANCHOR_ID
    anchor_dir.mkdir(parents=True, exist_ok=True)
    oof_path = anchor_dir / f"{B2_EVAL_ANCHOR_ID}_oof.npz"
    np.savez(
        oof_path,
        uids=folds.uids,
        topk=candidate["oof_topk"],
        fold=folds.folds,
    )
    manifest = {
        "anchor_id": B2_EVAL_ANCHOR_ID,
        "source_candidate_id": candidate["candidate_id"],
        "status": "materialized",
        "fold_protocol": AFAC_B2_FOLD_V1,
        "deployment_equivalent": False,
        "oof_path": rel_ref(oof_path, project_root),
        "oof_sha256": sha256_file(oof_path),
        "metrics": candidate["metrics"],
        "input_hashes": input_hashes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = anchor_dir / f"{B2_EVAL_ANCHOR_ID}_manifest.json"
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def materialize_online_anchor(
    out_dir: Path,
    candidate_id: str,
    candidate_path: Path,
    status: str = "no_submission_yet",
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Record the online anchor state for the B2 submission."""
    anchor_dir = out_dir / B2_ONLINE_ANCHOR_ID
    anchor_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "anchor_id": B2_ONLINE_ANCHOR_ID,
        "candidate_id": candidate_id,
        "status": status,
        "candidate_path": rel_ref(candidate_path, project_root) if project_root else str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "submitted_at": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = anchor_dir / f"{B2_ONLINE_ANCHOR_ID}_manifest.json"
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest
