# -*- coding: utf-8 -*-
"""AFAC v2.0 execution identity: input fingerprint vs execution id.

``input_fingerprint`` identifies identical inputs and deterministic
configuration; safe deterministic caches may be reused under the same
fingerprint.  ``execution_id`` identifies one concrete run instance and must
be unique for every real execution — it additionally mixes in the code
commit, branch, orchestrator/planner versions, contract hashes, an execution
nonce and the start time.  A v1 ``run_id`` (data-hash only) can therefore
never collide with a v2 ``execution_id``.

API keys and secrets never enter any hash or log field.
"""
from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from ..research.event_store import stable_hash

IDENTITY_VERSION = "afac_v2_execution_identity_v1"
ORCHESTRATOR_VERSION = "2.0.0"
PLANNER_VERSION = "v2_planner_1"


def code_commit(project_root: str | Path) -> str:
    """Best-effort current git commit; ``unknown`` outside a git worktree."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        commit = proc.stdout.strip()
        return commit if proc.returncode == 0 and commit else "unknown"
    except Exception:
        return "unknown"


def git_branch(project_root: str | Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        branch = proc.stdout.strip()
        return branch if proc.returncode == 0 and branch else "unknown"
    except Exception:
        return "unknown"


def build_input_fingerprint(
    *,
    data_hash: str,
    fold_hash: str,
    deterministic_config_hash: str,
    metric_contract_hash: str,
) -> dict[str, Any]:
    """Fingerprint of identical inputs + deterministic configuration."""
    components = {
        "data_hash": data_hash,
        "fold_hash": fold_hash,
        "deterministic_config_hash": deterministic_config_hash,
        "metric_contract_hash": metric_contract_hash,
    }
    return {
        "identity_version": IDENTITY_VERSION,
        "input_fingerprint": stable_hash({"kind": "input_fingerprint", **components}),
        "components": components,
    }


def build_execution_id(
    *,
    input_fingerprint: str,
    project_root: str | Path,
    planner_policy_hash: str,
    capability_registry_hash: str,
    metric_contract_hash: str,
    validation_contract_hash: str,
    llm_provider_config_hash: str,
    orchestrator_version: str = ORCHESTRATOR_VERSION,
    planner_version: str = PLANNER_VERSION,
    execution_nonce: str | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Unique-per-execution identity.

    Every call without an explicit ``execution_nonce`` draws a fresh uuid4,
    so two executions over identical inputs still receive distinct
    ``execution_id`` values.
    """
    nonce = execution_nonce or uuid.uuid4().hex
    started = float(started_at if started_at is not None else time.time())
    commit = code_commit(project_root)
    branch = git_branch(project_root)
    components = {
        "input_fingerprint": input_fingerprint,
        "code_commit": commit,
        "branch": branch,
        "orchestrator_version": orchestrator_version,
        "planner_version": planner_version,
        "planner_policy_hash": planner_policy_hash,
        "capability_registry_hash": capability_registry_hash,
        "metric_contract_hash": metric_contract_hash,
        "validation_contract_hash": validation_contract_hash,
        "llm_provider_config_hash": llm_provider_config_hash,
        "execution_nonce": nonce,
        "started_at": started,
    }
    return {
        "identity_version": IDENTITY_VERSION,
        "execution_id": stable_hash({"kind": "execution_id", **components}),
        "components": components,
    }


def llm_provider_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Sanitized provider config view that is safe to hash and log.

    Only provider type, model name, endpoint class, temperature, max tokens
    and planner policy version are included — never keys, URLs with
    credentials, or environment variable values.
    """
    base_url = str(config.get("base_url") or config.get("base_url_env") or "")
    endpoint_class = "env_var_reference" if base_url.endswith("_URL") or not base_url.startswith("http") else "https_endpoint"
    return {
        "provider": str(config.get("provider") or "aliyun_bailian_openai"),
        "model": str(config.get("model") or ""),
        "endpoint_class": endpoint_class,
        "temperature": float(config.get("temperature", 0.0)),
        "max_output_tokens": int(config.get("max_output_tokens", 1024)),
        "planner_policy_version": PLANNER_VERSION,
    }
