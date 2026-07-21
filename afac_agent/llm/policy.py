"""Conservative M6A shadow-policy loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import sha256_file


FORBIDDEN_ACTIONS = {
    "online_submission",
    "submit_online",
    "modify_champion",
    "register_champion",
    "overwrite_champion",
}


def load_llm_shadow_policy(policy_path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    required = [
        "allowed_actions",
        "forbidden_actions",
        "allowed_tool_risk_levels",
        "max_calls",
        "timeout_seconds",
        "max_prompt_chars",
        "max_output_chars",
        "allow_research_request",
        "allow_training_proposal",
        "allow_prediction_proposal",
        "force_human_approval_for_training",
        "force_human_approval_for_prediction",
        "force_auto_execution_false",
        "prompt_injection_policy",
        "raw_response_retention_policy",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"llm shadow policy missing fields: {', '.join(missing)}")
    if payload.get("force_auto_execution_false") is not True:
        raise ValueError("force_auto_execution_false must be true")
    if int(payload.get("max_calls", 0)) > 2:
        raise ValueError("max_calls must not exceed 2")
    if int(payload.get("timeout_seconds", 0)) > 120:
        raise ValueError("timeout_seconds must not exceed 120")
    allowed = set(map(str, payload.get("allowed_actions", [])))
    forbidden = set(map(str, payload.get("forbidden_actions", [])))
    unsafe_allowed = allowed & FORBIDDEN_ACTIONS
    if unsafe_allowed:
        raise ValueError(f"unsafe actions cannot be allowed: {sorted(unsafe_allowed)}")
    if not FORBIDDEN_ACTIONS <= forbidden:
        raise ValueError("online submission and champion mutation actions must be forbidden")
    if payload.get("force_human_approval_for_training") is not True:
        raise ValueError("training proposals must require human approval")
    if payload.get("force_human_approval_for_prediction") is not True:
        raise ValueError("prediction proposals must require human approval")
    return payload, sha256_file(policy_path)
