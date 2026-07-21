"""LLM providers for M6A.

Only the mock provider is required for automated tests.  local_ollama is
best-effort and returns provider_unavailable when not explicitly configured.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import LLMRequest, LLMResponse


class MockProvider:
    def __init__(self, mode: str = "agree") -> None:
        self.mode = mode
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.mode == "unavailable":
            return LLMResponse(
                status="provider_unavailable",
                provider="mock",
                model="mock",
                failure_reason="mock_provider_unavailable",
            )
        if self.mode == "timeout":
            return LLMResponse(
                status="timeout",
                provider="mock",
                model="mock",
                failure_reason="mock_timeout",
            )
        if self.mode == "non_json":
            return LLMResponse(status="completed", provider="mock", model="mock", text="not json")
        if self.mode == "retry_json" and self.calls == 1:
            return LLMResponse(status="completed", provider="mock", model="mock", text="not json")
        evidence = request.metadata.get("evidence_bundle", {})
        deterministic = evidence.get("deterministic_plan", {})
        action = "request_missing_input"
        tool: str | None = None
        reason_codes = ["missing_input", "missing_full_anchor_inputs"]
        risk = "low"
        research_needed = False
        research_query = ""
        if self.mode == "research":
            research_needed = True
            research_query = (
                "Look for low-risk A1 isolated or exact-hop ideas that do not use "
                "test truth and do not reopen closed branches."
            )
        elif self.mode == "unsafe_submission":
            action = "online_submission"
            tool = "FINALIZE_CURRENT_CHAMPION"
            risk = "critical"
            reason_codes = ["submission_auto_forbidden"]
        elif self.mode == "champion_mutation":
            action = "modify_champion"
            risk = "critical"
            reason_codes = ["champion_mutation_forbidden"]
        elif self.mode == "unregistered_tool":
            action = "run_adapter"
            tool = "UNKNOWN_TOOL"
            risk = "medium"
            reason_codes = ["unregistered_tool"]
        elif self.mode == "training":
            action = "run_training"
            tool = "A1_TRAIN_NEW_MODEL"
            risk = "high"
            reason_codes = ["training_requires_human_approval"]
        elif self.mode == "prediction":
            action = "generate_candidate"
            tool = "A1_V53Q1_PATCH_REPLAY_SAFE"
            risk = "high"
            reason_codes = ["prediction_requires_human_approval"]
        elif self.mode == "correct_smooth":
            action = "reopen_branch"
            tool = None
            risk = "high"
            reason_codes = ["branch_closed", "correct_smooth_reopen_forbidden"]
        payload: dict[str, Any] = {
            "proposal_version": "m6a_v1",
            "proposal_id": "PENDING",
            "task": deterministic.get("task", "A1"),
            "status": "completed",
            "primary_problem": deterministic.get("primary_problem", ""),
            "problem_interpretation": "Full-anchor inputs are missing; deterministic plan should remain authoritative.",
            "proposed_action": action,
            "proposed_tool": tool,
            "proposed_tool_version": None,
            "evidence_refs": deterministic.get("evidence_refs", []),
            "feedback_refs": deterministic.get("feedback_refs", []),
            "required_inputs": deterministic.get("required_inputs", []),
            "missing_inputs": deterministic.get("missing_inputs", []),
            "expected_information_gain": "Clarify missing canonical Fold and final v53Q-1 OOF inputs.",
            "expected_model_gain_status": "unavailable_without_full_anchor_oof",
            "risk_level": risk,
            "uncertainties": ["final v53Q-1 OOF is unavailable"],
            "assumptions": ["M5A plan remains authoritative"],
            "stop_conditions": deterministic.get("stop_conditions", []),
            "success_conditions": deterministic.get("success_conditions", []),
            "failure_conditions": deterministic.get("failure_conditions", []),
            "reason_codes": reason_codes,
            "human_readable_rationale": "Shadow proposal agrees with waiting for explicit full-anchor inputs.",
            "requires_human_approval": False,
            "auto_execution_allowed": True,
            "research_needed": research_needed,
            "research_query": research_query,
        }
        return LLMResponse(
            status="completed",
            provider="mock",
            model=f"mock-{self.mode}",
            text=json.dumps(payload, ensure_ascii=False),
        )


class LocalOllamaProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def generate(self, request: LLMRequest) -> LLMResponse:
        base_url = str(self.config.get("base_url") or "").rstrip("/")
        model = str(self.config.get("model") or request.model or "")
        if not base_url or not model:
            return LLMResponse(
                status="provider_unavailable",
                provider="local_ollama",
                model=model,
                failure_reason="local_ollama_not_configured",
            )
        payload = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{base_url}/api/generate"
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=request.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except socket.timeout:
            return LLMResponse(status="timeout", provider="local_ollama", model=model, failure_reason="timeout")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return LLMResponse(
                status="provider_unavailable",
                provider="local_ollama",
                model=model,
                failure_reason=f"local_ollama_unavailable: {exc}",
            )
        return LLMResponse(
            status="completed",
            provider="local_ollama",
            model=model,
            text=str(body.get("response", "")),
        )


def load_provider_config(project_root: Path, provider_config: str = "") -> dict[str, Any]:
    candidates = []
    if provider_config:
        candidates.append(Path(provider_config))
    candidates.append(project_root / "config" / "llm.local.json")
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def make_provider(name: str, *, project_root: Path, provider_config: str = "", mock_mode: str = "agree"):
    if name == "mock":
        return MockProvider(mock_mode)
    if name == "local_ollama":
        return LocalOllamaProvider(load_provider_config(project_root, provider_config))
    raise ValueError(f"unsupported provider: {name}")
