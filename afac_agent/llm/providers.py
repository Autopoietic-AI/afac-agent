"""LLM providers for M6A.

Only the mock provider is required for automated tests.  local_ollama is
best-effort and returns provider_unavailable when not explicitly configured.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base import LLMRequest, LLMResponse
from .utils import stable_hash


ALIYUN_BAILIAN_PROVIDER = "aliyun_bailian_openai"
ALIYUN_BAILIAN_DEFAULT_MODEL = "qwen3.6-max-preview"
ALIYUN_BAILIAN_ALLOWED_PREFIXES = ("qwen3.5-", "qwen3.6-")
DEFAULT_BAILIAN_CONFIG: dict[str, Any] = {
    "provider": ALIYUN_BAILIAN_PROVIDER,
    "model": ALIYUN_BAILIAN_DEFAULT_MODEL,
    "api_key_env": "DASHSCOPE_API_KEY",
    "base_url_env": "AFAC_BAILIAN_BASE_URL",
    "timeout_seconds": 180,
    "max_retries": 1,
    "max_output_tokens": 4096,
    "temperature": 0,
    "enable_thinking": True,
    "response_format": "json_object",
}


def is_allowed_bailian_model(model: str) -> bool:
    return str(model).startswith(ALIYUN_BAILIAN_ALLOWED_PREFIXES)


def redact_secret(text: Any) -> str:
    value = str(text)
    env_value = os.environ.get("DASHSCOPE_API_KEY", "")
    if env_value:
        value = value.replace(env_value, "[REDACTED_DASHSCOPE_API_KEY]")
    value = re.sub(r"sk-[A-Za-z0-9._-]+", "sk-[REDACTED]", value)
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        value,
    )
    if len(value) > 500:
        return value[:500] + "...[TRUNCATED]"
    return value


def _base_url_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    return parsed.netloc or ""


def _validate_api_key_shape(api_key: str) -> str:
    if not api_key or not api_key.strip():
        return "missing_api_key_environment_variable"
    if api_key != api_key.strip():
        return "api_key_has_surrounding_whitespace"
    if "\n" in api_key or "\r" in api_key:
        return "api_key_contains_newline"
    if '"' in api_key or "'" in api_key:
        return "api_key_contains_quote"
    return ""


def _sanitize_config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in sorted(config.items()):
        lower = str(key).lower()
        if "key" in lower or "secret" in lower or "token" in lower:
            if lower in {"api_key_env", "base_url_env"}:
                sanitized[str(key)] = value
            else:
                sanitized[str(key)] = "[REDACTED]"
            continue
        if lower == "base_url":
            sanitized[str(key)] = {"host": _base_url_host(str(value))}
            continue
        sanitized[str(key)] = value
    return sanitized


def provider_config_identity(project_root: Path, provider_config: str = "") -> dict[str, Any]:
    config = load_provider_config(project_root, provider_config)
    if config.get("provider") == ALIYUN_BAILIAN_PROVIDER:
        merged = {**DEFAULT_BAILIAN_CONFIG, **config}
        return {
            "provider": ALIYUN_BAILIAN_PROVIDER,
            "config": _sanitize_config_for_hash(merged),
            "api_key_env_present": bool(os.environ.get(str(merged.get("api_key_env")))),
            "base_url_env_present": bool(os.environ.get(str(merged.get("base_url_env")))),
        }
    if config:
        return {"provider_config": _sanitize_config_for_hash(config)}
    return {"provider_config": "unavailable"}


def provider_config_hash(project_root: Path, provider_config: str = "") -> str:
    return stable_hash(provider_config_identity(project_root, provider_config))


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


class AliyunBailianOpenAIProvider:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        client_factory: Any | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.config = {**DEFAULT_BAILIAN_CONFIG, **config}
        self.client_factory = client_factory
        self.environ = environ if environ is not None else os.environ

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = str(request.model or self.config.get("model") or ALIYUN_BAILIAN_DEFAULT_MODEL)
        audit = self._base_audit(model=model)
        if not is_allowed_bailian_model(model):
            return LLMResponse(
                status="provider_unavailable",
                provider=ALIYUN_BAILIAN_PROVIDER,
                model=model,
                failure_reason="configured_model_not_allowed",
                audit=audit,
            )
        api_key_env = str(self.config.get("api_key_env") or "DASHSCOPE_API_KEY")
        base_url_env = str(self.config.get("base_url_env") or "AFAC_BAILIAN_BASE_URL")
        api_key = self.environ.get(api_key_env, "")
        base_url = self.environ.get(base_url_env, "")
        key_shape_error = _validate_api_key_shape(api_key)
        if key_shape_error:
            return LLMResponse(
                status="provider_unavailable",
                provider=ALIYUN_BAILIAN_PROVIDER,
                model=model,
                failure_reason=key_shape_error,
                audit=audit,
            )
        if not base_url:
            return LLMResponse(
                status="provider_unavailable",
                provider=ALIYUN_BAILIAN_PROVIDER,
                model=model,
                failure_reason="missing_base_url_environment_variable",
                audit=audit,
            )
        audit["base_url_host"] = _base_url_host(base_url)
        try:
            factory = self.client_factory or self._load_openai_client
            client = factory(
                api_key=api_key,
                base_url=base_url,
                timeout=float(self.config.get("timeout_seconds", request.timeout_seconds)),
                max_retries=int(self.config.get("max_retries", 1)),
            )
        except Exception as exc:
            return LLMResponse(
                status="provider_unavailable",
                provider=ALIYUN_BAILIAN_PROVIDER,
                model=model,
                failure_reason=f"openai_sdk_unavailable: {redact_secret(exc)}",
                audit=audit,
            )

        if request.metadata.get("provider_check") is True:
            return self._generate_provider_check(client=client, request=request, model=model, audit=audit)

        response_format = self._response_format()
        messages = [
            {
                "role": "system",
                "content": "You are an AFAC shadow planner. Return only one valid JSON object. Do not use Markdown code fences.",
            },
            {"role": "user", "content": request.prompt},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": float(self.config.get("temperature", request.temperature)),
            "max_tokens": int(self.config.get("max_output_tokens", request.max_output_tokens)),
            "response_format": response_format,
        }
        if bool(self.config.get("enable_thinking", True)):
            kwargs["extra_body"] = {"enable_thinking": True}
            audit["thinking_enabled"] = True
        started = time.time()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "extra_body" in kwargs and self._is_unsupported_thinking_error(exc):
                audit["retry_count"] = 1
                audit["thinking_retry_without_extra_body"] = True
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("extra_body", None)
                try:
                    response = client.chat.completions.create(**retry_kwargs)
                except Exception as retry_exc:
                    return self._error_response(retry_exc, model=model, audit=audit, started=started)
            elif "response_format" in kwargs and self._is_response_format_incompatible(exc):
                audit["retry_count"] = 1
                audit["structured_output_parameter_incompatible"] = True
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("response_format", None)
                try:
                    response = client.chat.completions.create(**retry_kwargs)
                except Exception as retry_exc:
                    return self._error_response(retry_exc, model=model, audit=audit, started=started)
            else:
                return self._error_response(exc, model=model, audit=audit, started=started)
        audit["latency_seconds"] = round(time.time() - started, 6)
        self._fill_response_audit(audit, response)
        text = self._response_text(response)
        return LLMResponse(
            status="completed",
            provider=ALIYUN_BAILIAN_PROVIDER,
            model=model,
            text=text,
            audit=audit,
        )

    def _generate_provider_check(self, *, client: Any, request: LLMRequest, model: str, audit: dict[str, Any]) -> LLMResponse:
        audit["provider_check_mode"] = "streaming_basic"
        audit["response_format"] = "none"
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "这是接口连通测试，请简短回复OK。",
                }
            ],
            "stream": True,
            "extra_body": {"enable_thinking": True},
        }
        started = time.time()
        try:
            stream = client.chat.completions.create(**kwargs)
            text, stream_audit = self._consume_stream(stream)
        except Exception as exc:
            return self._error_response(exc, model=model, audit=audit, started=started)
        audit["latency_seconds"] = round(time.time() - started, 6)
        audit.update(stream_audit)
        audit["content_present"] = bool(text.strip())
        return LLMResponse(
            status="completed" if text.strip() else "invalid_output",
            provider=ALIYUN_BAILIAN_PROVIDER,
            model=model,
            text=text,
            failure_reason="" if text.strip() else "empty_provider_check_content",
            audit=audit,
        )

    def _base_audit(self, *, model: str) -> dict[str, Any]:
        return {
            "provider": ALIYUN_BAILIAN_PROVIDER,
            "model": model,
            "timeout_seconds": int(self.config.get("timeout_seconds", 180)),
            "response_format": self.config.get("response_format", "json_object"),
            "thinking_enabled": bool(self.config.get("enable_thinking", True)),
            "retry_count": 0,
        }

    @staticmethod
    def _consume_stream(stream: Any) -> tuple[str, dict[str, Any]]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        audit: dict[str, Any] = {}
        for chunk in stream:
            chunk_id = getattr(chunk, "id", None)
            if chunk_id is not None and "request_id" not in audit:
                audit["request_id"] = str(chunk_id)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    value = getattr(usage, attr, None)
                    if value is not None:
                        audit[attr] = value
            choices = getattr(chunk, "choices", []) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                audit["finish_reason"] = finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None and isinstance(choice, dict):
                delta = choice.get("delta")
            content = getattr(delta, "content", None) if delta is not None else None
            reasoning = getattr(delta, "reasoning_content", None) if delta is not None else None
            if isinstance(delta, dict):
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
            if content:
                content_parts.append(str(content))
            if reasoning:
                reasoning_parts.append(str(reasoning))
        reasoning_text = "".join(reasoning_parts)
        audit["reasoning_present"] = bool(reasoning_text)
        audit["reasoning_character_count"] = len(reasoning_text)
        if reasoning_text:
            audit["reasoning_hash"] = stable_hash(reasoning_text)
        return "".join(content_parts), audit

    @staticmethod
    def _load_openai_client(**kwargs: Any) -> Any:
        from openai import OpenAI

        return OpenAI(**kwargs)

    def _response_format(self) -> dict[str, str] | None:
        if self.config.get("response_format", "json_object") == "json_object":
            return {"type": "json_object"}
        return None

    @staticmethod
    def _response_text(response: Any) -> str:
        try:
            return str(response.choices[0].message.content or "")
        except Exception:
            if isinstance(response, dict):
                return str(
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )
            return ""

    @staticmethod
    def _fill_response_audit(audit: dict[str, Any], response: Any) -> None:
        for attr, key in [("id", "request_id"), ("status_code", "http_status")]:
            value = getattr(response, attr, None)
            if value is not None:
                audit[key] = str(value) if key == "request_id" else value
        try:
            audit["finish_reason"] = response.choices[0].finish_reason
        except Exception:
            pass
        try:
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            audit["reasoning_present"] = bool(reasoning)
            audit["reasoning_character_count"] = len(str(reasoning or ""))
            if reasoning:
                audit["reasoning_hash"] = stable_hash(str(reasoning))
        except Exception:
            audit.setdefault("reasoning_present", False)
            audit.setdefault("reasoning_character_count", 0)
        usage = getattr(response, "usage", None)
        if usage is not None:
            for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                value = getattr(usage, attr, None)
                if value is not None:
                    audit[attr] = value

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for attr in ["status_code", "http_status", "status"]:
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        return value if isinstance(value, int) else None

    @staticmethod
    def _is_unsupported_thinking_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "enable_thinking" in text and (
            "unsupported" in text or "unexpected" in text or "unknown" in text
        )

    def _is_response_format_incompatible(self, exc: Exception) -> bool:
        status_code = self._status_code(exc)
        text = str(exc).lower()
        return status_code == 400 and (
            "response_format" in text
            or "json_object" in text
            or "json mode" in text
        )

    def _error_response(
        self,
        exc: Exception,
        *,
        model: str,
        audit: dict[str, Any],
        started: float,
    ) -> LLMResponse:
        audit["latency_seconds"] = round(time.time() - started, 6)
        status_code = self._status_code(exc)
        if status_code is not None:
            audit["http_status"] = status_code
        text = redact_secret(exc)
        lower = text.lower()
        status = "failed"
        reason = "provider_error"
        if status_code == 401 or "authentication" in lower or "unauthorized" in lower:
            status = "provider_unavailable"
            reason = "authentication_failed"
        elif status_code == 403:
            status = "provider_unavailable"
            reason = "access_forbidden_or_model_not_enabled"
        elif status_code == 404 or "model not found" in lower or "does not exist" in lower:
            status = "provider_unavailable"
            reason = "configured_model_unavailable"
        elif status_code == 429:
            reason = "rate_limited"
        elif "timeout" in lower or exc.__class__.__name__.lower().endswith("timeout"):
            status = "timeout"
            reason = "provider_timeout"
        elif status_code is not None and status_code >= 500:
            reason = "provider_server_error"
        return LLMResponse(
            status=status,
            provider=ALIYUN_BAILIAN_PROVIDER,
            model=model,
            failure_reason=reason,
            warnings=[text],
            audit=audit,
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
    if name == ALIYUN_BAILIAN_PROVIDER:
        return AliyunBailianOpenAIProvider(load_provider_config(project_root, provider_config))
    raise ValueError(f"unsupported provider: {name}")
