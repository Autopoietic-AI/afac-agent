# -*- coding: utf-8 -*-
"""AFAC v2.0 LLM call contract and append-only call ledger.

Every real LLM call in a v2 run is appended to ``llm_calls.jsonl`` with the
full contract fields.  The ledger never records API keys, authorization
headers, environment variable values, or raw sensitive configuration — only
hashes of prompts/responses and sanitized error strings.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..llm.base import LLMRequest, LLMResponse
from ..llm.providers import redact_secret
from ..research.event_store import stable_hash

LEDGER_VERSION = "afac_v2_llm_ledger_v1"

LEDGER_FIELDS = (
    "call_id",
    "execution_id",
    "task",
    "stage",
    "provider",
    "model",
    "planner_policy_version",
    "prompt_template_id",
    "prompt_hash",
    "response_hash",
    "started_at",
    "ended_at",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "status",
    "error_type",
    "error_message_sanitized",
    "retry_count",
    "fallback_used",
)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


class LLMLedger:
    """Wrap an LLM provider and record every call to ``llm_calls.jsonl``."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        execution_id: str,
        task: str,
        provider: Any,
        provider_name: str,
        model: str,
        planner_policy_version: str,
        max_retries: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.run_dir / "llm_calls.jsonl"
        self.execution_id = execution_id
        self.task = task
        self.provider = provider
        self.provider_name = provider_name
        self.model = model
        self.planner_policy_version = planner_policy_version
        self.max_retries = max_retries
        self.calls_count = 0

    def call(
        self,
        *,
        stage: str,
        prompt_template_id: str,
        prompt: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one LLM call with retries and append the ledger record.

        Returns ``{"ok": bool, "text": str, "record": dict}``.
        """
        attempt = 0
        response: LLMResponse | None = None
        started = time.time()
        last_error = ""
        while attempt <= self.max_retries:
            request = LLMRequest(
                prompt=prompt,
                provider=self.provider_name,
                model=self.model,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata or {},
            )
            try:
                response = self.provider.generate(request)
            except Exception as exc:  # provider-level crash
                response = LLMResponse(
                    status="error",
                    provider=self.provider_name,
                    model=self.model,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            if response.status == "completed" and response.text:
                break
            last_error = response.failure_reason or response.status
            attempt += 1
        ended = time.time()
        ok = bool(response is not None and response.status == "completed" and response.text)
        text = response.text if ok and response else ""
        audit = response.audit if response else {}
        usage = audit.get("usage", {}) if isinstance(audit, dict) else {}
        self.calls_count += 1
        record = {
            "ledger_version": LEDGER_VERSION,
            "call_id": stable_hash(
                {
                    "execution_id": self.execution_id,
                    "stage": stage,
                    "prompt_hash": _sha_text(prompt),
                    "call_index": self.calls_count,
                }
            )[:24],
            "execution_id": self.execution_id,
            "task": self.task,
            "stage": stage,
            "provider": self.provider_name,
            "model": self.model,
            "planner_policy_version": self.planner_policy_version,
            "prompt_template_id": prompt_template_id,
            "prompt_hash": _sha_text(prompt),
            "response_hash": _sha_text(text) if text else "",
            "started_at": started,
            "ended_at": ended,
            "latency_ms": int((ended - started) * 1000),
            "input_tokens": int(usage.get("input_tokens") or _estimate_tokens(prompt)),
            "output_tokens": int(usage.get("output_tokens") or _estimate_tokens(text)),
            "status": "completed" if ok else (response.status if response else "error"),
            "error_type": "" if ok else (response.status if response else "exception"),
            "error_message_sanitized": "" if ok else redact_secret(last_error),
            "retry_count": attempt if ok else attempt,
            "fallback_used": False,
        }
        for field in LEDGER_FIELDS:
            record.setdefault(field, "")
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return {"ok": ok, "text": text, "record": record}

    def mark_fallback(self, *, stage: str, prompt_template_id: str, reason: str) -> dict[str, Any]:
        """Record a deterministic-fallback event for a stage (no LLM text)."""
        self.calls_count += 1
        now = time.time()
        record = {
            "ledger_version": LEDGER_VERSION,
            "call_id": stable_hash({"execution_id": self.execution_id, "stage": stage, "fallback": True, "call_index": self.calls_count})[:24],
            "execution_id": self.execution_id,
            "task": self.task,
            "stage": stage,
            "provider": self.provider_name,
            "model": self.model,
            "planner_policy_version": self.planner_policy_version,
            "prompt_template_id": prompt_template_id,
            "prompt_hash": "",
            "response_hash": "",
            "started_at": now,
            "ended_at": now,
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "status": "deterministic_fallback",
            "error_type": "",
            "error_message_sanitized": redact_secret(reason),
            "retry_count": 0,
            "fallback_used": True,
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record


def load_ledger(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
