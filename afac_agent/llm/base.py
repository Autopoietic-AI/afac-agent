"""Provider-neutral primitives for M6A shadow planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    provider: str
    model: str = ""
    timeout_seconds: int = 30
    max_output_tokens: int = 1024
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    status: str
    text: str = ""
    provider: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)
    failure_reason: str = ""


class LLMProvider(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a raw LLM response.  Providers must not execute tools."""
