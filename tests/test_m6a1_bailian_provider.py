from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from afac_agent.llm.base import LLMRequest, LLMResponse
from afac_agent.llm.providers import (
    ALIYUN_BAILIAN_DEFAULT_MODEL,
    ALIYUN_BAILIAN_PROVIDER,
    AliyunBailianOpenAIProvider,
    is_allowed_bailian_model,
    provider_config_hash,
    redact_secret,
)
from afac_agent.llm.shadow_planner import LLMShadowPlanner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
PROJECT_STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _proposal(action: str = "request_missing_input") -> str:
    return json.dumps(
        {
            "proposal_version": "m6a_v1",
            "proposal_id": "PENDING",
            "task": "A1",
            "status": "completed",
            "primary_problem": "missing_full_anchor_evaluation_inputs",
            "problem_interpretation": "Missing canonical Fold and final v53Q-1 OOF inputs.",
            "proposed_action": action,
            "proposed_tool": None,
            "proposed_tool_version": None,
            "evidence_refs": [],
            "feedback_refs": [],
            "required_inputs": ["canonical_fold_assignment"],
            "missing_inputs": ["final_v53q1_oof_proba"],
            "expected_information_gain": "Wait for full anchor inputs.",
            "expected_model_gain_status": "unavailable_without_full_anchor_oof",
            "risk_level": "low",
            "uncertainties": ["final v53Q-1 OOF unavailable"],
            "assumptions": ["M5A remains authoritative"],
            "stop_conditions": ["missing input"],
            "success_conditions": ["safe shadow proposal"],
            "failure_conditions": ["unsafe action"],
            "reason_codes": ["missing_input", "missing_full_anchor_inputs"],
            "human_readable_rationale": "Only advisory.",
            "requires_human_approval": False,
            "auto_execution_allowed": True,
            "research_needed": False,
            "research_query": "",
        },
        ensure_ascii=False,
    )


class _FakeCompletion:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletion(responses))


class _FakeFactory:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.kwargs: dict = {}
        self.client = _FakeClient(responses)

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.client


class _HTTPError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _TimeoutError(Exception):
    pass


def _response(content: str, *, request_id: str = "req-1"):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=7, total_tokens=17)
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
    return SimpleNamespace(id=request_id, choices=[choice], usage=usage)


def _stream_chunk(
    *,
    content: str = "",
    reasoning: str = "",
    finish_reason: str | None = None,
    usage: object | None = None,
    request_id: str = "stream-req-1",
):
    delta = SimpleNamespace(content=content or None, reasoning_content=reasoning or None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id=request_id, choices=[choice], usage=usage)


def _provider(*, responses: list[object], env: dict[str, str] | None = None, config: dict | None = None):
    factory = _FakeFactory(responses)
    provider = AliyunBailianOpenAIProvider(
        config or {},
        client_factory=factory,
        environ=env
        if env is not None
        else {
            "DASHSCOPE_API_KEY": "secret-value",
            "AFAC_BAILIAN_BASE_URL": "https://workspace.example.com/compatible-mode/v1",
        },
    )
    return provider, factory


def _request() -> LLMRequest:
    return LLMRequest(
        prompt="Return JSON.",
        provider=ALIYUN_BAILIAN_PROVIDER,
        model=ALIYUN_BAILIAN_DEFAULT_MODEL,
        timeout_seconds=180,
        max_output_tokens=256,
        temperature=0.0,
    )


def test_bailian_reads_env_missing_inputs_and_default_model() -> None:
    provider, factory = _provider(responses=[], env={})
    result = provider.generate(_request())
    assert result.status == "provider_unavailable"
    assert result.failure_reason == "missing_api_key_environment_variable"
    assert factory.kwargs == {}
    assert result.model == ALIYUN_BAILIAN_DEFAULT_MODEL

    provider, factory = _provider(responses=[], env={"DASHSCOPE_API_KEY": "secret-value"})
    result = provider.generate(_request())
    assert result.status == "provider_unavailable"
    assert result.failure_reason == "missing_base_url_environment_variable"
    assert factory.kwargs == {}


def test_key_shape_allows_non_sk_long_clean_credentials_and_rejects_dirty_values() -> None:
    clean_env = {
        "DASHSCOPE_API_KEY": "dashscope-valid-clean-token-" + "x" * 240,
        "AFAC_BAILIAN_BASE_URL": "https://workspace.example.com/compatible-mode/v1",
    }
    provider, factory = _provider(responses=[_response(_proposal())], env=clean_env)
    result = provider.generate(_request())
    assert result.status == "completed"
    assert factory.kwargs["api_key"] == clean_env["DASHSCOPE_API_KEY"]

    for bad_key, reason in [
        ("   ", "missing_api_key_environment_variable"),
        (" clean-but-padded ", "api_key_has_surrounding_whitespace"),
        ("line\nbreak", "api_key_contains_newline"),
        ('"quoted"', "api_key_contains_quote"),
    ]:
        provider, dirty_factory = _provider(
            responses=[],
            env={
                "DASHSCOPE_API_KEY": bad_key,
                "AFAC_BAILIAN_BASE_URL": "https://workspace.example.com/compatible-mode/v1",
            },
        )
        result = provider.generate(_request())
        assert result.status == "provider_unavailable"
        assert result.failure_reason == reason
        assert dirty_factory.kwargs == {}


def test_model_policy_and_no_fallback() -> None:
    assert is_allowed_bailian_model("qwen3.6-max-preview")
    assert is_allowed_bailian_model("qwen3.5-large")
    assert not is_allowed_bailian_model("qwen-plus")
    assert not is_allowed_bailian_model("gpt-4")
    provider, factory = _provider(responses=[], config={"model": "qwen-plus"})
    result = provider.generate(LLMRequest(prompt="{}", provider=ALIYUN_BAILIAN_PROVIDER))
    assert result.status == "provider_unavailable"
    assert result.failure_reason == "configured_model_not_allowed"
    assert factory.kwargs == {}


def test_successful_chat_completion_passes_json_format_thinking_and_usage() -> None:
    provider, factory = _provider(responses=[_response(_proposal())])
    result = provider.generate(_request())
    assert result.status == "completed"
    assert json.loads(result.text)["proposed_action"] == "request_missing_input"
    assert factory.kwargs["api_key"] == "secret-value"
    assert factory.kwargs["base_url"].startswith("https://workspace.example.com")
    call = factory.client.chat.completions.calls[0]
    assert call["model"] == ALIYUN_BAILIAN_DEFAULT_MODEL
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": True}
    assert result.audit["base_url_host"] == "workspace.example.com"
    assert result.audit["prompt_tokens"] == 10
    assert result.audit["completion_tokens"] == 7
    assert result.audit["total_tokens"] == 17
    assert "secret-value" not in json.dumps(result.audit, ensure_ascii=False)


def test_provider_check_uses_streaming_basic_call_without_json_response_format() -> None:
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)
    stream = [
        _stream_chunk(reasoning="thinking"),
        _stream_chunk(content="OK"),
        _stream_chunk(finish_reason="stop", usage=usage),
    ]
    provider, factory = _provider(responses=[stream])
    result = provider.generate(
        LLMRequest(
            prompt="provider check",
            provider=ALIYUN_BAILIAN_PROVIDER,
            model=ALIYUN_BAILIAN_DEFAULT_MODEL,
            metadata={"provider_check": True},
        )
    )
    assert result.status == "completed"
    assert result.text == "OK"
    call = factory.client.chat.completions.calls[0]
    assert call["stream"] is True
    assert "response_format" not in call
    assert call["extra_body"] == {"enable_thinking": True}
    assert result.audit["content_present"] is True
    assert result.audit["reasoning_present"] is True
    assert result.audit["reasoning_character_count"] == len("thinking")
    assert "reasoning_hash" in result.audit
    assert result.audit["total_tokens"] == 4


def test_enable_thinking_retry_happens_once_without_model_fallback() -> None:
    provider, factory = _provider(
        responses=[
            _HTTPError(400, "unsupported parameter enable_thinking"),
            _response(_proposal()),
        ]
    )
    result = provider.generate(_request())
    assert result.status == "completed"
    assert result.audit["retry_count"] == 1
    assert len(factory.client.chat.completions.calls) == 2
    assert "extra_body" in factory.client.chat.completions.calls[0]
    assert "extra_body" not in factory.client.chat.completions.calls[1]
    assert factory.client.chat.completions.calls[1]["model"] == ALIYUN_BAILIAN_DEFAULT_MODEL


def test_response_format_incompatibility_falls_back_without_changing_model() -> None:
    provider, factory = _provider(
        responses=[
            _HTTPError(400, "unsupported response_format json_object"),
            _response(_proposal()),
        ]
    )
    result = provider.generate(_request())
    assert result.status == "completed"
    assert result.audit["retry_count"] == 1
    assert result.audit["structured_output_parameter_incompatible"] is True
    assert "response_format" in factory.client.chat.completions.calls[0]
    assert "response_format" not in factory.client.chat.completions.calls[1]
    assert factory.client.chat.completions.calls[1]["model"] == ALIYUN_BAILIAN_DEFAULT_MODEL


def test_api_error_mapping_and_redaction() -> None:
    cases = [
        (_HTTPError(401, "authentication failed"), "provider_unavailable", "authentication_failed"),
        (_HTTPError(403, "forbidden"), "provider_unavailable", "access_forbidden_or_model_not_enabled"),
        (_HTTPError(404, "model not found"), "provider_unavailable", "configured_model_unavailable"),
        (_HTTPError(429, "rate limited"), "failed", "rate_limited"),
        (_HTTPError(500, "server error"), "failed", "provider_server_error"),
        (_TimeoutError("timeout"), "timeout", "provider_timeout"),
    ]
    for exc, status, reason in cases:
        provider, _factory = _provider(responses=[exc])
        result = provider.generate(_request())
        assert result.status == status
        assert result.failure_reason == reason
    header = "Author" + "ization: " + "Bear" + "er token " + ("s" + "k-secret-value")
    redacted = redact_secret(header)
    assert "secret-value" not in redacted
    assert ("Bear" + "er token") not in redacted


def test_provider_config_hash_excludes_api_key_value(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "llm.local.json"
    config.write_text(
        json.dumps(
            {
                "provider": ALIYUN_BAILIAN_PROVIDER,
                "model": ALIYUN_BAILIAN_DEFAULT_MODEL,
                "api_key_env": "DASHSCOPE_API_KEY",
                "base_url_env": "AFAC_BAILIAN_BASE_URL",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-one")
    monkeypatch.setenv("AFAC_BAILIAN_BASE_URL", "https://workspace.example.com/compatible-mode/v1")
    first = provider_config_hash(tmp_path, str(config))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-two")
    second = provider_config_hash(tmp_path, str(config))
    assert first == second


def _write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    plan = {
        "planner_version": "m5a_v1",
        "plan_id": "plan_synthetic",
        "task": "A1",
        "status": "waiting_for_input",
        "current_stage": "L0-L3_agent_integration",
        "primary_problem": "missing_full_anchor_evaluation_inputs",
        "problem_evidence": {},
        "selected_action": "request_missing_input",
        "selected_tool": None,
        "ranked_actions": [],
        "blocked_actions": [],
        "deferred_actions": [],
        "evidence_refs": [],
        "feedback_refs": [],
        "required_inputs": ["canonical_fold_assignment"],
        "missing_inputs": ["final_v53q1_oof_proba"],
        "expected_information_gain": "missing inputs",
        "expected_model_gain_status": "unavailable_without_full_anchor_oof",
        "risk_level": "low",
        "budget_cost": {"counts_as_experiment_round": False, "rounds_used": 0, "max_rounds": 12},
        "stop_conditions": [],
        "success_conditions": [],
        "failure_conditions": [],
        "reason_codes": ["missing_input", "missing_full_anchor_inputs"],
        "human_readable_rationale": "wait for inputs",
        "requires_human_approval": False,
        "auto_execution_allowed": False,
        "input_hashes": {},
        "policy_hash": "policy",
    }
    return {
        "plan": _write_json(tmp_path / "plan_decision.json", plan),
        "problem": _write_json(tmp_path / "problem_map.json", {"analysis_tier": "dataset_only", "problems": []}),
        "feedback": _write_json(
            tmp_path / "feedback.json",
            {
                "feedback_id": "fb1",
                "tool_name": "A1_V46A1_ISOLATED_AUDIT",
                "feedback_kind": "expert_scope_audit",
                "evaluation_tier": "expert_scope",
                "status": "completed",
                "recommendation": "request_missing_input",
            },
        ),
        "registry": _write_json(
            tmp_path / "tool_registry.json",
            {"tools": [{"name": "A1_V53Q1_PATCH_REPLAY_SAFE", "adapter_entrypoint": "x"}]},
        ),
        "state": _write_json(
            tmp_path / "project_state.json",
            {"task": "A1", "budget": {"rounds_used": 0, "max_rounds": 12}, "closed_branches": []},
        ),
        "history": _write_json(tmp_path / "history.json", []),
        "planner_policy": _write_json(tmp_path / "planner_policy.json", {"action_priority": []}),
        "llm_policy": _write_json(
            tmp_path / "llm_shadow_policy.json",
            json.loads((PROJECT_ROOT / "config" / "llm_shadow_policy.json").read_text(encoding="utf-8")),
        ),
    }


class _SequenceProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return self.responses.pop(0)


def test_shadow_planner_retries_invalid_schema_and_writes_provider_usage(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path / "中文 path")
    fake = _SequenceProvider(
        [
            LLMResponse(
                status="completed",
                provider=ALIYUN_BAILIAN_PROVIDER,
                model=ALIYUN_BAILIAN_DEFAULT_MODEL,
                text='{"status":"completed"}',
                audit={"provider": ALIYUN_BAILIAN_PROVIDER, "total_tokens": 3},
            ),
            LLMResponse(
                status="completed",
                provider=ALIYUN_BAILIAN_PROVIDER,
                model=ALIYUN_BAILIAN_DEFAULT_MODEL,
                text=_proposal(),
                audit={"provider": ALIYUN_BAILIAN_PROVIDER, "total_tokens": 17},
            ),
        ]
    )
    monkeypatch.setattr("afac_agent.llm.shadow_planner.make_provider", lambda *args, **kwargs: fake)
    result = LLMShadowPlanner(project_root=tmp_path).run(
        deterministic_plan_path=paths["plan"],
        problem_map_path=paths["problem"],
        feedback_paths=[paths["feedback"]],
        tool_registry_path=paths["registry"],
        project_state_path=paths["state"],
        history_path=paths["history"],
        planner_policy_path=paths["planner_policy"],
        llm_policy_path=paths["llm_policy"],
        provider_name=ALIYUN_BAILIAN_PROVIDER,
        out_root=tmp_path / "shadow runs",
    )
    assert result["status"] == "completed"
    assert fake.calls == 2
    usage_path = Path(result["artifacts"]["provider_usage"])
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    assert usage["provider"] == ALIYUN_BAILIAN_PROVIDER
    assert usage["audit"]["total_tokens"] == 17


def test_provider_check_missing_env_does_not_modify_state_or_print_secret(tmp_path: Path) -> None:
    before = {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
    env = os.environ.copy()
    env.pop("DASHSCOPE_API_KEY", None)
    env.pop("AFAC_BAILIAN_BASE_URL", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "afac_agent.main",
            "llm-provider-check",
            "--project_root",
            str(PROJECT_ROOT),
            "--provider",
            ALIYUN_BAILIAN_PROVIDER,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "provider_unavailable"
    assert payload["failure_reason"] == "missing_api_key_environment_variable"
    assert ("DASHSCOPE_API_KEY" + "=") not in proc.stdout
    assert before == {path: _sha256(path) for path in [CHAMPION, PROJECT_STATE, HISTORY]}
