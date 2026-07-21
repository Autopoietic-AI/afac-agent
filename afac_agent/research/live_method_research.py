# -*- coding: utf-8 -*-
"""Live-cached source-grounded method research agent for M6R-B2."""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Protocol

from afac_agent.llm.base import LLMRequest, LLMResponse
from afac_agent.llm.providers import ALIYUN_BAILIAN_DEFAULT_MODEL, ALIYUN_BAILIAN_PROVIDER, make_provider, redact_secret

from .event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash
from .local_conflict_checker import LocalConflictChecker
from .method_research import MethodCardExtractor, MethodCardValidator, MethodRanker, SourceChunker, SourceVerifier

LIVE_RESEARCH_VERSION = "m6r_b2"
NETWORK_MODES = {"live_cached", "cache_only", "disabled"}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)sk-[A-Za-z0-9_.\-]+", "sk-[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+", "[LOCAL_PATH_REDACTED]", text)
    text = re.sub(r"/(?:Users|home|mnt|tmp)/[^\s\"']+", "[LOCAL_PATH_REDACTED]", text)
    return text[:limit]


def _json_loads_lenient(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


class ResearchHttpClient(Protocol):
    def get_json(self, url: str, *, timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]: ...
    def get_text(self, url: str, *, timeout: int, headers: dict[str, str] | None = None) -> str: ...


class UrllibHttpClient:
    def get_json(self, url: str, *, timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
        return json.loads(self.get_text(url, timeout=timeout, headers=headers))

    def get_text(self, url: str, *, timeout: int, headers: dict[str, str] | None = None) -> str:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "AFAC-Agent-M6R-B2/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(1_000_000)
        return raw.decode("utf-8", errors="replace")


class ResearchCache:
    def __init__(self, *, root: str | Path):
        self.root = Path(root).resolve()

    def query_dir(self, provider_id: str, query_text: str) -> Path:
        return self.root / provider_id / stable_hash({"query": query_text})

    def content_dir(self, provider_id: str, canonical_url: str, content: str = "") -> Path:
        key = stable_hash({"url": canonical_url, "content_hash": stable_hash(content) if content else ""})
        return self.root / provider_id / key

    def read_query(self, provider_id: str, query_text: str) -> dict[str, Any] | None:
        path = self.query_dir(provider_id, query_text) / "query_record.json"
        return load_json(path) if path.exists() else None

    def write_query(self, provider_id: str, query_text: str, record: dict[str, Any]) -> Path:
        directory = self.query_dir(provider_id, query_text)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "query_record.json"
        path.write_text(json_dumps(record) + "\n", encoding="utf-8")
        return path

    def write_content(self, provider_id: str, *, canonical_url: str, title: str, source_type: str, content: str, source_level: str) -> dict[str, Any]:
        directory = self.content_dir(provider_id, canonical_url, content)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = ".md" if source_type in {"preprint", "peer_reviewed_paper", "official_repository", "official_documentation"} else ".txt"
        content_path = directory / f"content{suffix}"
        content_path.write_text(content, encoding="utf-8")
        content_hash = sha256_file(content_path)
        record = {
            "canonical_url": canonical_url,
            "title": title,
            "source_type": source_type,
            "source_level": source_level,
            "retrieved_at": "live_or_cached",
            "content_hash": content_hash,
            "cached_path": str(content_path),
            "verification_status": "cached_local_content",
        }
        (directory / "content_record.json").write_text(json_dumps(record) + "\n", encoding="utf-8")
        return record


class ResearchProvider:
    provider_id = "base"
    provider_type = "base"
    base_domains: list[str] = []
    purpose = ""
    enabled_modes = ["live_cached"]
    requires_auth = False
    credential_env_name = ""
    timeout_seconds = 20
    rate_limit_policy = "low_volume_smoke"
    cache_required = True
    source_level = "discovery_source"
    competition_status = "experimental"

    def __init__(self, *, http: ResearchHttpClient | None = None):
        self.http = http or UrllibHttpClient()

    def describe(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
            "base_domains": self.base_domains,
            "purpose": self.purpose,
            "enabled_modes": self.enabled_modes,
            "requires_auth": self.requires_auth,
            "credential_env_name": self.credential_env_name,
            "timeout_seconds": self.timeout_seconds,
            "rate_limit_policy": self.rate_limit_policy,
            "cache_required": self.cache_required,
            "source_level": self.source_level,
            "competition_status": self.competition_status,
        }

    def search(self, query: dict[str, Any], *, max_sources: int, cache: ResearchCache, network_mode: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAlexProvider(ResearchProvider):
    provider_id = "openalex"
    provider_type = "paper_metadata"
    base_domains = ["api.openalex.org"]
    purpose = "paper discovery and metadata"
    source_level = "discovery_source"

    def search(self, query: dict[str, Any], *, max_sources: int, cache: ResearchCache, network_mode: str) -> dict[str, Any]:
        query_text = str(query.get("query_text", ""))
        cached = cache.read_query(self.provider_id, query_text)
        if network_mode in {"disabled", "cache_only"}:
            return {"provider_id": self.provider_id, "status": "cache_hit" if cached else "cache_miss", "cache_hits": int(bool(cached)), "network_requests": 0, "sources": cached.get("sources", []) if cached else [], "errors": []}
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": query_text, "per-page": max_sources})
        try:
            payload = self.http.get_json(url, timeout=self.timeout_seconds)
            sources = []
            for item in _as_list(payload.get("results"))[:max_sources]:
                title = item.get("title") or ""
                landing = (item.get("primary_location") or {}).get("landing_page_url") or item.get("doi") or item.get("id") or ""
                abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
                content = f"# {title}\n\n{abstract}\n\nOpenAlex source: {landing}\n"
                record = cache.write_content(self.provider_id, canonical_url=landing or str(item.get("id")), title=title, source_type="preprint", content=content, source_level="discovery_source")
                sources.append({**record, "url": landing, "authors_or_organization": ", ".join([a.get("author", {}).get("display_name", "") for a in _as_list(item.get("authorships"))[:5]]), "year": item.get("publication_year", ""), "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name", ""), "source_level": "discovery_source"})
            qrecord = {"query_id": stable_hash({"provider": self.provider_id, "query": query_text}), "query_text": query_text, "provider_id": self.provider_id, "requested_at": "live_cached", "returned_urls": [s.get("canonical_url", "") for s in sources], "selected_urls": [s.get("canonical_url", "") for s in sources], "rejected_urls": [], "rejection_reasons": [], "cache_hits": int(bool(cached)), "network_requests": 1, "content_hashes": [s["content_hash"] for s in sources], "retrieval_status": "completed", "sources": sources}
            cache.write_query(self.provider_id, query_text, qrecord)
            return {"provider_id": self.provider_id, "status": "completed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": sources, "errors": []}
        except Exception as exc:
            return {"provider_id": self.provider_id, "status": "provider_failed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": cached.get("sources", []) if cached else [], "errors": [_safe_text(redact_secret(exc), 300)]}


class ArxivProvider(ResearchProvider):
    provider_id = "arxiv"
    provider_type = "paper_primary"
    base_domains = ["export.arxiv.org", "arxiv.org"]
    purpose = "arXiv paper retrieval"
    source_level = "primary_source"

    def search(self, query: dict[str, Any], *, max_sources: int, cache: ResearchCache, network_mode: str) -> dict[str, Any]:
        query_text = str(query.get("query_text", ""))
        cached = cache.read_query(self.provider_id, query_text)
        if network_mode in {"disabled", "cache_only"}:
            return {"provider_id": self.provider_id, "status": "cache_hit" if cached else "cache_miss", "cache_hits": int(bool(cached)), "network_requests": 0, "sources": cached.get("sources", []) if cached else [], "errors": []}
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": f"all:{query_text}", "start": 0, "max_results": max_sources})
        try:
            text = self.http.get_text(url, timeout=self.timeout_seconds)
            root = ET.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            sources = []
            for entry in root.findall("atom:entry", ns)[:max_sources]:
                title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
                summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
                canonical = entry.findtext("atom:id", default="", namespaces=ns) or ""
                authors = ", ".join([a.findtext("atom:name", default="", namespaces=ns) or "" for a in entry.findall("atom:author", ns)[:5]])
                content = f"# {title}\n\n{summary}\n\nArXiv: {canonical}\n"
                record = cache.write_content(self.provider_id, canonical_url=canonical, title=title, source_type="preprint", content=content, source_level="primary_source")
                sources.append({**record, "url": canonical, "authors_or_organization": authors, "year": (entry.findtext("atom:published", default="", namespaces=ns) or "")[:4], "venue": "arXiv", "source_level": "primary_source"})
            qrecord = {"query_id": stable_hash({"provider": self.provider_id, "query": query_text}), "query_text": query_text, "provider_id": self.provider_id, "requested_at": "live_cached", "returned_urls": [s.get("canonical_url", "") for s in sources], "selected_urls": [s.get("canonical_url", "") for s in sources], "rejected_urls": [], "rejection_reasons": [], "cache_hits": int(bool(cached)), "network_requests": 1, "content_hashes": [s["content_hash"] for s in sources], "retrieval_status": "completed", "sources": sources}
            cache.write_query(self.provider_id, query_text, qrecord)
            return {"provider_id": self.provider_id, "status": "completed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": sources, "errors": []}
        except Exception as exc:
            return {"provider_id": self.provider_id, "status": "provider_failed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": cached.get("sources", []) if cached else [], "errors": [_safe_text(redact_secret(exc), 300)]}


class GitHubRepositoryProvider(ResearchProvider):
    provider_id = "github_repository"
    provider_type = "repository"
    base_domains = ["api.github.com", "github.com"]
    purpose = "public repository search"
    requires_auth = False
    credential_env_name = "GITHUB_TOKEN"
    source_level = "primary_source"

    def search(self, query: dict[str, Any], *, max_sources: int, cache: ResearchCache, network_mode: str) -> dict[str, Any]:
        query_text = str(query.get("query_text", ""))
        cached = cache.read_query(self.provider_id, query_text)
        if network_mode in {"disabled", "cache_only"}:
            return {"provider_id": self.provider_id, "status": "cache_hit" if cached else "cache_miss", "cache_hits": int(bool(cached)), "network_requests": 0, "sources": cached.get("sources", []) if cached else [], "errors": []}
        headers = {"User-Agent": "AFAC-Agent-M6R-B2/0.1", "Accept": "application/vnd.github+json"}
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": query_text, "per_page": max_sources})
        try:
            payload = self.http.get_json(url, timeout=self.timeout_seconds, headers=headers)
            sources = []
            for item in _as_list(payload.get("items"))[:max_sources]:
                title = item.get("full_name") or item.get("name") or ""
                canonical = item.get("html_url") or ""
                content = f"# {title}\n\n{item.get('description') or ''}\n\nRepository: {canonical}\n"
                record = cache.write_content(self.provider_id, canonical_url=canonical, title=title, source_type="official_repository", content=content, source_level="primary_source")
                sources.append({**record, "url": canonical, "repository_url": canonical, "authors_or_organization": item.get("owner", {}).get("login", ""), "year": "", "venue": "GitHub", "source_level": "primary_source"})
            qrecord = {"query_id": stable_hash({"provider": self.provider_id, "query": query_text}), "query_text": query_text, "provider_id": self.provider_id, "requested_at": "live_cached", "returned_urls": [s.get("canonical_url", "") for s in sources], "selected_urls": [s.get("canonical_url", "") for s in sources], "rejected_urls": [], "rejection_reasons": [], "cache_hits": int(bool(cached)), "network_requests": 1, "content_hashes": [s["content_hash"] for s in sources], "retrieval_status": "completed", "sources": sources}
            cache.write_query(self.provider_id, query_text, qrecord)
            return {"provider_id": self.provider_id, "status": "completed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": sources, "errors": []}
        except Exception as exc:
            return {"provider_id": self.provider_id, "status": "provider_failed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": cached.get("sources", []) if cached else [], "errors": [_safe_text(redact_secret(exc), 300)]}


class DirectUrlFetchProvider(ResearchProvider):
    provider_id = "direct_url_fetch"
    provider_type = "direct_fetch"
    base_domains = ["registered-input-url"]
    purpose = "fetch already discovered primary pages"
    source_level = "primary_source"

    def search(self, query: dict[str, Any], *, max_sources: int, cache: ResearchCache, network_mode: str) -> dict[str, Any]:
        urls = [u for u in _as_list(query.get("urls")) if str(u).startswith(("http://", "https://"))][:max_sources]
        sources = []
        errors = []
        network_requests = 0
        for url in urls:
            cached = cache.read_query(self.provider_id, url)
            if network_mode in {"disabled", "cache_only"}:
                if cached:
                    sources.extend(cached.get("sources", []))
                continue
            try:
                network_requests += 1
                text = self.http.get_text(str(url), timeout=self.timeout_seconds)
                title = _title_from_html(text) or str(url)
                record = cache.write_content(self.provider_id, canonical_url=str(url), title=title, source_type="official_documentation", content=_strip_html(text)[:12000], source_level="primary_source")
                source = {**record, "url": str(url), "authors_or_organization": "official/public source", "year": "", "venue": urllib.parse.urlparse(str(url)).netloc, "source_level": "primary_source"}
                sources.append(source)
                cache.write_query(self.provider_id, str(url), {"query_id": stable_hash({"url": url}), "query_text": str(url), "provider_id": self.provider_id, "requested_at": "live_cached", "returned_urls": [str(url)], "selected_urls": [str(url)], "rejected_urls": [], "rejection_reasons": [], "cache_hits": int(bool(cached)), "network_requests": 1, "content_hashes": [source["content_hash"]], "retrieval_status": "completed", "sources": [source]})
            except Exception as exc:
                errors.append(_safe_text(redact_secret(exc), 300))
                if cached:
                    sources.extend(cached.get("sources", []))
        return {"provider_id": self.provider_id, "status": "completed" if sources else ("provider_failed" if errors else "cache_miss"), "cache_hits": 0, "network_requests": network_requests, "sources": sources, "errors": errors}


class MockLiveProvider(ResearchProvider):
    provider_id = "mock_live"
    provider_type = "mock"
    base_domains = ["example.org"]
    purpose = "tests only"
    source_level = "primary_source"

    def __init__(self, *, fail: bool = False, http: ResearchHttpClient | None = None):
        super().__init__(http=http)
        self.fail = fail

    def search(self, query: dict[str, Any], *, max_sources: int, cache: ResearchCache, network_mode: str) -> dict[str, Any]:
        query_text = str(query.get("query_text", ""))
        cached = cache.read_query(self.provider_id, query_text)
        if network_mode in {"disabled", "cache_only"}:
            return {"provider_id": self.provider_id, "status": "cache_hit" if cached else "cache_miss", "cache_hits": int(bool(cached)), "network_requests": 0, "sources": cached.get("sources", []) if cached else [], "errors": []}
        if self.fail:
            return {"provider_id": self.provider_id, "status": "provider_failed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": cached.get("sources", []) if cached else [], "errors": ["mock_failure"]}
        content = f"# Mock source\n\nA public primary source about {query_text} with an exact2 self-supervised signal."
        source = cache.write_content(self.provider_id, canonical_url=f"https://example.org/{stable_hash(query_text)[:8]}", title="Mock primary source", source_type="preprint", content=content, source_level="primary_source")
        source.update({"url": source["canonical_url"], "authors_or_organization": "mock", "year": 2026, "venue": "mock", "source_level": "primary_source"})
        cache.write_query(self.provider_id, query_text, {"query_id": stable_hash({"provider": self.provider_id, "query": query_text}), "query_text": query_text, "provider_id": self.provider_id, "requested_at": "mock", "returned_urls": [source["canonical_url"]], "selected_urls": [source["canonical_url"]], "rejected_urls": [], "rejection_reasons": [], "cache_hits": int(bool(cached)), "network_requests": 1, "content_hashes": [source["content_hash"]], "retrieval_status": "completed", "sources": [source]})
        return {"provider_id": self.provider_id, "status": "completed", "cache_hits": int(bool(cached)), "network_requests": 1, "sources": [source], "errors": []}


class ProviderRegistry:
    def __init__(self, providers: list[ResearchProvider] | None = None):
        defaults = providers or [OpenAlexProvider(), ArxivProvider(), GitHubRepositoryProvider(), DirectUrlFetchProvider()]
        self.providers = {provider.provider_id: provider for provider in defaults}

    def get(self, provider_id: str) -> ResearchProvider:
        if provider_id not in self.providers:
            raise ValueError(f"unregistered provider: {provider_id}")
        return self.providers[provider_id]

    def snapshot(self) -> dict[str, Any]:
        return {"registry_version": LIVE_RESEARCH_VERSION, "providers": [provider.describe() for provider in self.providers.values()], "provider_count": len(self.providers)}


class MockResearchLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.usage: list[dict[str, Any]] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.usage.append({"prompt_chars": len(request.prompt), "model": request.model, "metadata": request.metadata})
        if request.metadata.get("m6rb2_call") == "query_planner":
            return LLMResponse(status="completed", provider="mock", model="mock", text=json.dumps({"queries": [{"research_question": "How can exact2-only graph nodes use source-grounded self-supervised multi-hop information?", "query_text": "graph neural network self-supervised multi-hop exact 2-hop node classification", "query_type": "paper", "target_source_types": ["preprint", "official_repository"], "expected_evidence": ["mechanism", "minimal experiment"], "reason": "targets exact2-only new information"}]}, ensure_ascii=False), audit={"latency_seconds": 0.0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        return LLMResponse(status="completed", provider="mock", model="mock", text=json.dumps({"method_cards": [{"method_name": "Mock exact2 contrastive signal", "method_family": "contrastive", "target_problem_ids": ["p_exact2"], "target_scope": {"axis_id": "train_label_reachability", "value_id": "exact2_only"}, "target_mechanisms": ["multi_hop_signal_opportunity"], "core_hypothesis": "Use a source-grounded exact2 self-supervised signal before any training.", "mechanism_summary": "Mock source describes self-supervised multi-hop signal construction.", "required_inputs": ["canonical fold", "final OOF", "verified source chunks"], "information_source_type": "self_supervised_multi_hop_signal", "new_information_status": "new_information", "applicable_conditions": ["exact2_only"], "inapplicable_conditions": ["test truth routing"], "integration_point": "research design only", "minimal_experiment": {"mode": "read_only_design"}, "required_ablation": ["without exact2 signal"], "success_conditions": ["valid source support"], "failure_conditions": ["source support absent"], "stop_conditions": ["requires test truth"], "compute_cost": "low", "implementation_cost": "medium", "leakage_risk": "low", "expected_gain_status": "locally_unverified", "source_support_summary": "Supported by mock source."}]}, ensure_ascii=False), audit={"latency_seconds": 0.0, "prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4})


class LiveCachedMethodResearchRunner:
    def __init__(self, *, project_root: str | Path, registry: ProviderRegistry | None = None, llm_provider: Any | None = None):
        self.project_root = Path(project_root).resolve()
        self.registry = registry or ProviderRegistry()
        self.llm_provider = llm_provider

    def run(
        self,
        *,
        research_brief: str | Path,
        research_memory_root: str | Path,
        research_policy: str | Path,
        out_root: str | Path = "artifacts/live_method_research",
        cache_root: str | Path = "artifacts/research_cache",
        network_mode: str = "live_cached",
        provider: str = ALIYUN_BAILIAN_PROVIDER,
        model: str = ALIYUN_BAILIAN_DEFAULT_MODEL,
        max_queries: int | None = None,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        if network_mode not in NETWORK_MODES:
            return {"status": "failed", "failure_reason": "invalid_network_mode", "artifacts": {}}
        brief_path = _resolve(self.project_root, research_brief)
        memory_dir = self._resolve_memory_dir(research_memory_root)
        policy_path = _resolve(self.project_root, research_policy)
        missing = [name for name, path in {"research_brief": brief_path, "research_memory_root": memory_dir, "research_policy": policy_path}.items() if not path or not Path(path).exists()]
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        brief = load_json(brief_path)
        policy = load_json(policy_path)
        live_policy = policy.get("live_method_research", {})
        budgets = {
            "max_queries_per_problem": int(max_queries or live_policy.get("max_queries_per_problem", 3)),
            "max_sources_per_query": int(live_policy.get("max_sources_per_query", 5)),
            "max_verified_sources_per_problem": int(live_policy.get("max_verified_sources_per_problem", 4)),
        }
        cache = ResearchCache(root=_resolve(self.project_root, cache_root))
        input_hashes = {
            "research_brief": sha256_file(brief_path),
            "research_policy": sha256_file(policy_path),
            "research_memory_manifest": sha256_file(memory_dir / "research_manifest.json") if (memory_dir / "research_manifest.json").exists() else "",
        }
        run_id = stable_hash({"version": LIVE_RESEARCH_VERSION, "input_hashes": input_hashes, "network_mode": network_mode, "provider_registry": self.registry.snapshot(), "budgets": budgets})
        run_dir = _resolve(self.project_root, out_root) / run_id
        manifest_path = run_dir / "research_run_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": "duplicate", "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        started = time.time()
        llm = self.llm_provider or make_provider(provider, project_root=self.project_root)
        attempts = load_json(memory_dir / "method_attempt_ledger.json") if (memory_dir / "method_attempt_ledger.json").exists() else {"attempts": []}
        failures = load_json(memory_dir / "failure_ledger.json") if (memory_dir / "failure_ledger.json").exists() else {"failures": []}
        query_result = self._plan_queries(llm, brief, attempts, budgets, provider, model)
        if query_result["status"] != "completed":
            return {"status": query_result["status"], "failure_reason": query_result.get("failure_reason", "query_planning_failed"), "artifacts": {}}
        queries = [self._sanitize_query(q, brief) for q in query_result["queries"]][:budgets["max_queries_per_problem"]]
        retrieval_log, source_candidates, fallback_used = self._retrieve(queries, cache, network_mode, budgets)
        verified_primary = [s for s in source_candidates if s.get("source_level") == "primary_source"][:budgets["max_verified_sources_per_problem"]]
        effective_mode = "cache_only" if fallback_used and network_mode == "live_cached" else network_mode
        source_records = self._source_records(verified_primary, self.project_root)
        verification = SourceVerifier().verify(source_records)
        chunks = SourceChunker(max_chunk_chars=1800).chunk(verification["records"], manifest_base=self.project_root)
        extraction = self._extract_methods(llm, brief, verification["records"], chunks, attempts, provider, model)
        if extraction["status"] != "completed":
            return {"status": extraction["status"], "failure_reason": extraction.get("failure_reason", "method_extraction_failed"), "artifacts": {}}
        method_cards = [MethodCardExtractor()._finalize_card(card) for card in extraction["method_cards"][:6]]
        validations = [MethodCardValidator(allow_synthetic_test_sources=False).validate(method_card=card, research_brief=brief, source_records=verification["records"], chunks=chunks, policy=policy) for card in method_cards]
        checker = LocalConflictChecker()
        profile = load_json(memory_dir / "research_problem_profile.json") if (memory_dir / "research_problem_profile.json").exists() else {}
        conflicts = []
        for card in method_cards:
            conflict = checker.check(method_card=card, method_attempt_ledger=attempts, failure_ledger=failures, research_problem_profile=profile)
            conflict["method_id"] = card["method_id"]
            conflict["candidate_method_id"] = card["method_id"]
            conflicts.append(conflict)
        ranking = MethodRanker().rank(method_cards=method_cards, validations=validations, conflicts=conflicts, source_records=verification["records"], research_brief=brief, policy=policy)
        index = self._index(method_cards, ranking, conflicts)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        payloads = {
            "selected_research_briefs": {"items": [self._brief_summary(brief)], "view_hash": stable_hash(brief)},
            "research_queries": {"items": queries, "llm_audit": query_result["audit"], "view_hash": stable_hash(queries)},
            "provider_registry_snapshot": self.registry.snapshot(),
            "retrieval_log": {"items": retrieval_log, "fallback_used": fallback_used, "effective_network_mode": effective_mode, "view_hash": stable_hash(retrieval_log)},
            "source_registry": {"registry_version": LIVE_RESEARCH_VERSION, "sources": verification["records"], "view_hash": stable_hash(verification["records"])},
            "source_verification": verification,
            "method_cards_raw": {"items": extraction["raw_method_cards"], "llm_audit": extraction["audit"], "view_hash": stable_hash(extraction["raw_method_cards"])},
            "method_cards_validated": {"items": method_cards, "validations": validations, "view_hash": stable_hash({"cards": method_cards, "validations": validations})},
            "method_conflicts": {"items": conflicts, "view_hash": stable_hash(conflicts)},
            "method_ranking": ranking,
            "problem_method_index": index,
        }
        for name, payload in payloads.items():
            path = run_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        chunks_path = run_dir / "source_chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                safe = {**chunk, "content": chunk["content"][:1800]}
                handle.write(json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        artifacts["source_chunks"] = rel_ref(chunks_path, self.project_root)
        report_path = run_dir / "LIVE_METHOD_RESEARCH_REPORT.md"
        report_path.write_text(self._report(brief, queries, verification, ranking), encoding="utf-8")
        artifacts["live_method_research_report"] = rel_ref(report_path, self.project_root)
        manifest = {
            "run_version": LIVE_RESEARCH_VERSION,
            "run_id": run_id,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "read_only": True,
            "calls_llm": True,
            "calls_api": True,
            "uses_network": any(item.get("network_requests", 0) for item in retrieval_log),
            "executes_adapter": False,
            "trains_model": False,
            "generates_prediction": False,
            "counts_as_experiment_round": False,
            "mutates_project_state": False,
            "mutates_predictions": False,
            "network_mode": network_mode,
            "effective_network_mode": effective_mode,
            "source_freshness_limited": effective_mode == "cache_only",
            "qwen_call_count": query_result["call_count"] + extraction["call_count"],
            "qwen_usage": {"query_planner": query_result["audit"], "method_extractor": extraction["audit"]},
            "input_hashes": input_hashes,
            "provider_registry": self.registry.snapshot(),
            "primary_source_count": sum(1 for s in verification["records"] if s.get("source_level") == "primary_source" and s.get("verification_status") == "verified_local_content"),
            "method_card_count": len(method_cards),
            "conflict_check_executed": len(conflicts) == len(method_cards),
            "view_hash": stable_hash({name: payload.get("view_hash") for name, payload in payloads.items()} | {"chunks": stable_hash(chunks)}),
            "artifacts": artifacts,
        }
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["research_run_manifest"] = rel_ref(manifest_path, self.project_root)
        return {"status": "completed", "run_id": run_id, "view_hash": manifest["view_hash"], "artifacts": artifacts, "primary_source_count": manifest["primary_source_count"], "method_card_count": len(method_cards)}

    def _resolve_memory_dir(self, memory_root: str | Path) -> Path | None:
        path = _resolve(self.project_root, memory_root)
        if (path / "method_attempt_ledger.json").exists():
            return path
        if path.exists():
            kids = sorted([child for child in path.iterdir() if (child / "method_attempt_ledger.json").exists()])
            if len(kids) == 1:
                return kids[0]
        return None

    def _plan_queries(self, llm: Any, brief: dict[str, Any], attempts: dict[str, Any], budgets: dict[str, int], provider: str, model: str) -> dict[str, Any]:
        payload = {"brief": self._brief_summary(brief), "failed_methods": self._attempt_summary(attempts), "budget": budgets, "allowed_query_concepts": ["sparse graph node classification", "one-hop train-label reachability", "neighborhood reliability heterogeneity", "heterophily", "directed graph signals", "class-dependent graph routing", "self-supervised graph signal"], "forbidden_query_targets": ["canonical Fold recovery", "OOF recovery", "champion replay", "leaderboard", "test truth", "project files"]}
        prompt = (
            "Return one JSON object with key queries. Generate public method-research queries, not local-missing-input queries. "
            "Do not ask for canonical Fold, OOF recovery, champion replay, private artifacts, test truth, files, nodes, or predictions. "
            "Focus on source-grounded methods for sparse graph node classification and neighborhood reliability / heterophily / directed signals. "
            "Each query must be a general public literature or repository query.\n"
            + json_dumps(payload)
        )
        response = llm.generate(LLMRequest(prompt=prompt[:12000], provider=provider, model=model, timeout_seconds=180, max_output_tokens=2048, temperature=0, metadata={"m6rb2_call": "query_planner"}))
        if response.status != "completed":
            return {"status": response.status, "failure_reason": response.failure_reason, "audit": response.audit, "call_count": 1}
        try:
            parsed = _json_loads_lenient(response.text)
            queries = _as_list(parsed.get("queries"))
        except Exception:
            return {"status": "invalid_output", "failure_reason": "query_planner_json_invalid", "audit": response.audit, "call_count": 1}
        return {"status": "completed", "queries": queries[:budgets["max_queries_per_problem"]], "audit": self._audit(response), "call_count": 1}

    def _sanitize_query(self, query: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(query, dict):
            query = {
                "research_question": "Source-grounded method research for the current Research Brief scope.",
                "query_text": str(query),
                "query_type": "paper_repository",
                "target_source_types": ["preprint", "official_repository"],
                "expected_evidence": ["method mechanism", "applicable conditions", "minimal experiment"],
                "reason": "LLM returned a plain query string; normalized to structured query.",
            }
        allowed = {k: _safe_text(query.get(k), 1000) if isinstance(query.get(k), str) else query.get(k) for k in ["research_question", "query_text", "query_type", "target_source_types", "expected_evidence", "reason"]}
        forbidden = re.compile(r"\b(oof|out-of-fold|fold|v53|champion|leaderboard|test truth|test label|prediction|probability|proba)\b", re.I)
        if forbidden.search(str(allowed.get("query_text", ""))):
            scope_value = (brief.get("scope_refs", {}) or {}).get("value_id", "one_hop_available")
            allowed["query_text"] = (
                f"sparse graph neural network node classification {scope_value} "
                "neighborhood reliability heterophily directed graph self-supervised signal"
            )
            allowed["reason"] = "query sanitized away from local missing inputs toward public method research"
        allowed["query_type"] = allowed.get("query_type") or "paper_repository"
        allowed["target_source_types"] = allowed.get("target_source_types") or ["preprint", "official_repository"]
        allowed["expected_evidence"] = allowed.get("expected_evidence") or ["method mechanism", "applicable conditions", "minimal experiment"]
        allowed["query_id"] = stable_hash({"brief_id": brief.get("brief_id"), "query": allowed.get("query_text", "")})
        allowed["brief_id"] = brief.get("brief_id", "")
        allowed["target_problem_ids"] = brief.get("target_problem_ids", [])
        return allowed

    def _retrieve(self, queries: list[dict[str, Any]], cache: ResearchCache, network_mode: str, budgets: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        log: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        fallback_used = False
        preferred = [pid for pid in ["arxiv", "github_repository", "openalex"] if pid in self.registry.providers]
        provider_order = preferred + [pid for pid in self.registry.providers if pid not in preferred and pid != "direct_url_fetch"]
        for query in queries:
            for provider_id in provider_order:
                provider = self.registry.get(provider_id)
                result = provider.search(query, max_sources=budgets["max_sources_per_query"], cache=cache, network_mode=network_mode)
                log.append({"query_id": query["query_id"], "provider_id": provider_id, **{k: result.get(k) for k in ["status", "cache_hits", "network_requests", "errors"]}})
                sources.extend(_as_list(result.get("sources")))
                if result.get("status") == "provider_failed" and network_mode == "live_cached":
                    cache_result = provider.search(query, max_sources=budgets["max_sources_per_query"], cache=cache, network_mode="cache_only")
                    fallback_used = True
                    log.append({"query_id": query["query_id"], "provider_id": provider_id, "status": "cache_only_fallback", "cache_hits": cache_result.get("cache_hits", 0), "network_requests": 0, "errors": cache_result.get("errors", [])})
                    sources.extend(_as_list(cache_result.get("sources")))
                if len([s for s in sources if s.get("source_level") == "primary_source"]) >= budgets["max_verified_sources_per_problem"]:
                    break
            if len([s for s in sources if s.get("source_level") == "primary_source"]) >= budgets["max_verified_sources_per_problem"]:
                break
        unique = {}
        for source in sources:
            unique[source.get("content_hash") or source.get("canonical_url")] = source
        return log, list(unique.values()), fallback_used

    def _source_records(self, sources: list[dict[str, Any]], base: Path) -> list[dict[str, Any]]:
        records = []
        for source in sources:
            cached = Path(source.get("cached_path", ""))
            records.append({
                "source_id": source.get("source_id") or stable_hash({"url": source.get("canonical_url"), "hash": source.get("content_hash")}),
                "source_type": source.get("source_type", "other"),
                "title": source.get("title", ""),
                "authors_or_organization": source.get("authors_or_organization", "unknown public source"),
                "year": source.get("year", ""),
                "venue": source.get("venue", ""),
                "url": source.get("url", source.get("canonical_url", "")),
                "repository_url": source.get("repository_url", ""),
                "local_path": rel_ref(cached, base),
                "resolved_local_path": str(cached),
                "content_hash": source.get("content_hash", ""),
                "retrieved_at": source.get("retrieved_at", "live_or_cached"),
                "verification_status": "unverified",
                "verification_evidence": [],
                "license_or_usage_notes": "public metadata/content cached for method research only",
                "target_problem_ids": [],
                "target_scope_refs": {},
                "source_level": source.get("source_level", "unknown_source"),
                "synthetic_test_only": False,
            })
        return records

    def _extract_methods(self, llm: Any, brief: dict[str, Any], sources: list[dict[str, Any]], chunks: list[dict[str, Any]], attempts: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
        source_by_id = {source["source_id"]: source for source in sources}
        selected_chunks = self._select_chunks(chunks, max_chars=24000)
        prompt_payload = {
            "brief": self._brief_summary(brief),
            "sources": [{k: s.get(k, "") for k in ["source_id", "source_type", "title", "authors_or_organization", "year", "venue", "url", "source_level", "verification_status"]} for s in sources],
            "chunks": selected_chunks,
            "local_conflict_summary": self._attempt_summary(attempts),
            "instructions": (
                "Return JSON object with method_cards. Produce public method candidates grounded in the provided sources. "
                "Do not output cards about OOF recovery, Fold recovery, champion replay, project artifact recovery, leaderboard, or test truth. "
                "Each card must include non-empty core_hypothesis, required_inputs, applicable_conditions, inapplicable_conditions, "
                "success_conditions, failure_conditions, stop_conditions, source_refs/chunk_refs, and source_support_summary. "
                "Do not claim quantitative gain; expected_gain_status must be unknown, qualitative_only, or locally_unverified."
            ),
        }
        prompt = json_dumps(prompt_payload)
        response = llm.generate(LLMRequest(prompt=prompt[:36000], provider=provider, model=model, timeout_seconds=180, max_output_tokens=4096, temperature=0, metadata={"m6rb2_call": "method_extractor"}))
        call_count = 1
        if response.status != "completed":
            return {"status": response.status, "failure_reason": response.failure_reason, "audit": response.audit, "call_count": call_count}
        try:
            parsed = _json_loads_lenient(response.text)
        except Exception:
            return {"status": "invalid_output", "failure_reason": "method_extractor_json_invalid", "audit": response.audit, "call_count": call_count}
        raw_cards = _as_list(parsed.get("method_cards"))[:6]
        cards = []
        fallback_refs = [{"source_id": c["source_id"], "chunk_id": c["chunk_id"]} for c in selected_chunks[:1]]
        for raw in raw_cards:
            if not isinstance(raw, dict):
                continue
            refs = []
            for ref in _as_list(raw.get("source_refs")) + _as_list(raw.get("chunk_refs")):
                if isinstance(ref, dict) and ref.get("source_id") and ref.get("chunk_id"):
                    refs.append({"source_id": ref["source_id"], "chunk_id": ref["chunk_id"]})
            refs = refs or fallback_refs
            card = {
                "method_name": raw.get("method_name", raw.get("method_family", "source grounded method")),
                "method_family": raw.get("method_family", "source_grounded"),
                "target_problem_ids": raw.get("target_problem_ids") or brief.get("target_problem_ids", []),
                "target_scope_refs": raw.get("target_scope") or brief.get("scope_refs", {}),
                "bucket_axes": [raw.get("target_scope") or brief.get("scope_refs", {})],
                "mechanism_id": raw.get("target_mechanisms") or raw.get("mechanism_id") or ["neighborhood_reliability_heterogeneity"],
                "core_hypothesis": raw.get("core_hypothesis") or raw.get("hypothesis") or raw.get("mechanism_summary") or "Source-grounded method candidate for the current Research Brief scope.",
                "mechanism_summary": raw.get("mechanism_summary") or raw.get("core_hypothesis") or raw.get("hypothesis") or "Mechanism summary is source-grounded but locally unverified.",
                "required_inputs": raw.get("required_inputs") or ["verified public source chunks", "local validation protocol before any experiment"],
                "information_source_type": raw.get("information_source_type", "source_grounded_public_method"),
                "new_information_status": raw.get("new_information_status", "new_information"),
                "applicable_conditions": raw.get("applicable_conditions") or [f"current scope: {brief.get('scope_level', 'unknown')}"],
                "inapplicable_conditions": raw.get("inapplicable_conditions") or ["requires test truth", "requires unverified private artifacts"],
                "integration_point": raw.get("integration_point", "research_design"),
                "minimal_experiment": raw.get("minimal_experiment", {}),
                "required_ablation": raw.get("required_ablation", []),
                "success_conditions": raw.get("success_conditions") or ["source support is verified", "local conflict check does not block"],
                "failure_conditions": raw.get("failure_conditions") or ["insufficient source support", "same information as failed local route"],
                "stop_conditions": raw.get("stop_conditions") or ["requires test truth", "requires automatic prediction or submission"],
                "compute_cost": raw.get("compute_cost", "medium"),
                "implementation_cost": raw.get("implementation_cost", "medium"),
                "leakage_risk": raw.get("leakage_risk", "medium"),
                "expected_gain_status": raw.get("expected_gain_status", "locally_unverified"),
                "source_refs": refs,
                "chunk_refs": refs,
                "source_support_summary": raw.get("source_support_summary", ""),
                "status": "proposed",
                "source_type": source_by_id.get(refs[0]["source_id"], {}).get("source_type", "other") if refs else "other",
            }
            cards.append(card)
        return {"status": "completed", "raw_method_cards": raw_cards, "method_cards": cards, "audit": self._audit(response), "call_count": call_count}

    def _select_chunks(self, chunks: list[dict[str, Any]], *, max_chars: int) -> list[dict[str, Any]]:
        selected = []
        total = 0
        for chunk in chunks:
            content = _safe_text(chunk.get("content", ""), 8000)
            if total + len(content) > max_chars:
                continue
            selected.append({k: chunk[k] for k in ["chunk_id", "source_id", "section_title", "chunk_index"]} | {"content": content})
            total += len(content)
        return selected

    def _brief_summary(self, brief: dict[str, Any]) -> dict[str, Any]:
        summary = {k: brief.get(k) for k in ["brief_id", "brief_type", "target_problem_ids", "scope_level", "scope_refs", "primary_research_question", "evidence_gaps", "required_new_information", "success_conditions", "failure_conditions", "stop_conditions"]}
        for key in ["evidence_gaps", "required_new_information", "success_conditions", "failure_conditions", "stop_conditions"]:
            summary[key] = [_safe_text(item).replace("proba", "out-of-fold signal").replace("probability", "calibration signal") for item in _as_list(summary.get(key))]
        return summary

    def _attempt_summary(self, attempts: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for attempt in _as_list(attempts.get("attempts"))[:20]:
            rows.append({k: attempt.get(k) for k in ["method_id", "method_family", "information_source_type", "new_information_status", "branch_id", "outcome"]})
        return rows

    def _audit(self, response: LLMResponse) -> dict[str, Any]:
        audit = dict(response.audit)
        return {
            "provider": response.provider,
            "model": response.model,
            "latency_seconds": audit.get("latency_seconds"),
            "prompt_tokens": audit.get("prompt_tokens"),
            "completion_tokens": audit.get("completion_tokens"),
            "total_tokens": audit.get("total_tokens"),
            "finish_reason": audit.get("finish_reason"),
            "reasoning_present": bool(audit.get("reasoning_present")),
            "reasoning_character_count": int(audit.get("reasoning_character_count") or 0),
            "reasoning_hash": audit.get("reasoning_hash", ""),
        }

    def _index(self, cards: list[dict[str, Any]], ranking: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
        methods_by_problem: dict[str, list[str]] = {}
        for card in cards:
            for problem_id in _as_list(card.get("target_problem_ids")):
                methods_by_problem.setdefault(str(problem_id), []).append(card["method_id"])
        return {"index_version": LIVE_RESEARCH_VERSION, "methods_by_problem": methods_by_problem, "failures_by_method": {c["method_id"]: c.get("conflict_reasons", []) for c in conflicts if c.get("conflict_status") in {"exact_duplicate", "high_overlap"}}, "ranking_by_method": {item["method_id"]: {"rank": item["rank"], "selection_status": item["selection_status"]} for item in ranking["items"]}, "attempt_count": len(cards), "view_hash": stable_hash(methods_by_problem)}

    def _report(self, brief: dict[str, Any], queries: list[dict[str, Any]], verification: dict[str, Any], ranking: dict[str, Any]) -> str:
        return "\n".join(["# M6R-B2 Live-Cached Method Research Report", "", f"Brief: `{brief.get('brief_id', '')}`", f"Queries: `{len(queries)}`", f"Source summary: `{json.dumps(verification.get('summary', {}), ensure_ascii=False, sort_keys=True)}`", f"Method cards: `{len(ranking.get('items', []))}`", "", "No Adapter/training/prediction/submission was executed.", ""])


def _abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, locs in index.items():
        for loc in _as_list(locs):
            if isinstance(loc, int):
                positions.append((loc, str(word)))
    return " ".join(word for _, word in sorted(positions))


def _title_from_html(text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
    return _strip_html(match.group(1)).strip() if match else ""


def _strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()
