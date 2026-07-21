# -*- coding: utf-8 -*-
"""Source-grounded local method research foundation for M6R-B1.

This module is deliberately local and deterministic: no LLM/API/network,
no Adapter execution, no training, and no prediction generation.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash
from .local_conflict_checker import LocalConflictChecker

METHOD_RESEARCH_VERSION = "m6r_b1"
SOURCE_TYPES = {
    "peer_reviewed_paper",
    "preprint",
    "survey",
    "official_documentation",
    "official_repository",
    "mature_open_source_repository",
    "benchmark",
    "competition_solution",
    "other",
}
VERIFICATION_STATUSES = {
    "verified_local_content",
    "metadata_only",
    "unverified",
    "invalid",
    "duplicate",
    "conflicting_metadata",
    "synthetic_test_only",
}
FORMAL_SOURCE_STATUS = "verified_local_content"
DISABLED_PROVIDER_STATUS = "disabled"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _safe_local_ref(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.name


def _reasonable_url(value: str) -> bool:
    if not value:
        return True
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class SourceProvider:
    """Contract for future source providers."""

    provider_name = "base"
    network_enabled = False
    enabled = False

    def fetch(self, *, source_manifest: str | Path) -> list[dict[str, Any]]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "enabled": self.enabled,
            "network_enabled": self.network_enabled,
        }


class DisabledSourceProvider(SourceProvider):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def fetch(self, *, source_manifest: str | Path) -> list[dict[str, Any]]:
        return []

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update({"status": DISABLED_PROVIDER_STATUS, "reason": "M6R-B1 local mode only"})
        return payload


class WebSearchProvider(DisabledSourceProvider):
    def __init__(self) -> None:
        super().__init__("web_search")


class PaperSearchProvider(DisabledSourceProvider):
    def __init__(self) -> None:
        super().__init__("paper_search")


class RepositorySearchProvider(DisabledSourceProvider):
    def __init__(self) -> None:
        super().__init__("repository_search")


class LocalSourcePackProvider(SourceProvider):
    provider_name = "local_source_pack"
    network_enabled = False
    enabled = True

    def fetch(self, *, source_manifest: str | Path) -> list[dict[str, Any]]:
        manifest_path = Path(source_manifest).resolve()
        payload = load_json(manifest_path)
        base = manifest_path.parent
        candidates: list[dict[str, Any]] = []
        for raw in _as_list(payload.get("sources")):
            if not isinstance(raw, dict):
                continue
            local_path = _resolve(base, raw.get("local_path"))
            content_hash = raw.get("content_hash") or (sha256_file(local_path) if local_path and local_path.exists() else "")
            core = {
                "source_type": raw.get("source_type", ""),
                "title": raw.get("title", ""),
                "authors_or_organization": raw.get("authors_or_organization", ""),
                "year": raw.get("year", ""),
                "venue": raw.get("venue", ""),
                "url": raw.get("url", ""),
                "repository_url": raw.get("repository_url", ""),
                "local_path": raw.get("local_path", ""),
                "content_hash": content_hash,
                "synthetic_test_only": bool(raw.get("synthetic_test_only", False)),
            }
            source_id = raw.get("source_id") or stable_hash(core)
            candidate = {
                **raw,
                "source_id": source_id,
                "content_hash": content_hash,
                "resolved_local_path": str(local_path) if local_path else "",
                "manifest_path": str(manifest_path),
                "provider": self.provider_name,
                "retrieved_at": raw.get("retrieved_at", "local_manifest"),
            }
            candidates.append(candidate)
        return sorted(candidates, key=lambda item: item["source_id"])


class SourceVerifier:
    def verify(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        by_hash: dict[str, str] = {}
        by_title: dict[str, tuple[str, str]] = {}
        for candidate in candidates:
            status, evidence, warnings = self._verify_one(candidate)
            content_hash = candidate.get("content_hash", "")
            title_key = _norm_text(candidate.get("title")).lower()
            if status != "invalid" and content_hash:
                if content_hash in by_hash and by_hash[content_hash] != candidate.get("source_id"):
                    status = "duplicate"
                    evidence.append({"check": "duplicate_content_hash", "matched_source_id": by_hash[content_hash]})
                else:
                    by_hash[content_hash] = candidate.get("source_id", "")
            if title_key:
                prior = by_title.get(title_key)
                if prior and prior[0] != content_hash:
                    status = "conflicting_metadata"
                    evidence.append({"check": "title_hash_conflict", "matched_source_id": prior[1]})
                else:
                    by_title[title_key] = (content_hash, candidate.get("source_id", ""))
            record = {
                **{k: v for k, v in candidate.items() if k != "resolved_local_path"},
                "verification_status": status,
                "verification_evidence": evidence,
                "warnings": sorted(set(warnings)),
                "promotion_eligible": status == FORMAL_SOURCE_STATUS and not bool(candidate.get("synthetic_test_only")),
            }
            records.append(record)
        return {
            "verification_version": METHOD_RESEARCH_VERSION,
            "records": records,
            "summary": {
                status: sum(1 for record in records if record["verification_status"] == status)
                for status in sorted(VERIFICATION_STATUSES)
            },
            "view_hash": stable_hash(records),
        }

    def _verify_one(self, candidate: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[str]]:
        evidence: list[dict[str, Any]] = []
        warnings: list[str] = []
        required = ["source_type", "title", "authors_or_organization", "year"]
        missing = [key for key in required if _norm_text(candidate.get(key)) == ""]
        if candidate.get("source_type") not in SOURCE_TYPES:
            return "invalid", [{"check": "source_type", "status": "invalid"}], ["invalid_source_type"]
        if missing:
            return "invalid", [{"check": "required_metadata", "missing": missing}], ["missing_required_metadata"]
        if not candidate.get("local_path") and not candidate.get("url"):
            return "invalid", [{"check": "location", "status": "missing"}], ["missing_local_path_or_url"]
        if not _reasonable_url(_norm_text(candidate.get("url"))) or not _reasonable_url(_norm_text(candidate.get("repository_url"))):
            return "invalid", [{"check": "url_format", "status": "invalid"}], ["invalid_url"]
        memory_text = json.dumps(candidate, ensure_ascii=False).lower()
        if any(marker in memory_text for marker in ["llm internal knowledge", "model memory", "i know this without source"]):
            return "unverified", [{"check": "unverifiable_model_memory_statement", "status": "found"}], ["unverifiable_model_memory_statement"]
        local = Path(candidate.get("resolved_local_path", "")) if candidate.get("resolved_local_path") else None
        if local:
            if not local.exists():
                return "invalid", [{"check": "local_path_exists", "status": "missing"}], ["missing_local_content"]
            actual_hash = sha256_file(local)
            expected_hash = candidate.get("content_hash", "")
            evidence.append({"check": "local_path_exists", "status": "passed"})
            evidence.append({"check": "content_hash", "status": "passed" if actual_hash == expected_hash else "failed", "sha256": actual_hash})
            if actual_hash != expected_hash:
                return "invalid", evidence, ["content_hash_mismatch"]
            if candidate.get("synthetic_test_only"):
                return "synthetic_test_only", evidence + [{"check": "synthetic_test_only", "status": "true"}], []
            return "verified_local_content", evidence, []
        if candidate.get("url"):
            return "metadata_only", [{"check": "metadata_only", "status": "url_present_no_local_content"}], []
        return "unverified", evidence, ["missing_source_evidence"]


class SourceChunker:
    def __init__(self, *, max_chunk_chars: int = 1600) -> None:
        self.max_chunk_chars = max(200, int(max_chunk_chars))

    def chunk(self, source_records: list[dict[str, Any]], *, manifest_base: str | Path) -> list[dict[str, Any]]:
        base = Path(manifest_base).resolve()
        chunks: list[dict[str, Any]] = []
        for record in source_records:
            local_path = _resolve(base, record.get("local_path"))
            if not local_path or not local_path.exists():
                continue
            text = self._read_content(local_path)
            sections = self._sections(text, local_path.suffix.lower())
            chunk_index = 0
            for title, content in sections:
                for piece in self._split(content):
                    if not piece.strip():
                        continue
                    content_hash = stable_hash({"source_id": record["source_id"], "content": piece})
                    chunks.append({
                        "chunk_id": stable_hash({"source_id": record["source_id"], "chunk_index": chunk_index, "content_hash": content_hash}),
                        "source_id": record["source_id"],
                        "section_title": title,
                        "chunk_index": chunk_index,
                        "content": piece,
                        "content_hash": content_hash,
                        "character_count": len(piece),
                    })
                    chunk_index += 1
        return chunks

    def _read_content(self, path: Path) -> str:
        if path.suffix.lower() == ".json":
            return json_dumps(json.loads(path.read_text(encoding="utf-8")))
        return path.read_text(encoding="utf-8")

    def _sections(self, text: str, suffix: str) -> list[tuple[str, str]]:
        if suffix == ".json":
            return [("json", text)]
        if suffix in {".md", ".markdown"}:
            sections: list[tuple[str, list[str]]] = []
            current_title = "preamble"
            current_lines: list[str] = []
            for line in text.splitlines():
                if line.startswith("#"):
                    if current_lines:
                        sections.append((current_title, current_lines))
                    current_title = line.lstrip("#").strip() or "untitled"
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_lines:
                sections.append((current_title, current_lines))
            return [(title, "\n".join(lines).strip()) for title, lines in sections]
        return [("text", text.strip())]

    def _split(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self.max_chunk_chars:
            return [text]
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_chunk_chars)
            if end < len(text):
                split = text.rfind("\n", start, end)
                if split <= start:
                    split = text.rfind(" ", start, end)
                if split > start:
                    end = split
            pieces.append(text[start:end].strip())
            start = end
        return pieces


class MethodCardExtractor:
    def extract(
        self,
        *,
        source_manifest: dict[str, Any],
        source_records: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        manifest_path: str | Path,
    ) -> dict[str, Any]:
        base = Path(manifest_path).resolve().parent
        cards: list[dict[str, Any]] = []
        extraction_records: list[dict[str, Any]] = []
        chunk_by_source: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunk_by_source.setdefault(chunk["source_id"], []).append(chunk)
        for raw in _as_list(source_manifest.get("method_cards")):
            if isinstance(raw, dict):
                card = dict(raw)
                card.setdefault("extraction_mode", "structured_fixture")
                cards.append(self._finalize_card(card))
                extraction_records.append(self._record(card, "structured_fixture", "inline"))
        for rel_path in _as_list(source_manifest.get("method_card_files")):
            path = _resolve(base, rel_path)
            if path and path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payloads = payload if isinstance(payload, list) else [payload]
                for raw in payloads:
                    if isinstance(raw, dict):
                        card = dict(raw)
                        card.setdefault("extraction_mode", "structured_fixture")
                        cards.append(self._finalize_card(card))
                        extraction_records.append(self._record(card, "structured_fixture", str(rel_path)))
        for source in source_records:
            template = source.get("method_metadata")
            if not isinstance(template, dict):
                continue
            source_chunks = chunk_by_source.get(source["source_id"], [])
            if not source_chunks:
                continue
            first_chunk = source_chunks[0]
            card = {
                **template,
                "extraction_mode": "deterministic_template",
                "source_refs": template.get("source_refs") or [{"source_id": source["source_id"], "chunk_id": first_chunk["chunk_id"]}],
            }
            cards.append(self._finalize_card(card))
            extraction_records.append(self._record(card, "deterministic_template", source["source_id"]))
        return {
            "extraction_version": METHOD_RESEARCH_VERSION,
            "method_cards": sorted(cards, key=lambda item: item["method_id"]),
            "records": extraction_records,
            "view_hash": stable_hash(cards),
        }

    def _finalize_card(self, card: dict[str, Any]) -> dict[str, Any]:
        core = {k: v for k, v in card.items() if k not in {"method_id", "created_at", "updated_at"}}
        method_id = stable_hash({"method_card_version": METHOD_RESEARCH_VERSION, "core": core})
        card["method_id"] = card.get("method_id") or method_id
        card["expected_method_id"] = method_id
        card.setdefault("status", "proposed")
        card.setdefault("method_card_version", METHOD_RESEARCH_VERSION)
        return card

    def _record(self, card: dict[str, Any], mode: str, source: str) -> dict[str, Any]:
        core = {"method_id": card.get("method_id"), "mode": mode, "source": source}
        return {
            "extraction_record_id": stable_hash(core),
            "method_id": card.get("method_id", ""),
            "extraction_mode": mode,
            "source": source,
            "status": "extracted",
        }


class MethodCardValidator:
    def __init__(self, *, allow_synthetic_test_sources: bool = True) -> None:
        self.allow_synthetic_test_sources = allow_synthetic_test_sources

    def validate(
        self,
        *,
        method_card: dict[str, Any],
        research_brief: dict[str, Any],
        source_records: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        required = [
            "method_id",
            "target_problem_ids",
            "target_scope_refs",
            "core_hypothesis",
            "required_inputs",
            "information_source_type",
            "new_information_status",
            "applicable_conditions",
            "inapplicable_conditions",
            "minimal_experiment",
            "success_conditions",
            "failure_conditions",
            "stop_conditions",
            "source_refs",
        ]
        missing = [key for key in required if method_card.get(key) in (None, "", [])]
        expected = method_card.get("expected_method_id") or MethodCardExtractor()._finalize_card(dict(method_card)).get("expected_method_id")
        if method_card.get("method_id") != expected:
            return self._result("invalid", method_card, ["method_id_not_deterministic"], False)
        if missing:
            return self._result("invalid", method_card, [f"missing:{key}" for key in missing], False)
        if policy.get("allow_internal_model_knowledge_as_source") is not False and method_card.get("information_source_type") == "internal_model_knowledge":
            return self._result("invalid", method_card, ["unsafe_policy_internal_model_knowledge"], False)
        if method_card.get("information_source_type") == "internal_model_knowledge":
            return self._result("insufficient_source_support", method_card, ["internal_model_knowledge_forbidden"], False)
        brief_problems = set(map(str, _as_list(research_brief.get("target_problem_ids"))))
        card_problems = set(map(str, _as_list(method_card.get("target_problem_ids"))))
        if brief_problems and not (brief_problems & card_problems):
            return self._result("scope_mismatch", method_card, ["target_problem_ids_do_not_overlap_brief"], False)
        if not self._scope_matches(method_card.get("target_scope_refs", {}), research_brief.get("scope_refs", {})):
            return self._result("scope_mismatch", method_card, ["target_scope_refs_do_not_match_brief"], False)
        source_by_id = {record["source_id"]: record for record in source_records}
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        statuses: list[str] = []
        for ref in _as_list(method_card.get("source_refs")):
            if not isinstance(ref, dict) or not ref.get("source_id") or not ref.get("chunk_id"):
                return self._result("insufficient_source_support", method_card, ["source_ref_missing_source_or_chunk"], False)
            source = source_by_id.get(ref["source_id"])
            if not source:
                return self._result("insufficient_source_support", method_card, ["source_ref_unresolved"], False)
            if ref["chunk_id"] not in chunk_ids:
                return self._result("insufficient_source_support", method_card, ["chunk_ref_unresolved"], False)
            statuses.append(source.get("verification_status", "unverified"))
        if any(status in {"invalid", "duplicate", "conflicting_metadata", "unverified", "metadata_only"} for status in statuses):
            return self._result("unverified_source", method_card, [f"source_status:{status}" for status in sorted(set(statuses))], False)
        if any(status == "synthetic_test_only" for status in statuses):
            if self.allow_synthetic_test_sources:
                return self._result("valid", method_card, ["synthetic_test_only_not_promotion_eligible"], False)
            return self._result("unverified_source", method_card, ["synthetic_source_forbidden"], False)
        return self._result("valid", method_card, [], True)

    def _scope_matches(self, card_scope: dict[str, Any], brief_scope: dict[str, Any]) -> bool:
        if not brief_scope:
            return True
        if card_scope == brief_scope:
            return True
        for key, value in brief_scope.items():
            if key in card_scope and card_scope.get(key) == value:
                return True
        return False

    def _result(self, status: str, card: dict[str, Any], reasons: list[str], promotion_eligible: bool) -> dict[str, Any]:
        return {
            "validation_id": stable_hash({"method_id": card.get("method_id"), "status": status, "reasons": sorted(reasons)}),
            "method_id": card.get("method_id", ""),
            "validation_status": status,
            "valid": status == "valid",
            "promotion_eligible": promotion_eligible,
            "reasons": sorted(set(reasons)),
        }


class MethodRanker:
    def rank(
        self,
        *,
        method_cards: list[dict[str, Any]],
        validations: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        source_records: list[dict[str, Any]],
        research_brief: dict[str, Any],
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        weights = policy.get("method_research_ranking_weights", {})
        top_k = int(policy.get("method_research_top_k", 2))
        by_validation = {item["method_id"]: item for item in validations}
        by_conflict = {item.get("method_id", item.get("candidate_method_id", "")): item for item in conflicts}
        source_status = {record["source_id"]: record.get("verification_status") for record in source_records}
        rows: list[dict[str, Any]] = []
        for card in method_cards:
            validation = by_validation.get(card["method_id"], {})
            conflict = by_conflict.get(card["method_id"], {})
            components = self._components(card, validation, conflict, source_status, research_brief)
            raw = round(sum((components.get(k) or 0.0) * float(weights.get(k, 0.0)) for k in weights), 12)
            conflict_penalty = self._conflict_penalty(conflict.get("conflict_status"))
            closed_penalty = 0.5 if "closed_branch_conflict" in _as_list(conflict.get("conflict_reasons")) else 0.0
            penalty_total = round(conflict_penalty + closed_penalty, 12)
            status, reasons = self._selection(validation, conflict)
            rows.append({
                "ranking_record_id": stable_hash({"method_id": card["method_id"], "components": components, "validation": validation.get("validation_status"), "conflict": conflict.get("conflict_status")}),
                "method_id": card["method_id"],
                "method_family": card.get("method_family", ""),
                "ranking_components": components,
                "raw_score": raw,
                "penalty_total": penalty_total,
                "final_score": round(raw - penalty_total, 12),
                "rank": None,
                "selection_status": status,
                "selection_reasons": reasons,
                "conflict_status": conflict.get("conflict_status", "insufficient_information"),
            })
        rows.sort(key=lambda item: (-item["final_score"], item["method_id"]))
        eligible_seen = 0
        for index, row in enumerate(rows, 1):
            row["rank"] = index
            if row["selection_status"] == "deferred":
                eligible_seen += 1
                if eligible_seen <= top_k:
                    row["selection_status"] = "selected"
                    row["selection_reasons"].append("top_k_selected")
                else:
                    row["selection_reasons"].append("top_k_limit")
        return {
            "ranking_version": METHOD_RESEARCH_VERSION,
            "weights": weights,
            "top_k": top_k,
            "items": rows,
            "view_hash": stable_hash(rows),
        }

    def _components(self, card: dict[str, Any], validation: dict[str, Any], conflict: dict[str, Any], source_status: dict[str, str], brief: dict[str, Any]) -> dict[str, float | None]:
        card_problems = set(map(str, _as_list(card.get("target_problem_ids"))))
        brief_problems = set(map(str, _as_list(brief.get("target_problem_ids"))))
        statuses = [source_status.get(ref.get("source_id", ""), "unverified") for ref in _as_list(card.get("source_refs")) if isinstance(ref, dict)]
        return {
            "problem_fit": 1.0 if not brief_problems or card_problems & brief_problems else 0.0,
            "scope_fit": 1.0 if validation.get("validation_status") != "scope_mismatch" else 0.0,
            "mechanism_fit": 0.8 if card.get("mechanism_id") else None,
            "source_verification": max([self._source_score(status) for status in statuses], default=0.0),
            "source_maturity": self._maturity_score(card.get("source_type", "")),
            "new_information_value": {"new_information": 1.0, "new_representation_only": 0.25, "same_information_as_failed_route": 0.05}.get(card.get("new_information_status"), 0.2),
            "local_history_novelty": {"new_direction": 1.0, "partial_overlap": 0.55, "high_overlap": 0.1, "exact_duplicate": 0.0, "insufficient_information": None}.get(conflict.get("conflict_status"), None),
            "implementation_availability": 0.8 if card.get("repository_url") or card.get("implementation_notes") else 0.55,
            "compute_cost": {"low": 0.9, "medium": 0.5, "high": 0.1}.get(card.get("compute_cost", "medium"), 0.5),
            "implementation_cost": {"low": 0.9, "medium": 0.5, "high": 0.1}.get(card.get("implementation_cost", "medium"), 0.5),
            "leakage_risk": {"low": 0.9, "medium": 0.5, "high": 0.0}.get(card.get("leakage_risk", "medium"), 0.5),
            "deployment_risk": {"low": 0.9, "medium": 0.5, "high": 0.0}.get(card.get("deployment_risk", "medium"), 0.5),
            "conflict_penalty": -self._conflict_penalty(conflict.get("conflict_status")),
            "closed_branch_penalty": -0.5 if "closed_branch_conflict" in _as_list(conflict.get("conflict_reasons")) else 0.0,
        }

    def _source_score(self, status: str) -> float:
        return {"verified_local_content": 1.0, "synthetic_test_only": 0.35, "metadata_only": 0.2}.get(status, 0.0)

    def _maturity_score(self, source_type: str) -> float:
        return {
            "peer_reviewed_paper": 0.9,
            "survey": 0.9,
            "official_documentation": 0.85,
            "official_repository": 0.85,
            "mature_open_source_repository": 0.85,
            "benchmark": 0.75,
            "competition_solution": 0.75,
            "preprint": 0.65,
            "other": 0.4,
        }.get(source_type, 0.4)

    def _conflict_penalty(self, status: str | None) -> float:
        return {"exact_duplicate": 1.0, "high_overlap": 0.65, "partial_overlap": 0.2, "new_direction": 0.0, "insufficient_information": 0.4}.get(status or "", 0.4)

    def _selection(self, validation: dict[str, Any], conflict: dict[str, Any]) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if validation.get("validation_status") in {"invalid", "scope_mismatch", "unverified_source", "conflicting_source", "duplicate_method"}:
            return "blocked", [f"validation:{validation.get('validation_status')}"]
        if validation.get("validation_status") == "insufficient_source_support":
            return "insufficient_evidence", ["insufficient_source_support"]
        if conflict.get("conflict_status") in {"exact_duplicate", "high_overlap"}:
            return "blocked", [f"conflict:{conflict.get('conflict_status')}"]
        if "closed_branch_conflict" in _as_list(conflict.get("conflict_reasons")):
            return "blocked", ["closed_branch_conflict"]
        if validation.get("promotion_eligible") is False:
            reasons.append("not_promotion_eligible")
        reasons.append(f"conflict:{conflict.get('conflict_status', 'insufficient_information')}")
        return "deferred", reasons


class MethodResearchRunner:
    def __init__(self, *, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(
        self,
        *,
        research_brief: str | Path,
        research_memory_root: str | Path,
        source_manifest: str | Path,
        research_policy: str | Path,
        out_root: str | Path,
        dry_run: bool = False,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        required_paths = {
            "research_brief": Path(research_brief),
            "source_manifest": Path(source_manifest),
            "research_policy": Path(research_policy),
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        memory_dir = self._resolve_memory_dir(research_memory_root)
        if memory_dir is None:
            missing.append("research_memory_root")
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        brief = load_json(required_paths["research_brief"])
        policy = load_json(required_paths["research_policy"])
        self._validate_policy(policy)
        source_manifest_payload = load_json(required_paths["source_manifest"])
        input_hashes = {
            "research_brief": sha256_file(required_paths["research_brief"]),
            "source_manifest": sha256_file(required_paths["source_manifest"]),
            "research_policy": sha256_file(required_paths["research_policy"]),
            "research_memory_manifest": sha256_file(memory_dir / "research_manifest.json") if (memory_dir / "research_manifest.json").exists() else "",
            "method_attempt_ledger": sha256_file(memory_dir / "method_attempt_ledger.json") if (memory_dir / "method_attempt_ledger.json").exists() else "",
            "failure_ledger": sha256_file(memory_dir / "failure_ledger.json") if (memory_dir / "failure_ledger.json").exists() else "",
        }
        run_id = stable_hash({"version": METHOD_RESEARCH_VERSION, "input_hashes": input_hashes, "policy": policy.get("method_research_ranking_weights", {})})
        out = Path(out_root)
        out = out if out.is_absolute() else self.project_root / out
        run_dir = out / run_id
        manifest_path = run_dir / "research_run_manifest.json"
        if dry_run:
            return {"status": "dry_run", "run_id": run_id, "artifacts": {}, "missing_inputs": []}
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": "duplicate", "run_id": run_id, "artifacts": manifest.get("artifacts", {}), "missing_inputs": []}
        provider = LocalSourcePackProvider()
        candidates = provider.fetch(source_manifest=required_paths["source_manifest"])
        verification = SourceVerifier().verify(candidates)
        source_records = verification["records"]
        chunker = SourceChunker(max_chunk_chars=policy.get("source_chunk_max_chars", 1600))
        chunks = chunker.chunk(source_records, manifest_base=required_paths["source_manifest"].parent)
        extraction = MethodCardExtractor().extract(source_manifest=source_manifest_payload, source_records=source_records, chunks=chunks, manifest_path=required_paths["source_manifest"])
        method_cards = extraction["method_cards"]
        validator = MethodCardValidator(allow_synthetic_test_sources=True)
        validations = [
            validator.validate(method_card=card, research_brief=brief, source_records=source_records, chunks=chunks, policy=policy)
            for card in method_cards
        ]
        attempts = load_json(memory_dir / "method_attempt_ledger.json") if (memory_dir / "method_attempt_ledger.json").exists() else {"attempts": []}
        failures = load_json(memory_dir / "failure_ledger.json") if (memory_dir / "failure_ledger.json").exists() else {"failures": []}
        profile = load_json(memory_dir / "research_problem_profile.json") if (memory_dir / "research_problem_profile.json").exists() else {}
        checker = LocalConflictChecker()
        conflicts = []
        for card in method_cards:
            conflict = checker.check(method_card=card, method_attempt_ledger=attempts, failure_ledger=failures, research_problem_profile=profile)
            conflict["method_id"] = card["method_id"]
            conflict["candidate_method_id"] = card["method_id"]
            conflicts.append(conflict)
        ranking = MethodRanker().rank(method_cards=method_cards, validations=validations, conflicts=conflicts, source_records=source_records, research_brief=brief, policy=policy)
        index = self._problem_method_index(method_cards, conflicts, ranking)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        payloads = {
            "source_registry": {"registry_version": METHOD_RESEARCH_VERSION, "provider": provider.describe(), "sources": source_records, "view_hash": stable_hash(source_records)},
            "source_verification": verification,
            "method_cards": {"method_card_version": METHOD_RESEARCH_VERSION, "items": method_cards, "validations": validations, "extraction_records": extraction["records"], "view_hash": stable_hash({"cards": method_cards, "validations": validations})},
            "method_conflicts": {"conflict_version": METHOD_RESEARCH_VERSION, "items": conflicts, "view_hash": stable_hash(conflicts)},
            "method_ranking": ranking,
            "problem_method_index": index,
        }
        for name, payload in payloads.items():
            path = run_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        chunks_path = run_dir / "source_chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8", newline="\n") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        artifacts["source_chunks"] = rel_ref(chunks_path, self.project_root)
        report_path = run_dir / "METHOD_RESEARCH_REPORT.md"
        report_path.write_text(self._report(brief, verification, ranking), encoding="utf-8")
        artifacts["method_research_report"] = rel_ref(report_path, self.project_root)
        manifest = {
            "run_version": METHOD_RESEARCH_VERSION,
            "run_id": run_id,
            "created_at_epoch_seconds": time.time(),
            "read_only": True,
            "calls_llm": False,
            "calls_api": False,
            "uses_network": False,
            "executes_adapter": False,
            "trains_model": False,
            "generates_prediction": False,
            "counts_as_experiment_round": False,
            "mutates_project_state": False,
            "mutates_predictions": False,
            "input_hashes": input_hashes,
            "provider_contracts": {
                "local_source_pack": LocalSourcePackProvider().describe(),
                "web_search": WebSearchProvider().describe(),
                "paper_search": PaperSearchProvider().describe(),
                "repository_search": RepositorySearchProvider().describe(),
            },
            "source_count": len(source_records),
            "chunk_count": len(chunks),
            "method_card_count": len(method_cards),
            "conflict_check_executed": len(conflicts) == len(method_cards),
            "view_hash": stable_hash({name: payload.get("view_hash") for name, payload in payloads.items()} | {"chunks": stable_hash(chunks)}),
            "artifacts": artifacts,
        }
        manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["research_run_manifest"] = rel_ref(manifest_path, self.project_root)
        return {"status": "completed", "run_id": run_id, "view_hash": manifest["view_hash"], "artifacts": artifacts, "missing_inputs": []}

    def _resolve_memory_dir(self, memory_root: str | Path) -> Path | None:
        path = Path(memory_root)
        path = path if path.is_absolute() else self.project_root / path
        if (path / "method_attempt_ledger.json").exists() or (path / "research_manifest.json").exists():
            return path
        if path.exists():
            kids = sorted([child for child in path.iterdir() if (child / "method_attempt_ledger.json").exists()])
            if len(kids) == 1:
                return kids[0]
        return None

    def _validate_policy(self, policy: dict[str, Any]) -> None:
        if policy.get("method_research_network_enabled", False) is not False:
            raise ValueError("method research network must be disabled by default")
        if policy.get("method_research_llm_enabled", False) is not False:
            raise ValueError("method research LLM must be disabled by default")
        if policy.get("allow_unverified_method_promotion") is not False:
            raise ValueError("unverified method promotion must be disabled")

    def _problem_method_index(self, cards: list[dict[str, Any]], conflicts: list[dict[str, Any]], ranking: dict[str, Any]) -> dict[str, Any]:
        methods_by_problem: dict[str, list[str]] = {}
        for card in cards:
            for problem_id in _as_list(card.get("target_problem_ids")):
                methods_by_problem.setdefault(str(problem_id), []).append(card["method_id"])
        failures_by_method = {
            conflict["method_id"]: conflict.get("conflict_reasons", [])
            for conflict in conflicts
            if conflict.get("conflict_status") in {"exact_duplicate", "high_overlap", "insufficient_information"}
        }
        return {
            "index_version": METHOD_RESEARCH_VERSION,
            "methods_by_problem": {k: sorted(v) for k, v in sorted(methods_by_problem.items())},
            "failures_by_method": failures_by_method,
            "ranking_by_method": {item["method_id"]: {"rank": item["rank"], "selection_status": item["selection_status"]} for item in ranking["items"]},
            "attempt_count": len(cards),
            "view_hash": stable_hash({"methods_by_problem": methods_by_problem, "failures_by_method": failures_by_method}),
        }

    def _report(self, brief: dict[str, Any], verification: dict[str, Any], ranking: dict[str, Any]) -> str:
        selected = [item for item in ranking["items"] if item["selection_status"] == "selected"]
        return "\n".join([
            "# M6R-B1 Local Method Research Report",
            "",
            f"Brief: `{brief.get('brief_id', '')}`",
            f"Source verification summary: `{json.dumps(verification.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
            f"Selected methods: `{len(selected)}`",
            "",
            "No LLM/API/network/Adapter/training/prediction was executed.",
            "",
        ])
