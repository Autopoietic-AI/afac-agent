# -*- coding: utf-8 -*-
"""M6 Decision Core: research synthesis, critic, and deterministic admission."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from afac_agent.llm.base import LLMRequest, LLMResponse
from afac_agent.llm.providers import ALIYUN_BAILIAN_DEFAULT_MODEL, ALIYUN_BAILIAN_PROVIDER, make_provider

from .event_store import json_dumps, load_json, rel_ref, sha256_file, stable_hash
from .live_method_research import _json_loads_lenient, _safe_text
from .local_conflict_checker import LocalConflictChecker

DECISION_CORE_VERSION = "m6_decision_core_v1"
ABSOLUTE_PATH_PATTERN = re.compile(r"[A-Za-z]:\\|/home/|/Users/|/mnt/|/tmp/", re.IGNORECASE)
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9._-]+|Authorization\s*[:=]\s*Bearer", re.IGNORECASE)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(*values: Any) -> str:
    return " ".join(str(value or "") for value in values).lower()


def _contains_any(text: str, words: list[str]) -> bool:
    low = text.lower()
    return any(word.lower() in low for word in words)


def _safe_summary(value: Any, limit: int = 6000) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_summary(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_summary(v, limit) for v in value[:20]]
    if isinstance(value, str):
        return _safe_text(value, limit)
    return value


class SourceProblemRelevanceValidator:
    """Validate whether a verified source actually supports the Method Card."""

    def validate(
        self,
        *,
        method_card: dict[str, Any],
        source_records: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        research_brief: dict[str, Any],
    ) -> dict[str, Any]:
        source_by_id = {source.get("source_id"): source for source in source_records}
        chunks_by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
        refs = _as_list(method_card.get("source_refs"))
        reasons: list[str] = []
        source_evidence: list[dict[str, Any]] = []
        if not refs:
            return self._result(method_card, "insufficient_evidence", reasons=["missing_source_refs"])
        statuses: list[str] = []
        for ref in refs:
            if not isinstance(ref, dict):
                reasons.append("malformed_source_ref")
                continue
            source = source_by_id.get(ref.get("source_id"))
            chunk = chunks_by_id.get(ref.get("chunk_id"))
            if not source or not source.get("title") or not source.get("source_id"):
                reasons.append("incomplete_source_identity")
                continue
            if source.get("verification_status") != "verified_local_content":
                reasons.append("source_not_verified_local_content")
            evidence_text = _text(source.get("title"), source.get("venue"), source.get("source_type"), chunk.get("content") if chunk else "")
            card_text = _text(method_card.get("method_name"), method_card.get("method_family"), method_card.get("mechanism_id"), method_card.get("core_hypothesis"), method_card.get("mechanism_summary"))
            checks = self._checks(evidence_text=evidence_text, card_text=card_text, method_card=method_card, research_brief=research_brief, source=source)
            status = self._status_from_checks(checks, evidence_text, card_text)
            statuses.append(status)
            source_evidence.append({"source_id": source.get("source_id"), "title": source.get("title"), "relevance_status": status, "checks": checks})
        if "directly_relevant" in statuses:
            status = "directly_relevant"
        elif "partially_relevant" in statuses:
            status = "partially_relevant"
        elif "inspiration_only" in statuses:
            status = "inspiration_only"
        elif "irrelevant" in statuses:
            status = "irrelevant"
        else:
            status = "insufficient_evidence"
        return self._result(method_card, status, reasons=sorted(set(reasons)), evidence=source_evidence)

    def _checks(self, *, evidence_text: str, card_text: str, method_card: dict[str, Any], research_brief: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        graph = _contains_any(evidence_text, ["graph", "node classification", "gnn", "message passing", "heterophily", "directed graph"])
        route = _contains_any(evidence_text, ["routing", "multipath", "path", "stability"])
        transformer = _contains_any(evidence_text, ["transformer", "mixture-of-experts", "moe", "prompt"])
        heterophily = _contains_any(evidence_text, ["heterophily", "node classification", "multiplex graph"])
        scale = _contains_any(evidence_text + card_text, ["scalenet", "scale invariance"])
        mechanism_words = [str(x).lower() for x in _as_list(method_card.get("mechanism_id"))]
        mechanism_match = any(word.replace("_", " ") in evidence_text for word in mechanism_words) or (
            "neighborhood_reliability_heterogeneity" in mechanism_words and _contains_any(evidence_text, ["heterophily", "message passing", "routing", "stability", "directed"])
        )
        return {
            "task_type_match": graph or heterophily,
            "data_modality_match": graph,
            "target_scope_match": bool(research_brief.get("scope_refs")),
            "mechanism_match": mechanism_match,
            "required_input_match": True,
            "evaluation_match": _contains_any(evidence_text, ["node classification", "benchmark", "graph"]),
            "source_claim_support": self._claim_supported(card_text, evidence_text),
            "cross_domain_distance": "high" if transformer else ("medium" if route and not graph else "low"),
            "scale_closed_route_hint": scale,
        }

    def _claim_supported(self, card_text: str, evidence_text: str) -> bool:
        if _contains_any(card_text, ["heterophily", "node classification", "graph"]) and not _contains_any(evidence_text, ["heterophily", "node classification", "graph", "gnn", "message passing"]):
            return False
        tokens = {t for t in re.findall(r"[a-zA-Z][a-zA-Z_\-]{4,}", card_text.lower()) if t not in {"method", "source", "grounded", "current", "scope"}}
        evidence = evidence_text.lower()
        hits = sum(1 for token in tokens if token.replace("_", " ") in evidence or token in evidence)
        return hits >= 1

    def _status_from_checks(self, checks: dict[str, Any], evidence_text: str, card_text: str) -> str:
        if checks.get("scale_closed_route_hint"):
            return "partially_relevant"
        if checks.get("cross_domain_distance") == "high":
            return "inspiration_only"
        if checks["task_type_match"] and checks["data_modality_match"] and checks["mechanism_match"] and checks["source_claim_support"]:
            return "directly_relevant"
        if checks["task_type_match"] and (checks["mechanism_match"] or checks["source_claim_support"]):
            return "partially_relevant"
        if checks.get("cross_domain_distance") == "medium":
            return "inspiration_only"
        return "irrelevant"

    def _result(self, method_card: dict[str, Any], status: str, *, reasons: list[str], evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload = {"method_id": method_card.get("method_id"), "method_name": method_card.get("method_name"), "relevance_status": status, "reasons": reasons, "source_evidence": evidence or []}
        payload["relevance_id"] = stable_hash(payload)
        return payload


class MethodQualityGate:
    def __init__(self) -> None:
        self.relevance = SourceProblemRelevanceValidator()
        self.conflict_checker = LocalConflictChecker()

    def evaluate(
        self,
        *,
        method_cards: list[dict[str, Any]],
        validations: list[dict[str, Any]],
        source_records: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        research_brief: dict[str, Any],
        attempts: dict[str, Any],
        failures: dict[str, Any],
        project_state: dict[str, Any],
    ) -> dict[str, Any]:
        validation_by_id = {item.get("method_id"): item for item in validations}
        rows = []
        for card in method_cards:
            relevance = self.relevance.validate(method_card=card, source_records=source_records, chunks=chunks, research_brief=research_brief)
            conflict = self.conflict_checker.check(
                method_card=card,
                method_attempt_ledger=attempts,
                failure_ledger=failures,
                closed_branches=_as_list(project_state.get("closed_branches")),
            )
            quality = self._quality(card, validation_by_id.get(card.get("method_id"), {}), relevance, conflict, project_state)
            rows.append({"method_card": self._card_summary(card), "validation": validation_by_id.get(card.get("method_id"), {}), "relevance": relevance, "conflict": conflict, "quality": quality})
        summary = {
            "eligible": [r["method_card"] for r in rows if r["quality"]["final_status"] == "eligible"],
            "inspiration_only": [r["method_card"] for r in rows if r["quality"]["final_status"] == "inspiration_only"],
            "blocked": [r["method_card"] for r in rows if r["quality"]["final_status"] == "blocked"],
            "invalid": [r["method_card"] for r in rows if r["quality"]["final_status"] == "invalid"],
            "deferred": [r["method_card"] for r in rows if r["quality"]["final_status"] == "deferred"],
        }
        return {"gate_version": DECISION_CORE_VERSION, "items": rows, "summary": summary, "view_hash": stable_hash(rows)}

    def _quality(self, card: dict[str, Any], validation: dict[str, Any], relevance: dict[str, Any], conflict: dict[str, Any], project_state: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        status = "eligible"
        if validation.get("valid") is not True:
            status = "invalid"; reasons.append("method_card_validator_failed")
        rel_status = relevance.get("relevance_status")
        if rel_status in {"irrelevant", "insufficient_evidence"}:
            status = "blocked"; reasons.append("irrelevant_source" if rel_status == "irrelevant" else "insufficient_evidence")
        elif rel_status == "inspiration_only":
            status = "inspiration_only"; reasons.append("inspiration_only_source")
        elif rel_status == "partially_relevant" and status == "eligible":
            status = "deferred"; reasons.append("partial_source_relevance")
        if any("incomplete_source_identity" in r for r in relevance.get("reasons", [])):
            status = "blocked"; reasons.append("incomplete_source_identity")
        if any(ev.get("checks", {}).get("source_claim_support") is False for ev in relevance.get("source_evidence", [])):
            if status == "eligible":
                status = "deferred"
            reasons.append("unsupported_claim")
        conflict_status = conflict.get("conflict_status")
        if conflict_status in {"exact_duplicate", "high_overlap"}:
            status = "blocked"; reasons.append(conflict_status)
        if "closed_branch_conflict" in _as_list(conflict.get("conflict_reasons")):
            status = "blocked"; reasons.append("closed_branch_conflict")
        name_text = _text(card.get("method_name"), card.get("method_family"), card.get("core_hypothesis"))
        if "scalenet" in name_text or "scale invariance" in name_text:
            reasons.append("closed_scalenet_route_related")
            status = "blocked"
        if "correct smooth" in name_text or "smoothing" in name_text:
            reasons.append("correct_smooth_closed_route")
            status = "blocked"
        if card.get("new_information_status") in {"same_information_as_failed_route", "new_representation_only"}:
            status = "blocked"; reasons.append("missing_new_information")
        return {"final_status": status, "blocked_reasons": sorted(set(reasons)), "admissible_for_synthesis": status == "eligible", "can_support_formal_proposal": status == "eligible"}

    def _card_summary(self, card: dict[str, Any]) -> dict[str, Any]:
        return {k: card.get(k) for k in ["method_id", "method_name", "method_family", "target_problem_ids", "target_scope_refs", "bucket_axes", "mechanism_id", "information_source_type", "new_information_status", "source_refs", "chunk_refs", "expected_gain_status"]}


class DecisionCoreRunner:
    def __init__(self, *, project_root: str | Path, llm_provider: Any | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.llm_provider = llm_provider

    def run(
        self,
        *,
        research_brief: str | Path,
        live_research_run: str | Path,
        research_memory_root: str | Path,
        project_state: str | Path,
        tool_registry: str | Path,
        out_root: str | Path = "artifacts/decision_core_runs",
        provider: str = ALIYUN_BAILIAN_PROVIDER,
        model: str = ALIYUN_BAILIAN_DEFAULT_MODEL,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        paths = {
            "research_brief": self._resolve(research_brief),
            "live_research_run": self._resolve(live_research_run),
            "research_memory_root": self._resolve(research_memory_root),
            "project_state": self._resolve(project_state),
            "tool_registry": self._resolve(tool_registry),
        }
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing, "artifacts": {}}
        run_id = stable_hash({"version": DECISION_CORE_VERSION, "input_hashes": {k: self._hash_path(p) for k, p in paths.items()}})
        out_dir = self._resolve(out_root) / run_id
        manifest_path = out_dir / "decision_manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = load_json(manifest_path)
            return {"status": "duplicate", "run_id": run_id, "artifacts": manifest.get("artifacts", {})}
        required_live_files = [
            paths["live_research_run"] / "method_cards_validated.json",
            paths["live_research_run"] / "source_verification.json",
            paths["live_research_run"] / "source_chunks.jsonl",
            paths["research_memory_root"] / "method_attempt_ledger.json",
            paths["research_memory_root"] / "failure_ledger.json",
        ]
        missing_files = [rel_ref(path, self.project_root) for path in required_live_files if not path.exists()]
        if missing_files:
            return {"status": "waiting_for_input", "failure_reason": "missing_required_inputs", "missing_inputs": missing_files, "artifacts": {}}
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._execute(paths=paths, out_dir=out_dir, run_id=run_id, provider=provider, model=model)
        return {"status": "completed", "run_id": run_id, "artifacts": artifacts}

    def _execute(self, *, paths: dict[str, Path], out_dir: Path, run_id: str, provider: str, model: str) -> dict[str, str]:
        started = time.time()
        live = paths["live_research_run"]
        brief = load_json(paths["research_brief"])
        cards_payload = load_json(live / "method_cards_validated.json")
        source_payload = load_json(live / "source_verification.json")
        ranking_payload = load_json(live / "method_ranking.json") if (live / "method_ranking.json").exists() else {"items": []}
        chunks = self._load_chunks(live / "source_chunks.jsonl")
        attempts = load_json(paths["research_memory_root"] / "method_attempt_ledger.json")
        failures = load_json(paths["research_memory_root"] / "failure_ledger.json")
        success = load_json(paths["research_memory_root"] / "success_ledger.json") if (paths["research_memory_root"] / "success_ledger.json").exists() else {"successes": []}
        profile = load_json(paths["research_memory_root"] / "research_problem_profile.json") if (paths["research_memory_root"] / "research_problem_profile.json").exists() else {}
        state = load_json(paths["project_state"])
        tools = load_json(paths["tool_registry"])
        source_records = source_payload.get("records", source_payload.get("items", []))
        gate = MethodQualityGate().evaluate(method_cards=cards_payload.get("items", []), validations=cards_payload.get("validations", []), source_records=source_records, chunks=chunks, research_brief=brief, attempts=attempts, failures=failures, project_state=state)
        source_audit = {"audit_version": DECISION_CORE_VERSION, "items": [{"method_id": row["method_card"].get("method_id"), "method_name": row["method_card"].get("method_name"), "relevance": row["relevance"]} for row in gate["items"]], "view_hash": stable_hash(gate)}
        synthesis_input = self._synthesis_input(brief, gate, attempts, failures, success, state, tools, profile, ranking_payload)
        outbound_audit = self._outbound_audit(synthesis_input)
        if not outbound_audit["secret_scan_pass"]:
            raise RuntimeError("decision core outbound audit failed: possible secret")
        llm = self.llm_provider or make_provider(provider, project_root=self.project_root)
        synth = self._call_synthesis(llm, synthesis_input, provider, model)
        proposals_raw = synth.get("proposals", {})
        critic = self._call_critic(llm, proposals_raw, synthesis_input, provider, model)
        final = self._revise_or_block(proposals_raw, critic, gate)
        admission = self._admit(final, critic, gate, tools, state)
        payloads = {
            "outbound_audit": outbound_audit,
            "source_relevance_audit": source_audit,
            "m6b_method_eligibility": gate,
            "method_quality_gate": gate,
            "synthesis_input": synthesis_input,
            "m6b_experiment_proposals": proposals_raw,
            "experiment_proposals_raw": proposals_raw,
            "m6c_critic_review": critic,
            "critic_review": critic,
            "m6c_revised_proposals": final,
            "experiment_proposals_final": final,
            "m5_admission_decision": admission,
            "admission_decision": admission,
        }
        artifacts: dict[str, str] = {}
        for name, payload in payloads.items():
            path = out_dir / f"{name}.json"
            path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
            artifacts[name] = rel_ref(path, self.project_root)
        report = out_dir / "DECISION_CORE_REPORT.md"
        report.write_text(self._report(gate, final, critic, admission), encoding="utf-8")
        artifacts["decision_core_report"] = rel_ref(report, self.project_root)
        run_id = out_dir.name
        manifest = {
            "run_version": DECISION_CORE_VERSION,
            "run_id": run_id,
            "created_at_epoch_seconds": time.time(),
            "duration_seconds": round(time.time() - started, 6),
            "read_only": True,
            "calls_llm": True,
            "llm_call_count": 2,
            "qwen_usage": {"synthesis": synth.get("audit", {}), "critic": critic.get("audit", {})},
            "outbound_audit": outbound_audit,
            "executes_adapter": False,
            "trains_model": False,
            "generates_prediction": False,
            "counts_as_experiment_round": False,
            "mutates_project_state": False,
            "mutates_predictions": False,
            "view_hash": stable_hash(payloads),
            "artifacts": artifacts,
        }
        mp = out_dir / "decision_manifest.json"
        mp.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        artifacts["decision_manifest"] = rel_ref(mp, self.project_root)
        return artifacts

    def _synthesis_input(self, brief: dict[str, Any], gate: dict[str, Any], attempts: dict[str, Any], failures: dict[str, Any], success: dict[str, Any], state: dict[str, Any], tools: dict[str, Any], profile: dict[str, Any], ranking: dict[str, Any] | None = None) -> dict[str, Any]:
        eligible = [row for row in gate["items"] if row["quality"]["final_status"] == "eligible"]
        inspiration = [row for row in gate["items"] if row["quality"]["final_status"] == "inspiration_only"]
        blocked = [row for row in gate["items"] if row["quality"]["final_status"] in {"blocked", "invalid", "deferred"}]
        return _safe_summary({
            "research_brief": {k: brief.get(k) for k in ["brief_id", "brief_type", "target_problem_ids", "scope_level", "scope_refs", "primary_research_question", "evidence_gaps", "required_new_information", "success_conditions", "failure_conditions", "stop_conditions"]},
            "hierarchical_problem_summary": profile,
            "eligible_method_cards": [row["method_card"] for row in eligible],
            "inspiration_only_cards": [{"method": row["method_card"], "reason": row["quality"]["blocked_reasons"]} for row in inspiration],
            "blocked_cards": [{"method": row["method_card"], "reason": row["quality"]["blocked_reasons"]} for row in blocked],
            "m6rb2_ranking_summary": _as_list((ranking or {}).get("items"))[:8],
            "local_history": {"attempts": attempts.get("attempts", [])[:12], "failures": failures.get("failures", [])[:12], "successes": success.get("successes", [])[:8], "closed_branches": state.get("closed_branches", [])},
            "available_tools": [{"name": t.get("name"), "adapter": t.get("adapter"), "risk": t.get("risk_level"), "counts_as_round": t.get("counts_as_experiment_round")} for t in tools.get("tools", [])],
            "constraints": {"no_adapter_execution": True, "no_training": True, "no_prediction": True, "rounds_used": state.get("budget", {}).get("rounds_used"), "requires_oof": True, "full_anchor_inputs_missing": True, "max_proposals": 2},
        }, limit=8000)

    def _call_synthesis(self, llm: Any, synthesis_input: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
        prompt = (
            "Return one JSON object with primary_proposal and optional fallback_proposal. "
            "If no eligible Method Card exists, return diagnostic-only proposal that repairs source-method alignment. "
            "Do not propose Adapter execution, training, prediction, test truth use, or score gain. "
            "Each proposal must include proposal_id, target_problem_ids, target_scope, target_bucket_axes, target_classes, target_mechanisms, parent_candidate, baseline, core_hypothesis, expected_new_information, selected_method_components, source_refs, local_evidence_refs, feature_or_signal_changes, model_changes, training_changes, routing_or_gate_changes, adapter_or_tool, required_inputs, missing_inputs, minimal_experiment, required_ablation, OOF_evaluation_plan, bucket_metrics, bucket_class_metrics, success_conditions, failure_conditions, stop_conditions, estimated_runtime, estimated_gpu_memory, round_cost, risk_level.\n"
            + json_dumps(synthesis_input)
        )
        response = llm.generate(LLMRequest(prompt=prompt[:38000], provider=provider, model=model, timeout_seconds=180, max_output_tokens=4096, temperature=0, metadata={"m6_decision_core_call": "synthesis"}))
        parsed = self._parse_response(response, fallback={"primary_proposal": self._fallback_diagnostic(synthesis_input), "fallback_proposal": None})
        return {"proposals": parsed, "audit": self._audit(response)}

    def _call_critic(self, llm: Any, proposals: dict[str, Any], synthesis_input: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
        prompt = (
            "Return one JSON object reviewing the proposal. Choose verdict approve, revise, or reject. "
            "Be strict: no eligible Method Card means only diagnostic-only can pass; inspiration_only cannot support formal experiment. "
            "Fields: verdict, critical_issues, required_revisions, unsupported_claims, closed_branch_conflicts, missing_evidence, minimal_safe_revision, audit_scores.\n"
            + json_dumps({"proposal": proposals, "synthesis_context": synthesis_input})
        )
        response = llm.generate(LLMRequest(prompt=prompt[:36000], provider=provider, model=model, timeout_seconds=180, max_output_tokens=2048, temperature=0, metadata={"m6_decision_core_call": "critic"}))
        parsed = self._parse_response(response, fallback={"verdict": "revise", "critical_issues": ["critic_output_unavailable"], "required_revisions": ["diagnostic_only"], "unsupported_claims": [], "closed_branch_conflicts": [], "missing_evidence": ["critic unavailable"], "minimal_safe_revision": {"mode": "diagnostic_only"}, "audit_scores": {}})
        parsed["audit"] = self._audit(response)
        return parsed

    def _parse_response(self, response: LLMResponse, fallback: dict[str, Any]) -> dict[str, Any]:
        if response.status != "completed":
            return fallback
        try:
            parsed = _json_loads_lenient(response.text)
        except Exception:
            return fallback
        return parsed if isinstance(parsed, dict) else fallback

    def _outbound_audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = json_dumps(payload)
        field_names: list[str] = []

        def visit(value: Any, prefix: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    name = f"{prefix}.{key}" if prefix else str(key)
                    field_names.append(name)
                    visit(item, name)
            elif isinstance(value, list):
                for item in value[:20]:
                    visit(item, prefix)

        visit(payload)
        low = text.lower()
        return {
            "audit_version": DECISION_CORE_VERSION,
            "outbound_field_names": sorted(set(field_names))[:400],
            "outbound_character_count": len(text),
            "outbound_hash": stable_hash(payload),
            "secret_scan_pass": SECRET_PATTERN.search(text) is None,
            "raw_data_present": any(token in low for token in ["raw_node_features", "feature_values", "label_values", "probability_values", "matrix_rows", "array_values"]),
            "test_truth_present": any(token in low for token in ["test_label", "test_labels", "test_y", "ground_truth_test"]),
            "absolute_path_present": ABSOLUTE_PATH_PATTERN.search(text) is not None,
            "sample_identifier_present": any(token in low for token in ["node_id", "user_id", "item_id", "sample_id"]),
        }

    def _revise_or_block(self, proposals: dict[str, Any], critic: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
        primary = proposals.get("primary_proposal") or self._fallback_diagnostic({"research_brief": {}})
        fallback = proposals.get("fallback_proposal")
        if critic.get("verdict") == "reject":
            return {"status": "blocked", "primary_proposal": None, "fallback_proposal": None, "revision_applied": "critic_reject_no_executable_proposal"}
        eligible_count = len(gate["summary"]["eligible"])
        if critic.get("verdict") == "revise" or eligible_count == 0:
            primary = self._safe_revision(primary, reason="no_eligible_source_supported_method" if eligible_count == 0 else "critic_revise")
            return {"status": "revised", "primary_proposal": primary, "fallback_proposal": None, "revision_applied": "diagnostic_only_scope"}
        return {"status": "accepted", "primary_proposal": primary, "fallback_proposal": fallback, "revision_applied": "none"}

    def _safe_revision(self, proposal: dict[str, Any], *, reason: str) -> dict[str, Any]:
        revised = dict(proposal)
        revised["proposal_id"] = revised.get("proposal_id") or stable_hash({"diagnostic": reason, "proposal": proposal})
        revised["adapter_or_tool"] = ""
        revised["model_changes"] = []
        revised["training_changes"] = []
        revised["routing_or_gate_changes"] = []
        revised["feature_or_signal_changes"] = ["diagnostic-only source-method relevance repair"]
        revised["minimal_experiment"] = {"mode": "diagnostic_only", "action": "repair source refs and rerun relevance gate"}
        revised["round_cost"] = 0
        revised["risk_level"] = "low"
        revised.setdefault("missing_inputs", [])
        revised["success_conditions"] = ["at least one directly_relevant primary source supports a Method Card"]
        revised["failure_conditions"] = ["no directly_relevant source support"]
        revised["stop_conditions"] = ["requires Adapter execution", "requires training", "requires prediction", "requires test truth"]
        revised["revision_reason"] = reason
        return revised

    def _admit(self, final: dict[str, Any], critic: dict[str, Any], gate: dict[str, Any], tools: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        proposal = final.get("primary_proposal")
        if not proposal:
            return {"status": "blocked", "reason_codes": ["critic_rejected"], "m5_authoritative": True}
        reasons: list[str] = []
        tool = str(proposal.get("adapter_or_tool") or "")
        registered = {str(t.get("name")) for t in tools.get("tools", [])}
        if tool and tool not in registered:
            reasons.append("unregistered_adapter_or_tool")
        if proposal.get("round_cost", 0) not in {0, "0"}:
            reasons.append("round_cost_requires_human_approval")
        if proposal.get("risk_level") not in {"low", "medium"}:
            reasons.append("risk_requires_human_approval")
        if any("test truth" in str(x).lower() for x in _as_list(proposal.get("stop_conditions")) + _as_list(proposal.get("required_inputs"))):
            reasons.append("test_truth_guard_present")
        if len(gate["summary"]["eligible"]) == 0:
            reasons.append("no_eligible_method_card")
        mode = proposal.get("minimal_experiment", {}).get("mode") if isinstance(proposal.get("minimal_experiment"), dict) else ""
        if mode == "diagnostic_only" and not tool:
            return {"status": "admitted_diagnostic_only", "reason_codes": sorted(set(reasons + ["diagnostic_only", "llm_does_not_override_m5"])), "requires_human_approval": False, "m5_authoritative": True}
        if reasons:
            return {"status": "waiting_for_input" if "unregistered_adapter_or_tool" in reasons else "deferred", "reason_codes": sorted(set(reasons)), "requires_human_approval": True, "m5_authoritative": True}
        return {"status": "admitted", "reason_codes": ["m5_admitted"], "requires_human_approval": True, "m5_authoritative": True}

    def _fallback_diagnostic(self, synthesis_input: dict[str, Any]) -> dict[str, Any]:
        brief = synthesis_input.get("research_brief", {}) if isinstance(synthesis_input, dict) else {}
        core = {"brief": brief, "mode": "diagnostic_only"}
        return {"proposal_id": stable_hash(core), "target_problem_ids": brief.get("target_problem_ids", []), "target_scope": brief.get("scope_refs", {}), "target_bucket_axes": [brief.get("scope_refs", {})], "target_classes": [], "target_mechanisms": ["source_problem_relevance"], "parent_candidate": "v53Q-1", "baseline": "current frozen champion", "core_hypothesis": "Current Method Cards need source-problem relevance repair before any experiment.", "expected_new_information": "verified source-to-method alignment", "selected_method_components": [], "source_refs": [], "local_evidence_refs": [], "feature_or_signal_changes": ["diagnostic-only source relevance audit"], "model_changes": [], "training_changes": [], "routing_or_gate_changes": [], "adapter_or_tool": "", "required_inputs": ["directly relevant primary source"], "missing_inputs": ["directly relevant source-supported Method Card"], "minimal_experiment": {"mode": "diagnostic_only"}, "required_ablation": [], "OOF_evaluation_plan": {"status": "not_applicable_diagnostic_only"}, "bucket_metrics": [], "bucket_class_metrics": [], "success_conditions": ["eligible Method Card found"], "failure_conditions": ["only inspiration/irrelevant sources"], "stop_conditions": ["requires training", "requires prediction", "requires test truth"], "estimated_runtime": "minutes", "estimated_gpu_memory": "0GB", "round_cost": 0, "risk_level": "low"}

    def _audit(self, response: LLMResponse) -> dict[str, Any]:
        audit = dict(response.audit)
        return {"provider": response.provider, "model": response.model, "latency_seconds": audit.get("latency_seconds"), "prompt_tokens": audit.get("prompt_tokens"), "completion_tokens": audit.get("completion_tokens"), "total_tokens": audit.get("total_tokens"), "finish_reason": audit.get("finish_reason"), "reasoning_present": bool(audit.get("reasoning_present")), "reasoning_character_count": int(audit.get("reasoning_character_count") or 0), "reasoning_hash": audit.get("reasoning_hash", "")}

    def _load_chunks(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _hash_path(self, path: Path) -> str:
        if path.is_file():
            return sha256_file(path)
        manifest = path / "research_run_manifest.json"
        if manifest.exists():
            return sha256_file(manifest)
        decision = path / "decision_manifest.json"
        if decision.exists():
            return sha256_file(decision)
        return stable_hash(sorted(p.name for p in path.iterdir())) if path.exists() and path.is_dir() else ""

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def _report(self, gate: dict[str, Any], final: dict[str, Any], critic: dict[str, Any], admission: dict[str, Any]) -> str:
        return "\n".join(["# M6 Decision Core Report", "", f"eligible: `{len(gate['summary']['eligible'])}`", f"inspiration_only: `{len(gate['summary']['inspiration_only'])}`", f"blocked: `{len(gate['summary']['blocked'])}`", f"critic: `{critic.get('verdict')}`", f"final: `{final.get('status')}`", f"admission: `{admission.get('status')}`", "", "No Adapter/training/prediction was executed.", ""])


class MockDecisionLLM:
    def __init__(self, critic_verdict: str = "revise") -> None:
        self.calls = 0
        self.critic_verdict = critic_verdict

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if request.metadata.get("m6_decision_core_call") == "critic":
            payload = {"verdict": self.critic_verdict, "critical_issues": [] if self.critic_verdict == "approve" else ["needs diagnostic-only narrowing"], "required_revisions": ["diagnostic_only"] if self.critic_verdict == "revise" else [], "unsupported_claims": [], "closed_branch_conflicts": [], "missing_evidence": [], "minimal_safe_revision": {"mode": "diagnostic_only"}, "audit_scores": {"source_support": "partial"}}
        else:
            payload = {"primary_proposal": {"proposal_id": "mock-proposal", "target_problem_ids": ["p"], "target_scope": {}, "target_bucket_axes": [], "target_classes": [], "target_mechanisms": ["neighborhood_reliability_heterogeneity"], "parent_candidate": "v53Q-1", "baseline": "frozen champion", "core_hypothesis": "diagnostic", "expected_new_information": "source alignment", "selected_method_components": [], "source_refs": [], "local_evidence_refs": [], "feature_or_signal_changes": [], "model_changes": [], "training_changes": [], "routing_or_gate_changes": [], "adapter_or_tool": "", "required_inputs": [], "missing_inputs": [], "minimal_experiment": {"mode": "diagnostic_only"}, "required_ablation": [], "OOF_evaluation_plan": {}, "bucket_metrics": [], "bucket_class_metrics": [], "success_conditions": [], "failure_conditions": [], "stop_conditions": ["requires test truth"], "estimated_runtime": "minutes", "estimated_gpu_memory": "0GB", "round_cost": 0, "risk_level": "low"}, "fallback_proposal": None}
        return LLMResponse(status="completed", provider="mock", model="mock", text=json.dumps(payload, ensure_ascii=False), audit={"latency_seconds": 0.0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
