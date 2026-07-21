# -*- coding: utf-8 -*-
"""Deterministic local conflict checker for M6R-A v2.1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .event_store import load_json, stable_hash

CONFLICT_STATUSES = {"exact_duplicate", "high_overlap", "partial_overlap", "new_direction", "insufficient_information"}

COMPARE_FIELDS = [
    "method_family", "information_source_type", "new_information_status", "bucket_axes",
    "class_id", "mechanism_id", "feature_family", "training_objective", "loss_family",
    "teacher_identity", "parent_candidate", "candidate_scope", "gate_policy", "branch_id",
]


def _norm(v: Any) -> Any:
    if isinstance(v, list):
        return sorted([_norm(x) for x in v], key=lambda x: str(x))
    if isinstance(v, dict):
        return {k: _norm(v[k]) for k in sorted(v)}
    return v


def _as_set(v: Any) -> set[str]:
    if v is None or v == "": return set()
    if isinstance(v, list): return {str(_norm(x)) for x in v}
    return {str(_norm(v))}


def _overlap(a: Any, b: Any) -> str:
    sa, sb = _as_set(a), _as_set(b)
    if not sa or not sb: return "unknown"
    if sa == sb: return "identical"
    inter = sa & sb
    if not inter: return "disjoint"
    ratio = len(inter) / min(len(sa), len(sb))
    if ratio >= 0.8: return "high_overlap"
    if ratio >= 0.2: return "partial_overlap"
    return "low_overlap"


class LocalConflictChecker:
    """Compare a proposed Method Card against local attempts/failures/history.

    This checker deliberately compares structure and evidence fields, not only
    method names. It never calls LLM/API/network/Adapter code.
    """

    def check(
        self,
        *,
        method_card: dict[str, Any] | str | Path,
        method_attempt_ledger: dict[str, Any] | str | Path,
        failure_ledger: dict[str, Any] | str | Path,
        confirmed_history: dict[str, Any] | str | Path | None = None,
        closed_branches: list[str] | None = None,
        research_problem_profile: dict[str, Any] | str | Path | None = None,
    ) -> dict[str, Any]:
        card = self._load(method_card)
        attempts_payload = self._load(method_attempt_ledger)
        failures_payload = self._load(failure_ledger)
        history = self._load(confirmed_history) if confirmed_history else {}
        profile = self._load(research_problem_profile) if research_problem_profile else {}
        missing = [f for f in ["method_family", "information_source_type", "new_information_status"] if not card.get(f)]
        if missing:
            return self._result("insufficient_information", card, [], [], [], {}, {}, [f"missing_fields:{','.join(missing)}"], profile)
        attempts = attempts_payload.get("attempts", []) if isinstance(attempts_payload, dict) else []
        failures = failures_payload.get("failures", []) if isinstance(failures_payload, dict) else []
        closed = set(map(str, closed_branches or [])) | set(map(str, history.get("closed_branches", [])))
        matched_attempts=[]; matched_methods=[]; matched_branches=[]; shared={}; different={}; reasons=[]
        best_status = "new_direction"; best_score = -1
        for attempt in attempts:
            cmp = self._compare(card, attempt)
            score = len(cmp["shared_components"])
            status = self._status_from_compare(card, attempt, cmp)
            if score > best_score or self._rank(status) > self._rank(best_status):
                best_score = score; best_status = status
            if status != "new_direction" or score:
                matched_attempts.append(str(attempt.get("attempt_id", "")))
                matched_methods.append(str(attempt.get("method_id", "")))
                matched_branches.append(str(attempt.get("branch_id", "")))
                for k,v in cmp["shared_components"].items(): shared.setdefault(k,v)
                for k,v in cmp["different_components"].items(): different.setdefault(k,v)
                reasons.extend(cmp["reasons"])
            if attempt.get("outcome") == "materialized_reference" and status in {"exact_duplicate", "high_overlap"}:
                reasons.append("materialized_reference_conflict")
                best_status = "exact_duplicate" if status == "exact_duplicate" else best_status
        failure_types = {str(f.get("failure_type")) for f in failures}
        if card.get("new_information_status") == "same_information_as_failed_route":
            best_status = "high_overlap" if best_status != "exact_duplicate" else best_status
            reasons.append("same_information_as_failed_route")
        if card.get("new_information_status") == "new_representation_only":
            best_status = "high_overlap" if best_status != "exact_duplicate" else best_status
            reasons.append("new_representation_only")
        mechanisms = _as_set(card.get("mechanism_id"))
        if mechanisms & {"uniform_smoothing_damage", "over_smoothing"} and {"negative_net", "negative_macro_gain"} & failure_types:
            best_status = "high_overlap" if best_status != "exact_duplicate" else best_status
            reasons.append("uniform_smoothing_failure_overlap")
        branch = str(card.get("branch_id", ""))
        if branch and branch in closed:
            reasons.append("closed_branch_conflict")
            matched_branches.append(branch)
        return self._result(best_status, card, matched_methods, matched_attempts, sorted(set(matched_branches)), shared, different, reasons, profile)

    def _load(self, value: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        if value is None: return {}
        if isinstance(value, dict): return value
        return load_json(value)

    def _compare(self, card: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        shared={}; diff={}; reasons=[]
        for field in COMPARE_FIELDS:
            av = card.get(field)
            bv = attempt.get(field)
            if field == "bucket_axes":
                bv = attempt.get("target_scope_refs", {}).get("bucket_axes", attempt.get("bucket_axes"))
            if field == "mechanism_id":
                bv = attempt.get("target_mechanism_ids", attempt.get("mechanism_id"))
            if _norm(av) == _norm(bv) and av not in (None, "", []):
                shared[field] = _norm(av)
            elif av not in (None, "", []) or bv not in (None, "", []):
                diff[field] = {"candidate": _norm(av), "existing": _norm(bv)}
        for field in ["information_source_type", "bucket_axes", "mechanism_id", "training_objective", "parent_candidate"]:
            if field in shared: reasons.append(f"shared_{field}")
        return {"shared_components": shared, "different_components": diff, "reasons": reasons}

    def _status_from_compare(self, card: dict[str, Any], attempt: dict[str, Any], cmp: dict[str, Any]) -> str:
        shared = cmp["shared_components"]
        if all(f in shared for f in ["method_family", "information_source_type", "bucket_axes", "training_objective", "branch_id"] if card.get(f)) and card.get("configuration_identity") == attempt.get("configuration_identity"):
            return "exact_duplicate"
        if card.get("information_source_type") == attempt.get("information_source_type") and card.get("new_information_status") in {"same_information_as_failed_route", "new_representation_only"}:
            return "high_overlap"
        if card.get("new_information_status") == "new_information" and card.get("information_source_type") != attempt.get("information_source_type"):
            if "bucket_axes" in shared or card.get("method_family") == attempt.get("method_family"):
                return "partial_overlap"
            return "new_direction"
        if len(shared) >= 5:
            return "high_overlap"
        if card.get("method_family") == attempt.get("method_family") and card.get("new_information_status") == "new_information":
            return "partial_overlap"
        if _overlap(card.get("bucket_axes"), attempt.get("target_scope_refs", {}).get("bucket_axes")) in {"partial_overlap", "high_overlap", "identical"}:
            return "partial_overlap"
        return "new_direction"

    def _rank(self, status: str) -> int:
        return {"insufficient_information": 0, "new_direction": 1, "partial_overlap": 2, "high_overlap": 3, "exact_duplicate": 4}.get(status, 0)

    def _result(self, status: str, card: dict[str, Any], methods: list[str], attempts: list[str], branches: list[str], shared: dict[str, Any], diff: dict[str, Any], reasons: list[str], profile: dict[str, Any]) -> dict[str, Any]:
        return {"conflict_version": "m6r_a_v2_1", "conflict_id": stable_hash({"card": card, "status": status, "reasons": sorted(set(reasons))}), "conflict_status": status, "matched_method_ids": sorted(set(filter(None, methods))), "matched_attempt_ids": sorted(set(filter(None, attempts))), "matched_branch_ids": sorted(set(filter(None, branches))), "shared_components": shared, "different_components": diff, "information_source_overlap": "identical" if "information_source_type" in shared else "unknown", "target_scope_overlap": "identical" if "bucket_axes" in shared else "unknown", "mechanism_overlap": "identical" if "mechanism_id" in shared else "unknown", "objective_overlap": "identical" if "training_objective" in shared else "unknown", "parent_overlap": "identical" if "parent_candidate" in shared else "unknown", "configuration_overlap": "identical" if "configuration_identity" in shared else "unknown", "conflict_reasons": sorted(set(reasons)), "reopen_requirements": ["human approval", "verified new information", "canonical Fold/final OOF evidence"] if branches or "closed_branch_conflict" in reasons else [], "profile_memory_id": profile.get("memory_id", "")}
