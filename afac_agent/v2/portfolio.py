# -*- coding: utf-8 -*-
"""AFAC v2.0 candidate portfolio with guarded parent selection.

The portfolio tracks candidates under explicit roles (incumbent, best_overall,
best_macro, best_target_bucket, best_test_like, best_low_cost,
best_new_information, best_orthogonal, diagnostic).  Parent selection is
guarded M5-style: the default parent is always the incumbent; requesting any
other role as parent requires explicit justification (orthogonal_reason,
expected_complementarity, scope_difference) and is rejected otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PORTFOLIO_ROLES = (
    "incumbent",
    "best_overall",
    "best_macro",
    "best_target_bucket",
    "best_test_like",
    "best_low_cost",
    "best_new_information",
    "best_orthogonal",
    "diagnostic",
)

JUSTIFICATION_FIELDS = (
    "orthogonal_reason",
    "expected_complementarity",
    "scope_difference",
)


class ParentSelectionRejected(ValueError):
    """Raised when a non-incumbent parent is requested without full justification."""


@dataclass
class PortfolioCandidate:
    """A registered candidate and its assigned roles."""

    candidate_id: str
    record: dict[str, Any]
    roles: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class Portfolio:
    """Role-assigned candidate registry with guarded parent selection."""

    def __init__(self) -> None:
        self._candidates: dict[str, PortfolioCandidate] = {}
        self._role_index: dict[str, str] = {}

    def register_candidate(self, record: dict[str, Any]) -> PortfolioCandidate:
        """Register a candidate record and assign portfolio roles automatically.

        ``record`` must contain ``candidate_id`` and a ``metrics`` mapping.
        Recognized optional keys: ``roles`` (explicit role hints) and
        ``no_op`` (True marks the candidate diagnostic-only, excluded from
        portfolio roles).
        """
        candidate_id = str(record.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("record must contain a non-empty candidate_id")
        metrics = record.get("metrics") or {}
        if not isinstance(metrics, dict):
            raise ValueError("record['metrics'] must be a mapping")
        metrics = {str(k): float(v) for k, v in metrics.items()}

        roles = [str(r) for r in record.get("roles", []) if str(r) in PORTFOLIO_ROLES]
        if record.get("no_op"):
            roles = ["diagnostic"]
        elif not roles:
            roles = self._infer_roles(candidate_id, metrics)

        candidate = PortfolioCandidate(candidate_id=candidate_id, record=dict(record), roles=roles, metrics=metrics)
        self._candidates[candidate_id] = candidate
        for role in roles:
            self._assign_role(role, candidate)
        return candidate

    def _infer_roles(self, candidate_id: str, metrics: dict[str, float]) -> list[str]:
        if not self._candidates:
            return ["incumbent"]
        roles: list[str] = []
        standard = metrics.get("standard")
        if standard is not None and standard > self._best_metric("standard"):
            roles.append("best_overall")
        macro = metrics.get("macro")
        if macro is not None and macro > self._best_metric("macro"):
            roles.append("best_macro")
        return roles or ["diagnostic"]

    def _best_metric(self, key: str) -> float:
        values = [c.metrics[key] for c in self._candidates.values() if key in c.metrics]
        return max(values) if values else float("-inf")

    def _assign_role(self, role: str, candidate: PortfolioCandidate) -> None:
        current_id = self._role_index.get(role)
        if current_id is None:
            self._role_index[role] = candidate.candidate_id
            return
        current = self._candidates[current_id]
        if self._role_score(role, candidate) > self._role_score(role, current):
            current.roles = [r for r in current.roles if r != role]
            self._role_index[role] = candidate.candidate_id
        else:
            candidate.roles = [r for r in candidate.roles if r != role]

    @staticmethod
    def _role_score(role: str, candidate: PortfolioCandidate) -> float:
        key_by_role = {
            "incumbent": "standard",
            "best_overall": "standard",
            "best_macro": "macro",
            "best_target_bucket": "target_bucket",
            "best_test_like": "test_like",
            "best_low_cost": "low_cost",
            "best_new_information": "new_information",
            "best_orthogonal": "orthogonality",
            "diagnostic": "standard",
        }
        return candidate.metrics.get(key_by_role.get(role, "standard"), float("-inf"))

    def get_role_holder(self, role: str) -> PortfolioCandidate | None:
        candidate_id = self._role_index.get(role)
        return self._candidates.get(candidate_id) if candidate_id else None

    def get_parent(
        self,
        requested_role: str = "incumbent",
        orthogonal_reason: str | None = None,
        expected_complementarity: str | None = None,
        scope_difference: str | None = None,
    ) -> PortfolioCandidate:
        """Return the parent candidate for a new branch.

        The default parent is always the incumbent.  Requesting any other role
        requires ALL of ``orthogonal_reason``, ``expected_complementarity`` and
        ``scope_difference``; a missing justification raises
        :class:`ParentSelectionRejected`.
        """
        if requested_role not in PORTFOLIO_ROLES:
            raise ParentSelectionRejected(f"unknown portfolio role: {requested_role!r}")
        if requested_role != "incumbent":
            missing = [
                name
                for name, value in (
                    ("orthogonal_reason", orthogonal_reason),
                    ("expected_complementarity", expected_complementarity),
                    ("scope_difference", scope_difference),
                )
                if not value
            ]
            if missing:
                raise ParentSelectionRejected(
                    f"non-incumbent parent {requested_role!r} rejected: missing justification {missing}"
                )
        holder = self.get_role_holder(requested_role)
        if holder is None:
            raise ParentSelectionRejected(f"no candidate registered for role {requested_role!r}")
        return holder
