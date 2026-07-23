# -*- coding: utf-8 -*-
"""AFAC v2.0 online feedback records and misuse guardrails.

Online competition scores are scalar feedback only.  They may calibrate
validation, build direction confidence, assess deployment risk, and budget
submissions — but they must never drive node-level correction, item-level
correction, or test-label inference.  Every record is identity-verified
against the sha256 of the actually submitted file before it may be used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from afac_agent.research.event_store import load_json, sha256_file

ALLOWED_USES = (
    "validation_calibration",
    "direction_confidence",
    "deployment_risk",
    "submission_budgeting",
)

FORBIDDEN_USES = (
    "node_level_correction",
    "item_level_correction",
    "test_label_inference",
)

SCORE_TYPE = "competition_scalar_feedback"


class OnlineFeedbackMisuse(ValueError):
    """Raised when online feedback is used for a forbidden purpose."""


@dataclass
class OnlineFeedbackRecord:
    """One verified online submission result."""

    task: str
    version: str
    online_score: float
    score_type: str
    submission_file: str
    submission_sha256: str
    registered_at: str = ""
    identity_status: str = "unverified"
    allowed_uses: list[str] = field(default_factory=lambda: list(ALLOWED_USES))
    forbidden_uses: list[str] = field(default_factory=lambda: list(FORBIDDEN_USES))

    def verify_identity(self, repo_root: str | Path) -> bool:
        """Recompute the submitted file hash and mark identity verified on match."""
        actual = sha256_file(Path(repo_root) / self.submission_file)
        self.identity_status = "verified" if actual == self.submission_sha256 else "mismatch"
        return self.identity_status == "verified"


def load_baseline_feedback(knowledge_root: str | Path) -> list[OnlineFeedbackRecord]:
    """Load knowledge/v1_6_baseline/online_results.json into verified records.

    ``knowledge_root`` is the repository root that contains ``knowledge/`` and
    ``artifacts/``.  Each record's submitted file hash is recomputed and must
    match before the record's identity is marked verified; a mismatch raises
    ``ValueError``.
    """
    knowledge_root = Path(knowledge_root)
    payload = load_json(knowledge_root / "knowledge" / "v1_6_baseline" / "online_results.json")
    records: list[OnlineFeedbackRecord] = []
    for entry in payload.get("entries", []):
        record = OnlineFeedbackRecord(
            task=str(entry.get("task", "")),
            version=str(entry.get("submitted_run_id", "")),
            online_score=float(entry.get("online_score")),
            score_type=str(entry.get("score_type", SCORE_TYPE)),
            submission_file=str(entry.get("submitted_file_path", "")),
            submission_sha256=str(entry.get("submitted_file_sha256", "")),
            registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            allowed_uses=list(entry.get("allowed_uses", ALLOWED_USES)),
            forbidden_uses=list(entry.get("forbidden_uses", FORBIDDEN_USES)),
        )
        if not record.verify_identity(knowledge_root):
            raise ValueError(
                f"identity verification failed for {record.task}: {record.submission_file} "
                f"(recorded {record.submission_sha256})"
            )
        records.append(record)
    return records


def use_feedback(record: OnlineFeedbackRecord, purpose: str) -> OnlineFeedbackRecord:
    """Guardrail: permit only allowed uses of online feedback.

    Raises :class:`OnlineFeedbackMisuse` for forbidden purposes
    (node_level_correction, item_level_correction, test_label_inference) and
    for any purpose outside the record's allowed list.  Also refuses
    unverified records.
    """
    if purpose in record.forbidden_uses or purpose in FORBIDDEN_USES:
        raise OnlineFeedbackMisuse(
            f"online feedback for {record.task} may not be used for {purpose!r}; "
            "online scores are scalar competition feedback only"
        )
    if purpose not in record.allowed_uses:
        raise OnlineFeedbackMisuse(f"purpose {purpose!r} is not in the allowed uses for {record.task}")
    if record.identity_status != "verified":
        raise OnlineFeedbackMisuse(f"online feedback for {record.task} is not identity-verified")
    return record
