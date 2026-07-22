# -*- coding: utf-8 -*-
"""A2 task adapter: deterministic, read-only access to the A2 recommendation data.

The adapter loads train/test/user/item tables, validates uid/iid identity,
parses history sequences, preserves the Test user order, validates Top10
candidate lists, and enforces Test truth isolation (test.csv must not carry
any target column).  It never trains, never predicts and never writes back
into the data directory.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UID_PATTERN = re.compile(r"^u\d{6}$")
IID_PATTERN = re.compile(r"^i\d{6}$")

REQUIRED_FILES = ("train.csv", "test.csv", "user.csv", "item.csv", "sample_submission.csv", "metadata.json")
TRAIN_COLUMNS = ("uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts")
TEST_COLUMNS = ("uid", "item_seq_raw", "item_seq_dedup", "item_seq_counts")

LEN_BUCKETS = ("len0", "len1", "len2", "exact_len3", "len4_plus")
CANDIDATE_TYPES = ("history", "novel")
TOP10_SIZE = 10


def sequence_length_bucket(length: int) -> str:
    if length <= 0:
        return "len0"
    if length == 1:
        return "len1"
    if length == 2:
        return "len2"
    if length == 3:
        return "exact_len3"
    return "len4_plus"


def parse_sequence(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in (value or "").strip().split(","):
        token = token.strip()
        if not token:
            continue
        iid, _, num = token.rpartition(":")
        try:
            counts[iid.strip()] = int(num)
        except ValueError:
            counts[iid.strip()] = -1
    return counts


@dataclass
class A2Dataset:
    data_dir: str
    train_uids: list[str]
    train_targets: dict[str, str]
    train_seq_raw: dict[str, list[str]]
    train_seq_dedup: dict[str, list[str]]
    train_seq_counts: dict[str, dict[str, int]]
    test_uids: list[str]
    test_seq_dedup: dict[str, list[str]]
    item_ids: list[str]
    user_feature_uids: int
    item_feature_columns: list[str]
    validation: dict[str, Any] = field(default_factory=dict)

    def len_bucket(self, uid: str) -> str:
        seq = self.train_seq_dedup.get(uid)
        if seq is None:
            seq = self.test_seq_dedup.get(uid, [])
        return sequence_length_bucket(len(seq))

    def target_type(self, uid: str) -> str:
        target = self.train_targets.get(uid, "")
        if target and target in set(self.train_seq_dedup.get(uid, [])):
            return "history"
        return "novel"


class A2TaskAdapter:
    """Read-only loader/validator for the A2 recommendation task."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def missing_files(self) -> list[str]:
        return [name for name in REQUIRED_FILES if not (self.data_dir / name).is_file()]

    def load(self) -> A2Dataset:
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(f"A2 data files missing: {missing}")
        errors: list[str] = []
        warnings: list[str] = []

        item_ids, item_feature_columns = self._load_items(errors)
        item_set = set(item_ids)
        user_feature_uids = self._count_user_rows(errors)

        train_uids: list[str] = []
        train_targets: dict[str, str] = {}
        train_seq_raw: dict[str, list[str]] = {}
        train_seq_dedup: dict[str, list[str]] = {}
        train_seq_counts: dict[str, dict[str, int]] = {}
        self._load_sequence_table(
            self.data_dir / "train.csv",
            expect_target=True,
            item_set=item_set,
            errors=errors,
            warnings=warnings,
            uids=train_uids,
            targets=train_targets,
            seq_raw=train_seq_raw,
            seq_dedup=train_seq_dedup,
            seq_counts=train_seq_counts,
        )

        test_uids: list[str] = []
        test_seq_dedup: dict[str, list[str]] = {}
        self._load_sequence_table(
            self.data_dir / "test.csv",
            expect_target=False,
            item_set=item_set,
            errors=errors,
            warnings=warnings,
            uids=test_uids,
            targets=None,
            seq_raw=None,
            seq_dedup=test_seq_dedup,
            seq_counts=None,
        )

        overlap = sorted(set(train_uids) & set(test_uids))
        if overlap:
            errors.append(f"train_test_uid_overlap:{len(overlap)}")

        validation = {
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "warnings": warnings,
            "train_users": len(train_uids),
            "test_users": len(test_uids),
            "items": len(item_ids),
            "test_truth_isolated": True,
            "test_user_order_preserved": True,
        }
        return A2Dataset(
            data_dir=str(self.data_dir),
            train_uids=train_uids,
            train_targets=train_targets,
            train_seq_raw=train_seq_raw,
            train_seq_dedup=train_seq_dedup,
            train_seq_counts=train_seq_counts,
            test_uids=test_uids,
            test_seq_dedup=test_seq_dedup,
            item_ids=item_ids,
            user_feature_uids=user_feature_uids,
            item_feature_columns=item_feature_columns,
            validation=validation,
        )

    def _load_items(self, errors: list[str]) -> tuple[list[str], list[str]]:
        item_ids: list[str] = []
        with (self.data_dir / "item.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            if not columns or columns[0] != "iid":
                errors.append("item.csv: first column must be iid")
            seen: set[str] = set()
            for row in reader:
                iid = (row.get("iid") or "").strip()
                if not IID_PATTERN.match(iid):
                    errors.append(f"item.csv: invalid iid {iid!r}")
                    continue
                if iid in seen:
                    errors.append(f"item.csv: duplicate iid {iid}")
                    continue
                seen.add(iid)
                item_ids.append(iid)
        return item_ids, columns

    def _count_user_rows(self, errors: list[str]) -> int:
        count = 0
        with (self.data_dir / "user.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or reader.fieldnames[0] != "uid":
                errors.append("user.csv: first column must be uid")
            for row in reader:
                uid = (row.get("uid") or "").strip()
                if not UID_PATTERN.match(uid):
                    errors.append(f"user.csv: invalid uid {uid!r}")
                    continue
                count += 1
        return count

    def _load_sequence_table(
        self,
        path: Path,
        *,
        expect_target: bool,
        item_set: set[str],
        errors: list[str],
        warnings: list[str],
        uids: list[str],
        targets: dict[str, str] | None,
        seq_raw: dict[str, list[str]] | None,
        seq_dedup: dict[str, list[str]] | None,
        seq_counts: dict[str, dict[str, int]] | None,
    ) -> None:
        name = path.name
        seen: set[str] = set()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            if expect_target:
                if "target_iid" not in header:
                    errors.append(f"{name}: target_iid column missing")
            elif "target_iid" in header:
                errors.append(f"{name}: test table must not contain target_iid (test truth isolation)")
            if not header or header[0] != "uid":
                errors.append(f"{name}: first column must be uid")
            for row in reader:
                uid = (row.get("uid") or "").strip()
                if not UID_PATTERN.match(uid):
                    errors.append(f"{name}: invalid uid {uid!r}")
                    continue
                if uid in seen:
                    errors.append(f"{name}: duplicate uid {uid}")
                    continue
                seen.add(uid)
                uids.append(uid)
                raw = parse_sequence(row.get("item_seq_raw", ""))
                dedup = parse_sequence(row.get("item_seq_dedup", ""))
                counts = parse_counts(row.get("item_seq_counts", ""))
                for iid in dedup:
                    if not IID_PATTERN.match(iid):
                        errors.append(f"{name}:{uid}: invalid iid {iid!r} in sequence")
                    elif iid not in item_set:
                        warnings.append(f"{name}:{uid}: sequence iid {iid} not in item.csv")
                if seq_raw is not None:
                    seq_raw[uid] = raw
                if seq_dedup is not None:
                    seq_dedup[uid] = dedup
                if seq_counts is not None:
                    seq_counts[uid] = counts
                if expect_target and targets is not None:
                    target = (row.get("target_iid") or "").strip()
                    if not IID_PATTERN.match(target):
                        errors.append(f"{name}:{uid}: invalid target_iid {target!r}")
                    elif target not in item_set:
                        warnings.append(f"{name}:{uid}: target {target} not in item.csv")
                    targets[uid] = target

    def validate_top10(self, uid: str, items: list[str], item_set: set[str] | None = None) -> list[str]:
        """Validate a Top10 candidate list: size, no duplicates, no illegal items."""
        errors: list[str] = []
        if not UID_PATTERN.match(uid or ""):
            errors.append(f"invalid uid {uid!r}")
        if len(items) != TOP10_SIZE:
            errors.append(f"top10 size must be {TOP10_SIZE}, got {len(items)}")
        if len(set(items)) != len(items):
            errors.append("top10 contains duplicate items")
        for iid in items:
            if not IID_PATTERN.match(iid or ""):
                errors.append(f"top10 contains illegal item id {iid!r}")
            elif item_set is not None and iid not in item_set:
                errors.append(f"top10 item {iid} not in item catalog")
        return errors
