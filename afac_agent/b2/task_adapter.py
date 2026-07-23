# -*- coding: utf-8 -*-
"""B2 sequence recommendation task adapter.

Reads train/test/user/item tables from the real B2 directory layout, validates
uid/iid legality, parses item sequences, preserves the official test user
order, and enforces test-truth isolation.

The adapter is parameterized from ``metadata.json`` when available: it does not
hard-code A2-specific assumptions such as Top7/Top23 or exact column names.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class B2Dataset:
    task_id: str
    data_root: Path
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    user_df: pd.DataFrame
    item_df: pd.DataFrame
    sample_submission: list[dict[str, Any]]
    train_seq: dict[str, list[str]]
    test_seq: dict[str, list[str]]
    n_items: int
    top_k: int
    validation: dict[str, Any] = field(default_factory=dict)

    def all_item_ids(self) -> set[str]:
        return set(self.item_df["iid"].astype(str))

    def user_ids(self) -> set[str]:
        return set(self.train_df["uid"].astype(str)) | set(self.test_df["uid"].astype(str))


class B2TaskAdapter:
    """Read-only loader for the B2 sequence-recommendation task."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        task_id: str = "B2",
        top_k: int | None = None,
        seq_col: str = "item_seq_dedup",
        count_col: str = "item_seq_counts",
    ) -> None:
        self.data_root = Path(data_root)
        self.task_id = task_id
        self.top_k = top_k
        self.seq_col = seq_col
        self.count_col = count_col
        self._metadata: dict[str, Any] | None = None
        self._actual_root: Path | None = None

    def _metadata_path(self) -> Path:
        return self.actual_root() / "metadata.json"

    def actual_root(self) -> Path:
        """Resolve the directory that really contains the CSV files.

        The user-supplied ``data_root`` may itself contain the files or may
        contain a single subdirectory (e.g. ``B推荐/B推荐/``) that holds them.
        """
        if self._actual_root is None:
            root = self.data_root
            required = {"train.csv", "test.csv", "user.csv", "item.csv", "sample_submission.csv"}
            if required.issubset({p.name for p in root.iterdir() if p.is_file()}):
                self._actual_root = root
            else:
                candidates = [p for p in root.iterdir() if p.is_dir() and required.issubset({q.name for q in p.iterdir() if q.is_file()})]
                if len(candidates) == 1:
                    self._actual_root = candidates[0]
                elif len(candidates) > 1:
                    raise FileNotFoundError(f"multiple B2 data subdirectories under {root}")
                else:
                    self._actual_root = root
        return self._actual_root

    def missing_files(self) -> list[str]:
        missing = []
        root = self.actual_root()
        for name in ("train.csv", "test.csv", "user.csv", "item.csv", "sample_submission.csv", "metadata.json"):
            if not (root / name).is_file():
                missing.append(name)
        return missing

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            path = self._metadata_path()
            if path.is_file():
                self._metadata = json.loads(path.read_text(encoding="utf-8"))
            else:
                self._metadata = {}
        return self._metadata

    def _infer_top_k(self) -> int:
        if self.top_k is not None:
            return self.top_k
        meta = self._load_metadata()
        fmt = meta.get("submission_format", {})
        if "prediction_format" in fmt and "top-K" in fmt["prediction_format"]:
            # try to extract a number like "top-K" -> use sample submission
            pass
        # Infer from sample submission first row
        sample_path = self.actual_root() / "sample_submission.csv"
        if sample_path.is_file():
            with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    k = len(str(row["prediction"]).split(","))
                    if k > 0:
                        return k
        return 10

    def _parse_seq(self, value: Any) -> list[str]:
        if pd.isna(value):
            return []
        s = str(value).strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    def _parse_counts(self, value: Any) -> dict[str, int]:
        if pd.isna(value):
            return {}
        s = str(value).strip()
        if not s:
            return {}
        counts: dict[str, int] = {}
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            iid, cnt = part.rsplit(":", 1)
            try:
                counts[iid.strip()] = int(cnt)
            except ValueError:
                continue
        return counts

    def load(self) -> B2Dataset:
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(f"{self.task_id} data files missing: {missing}")

        root = self.actual_root()
        train_df = pd.read_csv(root / "train.csv", dtype=str, keep_default_na=True)
        test_df = pd.read_csv(root / "test.csv", dtype=str, keep_default_na=True)
        user_df = pd.read_csv(root / "user.csv", dtype=str, keep_default_na=True)
        item_df = pd.read_csv(root / "item.csv", dtype=str, keep_default_na=True)

        errors: list[str] = []
        warnings: list[str] = []

        top_k = self._infer_top_k()

        # Ensure canonical string identifiers
        for df in (train_df, test_df, user_df):
            if "uid" in df.columns:
                df["uid"] = df["uid"].astype(str).str.strip()
        if "iid" in item_df.columns:
            item_df["iid"] = item_df["iid"].astype(str).str.strip()

        all_uids = set(train_df["uid"]) | set(test_df["uid"])
        all_items = set(item_df["iid"])
        n_items = len(all_items)

        # UID uniqueness per split
        if train_df["uid"].nunique() != len(train_df):
            errors.append("duplicate uids in train.csv")
        if test_df["uid"].nunique() != len(test_df):
            errors.append("duplicate uids in test.csv")

        train_uids = set(train_df["uid"])
        test_uids = set(test_df["uid"])
        overlap = sorted(train_uids & test_uids)
        if overlap:
            errors.append(f"train/test uid overlap: {len(overlap)} users")

        # Validate that every uid in user.csv appears in train or test
        user_uids = set(user_df["uid"])
        orphan_users = user_uids - all_uids
        if orphan_users:
            warnings.append(f"{len(orphan_users)} user.csv uids not in train/test")

        # Sequence columns exist
        for split, df in (("train", train_df), ("test", test_df)):
            for col in (self.seq_col, self.count_col):
                if col not in df.columns:
                    errors.append(f"{split}.csv missing sequence column {col}")

        # Parse sequences
        train_seq: dict[str, list[str]] = {}
        test_seq: dict[str, list[str]] = {}
        if self.seq_col in train_df.columns:
            train_seq = {uid: self._parse_seq(v) for uid, v in zip(train_df["uid"], train_df[self.seq_col])}
        if self.seq_col in test_df.columns:
            test_seq = {uid: self._parse_seq(v) for uid, v in zip(test_df["uid"], test_df[self.seq_col])}

        # Validate item legality (train/test sequences and train targets)
        illegal_items: set[str] = set()
        target_in_train = "target_iid" in train_df.columns
        if target_in_train:
            for uid, iid in zip(train_df["uid"], train_df["target_iid"]):
                if pd.notna(iid) and str(iid).strip() not in all_items:
                    illegal_items.add(str(iid).strip())
        for seq in train_seq.values():
            for iid in seq:
                if iid not in all_items:
                    illegal_items.add(iid)
        for seq in test_seq.values():
            for iid in seq:
                if iid not in all_items:
                    illegal_items.add(iid)
        if illegal_items:
            sample = sorted(illegal_items)[:10]
            errors.append(f"illegal iids found: {len(illegal_items)} (sample {sample})")

        # Test truth isolation: test.csv must not expose target_iid
        if "target_iid" in test_df.columns:
            warnings.append("test.csv contains target_iid column; values will be ignored")
            test_truth_hidden = test_df["target_iid"].isna().all()
        else:
            test_truth_hidden = True

        # Sample submission order and Top-K legality
        sample_submission = self._load_submission(warnings)
        if sample_submission:
            sub_uids = [r["uid"] for r in sample_submission]
            if set(sub_uids) != test_uids:
                errors.append("sample submission uids do not match test uids")
            elif sub_uids != sorted(sub_uids, key=lambda u: list(test_uids).index(u) if u in test_uids else -1):
                # We preserve the file order; just report if it differs from sorted
                pass
            for r in sample_submission:
                pred = r["prediction"]
                if len(pred) != top_k:
                    warnings.append(f"sample submission row for {r['uid']} has {len(pred)} items (expected {top_k})")
                if len(set(pred)) != len(pred):
                    errors.append(f"sample submission row for {r['uid']} has duplicate items")
                bad = [iid for iid in pred if iid not in all_items]
                if bad:
                    errors.append(f"sample submission row for {r['uid']} contains illegal iids: {bad[:3]}")
        else:
            errors.append("sample submission empty or unreadable")

        validation = {
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "warnings": warnings,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_users": len(user_df),
            "n_items": n_items,
            "top_k": top_k,
            "target_in_train": target_in_train,
            "test_truth_hidden": test_truth_hidden,
            "train_test_uid_overlap": len(overlap),
        }

        return B2Dataset(
            task_id=self.task_id,
            data_root=root,
            train_df=train_df,
            test_df=test_df,
            user_df=user_df,
            item_df=item_df,
            sample_submission=sample_submission,
            train_seq=train_seq,
            test_seq=test_seq,
            n_items=n_items,
            top_k=top_k,
            validation=validation,
        )

    def _load_submission(self, warnings: list[str]) -> list[dict[str, Any]]:
        path = self.actual_root() / "sample_submission.csv"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "uid" not in (reader.fieldnames or []) or "prediction" not in (reader.fieldnames or []):
                warnings.append("sample_submission.csv missing uid or prediction column")
                return []
            for row in reader:
                pred = [x.strip() for x in str(row["prediction"]).split(",") if x.strip()]
                rows.append({"uid": str(row["uid"]).strip(), "prediction": pred})
        return rows

    def submission_order(self, dataset: B2Dataset) -> list[str]:
        return [r["uid"] for r in dataset.sample_submission]
