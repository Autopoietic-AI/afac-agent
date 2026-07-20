# -*- coding: utf-8 -*-
"""登记并审计当前线上冠军，不重新训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1_csv", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--online_score", type=float, required=True)
    parser.add_argument("--expected_rows", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    path = Path(args.a1_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "online_anchor_manifest.json"

    if not path.exists():
        result = {
            "status": "waiting_for_input",
            "missing_inputs": ["a1_csv"],
            "missing_files": {"a1_csv": str(path)},
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(3)

    frame = pd.read_csv(path)
    checks = {
        "columns_exact":
            list(frame.columns) == ["test_idx", "label"],
        "rows_exact":
            len(frame) == args.expected_rows,
        "unique_test_idx":
            not frame["test_idx"].duplicated().any(),
        "no_null":
            not frame.isna().any().any(),
        "label_range":
            bool(
                frame["label"].between(
                    0, args.num_classes - 1
                ).all()
            ),
    }
    passed = all(checks.values())
    current_hash = sha256(path)
    manifest = {
        "version": args.version,
        "online_score": args.online_score,
        "a1_csv": str(path.resolve()),
        "sha256": current_hash,
        "rows": int(len(frame)),
        "num_classes": args.num_classes,
        "checks": checks,
        "passed": passed,
        "registered_as_anchor": passed,
        "registration_status": "created",
        "idempotent": True,
    }

    if manifest_path.exists():
        try:
            previous = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            previous = {}
        same_anchor = (
            previous.get("version") == manifest["version"]
            and previous.get("online_score") == manifest["online_score"]
            and str(previous.get("sha256", "")).lower()
            == current_hash.lower()
            and previous.get("rows") == manifest["rows"]
            and previous.get("num_classes") == manifest["num_classes"]
            and previous.get("passed") == manifest["passed"]
        )
        manifest["registration_status"] = (
            "unchanged" if same_anchor else "updated"
        )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
