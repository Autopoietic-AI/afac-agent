# -*- coding: utf-8 -*-
"""统一A1 OOF解析器：Overall/Fold/Bucket/Class/Rescue。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold


def find_array(bundle, names):
    for name in names:
        if name in bundle:
            return bundle[name]
    raise KeyError(f"Cannot find any key from {names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--oof_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reference_oof_npz", default="")
    parser.add_argument("--fold_seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.npz_path)
    adj = sp.csr_matrix(
        (
            raw["adj_data"],
            raw["adj_indices"],
            raw["adj_indptr"],
        ),
        shape=tuple(raw["adj_shape"]),
    )
    labels = raw["labels"]
    train_idx = raw["train_idx"].astype(np.int64)
    y = labels[train_idx].astype(np.int64)

    out_degree = np.diff(adj.indptr)
    in_degree = np.asarray(adj.sum(axis=0)).ravel()
    isolated = (out_degree == 0) & (in_degree == 0)
    iso_train = isolated[train_idx]
    degree = out_degree + in_degree
    degree_train = degree[train_idx]

    pred_bundle = np.load(args.oof_npz)
    if "oof_proba" in pred_bundle:
        proba = pred_bundle["oof_proba"]
    elif "proba" in pred_bundle:
        proba = pred_bundle["proba"]
    elif "oof" in pred_bundle:
        proba = pred_bundle["oof"]
    else:
        proba = None

    if proba is not None:
        pred = proba.argmax(axis=1).astype(np.int64)
    else:
        pred = find_array(
            pred_bundle,
            ["oof_pred", "pred", "prediction"],
        ).astype(np.int64)

    if len(pred) != len(train_idx):
        raise ValueError(
            f"OOF rows {len(pred)} != train nodes {len(train_idx)}"
        )

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.fold_seed,
    )
    fold_id = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, valid) in enumerate(
        splitter.split(np.zeros(len(y)), y)
    ):
        fold_id[valid] = fold

    correct = pred == y
    overall = float(np.mean(correct))
    fold_rows = []
    for fold in range(args.folds):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "n": int(mask.sum()),
                "accuracy": float(np.mean(correct[mask])),
            }
        )

    class_rows = []
    for cls in sorted(np.unique(y)):
        mask = y == cls
        class_rows.append(
            {
                "class": int(cls),
                "n": int(mask.sum()),
                "correct": int(correct[mask].sum()),
                "error": int((~correct[mask]).sum()),
                "accuracy": float(np.mean(correct[mask])),
                "error_rate": float(np.mean(~correct[mask])),
            }
        )

    bucket_name = np.select(
        [
            iso_train,
            (~iso_train) & (degree_train == 1),
            (~iso_train)
            & (degree_train >= 2)
            & (degree_train <= 5),
            (~iso_train) & (degree_train >= 6),
        ],
        [
            "isolated",
            "degree_1",
            "degree_2_5",
            "degree_6_plus",
        ],
        default="other",
    )
    bucket_rows = []
    for bucket in sorted(np.unique(bucket_name)):
        mask = bucket_name == bucket
        bucket_rows.append(
            {
                "bucket": str(bucket),
                "n": int(mask.sum()),
                "accuracy": float(np.mean(correct[mask])),
                "error": int((~correct[mask]).sum()),
            }
        )

    rescue_damage = {}
    oracle = {}
    if args.reference_oof_npz:
        reference_bundle = np.load(args.reference_oof_npz)
        if "oof_proba" in reference_bundle:
            ref_pred = reference_bundle[
                "oof_proba"
            ].argmax(axis=1)
        elif "proba" in reference_bundle:
            ref_pred = reference_bundle[
                "proba"
            ].argmax(axis=1)
        else:
            ref_pred = find_array(
                reference_bundle,
                ["oof_pred", "pred", "prediction"],
            )
        ref_pred = ref_pred.astype(np.int64)
        changed = pred != ref_pred
        rescue = changed & (pred == y) & (ref_pred != y)
        damage = changed & (pred != y) & (ref_pred == y)
        neutral = changed & (pred != y) & (ref_pred != y)
        rescue_damage = {
            "changed": int(changed.sum()),
            "rescue": int(rescue.sum()),
            "damage": int(damage.sum()),
            "neutral": int(neutral.sum()),
            "net": int(rescue.sum() - damage.sum()),
            "precision": float(
                rescue.sum()
                / max(rescue.sum() + damage.sum(), 1)
            ),
        }
        oracle = {
            "reference_or_candidate_accuracy": float(
                np.mean((ref_pred == y) | (pred == y))
            )
        }

    summary = {
        "overall_accuracy": overall,
        "correct": int(correct.sum()),
        "error": int((~correct).sum()),
        "graph_visible_accuracy": float(
            np.mean(correct[~iso_train])
        ),
        "isolated_accuracy": float(
            np.mean(correct[iso_train])
        ),
        "rescue_damage": rescue_damage,
        "oracle": oracle,
    }

    pd.DataFrame(fold_rows).to_csv(
        out_dir / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(class_rows).to_csv(
        out_dir / "class_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(bucket_rows).to_csv(
        out_dir / "bucket_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out_dir / "oof_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
