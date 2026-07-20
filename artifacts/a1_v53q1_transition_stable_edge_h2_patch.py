# -*- coding: utf-8 -*-
"""
AFAC2026 A1 v53Q-1
Transition-Stable Edge-H2 Micro Patch
======================================

Inputs
------
1. Current A1 online anchor CSV (v43C + v46A-1).
2. v49A OOF meta scores.
3. v49A Test meta scores.

Frozen procedure
----------------
1. Keep only v49A `confidence_plus_edge` rows selected by the original
   frozen rule.
2. For each transition base_pred -> h2_pred, require on OOF:
   - support >= 3 selected rows;
   - decisive precision >= 2/3;
   - net rescue >= +1;
   - observed in >= 2 folds.
3. Apply only those transitions to selected Test rows.
4. Change no other Test node.

The script also performs leave-one-fold-out validation of the transition
gate. It never reads Test labels and never tunes on leaderboard feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_NAME = "confidence_plus_edge"


def transition_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["base_pred"].astype(int).astype(str)
        + "→"
        + frame["h2_pred"].astype(int).astype(str)
    )


def select_allowed_transitions(
    frame: pd.DataFrame,
    *,
    minimum_support: int,
    minimum_precision: float,
    minimum_net: int,
    minimum_folds: int,
) -> tuple[set[str], pd.DataFrame]:
    rows = []
    allowed: set[str] = set()
    for transition, group in frame.groupby("transition"):
        rescue = int(group["rescue"].sum())
        damage = int(group["damage"].sum())
        neutral = int(group["neutral"].sum())
        decisive = rescue + damage
        precision = rescue / decisive if decisive else 0.0
        net = rescue - damage
        folds = int(group["fold"].nunique())
        support = int(len(group))
        passed = (
            support >= minimum_support
            and decisive > 0
            and precision >= minimum_precision
            and net >= minimum_net
            and folds >= minimum_folds
        )
        if passed:
            allowed.add(str(transition))
        rows.append(
            {
                "transition": str(transition),
                "support": support,
                "rescue": rescue,
                "damage": damage,
                "neutral": neutral,
                "net": net,
                "decisive_precision": precision,
                "folds_observed": folds,
                "passed": passed,
            }
        )
    return allowed, pd.DataFrame(rows).sort_values(
        ["passed", "net", "support"],
        ascending=[False, False, False],
    )


def cross_fit_validate(
    selected_oof: pd.DataFrame,
    *,
    minimum_support: int,
    minimum_precision: float,
    minimum_net: int,
    minimum_folds: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for held_fold in sorted(selected_oof["fold"].unique()):
        fit = selected_oof[selected_oof["fold"] != held_fold]
        valid = selected_oof[selected_oof["fold"] == held_fold]
        allowed, _ = select_allowed_transitions(
            fit,
            minimum_support=minimum_support,
            minimum_precision=minimum_precision,
            minimum_net=minimum_net,
            minimum_folds=minimum_folds,
        )
        accepted = valid[valid["transition"].isin(allowed)]
        rescue = int(accepted["rescue"].sum())
        damage = int(accepted["damage"].sum())
        neutral = int(accepted["neutral"].sum())
        rows.append(
            {
                "held_fold": int(held_fold),
                "allowed_transitions": sorted(allowed),
                "changed": int(len(accepted)),
                "rescue": rescue,
                "damage": damage,
                "neutral": neutral,
                "net": rescue - damage,
            }
        )
    fold_frame = pd.DataFrame(rows)
    total_rescue = int(fold_frame["rescue"].sum())
    total_damage = int(fold_frame["damage"].sum())
    decisive = total_rescue + total_damage
    summary = {
        "changed": int(fold_frame["changed"].sum()),
        "rescue": total_rescue,
        "damage": total_damage,
        "neutral": int(fold_frame["neutral"].sum()),
        "net": total_rescue - total_damage,
        "decisive_precision": (
            total_rescue / decisive if decisive else 0.0
        ),
        "nonnegative_folds": int((fold_frame["net"] >= 0).sum()),
        "fold_nets": fold_frame["net"].astype(int).tolist(),
    }
    return fold_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--oof_meta_csv", required=True)
    parser.add_argument("--test_meta_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_name", default="A1_v53q1_transition_stable_edge_h2_SAFE.csv")
    parser.add_argument("--minimum_support", type=int, default=3)
    parser.add_argument("--minimum_precision", type=float, default=2.0 / 3.0)
    parser.add_argument("--minimum_net", type=int, default=1)
    parser.add_argument("--minimum_folds", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base_csv)
    oof = pd.read_csv(args.oof_meta_csv)
    test = pd.read_csv(args.test_meta_csv)

    required_base = {"test_idx", "label"}
    if set(base.columns) != required_base:
        raise ValueError(
            f"Base CSV columns must be exactly {required_base}, got {list(base.columns)}"
        )
    if base["test_idx"].duplicated().any():
        raise ValueError("Base CSV has duplicate test_idx")
    if base.isna().any().any():
        raise ValueError("Base CSV contains null values")

    selected_oof = oof[
        (oof["model_name"] == MODEL_NAME)
        & (oof["selected"].astype(bool))
    ].copy()
    selected_test = test[
        (test["model_name"] == MODEL_NAME)
        & (test["selected"].astype(bool))
    ].copy()
    selected_oof["transition"] = transition_series(selected_oof)
    selected_test["transition"] = transition_series(selected_test)

    fold_frame, cross_fit = cross_fit_validate(
        selected_oof,
        minimum_support=args.minimum_support,
        minimum_precision=args.minimum_precision,
        minimum_net=args.minimum_net,
        minimum_folds=args.minimum_folds,
    )

    allowed, transition_frame = select_allowed_transitions(
        selected_oof,
        minimum_support=args.minimum_support,
        minimum_precision=args.minimum_precision,
        minimum_net=args.minimum_net,
        minimum_folds=args.minimum_folds,
    )
    accepted_test = selected_test[
        selected_test["transition"].isin(allowed)
    ].copy()

    patch = {
        int(row.global_idx): int(row.h2_pred)
        for row in accepted_test.itertuples()
    }
    if len(patch) != len(accepted_test):
        raise ValueError("Duplicate Test global_idx among accepted candidates")

    patched = base.copy()
    original = patched.set_index("test_idx")["label"].to_dict()
    missing = sorted(set(patch) - set(original))
    if missing:
        raise ValueError(f"Patch nodes absent from base CSV: {missing}")

    patched["label"] = [
        patch.get(int(idx), int(label))
        for idx, label in zip(patched["test_idx"], patched["label"])
    ]
    changed = patched[
        patched["label"].astype(int)
        != base["label"].astype(int)
    ].copy()
    if len(changed) != len(patch):
        raise ValueError(
            f"Expected {len(patch)} changed rows, observed {len(changed)}"
        )

    output_csv = out_dir / args.output_name
    patched.to_csv(output_csv, index=False, encoding="utf-8-sig")

    diff_rows = []
    for row in accepted_test.sort_values("global_idx").itertuples():
        idx = int(row.global_idx)
        diff_rows.append(
            {
                "test_idx": idx,
                "base_label": int(original[idx]),
                "patched_label": int(row.h2_pred),
                "transition": str(row.transition),
                "v49a_rescue_score": float(row.rescue_score),
            }
        )
    diff_frame = pd.DataFrame(diff_rows)
    diff_frame.to_csv(
        out_dir / "v53q1_transition_stable_patch_diff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transition_frame.to_csv(
        out_dir / "v53q1_transition_oof_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_frame.to_csv(
        out_dir / "v53q1_crossfit_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "version": "v53Q-1",
        "model_name": MODEL_NAME,
        "base_rows": int(len(base)),
        "original_v49a_oof_selected": int(len(selected_oof)),
        "original_v49a_test_selected": int(len(selected_test)),
        "transition_gate": {
            "minimum_support": args.minimum_support,
            "minimum_precision": args.minimum_precision,
            "minimum_net": args.minimum_net,
            "minimum_folds": args.minimum_folds,
        },
        "allowed_transitions": sorted(allowed),
        "cross_fit": cross_fit,
        "test_changes": int(len(diff_frame)),
        "test_patch": diff_rows,
        "test_labels_used": False,
        "leaderboard_feedback_used_for_selection": False,
        "submission_created": False,
    }
    (out_dir / "v53q1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Output CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
