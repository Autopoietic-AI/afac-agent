# -*- coding: utf-8 -*-
"""AFAC Agent v1命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import AgentOrchestrator
from .paths import PathResolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        default=".",
    )
    parser.add_argument(
        "--state",
        default="config/project_state.json",
    )
    parser.add_argument(
        "--registry",
        default="config/tool_registry.json",
    )
    parser.add_argument(
        "--trajectory",
        default="output/trajectory_A1.json",
    )
    parser.add_argument("--paths_config", default="")
    parser.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument("--npz_path", default="")
    parser.add_argument("--edges_csv", default="")
    parser.add_argument("--fold_file", default="")
    parser.add_argument("--profile_out_dir", default="")
    parser.add_argument("--anchor_csv", default="")
    parser.add_argument("--anchor_oof_npz", default="")
    parser.add_argument("--reference_oof_npz", default="")
    args = parser.parse_args()

    resolver = PathResolver(args.project_root, args.paths_config or None)
    root = resolver.project_root
    orchestrator = AgentOrchestrator(
        project_root=root,
        state_path=root / args.state,
        registry_path=root / args.registry,
        trajectory_path=root / args.trajectory,
    )

    npz_path = resolver.a1_npz(args.npz_path)
    edges_csv = resolver.a1_edges_csv(args.edges_csv)
    fold_file = resolver.a1_fold_file(args.fold_file)
    profile_out_dir = resolver.a1_profile_out_dir(args.profile_out_dir)
    anchor_csv = resolver.a1_anchor_csv(args.anchor_csv)
    anchor_oof_npz = resolver.a1_anchor_oof_npz(args.anchor_oof_npz)
    reference_oof_npz = resolver.a1_reference_oof_npz(args.reference_oof_npz)
    variables = {
        "python": sys.executable,
        "root": str(root),
        "npz_path": str(npz_path or ""),
        "edges_csv": str(edges_csv or ""),
        "fold_file": str(fold_file or ""),
        "out_dir": str(profile_out_dir),
        "anchor_csv": str(anchor_csv),
        "anchor_oof_npz": str(anchor_oof_npz or ""),
        "reference_oof_npz": str(reference_oof_npz or ""),
    }
    outcome = orchestrator.run_once(
        variables=variables,
        execute=args.execute,
    )
    print(json.dumps(
        outcome,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
