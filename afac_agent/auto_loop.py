# -*- coding: utf-8 -*-
"""连续执行可运行的注册工具，直到需要输入、失败或达到轮数。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import AgentOrchestrator
from .paths import PathResolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--state", default="config/project_state.json")
    parser.add_argument("--registry", default="config/tool_registry.json")
    parser.add_argument("--trajectory", default="output/trajectory_A1.json")
    parser.add_argument("--paths_config", default="")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--npz_path", default="")
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
    anchor_csv = resolver.a1_anchor_csv(args.anchor_csv)
    anchor_oof_npz = resolver.a1_anchor_oof_npz(args.anchor_oof_npz)
    reference_oof_npz = resolver.a1_reference_oof_npz(args.reference_oof_npz)
    variables = {
        "python": sys.executable,
        "root": str(root),
        "npz_path": str(npz_path or ""),
        "anchor_csv": str(anchor_csv),
        "anchor_oof_npz": str(anchor_oof_npz or ""),
        "reference_oof_npz": str(reference_oof_npz or ""),
    }

    outcomes = []
    for _ in range(args.max_steps):
        outcome = orchestrator.run_once(
            variables=variables,
            execute=True,
        )
        outcomes.append(outcome)
        status = outcome["result"]["status"]
        if status != "success":
            break

    print(json.dumps(
        {
            "steps": len(outcomes),
            "outcomes": outcomes,
            "final_status": outcomes[-1]["result"]["status"]
            if outcomes else "no_action",
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
