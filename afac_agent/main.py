# -*- coding: utf-8 -*-
"""AFAC Agent v1命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.runner import AdapterRunner
from .feedback.builder import FeedbackBuilder
from .orchestrator import AgentOrchestrator
from .paths import PathResolver
from .registry import ToolRegistry


def _run_adapter(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main run-adapter")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--registry", default="config/tool_registry.json")
    parser.add_argument("--paths_config", default="")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--anchor_csv", default="")
    parser.add_argument("--v53q1_base_csv", default="")
    parser.add_argument("--v49a_oof_meta_csv", default="")
    parser.add_argument("--v49a_test_meta_csv", default="")
    parser.add_argument("--v53q1_audit_md", default="")
    parser.add_argument("--v53q1_patch_py", default="")
    parser.add_argument("--a1_npz", default="")
    parser.add_argument("--candidate_csv", default="")
    parser.add_argument("--parent_csv", default="")
    parser.add_argument("--candidate_oof_npz", default="")
    parser.add_argument("--audit_report", default="")
    parser.add_argument("--current_champion_csv", default="")
    parser.add_argument("--v46a1_base_csv", default="")
    parser.add_argument("--v53q1_patch_source", default="")
    parser.add_argument("--v49a_report", default="")
    parser.add_argument("--v49a_config", default="")
    parser.add_argument("--v49a_fold_results", default="")
    parser.add_argument("--adapter_output_root", default="")
    parser.add_argument("--allow-prediction-artifact", action="store_true")
    args = parser.parse_args(argv)

    resolver = PathResolver(args.project_root, args.paths_config or None)
    root = resolver.project_root
    registry = ToolRegistry(root / args.registry)
    tool = registry.get(args.tool)
    if args.adapter_output_root:
        tool.output_policy = dict(tool.output_policy or {})
        tool.output_policy["output_root"] = str(
            resolver.adapter_output_root(args.adapter_output_root)
        )
    if args.tool == "A1_V49A_EDGE_UTILITY_AUDIT":
        v49a_oof_meta_csv = resolver.a1_v49a_edge_oof_meta_csv(args.v49a_oof_meta_csv)
        v49a_test_meta_csv = resolver.a1_v49a_edge_test_meta_csv(args.v49a_test_meta_csv)
    else:
        v49a_oof_meta_csv = resolver.a1_v49a_oof_meta_csv(args.v49a_oof_meta_csv)
        v49a_test_meta_csv = resolver.a1_v49a_test_meta_csv(args.v49a_test_meta_csv)
    variables = {
        "python": sys.executable,
        "root": str(root),
        "anchor_csv": str(resolver.a1_anchor_csv(args.anchor_csv)),
        "v53q1_base_csv": str(resolver.a1_v53q1_base_csv(args.v53q1_base_csv) or ""),
        "v49a_oof_meta_csv": str(v49a_oof_meta_csv or ""),
        "v49a_test_meta_csv": str(v49a_test_meta_csv or ""),
        "v53q1_audit_md": str(resolver.a1_v53q1_audit_md(args.v53q1_audit_md)),
        "v53q1_patch_py": str(resolver.a1_v53q1_patch_py(args.v53q1_patch_py)),
        "v53q1_patch_source": str(
            resolver.resolve(args.v53q1_patch_source)
            or resolver.a1_v53q1_patch_py(args.v53q1_patch_py)
        ),
        "a1_npz": str(resolver.a1_npz(args.a1_npz) or ""),
        "candidate_csv": str(resolver.a1_v46a1_candidate_csv(args.candidate_csv) or ""),
        "parent_csv": str(resolver.a1_v46a1_parent_csv(args.parent_csv) or ""),
        "candidate_oof_npz": str(
            resolver.a1_v46a1_candidate_oof_npz(args.candidate_oof_npz) or ""
        ),
        "audit_report": str(resolver.a1_v46a1_audit_report(args.audit_report) or ""),
        "current_champion_csv": str(
            resolver.a1_current_champion_csv(args.current_champion_csv)
            if args.current_champion_csv
            else ""
        ),
        "v46a1_base_csv": str(
            resolver.a1_v49a_v46a1_base_csv(args.v46a1_base_csv) or ""
        ),
        "v49a_report": str(resolver.a1_v49a_report(args.v49a_report) or ""),
        "v49a_config": str(resolver.a1_v49a_config(args.v49a_config) or ""),
        "v49a_fold_results": str(
            resolver.a1_v49a_fold_results(args.v49a_fold_results) or ""
        ),
        "allow_prediction_artifact": (
            "true" if args.allow_prediction_artifact else ""
        ),
    }
    result = AdapterRunner(project_root=root).run(
        tool=tool,
        variables=variables,
        execute=args.execute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "dry_run", "duplicate"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _build_feedback(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main build-feedback")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--execution-result", required=True)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = PathResolver(args.project_root).project_root
    result = FeedbackBuilder(project_root=root).build(
        execution_result_path=args.execution_result,
        out_root=args.out_root or None,
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "dry_run"}:
        raise SystemExit(0)
    if result["status"] == "unavailable":
        raise SystemExit(3)
    raise SystemExit(2)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run-adapter":
        _run_adapter(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build-feedback":
        _build_feedback(sys.argv[2:])
        return

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
