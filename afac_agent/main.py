# -*- coding: utf-8 -*-
"""AFAC Agent v1命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .a1_closed_loop import run_a1_closed_loop
from .a2.integration import run_a2_integration
from .b1.closed_loop import run_b1_closed_loop
from .b2.closed_loop import run_b2_closed_loop
from .adapters.runner import AdapterRunner
from .feedback.builder import FeedbackBuilder
from .fusion_controller import run_fusion_controller
from .llm.base import LLMRequest
from .llm.providers import (
    ALIYUN_BAILIAN_DEFAULT_MODEL,
    ALIYUN_BAILIAN_PROVIDER,
    is_allowed_bailian_model,
    make_provider,
    redact_secret,
)
from .llm.shadow_planner import LLMShadowPlanner
from .evaluation_anchor_bootstrap import EvaluationAnchorBootstrap
from .m7_dry_run import M7DryRunOrchestrator
from .m7b_readiness import M7BReadinessRepair
from .orchestrator import AgentOrchestrator
from .paths import PathResolver
from .planning.deterministic_planner import DeterministicPlanner
from .registry import ToolRegistry
from .research import (
    DecisionCoreRunner,
    LiveCachedMethodResearchRunner,
    MethodResearchRunner,
    ResearchBriefBuilder,
    ResearchMemoryBuilder,
)


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
    parser.add_argument("--parent_oof_npz", default="")
    parser.add_argument("--canonical_fold_csv", default="")
    parser.add_argument("--anchor_manifest_json", default="")
    parser.add_argument("--node_bucket_csv", default="")
    parser.add_argument("--m2_profile_json", default="")
    parser.add_argument("--evaluation_policy_json", default="")
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
            resolver.resolve(args.candidate_oof_npz)
            or resolver.a1_v46a1_candidate_oof_npz("")
            or ""
        ),
        "parent_oof_npz": str(resolver.resolve(args.parent_oof_npz) or ""),
        "canonical_fold_csv": str(resolver.resolve(args.canonical_fold_csv) or ""),
        "anchor_manifest_json": str(resolver.resolve(args.anchor_manifest_json) or ""),
        "node_bucket_csv": str(resolver.resolve(args.node_bucket_csv) or ""),
        "m2_profile_json": str(resolver.resolve(args.m2_profile_json) or ""),
        "evaluation_policy_json": str(resolver.resolve(args.evaluation_policy_json) or ""),
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


def _plan_next(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main plan-next")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--problem-map", required=True)
    parser.add_argument("--feedback", action="append", default=[])
    parser.add_argument("--tool-registry", default="config/tool_registry.json")
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--history", default="history/confirmed_experiments_a1.json")
    parser.add_argument("--policy", default="config/planner_policy.json")
    parser.add_argument("--out-root", default="artifacts/plans")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)

    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = DeterministicPlanner(project_root=root).plan(
        problem_map_path=resolver.resolve(args.problem_map) or root / args.problem_map,
        feedback_paths=[
            resolver.resolve(path) or root / path
            for path in args.feedback
        ],
        tool_registry_path=resolver.resolve(args.tool_registry) or root / args.tool_registry,
        project_state_path=resolver.resolve(args.project_state) or root / args.project_state,
        history_path=resolver.resolve(args.history) or root / args.history,
        policy_path=resolver.resolve(args.policy) or root / args.policy,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "dry_run"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _shadow_plan(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main shadow-plan")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--deterministic-plan", required=True)
    parser.add_argument("--problem-map", required=True)
    parser.add_argument("--feedback", action="append", default=[])
    parser.add_argument("--tool-registry", default="config/tool_registry.json")
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--history", default="history/confirmed_experiments_a1.json")
    parser.add_argument("--planner-policy", default="config/planner_policy.json")
    parser.add_argument("--llm-policy", default="config/llm_shadow_policy.json")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "local_ollama", ALIYUN_BAILIAN_PROVIDER],
    )
    parser.add_argument("--provider-config", default="")
    parser.add_argument("--mock-mode", default="agree")
    parser.add_argument("--out-root", default="artifacts/llm_shadow_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)

    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = LLMShadowPlanner(project_root=root).run(
        deterministic_plan_path=(
            resolver.resolve(args.deterministic_plan)
            or root / args.deterministic_plan
        ),
        problem_map_path=resolver.resolve(args.problem_map) or root / args.problem_map,
        feedback_paths=[
            resolver.resolve(path) or root / path
            for path in args.feedback
        ],
        tool_registry_path=resolver.resolve(args.tool_registry) or root / args.tool_registry,
        project_state_path=resolver.resolve(args.project_state) or root / args.project_state,
        history_path=resolver.resolve(args.history) or root / args.history,
        planner_policy_path=resolver.resolve(args.planner_policy) or root / args.planner_policy,
        llm_policy_path=resolver.resolve(args.llm_policy) or root / args.llm_policy,
        provider_name=args.provider,
        provider_config=args.provider_config,
        mock_mode=args.mock_mode,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {
        "completed",
        "blocked",
        "invalid_output",
        "provider_unavailable",
        "timeout",
        "dry_run",
    }:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _research_memory_build(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main research-memory-build")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--problem-map", required=True)
    parser.add_argument("--feedback", action="append", default=[])
    parser.add_argument("--deterministic-plan", required=True)
    parser.add_argument("--shadow-comparison", required=True)
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--history", default="history/confirmed_experiments_a1.json")
    parser.add_argument("--research-policy", default="config/research_policy.json")
    parser.add_argument("--out-root", default="artifacts/research_memory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = ResearchMemoryBuilder(project_root=root).build(
        problem_map_path=resolver.resolve(args.problem_map) or root / args.problem_map,
        feedback_paths=[resolver.resolve(path) or root / path for path in args.feedback],
        deterministic_plan_path=resolver.resolve(args.deterministic_plan) or root / args.deterministic_plan,
        shadow_comparison_path=resolver.resolve(args.shadow_comparison) or root / args.shadow_comparison,
        project_state_path=resolver.resolve(args.project_state) or root / args.project_state,
        history_path=resolver.resolve(args.history) or root / args.history,
        research_policy_path=resolver.resolve(args.research_policy) or root / args.research_policy,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "dry_run"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _research_memory_update(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main research-memory-update")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--memory-root", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--research-policy", default="config/research_policy.json")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = ResearchMemoryBuilder(project_root=root).update(
        memory_root=resolver.resolve(args.memory_root) or root / args.memory_root,
        feedback_path=resolver.resolve(args.feedback) or root / args.feedback,
        experiment_manifest_path=resolver.resolve(args.experiment_manifest) or root / args.experiment_manifest,
        research_policy_path=resolver.resolve(args.research_policy) or root / args.research_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _research_brief(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main research-brief")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--memory-root", required=True)
    parser.add_argument("--queue-item", required=True)
    parser.add_argument("--out-root", default="artifacts/method_research")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = ResearchBriefBuilder(project_root=root).build(
        memory_root=resolver.resolve(args.memory_root) or root / args.memory_root,
        queue_item_id=args.queue_item,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "dry_run", "blocked"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _method_research_local(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main method-research-local")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--research-brief", required=True)
    parser.add_argument("--research-memory-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--research-policy", default="config/research_policy.json")
    parser.add_argument("--out-root", default="artifacts/method_research_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = MethodResearchRunner(project_root=root).run(
        research_brief=resolver.resolve(args.research_brief) or root / args.research_brief,
        research_memory_root=resolver.resolve(args.research_memory_root) or root / args.research_memory_root,
        source_manifest=resolver.resolve(args.source_manifest) or root / args.source_manifest,
        research_policy=resolver.resolve(args.research_policy) or root / args.research_policy,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "dry_run"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _live_method_research(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main live-method-research")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--research-brief", required=True)
    parser.add_argument("--research-memory-root", required=True)
    parser.add_argument("--research-policy", default="config/research_policy.json")
    parser.add_argument("--out-root", default="artifacts/live_method_research")
    parser.add_argument("--cache-root", default="artifacts/research_cache")
    parser.add_argument("--network-mode", default="live_cached", choices=["live_cached", "cache_only", "disabled"])
    parser.add_argument("--provider", default=ALIYUN_BAILIAN_PROVIDER)
    parser.add_argument("--model", default=ALIYUN_BAILIAN_DEFAULT_MODEL)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = LiveCachedMethodResearchRunner(project_root=root).run(
        research_brief=resolver.resolve(args.research_brief) or root / args.research_brief,
        research_memory_root=resolver.resolve(args.research_memory_root) or root / args.research_memory_root,
        research_policy=resolver.resolve(args.research_policy) or root / args.research_policy,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        cache_root=resolver.resolve(args.cache_root) or root / args.cache_root,
        network_mode=args.network_mode,
        provider=args.provider,
        model=args.model,
        max_queries=args.max_queries or None,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _decision_core_run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main decision-core-run")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--research-brief", required=True)
    parser.add_argument("--live-research-run", required=True)
    parser.add_argument("--research-memory-root", required=True)
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--tool-registry", default="config/tool_registry.json")
    parser.add_argument("--out-root", default="artifacts/decision_core_runs")
    parser.add_argument("--provider", default=ALIYUN_BAILIAN_PROVIDER)
    parser.add_argument("--model", default=ALIYUN_BAILIAN_DEFAULT_MODEL)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = DecisionCoreRunner(project_root=root).run(
        research_brief=resolver.resolve(args.research_brief) or root / args.research_brief,
        live_research_run=resolver.resolve(args.live_research_run) or root / args.live_research_run,
        research_memory_root=resolver.resolve(args.research_memory_root) or root / args.research_memory_root,
        project_state=resolver.resolve(args.project_state) or root / args.project_state,
        tool_registry=resolver.resolve(args.tool_registry) or root / args.tool_registry,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        provider=args.provider,
        model=args.model,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _m7_dry_run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main m7-dry-run")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--decision-run", required=True)
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--tool-registry", default="config/tool_registry.json")
    parser.add_argument("--out-root", default="artifacts/m7_dry_runs")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = M7DryRunOrchestrator(project_root=root).run(
        decision_run=resolver.resolve(args.decision_run) or root / args.decision_run,
        project_state=resolver.resolve(args.project_state) or root / args.project_state,
        tool_registry=resolver.resolve(args.tool_registry) or root / args.tool_registry,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed_dry_run", "ready_for_human_approval", "blocked", "invalid"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _m7b_readiness(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main m7b-readiness")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--paths_config", default="")
    parser.add_argument("--problem-map", default="artifacts/data_profile/a1_m2_v1/a1_problem_map.json")
    parser.add_argument("--data-profile", default="artifacts/data_profile/a1_m2_v1/a1_data_profile.json")
    parser.add_argument("--method-research-run", default="")
    parser.add_argument("--decision-run", default="")
    parser.add_argument("--m7a-run", default="")
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--tool-registry", default="config/tool_registry.json")
    parser.add_argument("--out-root", default="artifacts/m7b_readiness")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root, args.paths_config or None)
    root = resolver.project_root
    result = M7BReadinessRepair(project_root=root, paths_config=args.paths_config).run(
        problem_map=resolver.resolve(args.problem_map) or root / args.problem_map,
        data_profile=resolver.resolve(args.data_profile) or root / args.data_profile,
        method_research_run=resolver.resolve(args.method_research_run) or args.method_research_run,
        decision_run=resolver.resolve(args.decision_run) or args.decision_run,
        m7a_run=resolver.resolve(args.m7a_run) or args.m7a_run,
        project_state=resolver.resolve(args.project_state) or root / args.project_state,
        tool_registry=resolver.resolve(args.tool_registry) or root / args.tool_registry,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"ready_for_human_approval", "ready_for_experiment_design", "diagnostic_only", "waiting_for_anchor_rebuild_approval", "blocked"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _evaluation_anchor_bootstrap(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main evaluation-anchor-bootstrap")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--paths_config", default="")
    parser.add_argument("--a1-npz", required=True)
    parser.add_argument("--fold-candidate", required=True)
    parser.add_argument("--v43c-oof", required=True)
    parser.add_argument("--v46a-oof", required=True)
    parser.add_argument("--out-root", default="artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root, args.paths_config or None)
    root = resolver.project_root
    result = EvaluationAnchorBootstrap(project_root=root, paths_config=args.paths_config).run(
        a1_npz=resolver.resolve(args.a1_npz) or args.a1_npz,
        fold_candidate=resolver.resolve(args.fold_candidate) or args.fold_candidate,
        v43c_oof=resolver.resolve(args.v43c_oof) or args.v43c_oof,
        v46a_oof=resolver.resolve(args.v46a_oof) or args.v46a_oof,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"materialized_from_verified_components", "rebuild_required"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _fusion_controller(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main fusion-controller")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--anchor-dir", default="artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1")
    parser.add_argument("--v43c-oof", required=True)
    parser.add_argument("--v46a-oof", required=True)
    parser.add_argument("--out-root", default="artifacts/fusion_runs")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = run_fusion_controller(
        project_root=root,
        anchor_dir=resolver.resolve(args.anchor_dir) or root / args.anchor_dir,
        v43_oof=resolver.resolve(args.v43c_oof) or args.v43c_oof,
        v46_oof=resolver.resolve(args.v46a_oof) or args.v46a_oof,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _a1_closed_loop(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main a1-closed-loop")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--anchor-dir", default="artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1")
    parser.add_argument("--v43c-oof", required=True)
    parser.add_argument("--v46a-oof", required=True)
    parser.add_argument("--project-state", default="config/project_state.json")
    parser.add_argument("--tool-registry", default="config/tool_registry.json")
    parser.add_argument("--research-policy", default="config/research_policy.json")
    parser.add_argument("--out-root", default="artifacts/a1_closed_loop_runs")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-wall-clock-seconds", type=int, default=7200)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = run_a1_closed_loop(
        project_root=root,
        anchor_dir=resolver.resolve(args.anchor_dir) or root / args.anchor_dir,
        v43c_oof=resolver.resolve(args.v43c_oof) or args.v43c_oof,
        v46a_oof=resolver.resolve(args.v46a_oof) or args.v46a_oof,
        project_state=resolver.resolve(args.project_state) or root / args.project_state,
        tool_registry=resolver.resolve(args.tool_registry) or root / args.tool_registry,
        research_policy=resolver.resolve(args.research_policy) or root / args.research_policy,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        max_rounds=args.max_rounds,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "failed"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_input":
        raise SystemExit(3)
    raise SystemExit(2)


def _b1_closed_loop(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main b1-closed-loop")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", default="artifacts/b1_runs")
    parser.add_argument("--max-wall-clock-seconds", type=int, default=7200)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = run_b1_closed_loop(
        project_root=root,
        data_root=args.data_root,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
        max_rounds=args.max_rounds,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "validation_failed"}:
        raise SystemExit(0)
    if result["status"] in {"waiting_for_input", "waiting_for_data_intelligence"}:
        raise SystemExit(3)
    raise SystemExit(2)


def _legacy_out_root_guard(out_root: str, allow_legacy_output: bool) -> str | None:
    """Refuse legacy runs writing into v2 formal directories (replay shield)."""
    if "v2_formal_runs" in str(out_root).replace("\\", "/") and not allow_legacy_output:
        return (
            "REFUSED: out-root contains 'v2_formal_runs'. Legacy deterministic runners "
            "must not write into v2 formal directories. Use 'v2-run' for real v2 runs, "
            "'legacy-b2-closed-loop' with a legacy out-root for reproduction, or pass "
            "--allow-legacy-output explicitly."
        )
    return None


def _b2_closed_loop(argv: list[str], *, legacy_alias: bool = False) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main legacy-b2-closed-loop")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", default="artifacts/b2_runs")
    parser.add_argument("--max-wall-clock-seconds", type=int, default=7200)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--allow-legacy-output", action="store_true")
    args = parser.parse_args(argv)
    guard_error = _legacy_out_root_guard(args.out_root, args.allow_legacy_output)
    if guard_error:
        print(json.dumps({"status": "refused", "reason": guard_error}, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if legacy_alias:
        warning = (
            "WARNING: 'b2-closed-loop' is the LEGACY v1 deterministic runner "
            "(b2_autonomous_recommendation_loop_v1). It performs NO LLM calls, NO problem "
            "selection, NO M6B/M6C/M5, and its result is NOT a v2 run. "
            "Use 'v2-run --task B2' for real v2 executions."
        )
        print(warning, file=sys.stderr)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = run_b2_closed_loop(
        project_root=root,
        data_root=args.data_root,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
        max_rounds=args.max_rounds,
        force_rebuild=args.force_rebuild,
    )
    if legacy_alias:
        result["legacy_alias_warning"] = "b2-closed-loop is a legacy alias; NOT a v2 run"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"completed", "duplicate", "validation_failed"}:
        raise SystemExit(0)
    if result["status"] in {"waiting_for_input", "waiting_for_data_intelligence"}:
        raise SystemExit(3)
    raise SystemExit(2)


def _v2_run(argv: list[str]) -> None:
    """Real v2 entrypoint: V2AutonomousResearchOrchestrator (never the legacy runner)."""
    from .v2.orchestrator import V2AutonomousResearchOrchestrator

    parser = argparse.ArgumentParser(prog="afac_agent.main v2-run")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--task", required=True, choices=["B1", "B2"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", default="artifacts/v2_runs")
    parser.add_argument("--max-wall-clock-seconds", type=float, default=7200.0)
    parser.add_argument("--require-llm", action="store_true", default=True)
    parser.add_argument("--no-require-llm", dest="require_llm", action="store_false")
    parser.add_argument("--allow-deterministic-fallback", action="store_true")
    parser.add_argument("--force-new-execution", action="store_true")
    parser.add_argument("--resume-execution-id", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-users", type=int, default=512)
    parser.add_argument("--smoke-max-items", type=int, default=1000)
    parser.add_argument("--smoke-max-seconds", type=float, default=300.0)
    parser.add_argument("--no-deployment", action="store_true")
    parser.add_argument("--dry-run-orchestration", action="store_true")
    parser.add_argument("--limit-users", type=int, default=0, help="cap train users in formal mode (0 = all; for bounded control-flow smokes)")
    parser.add_argument("--deployment-reserve-seconds", type=float, default=300.0)
    parser.add_argument("--no-full-cv", dest="allow_full_cv", action="store_false", default=True)
    parser.add_argument("--provider", default=ALIYUN_BAILIAN_PROVIDER)
    parser.add_argument("--provider-config", default="")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    orchestrator = V2AutonomousResearchOrchestrator(
        project_root=root,
        task=args.task,
        data_root=args.data_root,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
        require_llm=args.require_llm,
        allow_deterministic_fallback=args.allow_deterministic_fallback,
        force_new_execution=args.force_new_execution,
        resume_execution_id=args.resume_execution_id or None,
        smoke=args.smoke,
        smoke_max_users=args.smoke_max_users,
        smoke_max_items=args.smoke_max_items,
        smoke_max_seconds=args.smoke_max_seconds,
        no_deployment=args.no_deployment,
        dry_run_orchestration=args.dry_run_orchestration,
        provider_name=args.provider,
        provider_config=args.provider_config,
        formal_max_users=args.limit_users,
        deployment_reserve_seconds=args.deployment_reserve_seconds,
        allow_full_cv=args.allow_full_cv,
    )
    result = orchestrator.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result["status"] in {"completed", "completed_smoke"}:
        raise SystemExit(0)
    if result["status"] in {"blocked_missing_llm", "blocked_llm_error"}:
        raise SystemExit(4)
    raise SystemExit(2)


def _a2_integration(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main a2-integration")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--a2-data-dir", default="")
    parser.add_argument("--a2-runs-root", default="")
    parser.add_argument("--out-root", default="artifacts/a2_integration")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    result = run_a2_integration(
        project_root=root,
        a2_data_dir=args.a2_data_dir,
        a2_runs_root=args.a2_runs_root,
        out_root=resolver.resolve(args.out_root) or root / args.out_root,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"ready_for_experiment_design", "duplicate", "validation_failed"}:
        raise SystemExit(0)
    if result["status"] == "waiting_for_explicit_asset":
        raise SystemExit(3)
    raise SystemExit(2)


def _llm_provider_check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="afac_agent.main llm-provider-check")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--provider", required=True, choices=[ALIYUN_BAILIAN_PROVIDER])
    parser.add_argument("--provider-config", default="")
    parser.add_argument("--model", default=ALIYUN_BAILIAN_DEFAULT_MODEL)
    args = parser.parse_args(argv)

    resolver = PathResolver(args.project_root)
    root = resolver.project_root
    model = args.model
    if not is_allowed_bailian_model(model):
        result = {
            "status": "provider_unavailable",
            "provider": args.provider,
            "model": model,
            "failure_reason": "configured_model_not_allowed",
            "response_schema_pass": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    provider = make_provider(
        args.provider,
        project_root=root,
        provider_config=args.provider_config,
    )
    response = provider.generate(
        LLMRequest(
            prompt="这是接口连通测试，请简短回复OK。",
            provider=args.provider,
            model=model,
            timeout_seconds=180,
            max_output_tokens=128,
            temperature=0.0,
            metadata={"provider_check": True},
        )
    )
    schema_pass = False
    if response.status == "completed":
        schema_pass = bool(response.audit.get("content_present"))
    result = {
        "status": response.status,
        "provider": response.provider or args.provider,
        "model": response.model or model,
        "failure_reason": response.failure_reason,
        "response_schema_pass": schema_pass,
        "content_present": bool(response.audit.get("content_present")),
        "audit": response.audit,
        "warnings": [redact_secret(item) for item in response.warnings],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if response.status in {"completed", "provider_unavailable", "timeout", "invalid_output"}:
        raise SystemExit(0)
    raise SystemExit(2)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run-adapter":
        _run_adapter(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build-feedback":
        _build_feedback(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "plan-next":
        _plan_next(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "shadow-plan":
        _shadow_plan(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "llm-provider-check":
        _llm_provider_check(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "research-memory-build":
        _research_memory_build(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "research-memory-update":
        _research_memory_update(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "research-brief":
        _research_brief(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "method-research-local":
        _method_research_local(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "live-method-research":
        _live_method_research(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "decision-core-run":
        _decision_core_run(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "m7-dry-run":
        _m7_dry_run(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "m7b-readiness":
        _m7b_readiness(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "evaluation-anchor-bootstrap":
        _evaluation_anchor_bootstrap(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "fusion-controller":
        _fusion_controller(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "a1-closed-loop":
        _a1_closed_loop(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "a2-integration":
        _a2_integration(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "b1-closed-loop":
        _b1_closed_loop(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "b2-closed-loop":
        _b2_closed_loop(sys.argv[2:], legacy_alias=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "legacy-b2-closed-loop":
        _b2_closed_loop(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "v2-run":
        _v2_run(sys.argv[2:])
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
