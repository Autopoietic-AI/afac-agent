from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from conftest import PROJECT_ROOT

from afac_agent.research.memory_views import ResearchMemoryBuilder

CHAMPION = PROJECT_ROOT / "artifacts" / "A1_v53q1_transition_stable_edge_h2_SAFE.csv"
PROJECT_STATE = PROJECT_ROOT / "config" / "project_state.json"
HISTORY = PROJECT_ROOT / "history" / "confirmed_experiments_a1.json"
PROBLEM_MAP = PROJECT_ROOT / "artifacts" / "data_profile" / "a1_m2_v1" / "a1_problem_map.json"
PLAN = PROJECT_ROOT / "artifacts" / "plans" / "33acfe3cb9830b132ab1c5643550384c6f9afa34e6bd41c9cdc4ba8aa0329bfa" / "plan_decision.json"
SHADOW = PROJECT_ROOT / "artifacts" / "llm_shadow_runs" / "33acfe3cb9830b132ab1c5643550384c6f9afa34e6bd41c9cdc4ba8aa0329bfa" / "bd1d1f6374fe16e95cc72d933a6d347384b143b579b88db9c80ec5ff2fa0866d" / "shadow_plan_comparison.json"
POLICY = PROJECT_ROOT / "config" / "research_policy.json"
FEEDBACKS = [
    PROJECT_ROOT / "artifacts" / "feedback_runs" / "A1_V46A1_ISOLATED_AUDIT" / "f3eb038640e55d3f2ba4892dc811df74005209aa59b978fbce77df99107a97aa" / "experiment_feedback.json",
    PROJECT_ROOT / "artifacts" / "feedback_runs" / "A1_V49A_EDGE_UTILITY_AUDIT" / "9a0a58051830f5fb1cd61138400020c1dacd38ee7d0a8b754849c053aa194a7f" / "experiment_feedback.json",
    PROJECT_ROOT / "artifacts" / "feedback_runs" / "A1_V53Q1_PATCH_REPLAY_SAFE" / "743b1f987c5a0e6c270446ed708ce4a867e530cc43044ca86790fdf78c63401b" / "experiment_feedback.json",
    PROJECT_ROOT / "artifacts" / "feedback_runs" / "A1_OOF_CANDIDATE_EVALUATOR" / "6a2ee51bff1fc83ff22ca5485b7ce420e0fd7f5194d92728a17446a4c23eeac6" / "experiment_feedback.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(out_root: Path) -> tuple[dict, Path]:
    result = ResearchMemoryBuilder(project_root=PROJECT_ROOT).build(
        problem_map_path=PROBLEM_MAP,
        feedback_paths=FEEDBACKS,
        deterministic_plan_path=PLAN,
        shadow_comparison_path=SHADOW,
        project_state_path=PROJECT_STATE,
        history_path=HISTORY,
        research_policy_path=POLICY,
        out_root=out_root,
    )
    assert result["status"] in {"completed", "duplicate"}
    return result, out_root / result["memory_id"]


def _load(memory_dir: Path, name: str) -> dict:
    return json.loads((memory_dir / name).read_text(encoding="utf-8"))


def test_real_memory_smoke_separates_blockers_and_scientific_problems(tmp_path: Path) -> None:
    before = {p: _sha256(p) for p in [CHAMPION, PROJECT_STATE, HISTORY]}
    result, memory_dir = _build(tmp_path / "research memory \u4e2d\u6587 with space")
    manifest = _load(memory_dir, "research_manifest.json")
    global_view = _load(memory_dir, "global_problem_view.json")
    profile = _load(memory_dir, "research_problem_profile.json")

    assert result["event_count"] >= 10
    assert manifest["calls_llm"] is False
    assert manifest["calls_api"] is False
    assert manifest["executes_adapter"] is False
    assert manifest["trains_model"] is False
    assert manifest["generates_prediction"] is False
    assert manifest["counts_as_experiment_round"] is False
    assert global_view["execution_blockers"][0]["blocker_id"] == "missing_full_anchor_evaluation_inputs"
    assert global_view["scientific_problems"][0]["problem_id"] != "blocker::missing_full_anchor_evaluation_inputs"
    assert global_view["scientific_problems"][0]["globalization_status"] == "unverified"
    assert set(profile["scope_levels"]) == {"global", "bucket", "bucket_class", "cross_bucket_class", "error_mechanism"}
    assert {p: _sha256(p) for p in before} == before
    assert json.loads(PROJECT_STATE.read_text(encoding="utf-8"))["budget"]["rounds_used"] == 0


def test_bucket_bucket_class_cross_bucket_and_mechanism_views_are_safe(tmp_path: Path) -> None:
    _, memory_dir = _build(tmp_path / "memory")
    bucket_view = _load(memory_dir, "bucket_problem_view.json")
    bucket_class = _load(memory_dir, "bucket_class_problem_view.json")
    cross = _load(memory_dir, "cross_bucket_class_view.json")
    mechanisms = _load(memory_dir, "error_mechanism_view.json")

    bucket_ids = {item["bucket_id"] for item in bucket_view["buckets"]}
    assert {"isolated", "one_hop_available", "exact2_only", "exact3_4_only", "no_visible_train_within_4_hops", "graph_visible"}.issubset(bucket_ids)
    assert any(item["bucket_id"] == "one_hop_available" and item["sample_count"] == 10366 for item in bucket_view["buckets"])
    assert bucket_class["bucket_class_problems"]
    assert all(row["parent_metric"] is None and row["candidate_metric"] is None for row in bucket_class["bucket_class_problems"])
    assert all(row["evidence_status"] == "unavailable" for row in bucket_class["bucket_class_problems"])
    assert {row["class_id"] for row in cross["class_audits"]} == {1, 4, 8}
    assert all(row["class_global_problem_status"] == "uncertain" for row in cross["class_audits"])
    mechanism_ids = {row["mechanism_id"] for row in mechanisms["mechanisms"]}
    assert {"information_source_missing", "neighbor_unreliability", "uniform_smoothing_damage", "multi_hop_signal_opportunity"}.issubset(mechanism_ids)


def test_method_failure_success_ledgers_preserve_feedback_semantics(tmp_path: Path) -> None:
    _, memory_dir = _build(tmp_path / "memory")
    attempts = _load(memory_dir, "method_attempt_ledger.json")["attempts"]
    failures = _load(memory_dir, "failure_ledger.json")["failures"]
    successes = _load(memory_dir, "success_ledger.json")["successes"]

    by_tool = {a["branch_id"]: a for a in attempts}
    assert by_tool["A1_V53Q1_PATCH_REPLAY_SAFE"]["outcome"] == "materialized_reference"
    assert by_tool["A1_V53Q1_PATCH_REPLAY_SAFE"]["verdict"] == "reference_only"
    assert by_tool["A1_V49A_EDGE_UTILITY_AUDIT"]["new_information_status"] == "new_information"
    assert by_tool["A1_V46A1_ISOLATED_AUDIT"]["outcome"] == "inconclusive"
    assert "missing_oof" in by_tool["A1_V46A1_ISOLATED_AUDIT"]["reason_codes"]
    assert by_tool["A1_OOF_CANDIDATE_EVALUATOR"]["outcome"] == "failure"
    assert by_tool["A1_OOF_CANDIDATE_EVALUATOR"]["rescue"] == 8
    assert by_tool["A1_OOF_CANDIDATE_EVALUATOR"]["damage"] == 15
    assert by_tool["A1_OOF_CANDIDATE_EVALUATOR"]["net"] == -7
    failure_types = {f["failure_type"] for f in failures}
    assert {"negative_overall_gain", "negative_macro_gain", "negative_net", "parent_unverified", "missing_oof"}.issubset(failure_types)
    assert any(f["failure_type"] == "negative_net" and f["oracle_gain_does_not_reopen"] for f in failures)
    assert any(s["success_mode"] == "reference_verified" for s in successes)


def test_queue_policy_topk_brief_and_non_topk_block(tmp_path: Path) -> None:
    _, memory_dir = _build(tmp_path / "memory")
    queue = _load(memory_dir, "research_queue.json")
    selected = [item for item in queue["items"] if item.get("deep_research_selected")]
    assert len([i for i in selected if i["scope_level"] == "global"]) <= 1
    assert len([i for i in selected if i["scope_level"] == "bucket"]) <= 2
    assert len([i for i in selected if i["scope_level"] == "bucket_class"]) <= 1
    assert all("priority_components" in item for item in queue["items"])

    chosen = selected[0]
    cmd = [sys.executable, "-m", "afac_agent.main", "research-brief", "--project_root", str(PROJECT_ROOT), "--memory-root", str(memory_dir), "--queue-item", chosen["queue_item_id"], "--out-root", str(tmp_path / "briefs \u4e2d\u6587 with space")]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "completed"
    brief = json.loads((Path(result["artifacts"]["research_brief"])).read_text(encoding="utf-8")) if Path(result["artifacts"]["research_brief"]).is_absolute() else json.loads((PROJECT_ROOT / result["artifacts"]["research_brief"]).read_text(encoding="utf-8"))
    assert brief["scope_level"] == chosen["scope_level"]
    assert "automatic training" in brief["excluded_method_families"]

    not_selected = next(item for item in queue["items"] if not item.get("deep_research_selected"))
    blocked_cmd = cmd.copy()
    blocked_cmd[blocked_cmd.index("--queue-item") + 1] = not_selected["queue_item_id"]
    blocked_cmd[blocked_cmd.index("--out-root") + 1] = str(tmp_path / "blocked")
    blocked = subprocess.run(blocked_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert blocked.returncode == 0
    assert json.loads(blocked.stdout)["status"] == "blocked"


def test_duplicate_update_and_memory_hash_determinism(tmp_path: Path) -> None:
    first, memory_dir = _build(tmp_path / "memory")
    second, _ = _build(tmp_path / "memory")
    assert second["status"] == "duplicate"
    assert second["memory_id"] == first["memory_id"]

    before_count = len((memory_dir / "research_events.jsonl").read_text(encoding="utf-8").splitlines())
    result = ResearchMemoryBuilder(project_root=PROJECT_ROOT).update(
        memory_root=memory_dir,
        feedback_path=FEEDBACKS[0],
        experiment_manifest_path=memory_dir / "research_manifest.json",
        research_policy_path=POLICY,
    )
    assert result["status"] == "duplicate"
    after_count = len((memory_dir / "research_events.jsonl").read_text(encoding="utf-8").splitlines())
    assert after_count == before_count

    changed_problem = tmp_path / "problem_changed.json"
    payload = json.loads(PROBLEM_MAP.read_text(encoding="utf-8"))
    payload["problems"][0]["node_count"] += 1
    changed_problem.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    changed = ResearchMemoryBuilder(project_root=PROJECT_ROOT).build(
        problem_map_path=changed_problem,
        feedback_paths=FEEDBACKS,
        deterministic_plan_path=PLAN,
        shadow_comparison_path=SHADOW,
        project_state_path=PROJECT_STATE,
        history_path=HISTORY,
        research_policy_path=POLICY,
        out_root=tmp_path / "changed",
    )
    assert changed["memory_id"] != first["memory_id"]


def test_research_policy_safety_and_doctor(tmp_path: Path) -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["allow_automatic_experiment_execution"] = True
    bad_policy = tmp_path / "bad_policy.json"
    bad_policy.write_text(json.dumps(policy), encoding="utf-8")
    result = ResearchMemoryBuilder(project_root=PROJECT_ROOT).build(
        problem_map_path=PROBLEM_MAP,
        feedback_paths=FEEDBACKS,
        deterministic_plan_path=PLAN,
        shadow_comparison_path=SHADOW,
        project_state_path=PROJECT_STATE,
        history_path=HISTORY,
        research_policy_path=bad_policy,
        out_root=tmp_path / "memory",
    )
    assert result["status"] == "failed"
    assert "allow_automatic_experiment_execution" in result["error"]

    doctor = subprocess.run([sys.executable, "-m", "afac_agent.doctor", "--project_root", str(PROJECT_ROOT), "--json"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    assert report["checks"]["research_policy_config"]["passed"] is True
    assert report["checks"]["research_memory_output_root"]["passed"] is True
    assert report["checks"]["method_research_output_root"]["passed"] is True


def test_cli_dry_run_writes_nothing_and_missing_inputs(tmp_path: Path) -> None:
    out = tmp_path / "research memory cli \u4e2d\u6587 with space"
    cmd = [sys.executable, "-m", "afac_agent.main", "research-memory-build", "--project_root", str(PROJECT_ROOT), "--problem-map", str(PROBLEM_MAP), "--deterministic-plan", str(PLAN), "--shadow-comparison", str(SHADOW), "--project-state", str(PROJECT_STATE), "--history", str(HISTORY), "--research-policy", str(POLICY), "--out-root", str(out)]
    for fb in FEEDBACKS:
        cmd.extend(["--feedback", str(fb)])
    dry = subprocess.run([*cmd, "--dry-run"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["status"] == "dry_run"
    assert not out.exists()

    missing = subprocess.run([sys.executable, "-m", "afac_agent.main", "research-memory-build", "--project_root", str(PROJECT_ROOT), "--problem-map", str(tmp_path / "missing.json"), "--deterministic-plan", str(PLAN), "--shadow-comparison", str(SHADOW), "--project-state", str(PROJECT_STATE), "--history", str(HISTORY), "--research-policy", str(POLICY), "--out-root", str(out)], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["status"] == "waiting_for_input"


def test_same_event_log_materializes_same_view_hash(tmp_path: Path) -> None:
    _, memory_dir = _build(tmp_path / "memory")
    from afac_agent.research.event_store import ResearchEventStore
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    events = ResearchEventStore(memory_dir / "research_events.jsonl").load()
    builder = ResearchMemoryBuilder(project_root=PROJECT_ROOT)
    first = builder.materialize(events, policy=policy, memory_id="x")
    second = builder.materialize(events, policy=policy, memory_id="x")
    assert {k: v["view_hash"] for k, v in first.items()} == {k: v["view_hash"] for k, v in second.items()}



def test_v21_bucket_axis_registry_overlap_and_scope_identity(tmp_path: Path) -> None:
    _, memory_dir = _build(tmp_path / "memory")
    registry = _load(memory_dir, "bucket_axis_registry.json")
    audit = _load(memory_dir, "bucket_overlap_audit.json")
    bucket_view = _load(memory_dir, "bucket_problem_view.json")
    axes = {a["axis_id"]: a for a in registry["bucket_axis_registry"]}

    assert axes["connectivity_visibility"]["allowed_values"][:2] == ["graph_visible", "isolated"]
    assert {"one_hop_available", "exact2_only", "exact3_4_only", "no_visible_train_within_4_hops"}.issubset(set(axes["train_label_reachability"]["allowed_values"]))
    assert axes["connectivity_visibility"]["mutually_exclusive_within_axis"] is True
    assert "train_label_reachability" in axes["connectivity_visibility"]["can_intersect_with_axes"]
    assert registry["axis_value_counts"]["connectivity_visibility"] == {"graph_visible": 10960, "isolated": 2792}
    assert registry["axis_value_counts"]["train_label_reachability"]["one_hop_available"] == 10366
    assert audit["axis_within_audit"]["connectivity_visibility"]["duplicate_count"] == 0
    assert audit["axis_within_audit"]["connectivity_visibility"]["uncovered_count"] == 0
    rel = audit["specific_relations"]["isolated_vs_no_visible_train_within_4_hops"]
    assert rel["intersection_count"] == 2792
    assert rel["scope_overlap_status"] == "subset"
    assert audit["specific_relations"]["graph_visible_and_no_visible_train_within_4_hops"]["intersection_count"] == 53

    graph = next(b for b in bucket_view["buckets"] if b["axis_id"] == "connectivity_visibility" and b["bucket_id"] == "graph_visible")
    exact2 = next(b for b in bucket_view["buckets"] if b["axis_id"] == "train_label_reachability" and b["bucket_id"] == "exact2_only")
    assert graph["scope_id"] != exact2["scope_id"]
    assert graph["bucket_axes"] == [{"axis_id": "connectivity_visibility", "value_id": "graph_visible"}]
    assert exact2["bucket_axes"] == [{"axis_id": "train_label_reachability", "value_id": "exact2_only"}]


def test_v21_priority_components_policy_and_overlap_explainability(tmp_path: Path) -> None:
    _, memory_dir = _build(tmp_path / "memory")
    queue = _load(memory_dir, "research_queue.json")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    required = set(policy["priority_component_weights"])
    items = queue["items"]
    for item in items:
        assert required.issubset(set(item["priority_components"]))
        assert item["priority_components"]["error_headroom_component"] is None
        assert item["priority_components"]["expected_score_impact_component"] is None
        assert item["priority_components"]["macro_importance_component"] is None
        assert "headroom_unavailable_not_imputed" in item["reason_codes"]
        assert "raw_priority_score" in item and "final_priority_score" in item
    one_hop = next(i for i in items if i["scope_refs"].get("value_id") == "one_hop_available")
    exact2 = next(i for i in items if i["scope_refs"].get("value_id") == "exact2_only")
    assert one_hop["affected_count"] == 10366
    assert exact2["affected_count"] == 527
    assert one_hop["priority_components"]["affected_count_component"] <= 1.0
    assert queue["priority_policy"]["affected_count_transform"] == policy["affected_count_transform"]

    changed_policy = json.loads(POLICY.read_text(encoding="utf-8"))
    changed_policy["priority_component_weights"]["affected_count_component"] = 0.0
    changed_policy["priority_component_weights"]["novelty_information_gap_component"] = 0.5
    changed_policy_path = tmp_path / "policy.json"
    changed_policy_path.write_text(json.dumps(changed_policy), encoding="utf-8")
    changed = ResearchMemoryBuilder(project_root=PROJECT_ROOT).build(
        problem_map_path=PROBLEM_MAP,
        feedback_paths=FEEDBACKS,
        deterministic_plan_path=PLAN,
        shadow_comparison_path=SHADOW,
        project_state_path=PROJECT_STATE,
        history_path=HISTORY,
        research_policy_path=changed_policy_path,
        out_root=tmp_path / "changed_memory",
    )
    changed_queue = _load(tmp_path / "changed_memory" / changed["memory_id"], "research_queue.json")
    assert changed_queue["priority_policy"]["weights"] != queue["priority_policy"]["weights"]


def test_v21_local_conflict_checker_synthetic_cases(tmp_path: Path) -> None:
    from afac_agent.research.local_conflict_checker import LocalConflictChecker

    axes = [{"axis_id": "connectivity_visibility", "value_id": "isolated"}]
    exact2_axes = [{"axis_id": "train_label_reachability", "value_id": "exact2_only"}]
    attempts = {
        "attempts": [
            {"attempt_id": "a1", "method_id": "method::mlp_iso", "branch_id": "isolated_mlp", "method_family": "mlp", "information_source_type": "existing_node_attributes", "new_information_status": "same_information_as_failed_route", "target_scope_refs": {"bucket_axes": axes}, "target_mechanism_ids": ["information_source_missing"], "training_objective": "supervised_ce", "loss_family": "ce", "parent_candidate": "v53Q-1", "configuration_identity": "cfg1", "outcome": "failure"},
            {"attempt_id": "a2", "method_id": "method::exact2_dir", "branch_id": "exact2_dir", "method_family": "gnn", "information_source_type": "directed_path_signal", "new_information_status": "new_information", "target_scope_refs": {"bucket_axes": exact2_axes}, "target_mechanism_ids": ["multi_hop_signal_opportunity"], "training_objective": "edge_utility", "loss_family": "ce", "configuration_identity": "cfg2", "outcome": "inconclusive"},
            {"attempt_id": "a3", "method_id": "method::v53_replay", "branch_id": "A1_V53Q1_PATCH_REPLAY_SAFE", "method_family": "champion_replay", "information_source_type": "other", "new_information_status": "same_information_as_failed_route", "target_scope_refs": {"bucket_axes": []}, "training_objective": "none", "configuration_identity": "cfg3", "outcome": "materialized_reference"},
        ]
    }
    failures = {"failures": [{"failure_type": "negative_net"}, {"failure_type": "negative_macro_gain"}]}
    checker = LocalConflictChecker()
    base = {"method_id": "candidate", "method_family": "mlp", "information_source_type": "existing_node_attributes", "new_information_status": "same_information_as_failed_route", "bucket_axes": axes, "mechanism_id": ["information_source_missing"], "training_objective": "supervised_ce", "loss_family": "ce", "parent_candidate": "v53Q-1", "branch_id": "isolated_mlp", "configuration_identity": "cfg1", "status": "proposed"}

    assert checker.check(method_card=base, method_attempt_ledger=attempts, failure_ledger=failures)["conflict_status"] == "exact_duplicate"
    renamed = {**base, "method_id": "different_name", "configuration_identity": "cfgX"}
    high = checker.check(method_card=renamed, method_attempt_ledger=attempts, failure_ledger=failures)
    assert high["conflict_status"] == "high_overlap"
    assert "same_information_as_failed_route" in high["conflict_reasons"]
    partial = checker.check(method_card={**base, "method_family": "gnn", "information_source_type": "directed_path_signal", "new_information_status": "new_information", "bucket_axes": exact2_axes, "mechanism_id": ["multi_hop_signal_opportunity"], "branch_id": "new_exact2", "configuration_identity": "cfg_new"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert partial["conflict_status"] in {"partial_overlap", "high_overlap"}
    repl = checker.check(method_card={**base, "method_family": "transformer", "new_information_status": "new_representation_only", "branch_id": "transformer_iso", "configuration_identity": "cfg_t"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert repl["conflict_status"] == "high_overlap"
    assert "new_representation_only" in repl["conflict_reasons"]
    ext = checker.check(method_card={**base, "information_source_type": "external_pretraining", "new_information_status": "new_information", "branch_id": "ext_iso", "configuration_identity": "cfg_ext"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert ext["conflict_status"] in {"partial_overlap", "new_direction"}
    new = checker.check(method_card={"method_id": "proto", "method_family": "prototype", "information_source_type": "prototype_signal", "new_information_status": "new_information", "bucket_axes": [{"axis_id": "train_label_reachability", "value_id": "one_hop_available"}], "mechanism_id": ["calibration_error"], "training_objective": "prototype", "branch_id": "proto_new", "status": "proposed"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert new["conflict_status"] == "new_direction"
    insuff = checker.check(method_card={"method_id": "bad"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert insuff["conflict_status"] == "insufficient_information"
    closed = checker.check(method_card={**base, "branch_id": "closed_branch_x", "configuration_identity": "cfg_closed"}, method_attempt_ledger=attempts, failure_ledger=failures, closed_branches=["closed_branch_x"])
    assert "closed_branch_conflict" in closed["conflict_reasons"]
    materialized = checker.check(method_card={"method_id": "replay", "method_family": "champion_replay", "information_source_type": "other", "new_information_status": "same_information_as_failed_route", "bucket_axes": [], "training_objective": "none", "branch_id": "A1_V53Q1_PATCH_REPLAY_SAFE", "configuration_identity": "cfg3", "status": "proposed"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert "materialized_reference_conflict" in materialized["conflict_reasons"]
    smooth = checker.check(method_card={"method_id": "correct_smooth_retry", "method_family": "smoothing", "information_source_type": "one_hop_topology", "new_information_status": "same_information_as_failed_route", "bucket_axes": [{"axis_id": "connectivity_visibility", "value_id": "graph_visible"}], "mechanism_id": "uniform_smoothing_damage", "training_objective": "uniform_smoothing", "branch_id": "correct_smooth", "status": "proposed"}, method_attempt_ledger=attempts, failure_ledger=failures)
    assert "uniform_smoothing_failure_overlap" in smooth["conflict_reasons"]
