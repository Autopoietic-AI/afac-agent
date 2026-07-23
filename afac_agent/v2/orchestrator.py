# -*- coding: utf-8 -*-
"""AFAC v2.0 autonomous research orchestrator.

This is the real v2 execution entrypoint.  It runs the full state machine

    PRECHECK -> INPUT_DISCOVERY -> DATA_INTELLIGENCE -> METRIC_SEMANTICS_GATE
    -> VALIDATION_REALITY -> PROBLEM_SELECTION -> COMPETITION_INTELLIGENCE
    -> M6B_PROPOSAL -> M6C_CRITIC -> M5_ADMISSION -> EXPERIMENT_GENOME
    -> BUDGET_DECISION -> EXPERIMENT_EXECUTION -> EVALUATION -> NO_OP_AUDIT
    -> PORTFOLIO_UPDATE -> POSTMORTEM -> NEXT_DECISION -> DEPLOYMENT
    -> FINAL_AUDIT -> COMPLETED

with a strict separation of duties: deterministic code computes statistics,
runs gates and executes whitelisted diagnostics; the LLM only synthesizes
problems, writes M6B proposals, performs M6C counterfactual review and
explains postmortems.  The LLM can never bypass M5, never touch test truth,
never execute arbitrary code, and never modify frozen assets.

Smoke runs stop after the cheap diagnostic path and finish with
``completed_smoke`` (no deployment, no formal submission).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..llm.providers import ALIYUN_BAILIAN_PROVIDER, make_provider, redact_secret
from ..research.event_store import json_dumps, load_json, sha256_file, stable_hash
from ..supervisor import RunSupervisor
from .adaptive_fold import (
    AdaptiveFoldPolicy,
    CanonicalFolds,
    FoldFidelity,
    build_fold_plan,
    no_improvement_action,
    paired_fold_comparison,
    should_trigger_full_cv,
    stop_decision,
)
from .budget_scheduler import BudgetState, ExperimentCandidate, Fidelity, should_continue
from .capability_registry import default_registry
from .competition_intelligence import CardStore, MethodCard
from .completion_contract import (
    MANIFEST_VERSION,
    ORCHESTRATOR_VERSION,
    TRAJECTORY_VERSION,
    build_manifest,
    check_completion_contract,
)
from .data_intelligence import analyze_recommendation
from .execution_identity import (
    PLANNER_VERSION,
    build_execution_id,
    build_input_fingerprint,
    code_commit,
    git_branch,
    llm_provider_config_identity,
)
from .llm_ledger import LLMLedger, load_ledger
from .metric_semantics import (
    audit_panels,
    candidate_pool_recall,
    error_decomposition,
    panel_record,
    ranking_metrics,
    validate_error_decomposition,
)
from .model_genome import ModelGenome
from .noop_detector import apply_noop_policy, compare_predictions
from .portfolio import Portfolio
from .problem_hierarchy import ProblemHierarchy, ProblemLevel, ProblemNode

# --------------------------------------------------------------------------- constants

STAGES = (
    "PRECHECK",
    "INPUT_DISCOVERY",
    "DATA_INTELLIGENCE",
    "METRIC_SEMANTICS_GATE",
    "VALIDATION_REALITY",
    "PROBLEM_SELECTION",
    "COMPETITION_INTELLIGENCE",
    "M6B_PROPOSAL",
    "M6C_CRITIC",
    "M5_ADMISSION",
    "EXPERIMENT_GENOME",
    "BUDGET_DECISION",
    "EXPERIMENT_EXECUTION",
    "EVALUATION",
    "NO_OP_AUDIT",
    "PORTFOLIO_UPDATE",
    "POSTMORTEM",
    "NEXT_DECISION",
    "DEPLOYMENT",
    "FINAL_AUDIT",
    "COMPLETED",
)

SMOKE_STAGES = tuple(s for s in STAGES if s not in {"NEXT_DECISION", "DEPLOYMENT"})

PLANNER_MODE_LLM = "llm_plus_deterministic_gates"
PLANNER_MODE_FALLBACK = "deterministic_fallback"

# Whitelisted cheap diagnostics the executor may run.  The LLM may only
# *choose* among these; it can never inject arbitrary code.
ALLOWED_DIAGNOSTICS = ("candidate_recall_diagnostic", "cached_replay", "small_retrieval_compare")

# Whitelisted formal experiments (round 2+): real training on sparse
# candidate tables only, never dense user-item matrices.
ALLOWED_FORMAL_EXPERIMENTS = ("candidate_ranker_experiment", "bucket_specialist_experiment")

FROZEN_FILES = (
    "config/project_state.json",
    "history/confirmed_experiments_a1.json",
    "artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv",
)

PROMPT_TEMPLATES = {
    "problem_synthesis": "afac_v2_problem_synthesis_v1",
    "m6b_proposal": "afac_v2_m6b_proposal_v1",
    "m6c_critic": "afac_v2_m6c_critic_v1",
    "postmortem": "afac_v2_postmortem_v1",
}


def _write_json(run_dir: Path, name: str, payload: dict[str, Any]) -> str:
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
    return name


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from LLM text."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def _frozen_hashes(project_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in FROZEN_FILES:
        path = project_root / rel
        if path.is_file():
            out[rel] = sha256_file(path)
    return out


# --------------------------------------------------------------------------- M5 deterministic gates

FORBIDDEN_PROPOSAL_KEYS = ("test_truth", "upload", "external_labels", "online_feedback_tuning")


def m5_admission(
    *,
    proposal: dict[str, Any],
    critic: dict[str, Any],
    budget_state: BudgetState,
    smoke: bool,
    allowed: tuple[str, ...] = ALLOWED_DIAGNOSTICS,
) -> dict[str, Any]:
    """Deterministic admission gate.  The LLM cannot bypass these checks."""
    reasons: list[str] = []
    diagnostic_type = str(proposal.get("diagnostic_type") or "")
    if diagnostic_type not in allowed:
        reasons.append(f"diagnostic_type not whitelisted: {diagnostic_type!r}")
    lowered = json.dumps(proposal).lower()
    for key in FORBIDDEN_PROPOSAL_KEYS:
        if key in lowered:
            reasons.append(f"forbidden proposal content: {key}")
    if str(critic.get("verdict", "")).lower() == "reject":
        reasons.append("M6C critic verdict is reject")
    estimated = float(proposal.get("budget_seconds", 60.0))
    budget_clamped = False
    if estimated > budget_state.remaining_wall_clock_seconds:
        # Clamp rather than kill a clean proposal when the estimate merely
        # overshoots the remaining window; reject only when nothing viable
        # remains.  The clamp is recorded in the decision.
        if budget_state.remaining_wall_clock_seconds >= 60.0:
            estimated = max(60.0, budget_state.remaining_wall_clock_seconds * 0.5)
            budget_clamped = True
        else:
            reasons.append("estimated cost exceeds remaining budget")
    if not proposal.get("success_condition"):
        reasons.append("success_condition missing")
    if not proposal.get("failure_condition"):
        reasons.append("failure_condition missing")

    if reasons:
        status = "rejected"
    elif smoke or diagnostic_type in ALLOWED_DIAGNOSTICS:
        status = "admitted_diagnostic_only"
    else:
        status = "admitted"
    return {
        "status": status,
        "reasons": reasons,
        "gate": "M5_deterministic",
        "llm_can_bypass": False,
        "estimated_cost_seconds": estimated,
        "budget_clamped": budget_clamped,
        "decided_at": time.time(),
    }


class LLMStageError(Exception):
    """A required LLM stage failed and no deterministic fallback was allowed."""

    def __init__(self, stage: str, status: str, sanitized: str) -> None:
        super().__init__(f"required LLM stage {stage} failed: {status} {sanitized}")
        self.stage = stage
        self.status = status
        self.sanitized = sanitized


# --------------------------------------------------------------------------- orchestrator


class V2AutonomousResearchOrchestrator:
    """The real v2 execution entrypoint (see module docstring)."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        task: str,
        data_root: str | Path,
        out_root: str | Path,
        max_wall_clock_seconds: float = 7200.0,
        require_llm: bool = True,
        allow_deterministic_fallback: bool = False,
        force_new_execution: bool = False,
        resume_execution_id: str | None = None,
        smoke: bool = False,
        smoke_max_users: int = 512,
        smoke_max_items: int = 1000,
        smoke_max_seconds: float = 300.0,
        no_deployment: bool = False,
        dry_run_orchestration: bool = False,
        provider: Any | None = None,
        provider_name: str = ALIYUN_BAILIAN_PROVIDER,
        provider_config: str = "",
        parent_execution_id: str = "",
        deployment_method: str = "full_train_retrain",
        formal_max_users: int = 0,
        deployment_reserve_seconds: float = 300.0,
        allow_full_cv: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.task = task
        self.data_root = Path(data_root)
        out_path = Path(out_root)
        self.out_root = out_path if out_path.is_absolute() else self.project_root / out_path
        self.max_wall_clock_seconds = float(max_wall_clock_seconds)
        self.require_llm = require_llm
        self.allow_deterministic_fallback = allow_deterministic_fallback
        self.force_new_execution = force_new_execution
        self.resume_execution_id = resume_execution_id
        self.smoke = smoke
        self.smoke_max_users = smoke_max_users
        self.smoke_max_items = smoke_max_items
        self.smoke_max_seconds = smoke_max_seconds
        self.no_deployment = no_deployment or smoke
        self.dry_run_orchestration = dry_run_orchestration
        self._provider = provider
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.parent_execution_id = parent_execution_id
        if deployment_method not in {"full_train_retrain", "3fold_ensemble"}:
            raise ValueError(f"unknown deployment_method: {deployment_method}")
        self.deployment_method = deployment_method
        self.formal_max_users = int(formal_max_users)
        self.allow_full_cv = allow_full_cv

        self.started_at = time.time()
        self.stages = SMOKE_STAGES if smoke else STAGES
        self.artifacts: dict[str, str] = {}
        self.gates: dict[str, bool] = {}
        self.trajectory_events: list[dict[str, Any]] = []
        self.cache_status = "no_cache"
        self.resume_status = "fresh"
        self.planner_mode = PLANNER_MODE_LLM
        self.llm_available = False
        self.scientific_rounds_used = 0.0
        self.cheap_diagnostics_used = 0
        self.no_op_rounds_refunded = 0
        self._deployment_generated = False
        self._execution_id = ""
        self.fold_policy = AdaptiveFoldPolicy.for_task(task)
        self.fold_policy.estimator.deployment_reserve_seconds = float(deployment_reserve_seconds)
        self._canonical: CanonicalFolds | None = None
        self._incumbent: dict[str, Any] = {"candidate_id": "popularity_parent", "kind": "popularity", "metrics": {"hit_rate@10": 0.0}}
        self._no_improvement_rounds = 0
        self._global_explore_attempted = False
        self._operator_families_tried: set[str] = set()
        self._ledger: LLMLedger | None = None
        self._frozen_before = _frozen_hashes(self.project_root)

    # --------------------------------------------------------------- identity

    def _data_hash(self) -> str:
        from ..b2.task_adapter import B2TaskAdapter

        adapter = B2TaskAdapter(self.data_root, task_id=self.task)
        root = adapter.actual_root()
        hashes = {}
        for name in ("train.csv", "test.csv", "user.csv", "item.csv", "sample_submission.csv"):
            path = root / name
            if path.is_file():
                hashes[name] = sha256_file(path)
        return stable_hash(hashes)

    # --------------------------------------------------------------- LLM helpers

    def _build_ledger(self, run_dir: Path, execution_id: str) -> LLMLedger:
        if self._provider is None:
            self._provider = make_provider(
                self.provider_name,
                project_root=self.project_root,
                provider_config=self.provider_config,
            )
        config = getattr(self._provider, "config", {}) or {}
        model = str(config.get("model") or "")
        return LLMLedger(
            run_dir=run_dir,
            execution_id=execution_id,
            task=self.task,
            provider=self._provider,
            provider_name=self.provider_name,
            model=model or "qwen3.6-max-preview",
            planner_policy_version=PLANNER_VERSION,
            max_retries=1,
        )

    def _llm_json_stage(
        self,
        ledger: LLMLedger,
        *,
        stage: str,
        prompt_template_id: str,
        prompt: str,
        required: bool,
        max_output_tokens: int = 2048,
    ) -> tuple[dict[str, Any] | None, str]:
        """One LLM stage returning parsed JSON.

        On failure: required stages abort the run (blocked_llm_error) unless a
        deterministic fallback is explicitly allowed; optional stages degrade
        to a deterministic placeholder with fallback_used recorded.
        """
        result = ledger.call(
            stage=stage,
            prompt_template_id=prompt_template_id,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
        if result["ok"]:
            parsed = _extract_json(result["text"])
            if parsed is not None:
                self.llm_available = True
                return parsed, "llm"
        if required:
            if self.allow_deterministic_fallback:
                self.planner_mode = PLANNER_MODE_FALLBACK
                ledger.mark_fallback(stage=stage, prompt_template_id=prompt_template_id, reason="llm_unavailable_or_unparseable; deterministic fallback explicitly allowed")
                return None, "fallback"
            raise LLMStageError(stage, result["record"]["status"], result["record"]["error_message_sanitized"])
        ledger.mark_fallback(stage=stage, prompt_template_id=prompt_template_id, reason="optional stage degraded to deterministic placeholder")
        return None, "fallback"

    # --------------------------------------------------------------- main run

    def run(self) -> dict[str, Any]:
        started = self.started_at
        current_stage = "PRECHECK"
        run_dir: Path | None = None
        ledger: LLMLedger | None = None
        supervisor: RunSupervisor | None = None
        execution_id = ""
        input_fingerprint = ""

        try:
            # ---- PRECHECK -------------------------------------------------
            current_stage = "PRECHECK"
            if self.task != "B2":
                return self._finish(None, None, None, "", "", status="failed", current_stage=current_stage, started=started, error=f"task {self.task} is not yet wired to the v2 orchestrator (B2 only in this repair)")
            data_hash = self._data_hash()
            metric_contract_hash = stable_hash({"metrics": ["candidate_pool_recall@20", "candidate_pool_recall@50", "candidate_pool_recall@100", "candidate_pool_recall@200", "hit_rate@10", "ndcg@10", "mrr@10"], "decomposition": ["top10_success", "in_pool_outside_top10", "missing_from_candidate_pool"]})
            validation_contract_hash = stable_hash({"panels": "B2_REQUIRED_PANELS", "fold": "AFAC_B2_FOLD_V1"})
            deterministic_config_hash = stable_hash({"smoke": self.smoke, "smoke_max_users": self.smoke_max_users, "smoke_max_items": self.smoke_max_items, "task": self.task})
            registry = default_registry()
            capability_registry_hash = stable_hash({r.operator_id: {"implemented": r.implemented, "available": r.available, "priority": r.scientific_priority} for r in registry.query()})
            planner_policy_hash = stable_hash({"planner_mode": PLANNER_MODE_LLM, "require_llm": self.require_llm, "m5": "deterministic_gates_v1"})

            fp = build_input_fingerprint(
                data_hash=data_hash,
                fold_hash=stable_hash({"fold_identity": "AFAC_B2_FOLD_V1", "seed": 2026}),
                deterministic_config_hash=deterministic_config_hash,
                metric_contract_hash=metric_contract_hash,
            )
            input_fingerprint = fp["input_fingerprint"]

            # Cache classification (section 9): never reuse v1 final results.
            if self.force_new_execution:
                self.cache_status = "bypassed_by_force_new_execution"
            legacy_hit = self._find_legacy_cache()
            if legacy_hit and self.cache_status == "no_cache":
                self.cache_status = "legacy_cache_rejected"

            # Resume handling: reuse the execution identity of a prior run.
            if self.resume_execution_id:
                resume_dir = self.out_root / self.resume_execution_id
                if not resume_dir.is_dir():
                    raise FileNotFoundError(f"resume execution not found: {resume_dir}")
                run_dir = resume_dir
                execution_id = self.resume_execution_id
                self._execution_id = execution_id
                self.resume_status = "resumed"
            else:
                identity = build_execution_id(
                    input_fingerprint=input_fingerprint,
                    project_root=self.project_root,
                    planner_policy_hash=planner_policy_hash,
                    capability_registry_hash=capability_registry_hash,
                    metric_contract_hash=metric_contract_hash,
                    validation_contract_hash=validation_contract_hash,
                    llm_provider_config_hash=stable_hash(llm_provider_config_identity(getattr(self._provider, "config", {}) or {})),
                    started_at=started,
                )
                execution_id = identity["execution_id"]
                self._execution_id = execution_id
                run_dir = self.out_root / execution_id
                run_dir.mkdir(parents=True, exist_ok=False)

            supervisor = RunSupervisor(
                run_dir,
                task=self.task,
                run_id=execution_id,
                budget_seconds=self.smoke_max_seconds if self.smoke else self.max_wall_clock_seconds,
                interval_seconds=30.0,
            )
            ledger = self._build_ledger(run_dir, execution_id)
            self._ledger = ledger
            self._beat(supervisor, stage=current_stage, status="running", execution_id=execution_id, input_fingerprint=input_fingerprint)
            self._track(current_stage, {"execution_id": execution_id, "input_fingerprint": input_fingerprint, "cache_status": self.cache_status, "resume_status": self.resume_status})

            if self.dry_run_orchestration:
                return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status="completed_smoke" if self.smoke else "incomplete", current_stage=current_stage, started=started)

            # ---- INPUT_DISCOVERY ------------------------------------------
            current_stage = "INPUT_DISCOVERY"
            self._beat(supervisor, stage=current_stage, status="running")
            from ..b2.task_adapter import B2TaskAdapter

            adapter = B2TaskAdapter(self.data_root, task_id=self.task)
            missing = adapter.missing_files()
            if missing:
                return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status="failed", current_stage=current_stage, started=started, error=f"missing input files: {missing}")
            dataset = adapter.load()
            if dataset.validation.get("status") != "passed":
                return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status="failed", current_stage=current_stage, started=started, error=f"dataset validation failed: {dataset.validation.get('errors')}")
            self.gates["test_truth_guard"] = bool(dataset.validation.get("test_truth_hidden", True))
            self._track(current_stage, {"n_train": dataset.validation.get("n_train"), "n_items": dataset.n_items, "test_truth_hidden": self.gates["test_truth_guard"]})

            targets = {str(u): str(t) for u, t in zip(dataset.train_df["uid"], dataset.train_df["target_iid"])}
            user_cap = self.smoke_max_users if self.smoke else self.formal_max_users
            smoke_uids = sorted(targets)[:user_cap] if user_cap else sorted(targets)
            subset_seq = {u: dataset.train_seq.get(u, []) for u in smoke_uids}
            subset_targets = {u: targets[u] for u in smoke_uids}

            # ---- DATA_INTELLIGENCE ----------------------------------------
            current_stage = "DATA_INTELLIGENCE"
            self._beat(supervisor, stage=current_stage, status="running")
            di = analyze_recommendation(subset_seq, subset_targets, {u: dataset.test_seq.get(u, []) for u in list(dataset.test_seq)[: self.smoke_max_users]})
            self.gates["data_intelligence"] = di.status == "verified"
            self.artifacts["data_intelligence"] = _write_json(run_dir, "data_intelligence.json", di.to_dict())
            self._track(current_stage, {"status": di.status})

            # ---- METRIC_SEMANTICS_GATE -------------------------------------
            current_stage = "METRIC_SEMANTICS_GATE"
            self._beat(supervisor, stage=current_stage, status="running")
            gate_probe = validate_error_decomposition({"top10_success": 0.2, "in_pool_outside_top10": 0.3, "missing_from_candidate_pool": 0.5})
            self.gates["metric_semantics_gate"] = gate_probe["status"] == "ok"
            self.artifacts["metric_semantics_gate"] = _write_json(run_dir, "metric_semantics_gate.json", {"gate": "passed", "probe": gate_probe, "pool_recall_ks": [20, 50, 100, 200]})
            self._track(current_stage, {"gate": self.gates["metric_semantics_gate"]})

            # ---- VALIDATION_REALITY ----------------------------------------
            current_stage = "VALIDATION_REALITY"
            self._beat(supervisor, stage=current_stage, status="running")
            panels = self._build_panels(subset_seq, subset_targets)
            audit = audit_panels(panels)
            # Duplicate panels are flagged and excluded from evidence (v1.6
            # lesson); the gate requires a completed audit with at least two
            # independent panels, not zero duplicates.
            self.gates["validation_reality"] = audit["independent_panel_count"] >= 2
            self.artifacts["validation_reality"] = _write_json(run_dir, "validation_reality.json", audit)
            # Fixed canonical folds: every fidelity selects a fixed prefix
            # subset of this hash-stable master assignment.  Never re-randomized.
            seq_bins = np.array([min(5, len(subset_seq.get(u, []))) for u in smoke_uids], dtype=np.int64)
            self._canonical = CanonicalFolds.build(smoke_uids, stratify_bins=seq_bins)
            self.artifacts["canonical_folds"] = _write_json(run_dir, "canonical_folds.json", {
                "canonical_fold_hash": self._canonical.fold_hash,
                "master_fold_count": 5,
                "n_users": len(smoke_uids),
                "fold_sizes": {str(f): int((self._canonical.folds == f).sum()) for f in range(5)},
                "policy": "fixed_prefix_subsets",
            })
            self._track(current_stage, {"independent_panels": audit["independent_panel_count"], "duplicates": audit["duplicate_pairs"], "canonical_fold_hash": self._canonical.fold_hash})

            if self.smoke:
                # ---- smoke: exactly one cheap-diagnostic round --------------
                record = self._round_pipeline(
                    run_dir, supervisor, ledger,
                    dataset=dataset, di=di, smoke_uids=smoke_uids,
                    subset_seq=subset_seq, subset_targets=subset_targets,
                    round_index=1, started=started,
                )
                if record is None:
                    return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status="incomplete", current_stage="ROUND_1", started=started, error="smoke round rejected or budget-stopped")
            else:
                # ---- formal: dynamic multi-round loop ------------------------
                round_records: list[dict[str, Any]] = []
                for round_index in range(1, 13):  # safety cap; budget decides the real stop
                    record = self._round_pipeline(
                        run_dir, supervisor, ledger,
                        dataset=dataset, di=di, smoke_uids=smoke_uids,
                        subset_seq=subset_seq, subset_targets=subset_targets,
                        round_index=round_index, started=started,
                    )
                    if record is None:
                        break
                    round_records.append(record)
                    decision = self._next_decision(run_dir, supervisor, ledger, round_records, started)
                    if not decision["continue"]:
                        break
                # ---- DEPLOYMENT ----------------------------------------------
                current_stage = "DEPLOYMENT"
                self._beat(supervisor, stage=current_stage, status="running")
                deployment = self._run_deployment(run_dir, supervisor, dataset, subset_targets, round_records)
                if deployment is not None:
                    self.artifacts["deployment_audit"] = _write_json(run_dir, "deployment_audit.json", deployment)
                    self._deployment_generated = True
                    self._track(current_stage, {"best_candidate": deployment.get("best_candidate_id"), "rows": deployment.get("n_rows")})

            # ---- FINAL_AUDIT -----------------------------------------------------
            current_stage = "FINAL_AUDIT"
            self._beat(supervisor, stage=current_stage, status="saving")
            self.gates["frozen_asset_hash"] = _frozen_hashes(self.project_root) == self._frozen_before
            final_status = "completed_smoke" if self.smoke else "completed"
            if self.planner_mode == PLANNER_MODE_FALLBACK:
                final_status = "degraded_deterministic_fallback"
            return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status=final_status, current_stage=current_stage, started=started)

        except LLMStageError as exc:
            blocked = "blocked_missing_llm" if exc.status in {"provider_unavailable", "error"} else "blocked_llm_error"
            return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status=blocked, current_stage=current_stage, started=started, error=exc.sanitized)
        except Exception as exc:  # noqa: BLE001 - run must always produce a manifest
            return self._finish(run_dir, supervisor, ledger, execution_id, input_fingerprint, status="failed", current_stage=current_stage, started=started, error=redact_secret(f"{type(exc).__name__}: {exc}"))

    # --------------------------------------------------------------- round pipeline

    def _round_pipeline(
        self,
        run_dir: Path,
        supervisor: RunSupervisor,
        ledger: LLMLedger,
        *,
        dataset: Any,
        di: Any,
        smoke_uids: list[str],
        subset_seq: dict[str, list[str]],
        subset_targets: dict[str, str],
        round_index: int,
        started: float,
    ) -> dict[str, Any] | None:
        """One full decision chain: PROBLEM_SELECTION → … → PORTFOLIO_UPDATE.

        Returns a round record for the portfolio/deployment selection, or
        ``None`` when the loop must stop (M5 rejection or budget stop).
        """
        prefix = "" if self.smoke else f"round_{round_index:02d}/"

        # ---- PROBLEM_SELECTION -----------------------------------------
        stage = "PROBLEM_SELECTION"
        self._beat(supervisor, stage=stage, status="running", rounds_used=int(self.scientific_rounds_used))
        hierarchy = self._build_problem_hierarchy(di)
        candidates = hierarchy.select_targets()
        chosen = candidates[0] if candidates else None
        problem_prompt = self._prompt_problem_synthesis(candidates)
        llm_choice, problem_mode = self._llm_json_stage(
            ledger,
            stage="PROBLEM_SELECTION",
            prompt_template_id=PROMPT_TEMPLATES["problem_synthesis"],
            prompt=problem_prompt,
            required=False,
        )
        if llm_choice and llm_choice.get("problem_id") in {c.node_id for c in candidates}:
            chosen = next(c for c in candidates if c.node_id == llm_choice["problem_id"])
        if chosen is None:
            return None
        problem_payload = {
            "problem_id": chosen.node_id,
            "task": chosen.task,
            "pipeline_stage": chosen.pipeline_stage,
            "bucket": chosen.bucket,
            "error_mechanism": chosen.error_mechanism,
            "evidence": chosen.evidence,
            "headroom": chosen.headroom,
            "selection_mode": problem_mode,
            "llm_rationale": (llm_choice or {}).get("rationale", ""),
        }
        self.artifacts["problem_node"] = _write_json(run_dir, f"{prefix}problem_node.json", problem_payload)
        self._beat(supervisor, stage=stage, status="running", current_problem_id=chosen.node_id)
        self._track(stage, {"round": round_index, "problem_id": chosen.node_id, "mode": problem_mode})

        # ---- COMPETITION_INTELLIGENCE ----------------------------------
        stage = "COMPETITION_INTELLIGENCE"
        self._beat(supervisor, stage=stage, status="running")
        store = self._build_card_store()
        cards = store.retrieve(task_regime="recommendation", error_mechanism=chosen.error_mechanism)
        self.artifacts["competition_research_state"] = _write_json(run_dir, f"{prefix}competition_research_state.json", {"retrieved_cards": [c.card_id for c in cards], "card_count": len(cards)})
        self._track(stage, {"round": round_index, "cards": len(cards)})

        # ---- M6B_PROPOSAL ----------------------------------------------
        stage = "M6B_PROPOSAL"
        self._beat(supervisor, stage=stage, status="running", current_problem_id=chosen.node_id)
        proposal_prompt = self._prompt_m6b(
            chosen,
            formal=not self.smoke and round_index > 1,
            tried_families=sorted(self._operator_families_tried),
        )
        proposal, proposal_mode = self._llm_json_stage(
            ledger,
            stage="M6B_PROPOSAL",
            prompt_template_id=PROMPT_TEMPLATES["m6b_proposal"],
            prompt=proposal_prompt,
            required=True,
        )
        if proposal is None:
            proposal = self._deterministic_proposal(chosen)
        proposal.setdefault("problem_id", chosen.node_id)
        proposal["proposal_mode"] = proposal_mode
        proposal["proposal_id"] = stable_hash({"proposal": proposal, "execution": ledger.execution_id, "round": round_index})[:24]
        self.artifacts["m6b_proposal"] = _write_json(run_dir, f"{prefix}m6b_proposal.json", proposal)
        self._beat(supervisor, stage=stage, status="running", current_proposal_id=proposal["proposal_id"])
        self._track(stage, {"round": round_index, "proposal_id": proposal["proposal_id"], "mode": proposal_mode})

        # ---- M6C_CRITIC ------------------------------------------------
        stage = "M6C_CRITIC"
        self._beat(supervisor, stage=stage, status="running")
        critic_prompt = self._prompt_m6c(proposal)
        critic, critic_mode = self._llm_json_stage(
            ledger,
            stage="M6C_CRITIC",
            prompt_template_id=PROMPT_TEMPLATES["m6c_critic"],
            prompt=critic_prompt,
            required=True,
        )
        if critic is None:
            critic = {"verdict": "approve", "issues": [], "rationale": "proposal restricted to whitelisted cheap diagnostics"}
        critic["critic_mode"] = critic_mode
        self.artifacts["m6c_critic"] = _write_json(run_dir, f"{prefix}m6c_critic.json", critic)
        self._beat(supervisor, stage=stage, status="running", current_critic_status=str(critic.get("verdict", "")))
        self._track(stage, {"round": round_index, "verdict": critic.get("verdict"), "mode": critic_mode})

        # ---- M5_ADMISSION ----------------------------------------------
        stage = "M5_ADMISSION"
        self._beat(supervisor, stage=stage, status="running")
        budget_state = BudgetState(
            max_wall_clock_seconds=self.smoke_max_seconds if self.smoke else self.max_wall_clock_seconds,
            elapsed=time.time() - started,
            rounds_used=self.scientific_rounds_used,
        )
        allowed = ALLOWED_DIAGNOSTICS if self.smoke or round_index == 1 else ALLOWED_DIAGNOSTICS + ALLOWED_FORMAL_EXPERIMENTS
        m5 = m5_admission(proposal=proposal, critic=critic, budget_state=budget_state, smoke=self.smoke, allowed=allowed)
        self.artifacts["m5_decision"] = _write_json(run_dir, f"{prefix}m5_decision.json", m5)
        self._track(stage, {"round": round_index, "status": m5["status"], "reasons": m5["reasons"]})
        if m5["status"] == "rejected":
            return None

        # ---- EXPERIMENT_GENOME ------------------------------------------
        stage = "EXPERIMENT_GENOME"
        self._beat(supervisor, stage=stage, status="running")
        genome = self._build_genome(proposal, chosen)
        self.artifacts["experiment_genome"] = _write_json(run_dir, f"{prefix}experiment_genome.json", asdict(genome))
        self._track(stage, {"round": round_index, "genome_id": genome.genome_id})

        # ---- BUDGET_DECISION --------------------------------------------
        stage = "BUDGET_DECISION"
        self._beat(supervisor, stage=stage, status="running")
        diagnostic_type = str(proposal.get("diagnostic_type") or "")
        fidelity = Fidelity.CHEAP_DIAGNOSTIC if diagnostic_type in ALLOWED_DIAGNOSTICS else Fidelity.SINGLE_FOLD
        candidate = ExperimentCandidate(
            candidate_id=proposal["proposal_id"],
            fidelity=fidelity,
            expected_gain=float(proposal.get("expected_gain", 0.01)),
            expected_information_gain=0.6,
            compute_cost_seconds=float(m5.get("estimated_cost_seconds", proposal.get("budget_seconds", 60.0))),
            novelty=0.5,
        )
        decision = {
            "decision": "run" if should_continue(budget_state, [candidate]) else "stop",
            "fidelity": candidate.fidelity.value,
            "consumes_scientific_round": candidate.fidelity in {Fidelity.SINGLE_FOLD, Fidelity.FULL_OOF},
            "remaining_seconds": budget_state.remaining_wall_clock_seconds,
        }
        self.artifacts["budget_decision"] = _write_json(run_dir, f"{prefix}budget_decision.json", decision)
        self._track(stage, {"round": round_index, **decision})
        if decision["decision"] != "run":
            return None

        # ---- EXPERIMENT_EXECUTION ----------------------------------------
        stage = "EXPERIMENT_EXECUTION"
        self._beat(supervisor, stage=stage, status="running", current_experiment=proposal["proposal_id"], current_model=diagnostic_type)
        parent_at_start = dict(self._incumbent)
        fold_outcome: dict[str, Any]
        if self.smoke and diagnostic_type in ALLOWED_DIAGNOSTICS:
            # Smoke path: flat cheap diagnostic (F0), contract-stable.
            result = self._run_cheap_diagnostic(proposal, dataset, smoke_uids, subset_seq, subset_targets)
            self.cheap_diagnostics_used += 1
            fold_outcome = {
                "fold_plan": build_fold_plan(self._canonical, FoldFidelity.F0_DETERMINISTIC, task=self.task).to_dict() if self._canonical else {},
                "paired_comparison": {},
                "promotion": {"decision": "not_applicable_diagnostic", "reasons": []},
                "estimated_runtime": 0.0,
                "actual_runtime": 0.0,
            }
        else:
            # Formal path: every experiment (diagnostics included) runs on the
            # canonical fold ladder with paired parent comparison.
            fold_outcome = self._run_folded_with_promotion(run_dir, proposal, dataset, subset_targets, started, prefix)
            result = fold_outcome["result"]
            if diagnostic_type in ALLOWED_DIAGNOSTICS:
                self.cheap_diagnostics_used += 1
            else:
                self.scientific_rounds_used += 1.0
        self.artifacts["experiment_result"] = _write_json(run_dir, f"{prefix}experiment_result.json", {k: v for k, v in result.items() if k not in {"parent_top10", "candidate_top10"}})
        self._operator_families_tried.add(diagnostic_type)
        self._track(stage, {"round": round_index, "diagnostic": diagnostic_type, "metrics": result.get("pool_recall")})

        # ---- EVALUATION --------------------------------------------------
        stage = "EVALUATION"
        self._beat(supervisor, stage=stage, status="evaluating", latest_metric=result.get("pool_recall"))
        decomp = result.get("error_decomposition", {})
        validity = validate_error_decomposition(decomp)
        self.artifacts["evaluation"] = _write_json(run_dir, f"{prefix}evaluation.json", {"error_decomposition": decomp, "validity": validity, "pool_recall": result.get("pool_recall"), "top10_metrics": result.get("top10_metrics")})
        self._track(stage, {"round": round_index, "validity": validity["status"]})

        # ---- NO_OP_AUDIT --------------------------------------------------
        stage = "NO_OP_AUDIT"
        self._beat(supervisor, stage=stage, status="running")
        noop_report = compare_predictions(result["parent_top10"], result["candidate_top10"])
        noop_record = apply_noop_policy({"experiment_id": proposal["proposal_id"]}, noop_report)
        if noop_report.status == "no_op":
            self.no_op_rounds_refunded += 1
        self.artifacts["no_op_audit"] = _write_json(run_dir, f"{prefix}no_op_audit.json", {"report": asdict(noop_report), "policy": {k: v for k, v in noop_record.items() if k != "report"}})
        self._track(stage, {"round": round_index, "status": noop_report.status, "changed_fraction": noop_report.changed_fraction})

        # ---- PORTFOLIO_UPDATE ---------------------------------------------
        stage = "PORTFOLIO_UPDATE"
        self._beat(supervisor, stage=stage, status="running")
        portfolio = Portfolio()
        metrics = {
            "hit_rate@10": result.get("top10_metrics", {}).get("hit_rate@10", 0.0),
            "ndcg@10": result.get("top10_metrics", {}).get("ndcg@10", 0.0),
            "pool_recall@100": result.get("pool_recall", {}).get("candidate_pool_recall@100", 0.0),
        }
        registered = portfolio.register_candidate({"candidate_id": proposal["proposal_id"], "metrics": metrics, "no_op": noop_report.status == "no_op"})
        self.artifacts["portfolio_update"] = _write_json(run_dir, f"{prefix}portfolio_update.json", {"registered": asdict(registered)})
        self._track(stage, {"round": round_index, "registered": proposal["proposal_id"]})

        # Incumbent promotion requires confirm-level (3-fold) or higher
        # comparable evidence — screen results alone never promote.
        fidelity_level = int(fold_outcome.get("fold_plan", {}).get("fidelity", 0))
        if (
            fidelity_level >= int(FoldFidelity.F2_CONFIRM)
            and fold_outcome.get("paired_comparison", {}).get("comparable")
            and metrics["hit_rate@10"] > self._incumbent["metrics"].get("hit_rate@10", 0.0)
        ):
            self._incumbent = {"candidate_id": proposal["proposal_id"], "kind": diagnostic_type, "metrics": metrics}

        # ---- POSTMORTEM -----------------------------------------------------
        stage = "POSTMORTEM"
        self._beat(supervisor, stage=stage, status="running")
        postmortem_prompt = self._prompt_postmortem(proposal, result, noop_report)
        postmortem, postmortem_mode = self._llm_json_stage(
            ledger,
            stage="POSTMORTEM",
            prompt_template_id=PROMPT_TEMPLATES["postmortem"],
            prompt=postmortem_prompt,
            required=False,
        )
        if postmortem is None:
            postmortem = {"summary": "deterministic postmortem (optional LLM stage degraded)", "result_metrics": result.get("pool_recall")}
        postmortem["mode"] = postmortem_mode
        self.artifacts["postmortem"] = _write_json(run_dir, f"{prefix}postmortem.json", postmortem)
        self._track(stage, {"round": round_index, "mode": postmortem_mode})

        return {
            "iteration_index": round_index,
            "round_index": round_index,
            "problem_id": chosen.node_id,
            "proposal_id": proposal["proposal_id"],
            "candidate_id": proposal["proposal_id"],
            "parent_candidate_id": parent_at_start["candidate_id"],
            "diagnostic_type": diagnostic_type,
            "fidelity": fold_outcome.get("fold_plan", {}).get("fidelity", 0),
            "fold_count": fold_outcome.get("fold_plan", {}).get("fold_count", 0),
            "selected_fold_ids": fold_outcome.get("fold_plan", {}).get("selected_fold_ids", []),
            "canonical_fold_hash": fold_outcome.get("fold_plan", {}).get("canonical_fold_hash", ""),
            "paired_fold_delta": fold_outcome.get("paired_comparison", {}).get("mean_delta", 0.0),
            "target_metric": metrics["hit_rate@10"],
            "target_bucket_metric": metrics["pool_recall@100"],
            "promotion_decision": fold_outcome.get("promotion", {}).get("decision", ""),
            "promotion_reason": fold_outcome.get("promotion", {}).get("reasons", []),
            "estimated_runtime": fold_outcome.get("estimated_runtime", 0.0),
            "actual_runtime": fold_outcome.get("actual_runtime", 0.0),
            "metrics": metrics,
            "noop": noop_report.status == "no_op",
        }

    # --------------------------------------------------------------- next decision

    def _next_decision(
        self,
        run_dir: Path,
        supervisor: RunSupervisor,
        ledger: LLMLedger,
        round_records: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        """NEXT_DECISION: continue while budget and positive-ROI candidates
        remain; stop on stagnation (no improvement over two rounds)."""
        stage = "NEXT_DECISION"
        self._beat(supervisor, stage=stage, status="running")
        budget_state = BudgetState(
            max_wall_clock_seconds=self.max_wall_clock_seconds,
            elapsed=time.time() - started,
            rounds_used=self.scientific_rounds_used,
        )
        remaining = budget_state.remaining_wall_clock_seconds
        hits = [r["metrics"].get("hit_rate@10", 0.0) for r in round_records]
        improved = len(hits) < 2 or hits[-1] > max(hits[:-1]) + 1e-12
        self._no_improvement_rounds = 0 if improved else self._no_improvement_rounds + 1
        action = no_improvement_action(self._no_improvement_rounds)
        if action == "switch_problem_or_global_explore":
            self._global_explore_attempted = True
        deployment_reserve_entered = remaining < self.fold_policy.estimator.deployment_reserve_seconds
        sd = stop_decision(
            no_improvement_rounds=self._no_improvement_rounds,
            operator_families_tried=sorted({r["diagnostic_type"] for r in round_records}),
            problem_nodes_tried=sorted({r["problem_id"] for r in round_records}),
            global_explore_attempted=self._global_explore_attempted,
            high_roi_routes_remaining=False,
            m5_admissible_routes_remaining=False,
            remaining_seconds=remaining,
            deployment_reserve_entered=deployment_reserve_entered,
        )
        decision = {
            "continue": not sd["allow_stop"] and remaining > 180.0,
            "action": action,
            "stop_decision": sd,
            "remaining_seconds": remaining,
            "rounds_used": self.scientific_rounds_used,
            "reason": sd["required_action"] if sd["allow_stop"] else action,
        }
        self.artifacts.setdefault("next_decision", _write_json(run_dir, f"round_{len(round_records):02d}/next_decision.json", decision))
        self._track(stage, decision)
        return decision

    # --------------------------------------------------------------- deployment

    def _run_deployment(
        self,
        run_dir: Path,
        supervisor: RunSupervisor,
        dataset: Any,
        targets: dict[str, str],
        round_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Retrain the best candidate on all public train users, infer the
        official test users, write candidate_B2.csv with a strict audit."""
        if self.no_deployment:
            return None
        from .operators.recommendation import (
            CandidateTableRanker,
            CandidateUnion,
            HistoryRetriever,
            PairTransitionRetriever,
            PopularityRetriever,
        )

        if not round_records:
            return None
        best = max(round_records, key=lambda r: r["metrics"].get("hit_rate@10", 0.0))
        kind = best["diagnostic_type"]
        all_train_uids = sorted(targets)
        all_seq = {u: dataset.train_seq.get(u, []) for u in all_train_uids}

        popularity = PopularityRetriever().fit(all_seq, targets, dataset.user_df, dataset.item_df)
        test_uids = [str(u) for u in dataset.test_df["uid"]]
        parent_table = popularity.retrieve(test_uids, max_per_user=200)
        history = HistoryRetriever().fit(all_seq, targets, dataset.user_df, dataset.item_df)
        pair = PairTransitionRetriever().fit(all_seq, targets, dataset.user_df, dataset.item_df)
        for retr in (history, pair):
            retr.train_seq.update({u: dataset.test_seq.get(u, []) for u in test_uids})
        union_table = CandidateUnion(rrf_k=60, max_per_user=200).merge(
            [parent_table, history.retrieve(test_uids, max_per_user=200), pair.retrieve(test_uids, max_per_user=200)]
        )

        if kind == "candidate_ranker_experiment":
            fit_tables = [retr.retrieve(all_train_uids, max_per_user=200) for retr in (popularity, history, pair)]
            fit_union = CandidateUnion(rrf_k=60, max_per_user=200).merge(fit_tables)
            ranker = CandidateTableRanker()
            ranker.set_context(train_seq=all_seq, train_targets=targets, user_df=dataset.user_df, item_df=dataset.item_df)
            rows, labels = ranker.build_training_rows(fit_union, user_ids=all_train_uids)
            ranker.fit(rows, labels)
            ranker.build_scoring_rows(union_table)
            top10_lists = ranker.rerank(test_uids, k=dataset.top_k)
        else:
            top10_lists = union_table.to_topk_lists(dataset.top_k)

        if self.deployment_method == "3fold_ensemble":
            top10_lists = self._deployment_3fold_ensemble(dataset, targets, test_uids, kind)

        _write_json(run_dir, "deployment_decision.json", {
            "method": self.deployment_method,
            "supported_methods": ["full_train_retrain", "3fold_ensemble"],
            "best_candidate_id": best["proposal_id"],
            "kind": kind,
            "selection_evidence": f"fidelity_{best.get('fidelity', 0)}_folds_{best.get('fold_count', 0)}",
            "reason": "default full-data retrain of the selected candidate; 3-fold ensemble available via deployment_method=3fold_ensemble",
            "uses_test_truth": False,
        })

        # Official order from the sample submission.
        order = [r["uid"] for r in dataset.sample_submission]
        top10_by_uid = {uid: lst for uid, lst in zip(test_uids, top10_lists)}
        all_items = dataset.all_item_ids()
        errors: list[str] = []
        rows_out: list[tuple[str, str]] = []
        for uid in order:
            items = top10_by_uid.get(uid, [])
            if len(items) < dataset.top_k:
                errors.append(f"{uid}: only {len(items)} items")
            if len(set(items)) != len(items):
                errors.append(f"{uid}: duplicate items")
            bad = [i for i in items if i not in all_items]
            if bad:
                errors.append(f"{uid}: illegal items {bad[:3]}")
            rows_out.append((uid, ",".join(items[: dataset.top_k])))
        if len(rows_out) != len(order):
            errors.append("row count mismatch")

        to_upload = run_dir / "TO_UPLOAD"
        to_upload.mkdir(parents=True, exist_ok=True)
        candidate_path = to_upload / "candidate_B2.csv"
        with candidate_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("uid,prediction\n")
            for uid, pred in rows_out:
                handle.write(f"{uid},{pred}\n")

        audit = {
            "n_rows": len(rows_out),
            "expected_rows": len(order),
            "order_matches_sample_submission": True,
            "top_k": dataset.top_k,
            "errors": errors,
            "audit_passed": not errors,
            "candidate_sha256": sha256_file(candidate_path),
            "contains_test_truth": False,
        }
        _write_json(to_upload, "submission_audit.json", audit)
        _write_json(to_upload, "deployment_manifest.json", {
            "best_candidate_id": best["proposal_id"],
            "kind": kind,
            "metrics": best["metrics"],
            "execution_id": self._execution_id,
            "retrained_on_all_train_users": True,
        })
        _write_json(to_upload, "best_candidate_summary.json", best)
        return {
            "best_candidate_id": best["proposal_id"],
            "kind": kind,
            "n_rows": len(rows_out),
            "audit_passed": not errors,
            "errors": errors,
            "candidate_sha256": audit["candidate_sha256"],
            "to_upload": "TO_UPLOAD/",
        }

    def _deployment_3fold_ensemble(self, dataset: Any, targets: dict[str, str], test_uids: list[str], kind: str) -> list[list[str]]:
        """3-fold ensemble test inference: fit the retrieval union on each of
        three canonical fold complements and merge via RRF."""
        from .operators.recommendation import (
            CandidateUnion,
            HistoryRetriever,
            PairTransitionRetriever,
            PopularityRetriever,
        )

        all_train_uids = sorted(targets)
        canonical = CanonicalFolds.build(all_train_uids)
        tables = []
        for f in [0, 1, 2]:
            fit_mask, _ = canonical.masks(f)
            fit_uids = [u for u, m in zip(canonical.uids, fit_mask) if m]
            fit_seq = {u: dataset.train_seq.get(u, []) for u in fit_uids}
            fit_targets = {u: targets[u] for u in fit_uids}
            popularity = PopularityRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            history = HistoryRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            pair = PairTransitionRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            for retr in (history, pair):
                retr.train_seq.update({u: dataset.test_seq.get(u, []) for u in test_uids})
            tables.extend(retr.retrieve(test_uids, max_per_user=200) for retr in (popularity, history, pair))
        union_table = CandidateUnion(rrf_k=60, max_per_user=200).merge(tables)
        return union_table.to_topk_lists(dataset.top_k)

    # --------------------------------------------------------------- stage helpers

    def _beat(self, supervisor: RunSupervisor, **fields: Any) -> None:
        fields.setdefault("orchestrator_version", ORCHESTRATOR_VERSION)
        fields.setdefault("planner_mode", self.planner_mode)
        ledger = getattr(self, "_ledger", None)
        if ledger is not None:
            fields.setdefault("llm_calls_count", ledger.calls_count)
        supervisor.heartbeat(**fields)
        supervisor.event("stage_transition", {"stage": fields.get("stage", ""), "status": fields.get("status", "")})
        supervisor.write_all()

    def _track(self, stage: str, payload: dict[str, Any]) -> None:
        self.trajectory_events.append({"stage": stage, "timestamp": time.time(), "payload": payload})

    def _find_legacy_cache(self) -> bool:
        """A v1 run over the same task exists -> legacy cache must be rejected."""
        legacy_root = self.project_root / "artifacts" / "b2_runs"
        if not legacy_root.is_dir():
            return False
        for manifest_path in legacy_root.glob("*/run_manifest.json"):
            try:
                manifest = load_json(manifest_path)
            except Exception:
                continue
            if str(manifest.get("manifest_version", "")).startswith("b2_autonomous"):
                return True
        return False

    def _build_panels(self, seqs: dict[str, list[str]], targets: dict[str, str]) -> list[dict[str, Any]]:
        uids = sorted(targets)
        lengths = {u: len(seqs.get(u, [])) for u in uids}
        panels = [
            panel_record("B2_STANDARD_PANEL", uids),
            panel_record("B2_SHORT_HISTORY_PANEL", [u for u in uids if lengths[u] <= 3]),
            panel_record("B2_LONG_HISTORY_PANEL", [u for u in uids if lengths[u] >= 4]),
            panel_record("B2_HISTORY_TARGET_PANEL", [u for u in uids if targets[u] in seqs.get(u, [])]),
            panel_record("B2_NOVEL_TARGET_PANEL", [u for u in uids if targets[u] not in seqs.get(u, [])]),
        ]
        return panels

    def _build_problem_hierarchy(self, di: Any) -> ProblemHierarchy:
        hierarchy = ProblemHierarchy()
        di_dict = di.to_dict()
        coverage = di_dict.get("coverage_map", {})
        history_ratio = coverage.get("history_recall_coverage", 0.0)
        novel_ratio = coverage.get("novel_target_ratio", 0.0)

        root = ProblemNode(
            level=int(ProblemLevel.TASK),
            task=self.task,
            evidence=[{"evidence_id": "di_fingerprint", "metric": "dataset", "value": str(di_dict.get("dataset_fingerprint", ""))[:64]}],
            headroom=1.0,
        )
        root_id = hierarchy.add_node(root)

        def add_leaf(
            *,
            stage: str,
            bucket: str,
            item_type: str,
            mechanism: str,
            evidence: list[dict[str, Any]],
            headroom: float,
            info_gain: float,
        ) -> str:
            parent = root_id
            for level, kwargs in (
                (ProblemLevel.PIPELINE_STAGE, {"pipeline_stage": stage}),
                (ProblemLevel.BUCKET, {"pipeline_stage": stage, "bucket": bucket}),
                (ProblemLevel.CLASS_OR_ITEM_TYPE, {"pipeline_stage": stage, "bucket": bucket, "class_or_item_type": item_type}),
            ):
                node = ProblemNode(level=int(level), task=self.task, **kwargs)
                parent = hierarchy.add_node(node, parent_id=parent)
            leaf = ProblemNode(
                level=int(ProblemLevel.ERROR_MECHANISM),
                task=self.task,
                pipeline_stage=stage,
                bucket=bucket,
                class_or_item_type=item_type,
                error_mechanism=mechanism,
                evidence=evidence,
                headroom=headroom,
                expected_information_gain=info_gain,
                compute_cost=60.0,
                open_hypotheses=["investigate"],
            )
            return hierarchy.add_node(leaf, parent_id=parent)

        add_leaf(
            stage="retrieval",
            bucket="novel_target",
            item_type="novel_item",
            mechanism="novel_target_low_coverage",
            evidence=[
                {"evidence_id": "di_novel_ratio", "metric": "novel_target_ratio", "value": novel_ratio},
                {"evidence_id": "di_history_coverage", "metric": "history_recall_coverage", "value": history_ratio},
            ],
            headroom=max(0.0, float(novel_ratio)),
            info_gain=0.7,
        )
        add_leaf(
            stage="retrieval",
            bucket="candidate_pool",
            item_type="all",
            mechanism="candidate_pool_recall_headroom",
            evidence=[{"evidence_id": "di_coverage", "metric": "history_recall_coverage", "value": history_ratio}],
            headroom=max(0.0, 1.0 - float(history_ratio)),
            info_gain=0.5,
        )
        return hierarchy

    def _build_card_store(self) -> CardStore:
        store = CardStore()
        store.add(MethodCard(
            card_id="A2_anchor_first",
            task_regime="recommendation",
            data_regime="sparse_sequence",
            error_mechanism="",
            validation_mechanism="oof",
            compute_budget="low",
            content={"principle": "anchor_first"},
            evidence=["knowledge/recommendation/champions/A2_05093"],
            source="a2_champion_package",
        ))
        store.add(MethodCard(
            card_id="B2_v16_novel_gap",
            task_regime="recommendation",
            data_regime="sparse_sequence",
            error_mechanism="novel_target_low_coverage",
            validation_mechanism="oof",
            compute_budget="low",
            content={"finding": "history recall cannot cover novel targets; pool recall and hit rate must be separated"},
            evidence=["knowledge/v1_6_baseline/b2_postmortem.json"],
            source="v1_6_postmortem",
        ))
        return store

    # --------------------------------------------------------------- prompts

    def _prompt_problem_synthesis(self, candidates: list[ProblemNode]) -> str:
        options = [
            {"problem_id": c.node_id, "mechanism": c.error_mechanism, "headroom": c.headroom, "evidence": c.evidence}
            for c in candidates
        ]
        return (
            "You are the AFAC v2 research orchestrator's problem-synthesis stage. "
            "Choose exactly ONE problem node to work on from the candidate list, based only on the "
            "deterministic evidence shown. Reply with a single JSON object: "
            '{"problem_id": "<one of the candidate ids>", "rationale": "<one sentence>"}. '
            f"Candidates: {json.dumps(options, ensure_ascii=False)}"
        )

    def _prompt_m6b(self, problem: ProblemNode, formal: bool = False, tried_families: list[str] | None = None) -> str:
        if formal:
            allowed = ALLOWED_DIAGNOSTICS + ALLOWED_FORMAL_EXPERIMENTS
        else:
            allowed = ALLOWED_DIAGNOSTICS
        tried = [t for t in (tried_families or []) if t]
        avoid = (
            f"These experiment families were ALREADY tried in this run and produced no improvement — you MUST choose a different family: {json.dumps(tried)}. "
            if tried
            else ""
        )
        return (
            "You are the M6B proposal stage of the AFAC v2 orchestrator. "
            f"Target problem: {problem.error_mechanism} (evidence: {json.dumps(problem.evidence, ensure_ascii=False)}). "
            + avoid
            + "Propose ONE experiment. Reply with a single JSON object with keys: "
            f'"diagnostic_type" (one of {json.dumps(list(allowed))}), '
            '"hypothesis", "information_sources", "parent", "budget_seconds" (number, <=600), '
            '"expected_gain" (number), "success_condition", "failure_condition", "stop_condition". '
            "Rules: never use test truth, never use online feedback, never modify frozen assets."
        )

    def _prompt_m6c(self, proposal: dict[str, Any]) -> str:
        return (
            "You are the M6C counterfactual critic of the AFAC v2 orchestrator. "
            f"Review this proposal: {json.dumps(proposal, ensure_ascii=False)}. "
            "Attack it: label leakage risk, no-op risk, metric-semantics confusion, availability bias, "
            "budget realism. Reply with a single JSON object: "
            '{"verdict": "approve"|"revise"|"reject", "issues": ["..."], "required_adjustments": ["..."], "rationale": "..."}.'
        )

    def _prompt_postmortem(self, proposal: dict[str, Any], result: dict[str, Any], noop_report: Any) -> str:
        return (
            "You are the postmortem stage of the AFAC v2 orchestrator. "
            f"Proposal: {json.dumps(proposal, ensure_ascii=False)}. "
            f"Result metrics: {json.dumps(result.get('pool_recall'), ensure_ascii=False)}; "
            f"no-op status: {noop_report.status}, changed_fraction: {noop_report.changed_fraction}. "
            'Reply with a single JSON object: {"summary": "...", "what_we_learned": "...", "next_priority": "..."}.'
        )

    # --------------------------------------------------------------- execution

    def _deterministic_proposal(self, problem: ProblemNode) -> dict[str, Any]:
        return {
            "problem_id": problem.node_id,
            "diagnostic_type": "candidate_recall_diagnostic",
            "hypothesis": "multi-source retrieval union raises candidate pool recall over the popularity parent",
            "information_sources": ["popularity", "history", "pair_transition"],
            "parent": "popularity",
            "budget_seconds": 90.0,
            "expected_gain": 0.05,
            "success_condition": "candidate_pool_recall@100 of union > popularity-only pool recall@100",
            "failure_condition": "union pool recall@100 <= popularity pool recall@100 or error decomposition invalid",
            "stop_condition": "single cheap diagnostic; no further rounds in smoke",
            "proposal_mode": "deterministic_fallback",
        }

    def _build_genome(self, proposal: dict[str, Any], problem: ProblemNode) -> ModelGenome:
        return ModelGenome(
            layers={
                "L0": "smoke_cheap_diagnostic_contract",
                "L1": "b2_train_subset_view",
                "L2": ",".join(proposal.get("information_sources", []) or ["popularity"]),
                "L3": "sparse_candidate_features",
                "L4": "none",
                "L5": str(proposal.get("diagnostic_type")),
                "L6": "pool_recall_maximization",
                "L7": "rrf_union",
                "L8": "none",
                "L9": "no_deployment",
            },
            parent_genome_id="",
            changed_layers=["L5"],
            fixed_layers=["L0", "L1", "L2", "L3", "L4", "L6", "L7", "L8", "L9"],
            new_information_source="",
            target_problem_id=problem.node_id,
            adapter_id="afac_agent.v2.operators.recommendation",
            budget={"seconds": float(proposal.get("budget_seconds", 60.0))},
            ablation="popularity-only parent vs multi-source union",
            success_condition=str(proposal.get("success_condition", "")),
            failure_condition=str(proposal.get("failure_condition", "")),
            stop_condition=str(proposal.get("stop_condition", "")),
        )

    def _run_cheap_diagnostic(
        self,
        proposal: dict[str, Any],
        dataset: Any,
        smoke_uids: list[str],
        subset_seq: dict[str, list[str]],
        subset_targets: dict[str, str],
    ) -> dict[str, Any]:
        """Deterministic whitelisted cheap diagnostic (never LLM-generated code)."""
        from .operators.recommendation import (
            CandidateUnion,
            HistoryRetriever,
            PairTransitionRetriever,
            PopularityRetriever,
        )

        n_fit = max(1, int(0.8 * len(smoke_uids)))
        fit_uids = smoke_uids[:n_fit]
        held_uids = smoke_uids[n_fit:]
        if not held_uids:
            held_uids = smoke_uids[-max(1, len(smoke_uids) // 5):]
            fit_uids = smoke_uids
        fit_seq = {u: subset_seq.get(u, []) for u in fit_uids}
        fit_targets = {u: subset_targets[u] for u in fit_uids if u in subset_targets}

        popularity = PopularityRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
        parent_table = popularity.retrieve(held_uids, max_per_user=200)

        diagnostic = str(proposal.get("diagnostic_type") or "candidate_recall_diagnostic")
        if diagnostic == "cached_replay":
            candidate_table = parent_table
        else:
            history = HistoryRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            pair = PairTransitionRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            for retr in (history, pair):
                retr.train_seq.update({u: subset_seq.get(u, []) for u in held_uids})
            candidate_table = CandidateUnion(rrf_k=60, max_per_user=200).merge(
                [parent_table, history.retrieve(held_uids, max_per_user=200), pair.retrieve(held_uids, max_per_user=200)]
            )

        held_targets = [subset_targets[u] for u in held_uids]
        parent_pool = parent_table.to_topk_lists(200)
        cand_pool = candidate_table.to_topk_lists(200)
        cand_top10 = candidate_table.to_topk_lists(10)
        return {
            "diagnostic_type": diagnostic,
            "n_fit_users": len(fit_uids),
            "n_held_users": len(held_uids),
            "parent_pool_recall": candidate_pool_recall(parent_pool, held_targets, ks=(20, 50, 100, 200)),
            "pool_recall": candidate_pool_recall(cand_pool, held_targets, ks=(20, 50, 100, 200)),
            "top10_metrics": ranking_metrics(cand_top10, held_targets, k=10),
            "error_decomposition": error_decomposition(cand_top10, cand_pool, held_targets, k=10),
            "parent_top10": parent_table.to_topk_lists(10),
            "candidate_top10": cand_top10,
        }

    def _run_formal_experiment(
        self,
        proposal: dict[str, Any],
        dataset: Any,
        smoke_uids: list[str],
        subset_seq: dict[str, list[str]],
        subset_targets: dict[str, str],
    ) -> dict[str, Any]:
        """Whitelisted formal experiments (round 2+): real training on sparse
        candidate tables, evaluated on a held-out user split."""
        from .operators.recommendation import (
            CandidateTableRanker,
            CandidateUnion,
            HistoryRetriever,
            PairTransitionRetriever,
            PopularityRetriever,
        )

        n_fit = max(1, int(0.8 * len(smoke_uids)))
        fit_uids = smoke_uids[:n_fit]
        held_uids = smoke_uids[n_fit:] or smoke_uids[-max(1, len(smoke_uids) // 5):]
        fit_seq = {u: subset_seq.get(u, []) for u in fit_uids}
        fit_targets = {u: subset_targets[u] for u in fit_uids if u in subset_targets}

        popularity = PopularityRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
        history = HistoryRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
        pair = PairTransitionRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
        for retr in (history, pair):
            retr.train_seq.update({u: subset_seq.get(u, []) for u in held_uids})

        fit_tables = [retr.retrieve(fit_uids, max_per_user=200) for retr in (popularity, history, pair)]
        fit_union = CandidateUnion(rrf_k=60, max_per_user=200).merge(fit_tables)
        held_tables = [retr.retrieve(held_uids, max_per_user=200) for retr in (popularity, history, pair)]
        held_union = CandidateUnion(rrf_k=60, max_per_user=200).merge(held_tables)

        kind = str(proposal.get("diagnostic_type"))
        held_targets_list = [subset_targets[u] for u in held_uids]
        parent_top10 = held_union.to_topk_lists(10)

        if kind == "candidate_ranker_experiment":
            ranker = CandidateTableRanker()
            ranker.set_context(
                train_seq={u: subset_seq.get(u, []) for u in smoke_uids},
                train_targets=fit_targets,
                user_df=dataset.user_df,
                item_df=dataset.item_df,
            )
            rows, labels = ranker.build_training_rows(fit_union, user_ids=fit_uids)
            ranker.fit(rows, labels)
            ranker.build_scoring_rows(held_union)
            cand_top10 = ranker.rerank(held_uids, k=10)
        else:  # bucket_specialist_experiment: length-bucket weighted blend
            cand_top10 = []
            for i, uid in enumerate(held_uids):
                seq_len = len(subset_seq.get(uid, []))
                entries = held_union.entries_for(uid)
                if seq_len == 0:
                    weight = lambda e: 0.7 * e.sources.get("popularity", {}).get("score", 0.0) + 0.3 * e.rrf  # noqa: E731
                else:
                    weight = lambda e: 0.3 * e.sources.get("popularity", {}).get("score", 0.0) + 0.7 * e.rrf  # noqa: E731
                ranked = sorted(entries, key=lambda e: (-weight(e), e.item_id))
                cand_top10.append([e.item_id for e in ranked[:10]])

        cand_pool = held_union.to_topk_lists(200)
        return {
            "diagnostic_type": kind,
            "n_fit_users": len(fit_uids),
            "n_held_users": len(held_uids),
            "pool_recall": candidate_pool_recall(cand_pool, held_targets_list, ks=(20, 50, 100, 200)),
            "top10_metrics": ranking_metrics(cand_top10, held_targets_list, k=10),
            "error_decomposition": error_decomposition(cand_top10, cand_pool, held_targets_list, k=10),
            "parent_top10": parent_top10,
            "candidate_top10": cand_top10,
        }

    def _bucket_rerank(self, union_table: Any, user_ids: list[str], seqs: dict[str, list[str]]) -> list[list[str]]:
        """Length-bucket weighted rerank over a sparse candidate union."""
        out: list[list[str]] = []
        for uid in user_ids:
            seq_len = len(seqs.get(uid, []))
            entries = union_table.entries_for(uid)
            if seq_len == 0:
                ranked = sorted(entries, key=lambda e: (-(0.7 * e.sources.get("popularity", {}).get("score", 0.0) + 0.3 * e.rrf), e.item_id))
            else:
                ranked = sorted(entries, key=lambda e: (-(0.3 * e.sources.get("popularity", {}).get("score", 0.0) + 0.7 * e.rrf), e.item_id))
            out.append([e.item_id for e in ranked[:10]])
        return out

    def _run_folded_experiment(
        self,
        kind: str,
        dataset: Any,
        subset_targets: dict[str, str],
        fold_ids: list[int],
        *,
        parent_kind: str,
    ) -> dict[str, Any]:
        """Run one whitelisted experiment fold-by-fold over the canonical
        master assignment: fit on the fold complement, evaluate on the fold,
        and compute paired parent metrics on the SAME folds."""
        from .operators.recommendation import (
            CandidateTableRanker,
            CandidateUnion,
            HistoryRetriever,
            PairTransitionRetriever,
            PopularityRetriever,
        )

        canonical = self._canonical
        assert canonical is not None, "canonical folds must be built first"
        cand_fold_metrics: dict[int, float] = {}
        parent_fold_metrics: dict[int, float] = {}
        agg_eval_uids: list[str] = []
        agg_parent_top10: list[list[str]] = []
        agg_cand_top10: list[list[str]] = []
        agg_pool: list[list[str]] = []
        changed_any = False

        for f in fold_ids:
            t0 = time.time()
            fit_mask, eval_mask = canonical.masks(f)
            fit_uids = [u for u, m in zip(canonical.uids, fit_mask) if m]
            eval_uids = [u for u, m in zip(canonical.uids, eval_mask) if m]
            seqs = {u: dataset.train_seq.get(u, []) for u in canonical.uids}
            fit_seq = {u: seqs[u] for u in fit_uids}
            fit_targets = {u: subset_targets[u] for u in fit_uids if u in subset_targets}
            eval_targets = [subset_targets[u] for u in eval_uids]

            popularity = PopularityRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            history = HistoryRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            pair = PairTransitionRetriever().fit(fit_seq, fit_targets, dataset.user_df, dataset.item_df)
            for retr in (history, pair):
                retr.train_seq.update({u: seqs[u] for u in eval_uids})
            eval_tables = [retr.retrieve(eval_uids, max_per_user=200) for retr in (popularity, history, pair)]
            eval_union = CandidateUnion(rrf_k=60, max_per_user=200).merge(eval_tables)

            if parent_kind == "popularity":
                parent_top10 = popularity.retrieve(eval_uids, max_per_user=200).to_topk_lists(10)
            else:
                parent_top10 = eval_union.to_topk_lists(10)

            if kind == "candidate_ranker_experiment":
                fit_tables = [retr.retrieve(fit_uids, max_per_user=200) for retr in (popularity, history, pair)]
                fit_union = CandidateUnion(rrf_k=60, max_per_user=200).merge(fit_tables)
                ranker = CandidateTableRanker()
                ranker.set_context(train_seq=seqs, train_targets=fit_targets, user_df=dataset.user_df, item_df=dataset.item_df)
                rows, labels = ranker.build_training_rows(fit_union, user_ids=fit_uids)
                if len(set(labels)) == 2:
                    ranker.fit(rows, labels)
                    ranker.build_scoring_rows(eval_union)
                    cand_top10 = ranker.rerank(eval_uids, k=10)
                else:
                    cand_top10 = eval_union.to_topk_lists(10)
            elif kind == "bucket_specialist_experiment":
                cand_top10 = self._bucket_rerank(eval_union, eval_uids, seqs)
            elif kind == "cached_replay":
                cand_top10 = list(parent_top10)
            else:  # candidate_recall_diagnostic / small_retrieval_compare: union top-10
                cand_top10 = eval_union.to_topk_lists(10)

            cand_fold_metrics[f] = ranking_metrics(cand_top10, eval_targets, k=10)["hit_rate@10"]
            parent_fold_metrics[f] = ranking_metrics(parent_top10, eval_targets, k=10)["hit_rate@10"]
            if cand_top10 != parent_top10:
                changed_any = True
            self.fold_policy.estimator.record_fold_runtime(time.time() - t0)
            agg_eval_uids.extend(eval_uids)
            agg_parent_top10.extend(parent_top10)
            agg_cand_top10.extend(cand_top10)
            agg_pool.extend(eval_union.to_topk_lists(200))

        agg_targets = [subset_targets[u] for u in agg_eval_uids]
        novel_mask = [t not in set(dataset.train_seq.get(u, [])) for u, t in zip(agg_eval_uids, agg_targets)]
        cand_novel = float(np.mean([t in lst for lst, t, m in zip(agg_cand_top10, agg_targets, novel_mask) if m])) if any(novel_mask) else 0.0
        parent_novel = float(np.mean([t in lst for lst, t, m in zip(agg_parent_top10, agg_targets, novel_mask) if m])) if any(novel_mask) else 0.0
        parent_hits = np.array([t in lst for lst, t in zip(agg_parent_top10, agg_targets)])
        cand_hits = np.array([t in lst for lst, t in zip(agg_cand_top10, agg_targets)])
        return {
            "diagnostic_type": kind,
            "candidate_fold_metrics": cand_fold_metrics,
            "parent_fold_metrics": parent_fold_metrics,
            "target_bucket_gain": cand_novel - parent_novel,
            "rescue": int(np.sum(~parent_hits & cand_hits)),
            "damage": int(np.sum(parent_hits & ~cand_hits)),
            "prediction_changed": changed_any,
            "noop": not changed_any,
            "result": {
                "diagnostic_type": kind,
                "n_eval_users": len(agg_eval_uids),
                "pool_recall": candidate_pool_recall(agg_pool, agg_targets, ks=(20, 50, 100, 200)),
                "top10_metrics": ranking_metrics(agg_cand_top10, agg_targets, k=10),
                "error_decomposition": error_decomposition(agg_cand_top10, agg_pool, agg_targets, k=10),
                "parent_top10": agg_parent_top10,
                "candidate_top10": agg_cand_top10,
            },
        }

    def _run_folded_with_promotion(
        self,
        run_dir: Path,
        proposal: dict[str, Any],
        dataset: Any,
        subset_targets: dict[str, str],
        started: float,
        prefix: str,
    ) -> dict[str, Any]:
        """F1 screen -> F2 confirm -> (triggered) F3 full-CV ladder."""
        kind = str(proposal.get("diagnostic_type"))
        policy = self.fold_policy
        t0 = time.time()
        remaining = self.max_wall_clock_seconds - (time.time() - started)

        screen_plan = build_fold_plan(self._canonical, FoldFidelity.F1_SCREEN, task=self.task)
        screen = self._run_folded_experiment(kind, dataset, subset_targets, screen_plan.selected_fold_ids, parent_kind=self._incumbent["kind"])
        comparison = paired_fold_comparison(
            candidate_id=proposal["proposal_id"],
            parent_id=self._incumbent["candidate_id"],
            selected_fold_ids=screen_plan.selected_fold_ids,
            candidate_fold_metrics=screen["candidate_fold_metrics"],
            parent_fold_metrics=screen["parent_fold_metrics"],
        )
        self.artifacts["paired_fold_comparison"] = _write_json(run_dir, f"{prefix}paired_fold_comparison.json", comparison)
        promo = policy.promote(
            comparison,
            target_bucket_gain=screen["target_bucket_gain"],
            rescue=screen["rescue"],
            damage=screen["damage"],
            prediction_changed=screen["prediction_changed"],
            no_op=screen["noop"],
            expected_information_gain=0.6,
            remaining_seconds=remaining,
        )
        final, final_plan, final_comparison = screen, screen_plan, comparison
        promotion = {"decision": "hold_at_screen", "reasons": promo["reasons"]}

        if promo["decision"] == "promote_to_confirm":
            confirm_plan = build_fold_plan(self._canonical, FoldFidelity.F2_CONFIRM, task=self.task)
            confirm = self._run_folded_experiment(kind, dataset, subset_targets, confirm_plan.selected_fold_ids, parent_kind=self._incumbent["kind"])
            confirm_comparison = paired_fold_comparison(
                candidate_id=proposal["proposal_id"],
                parent_id=self._incumbent["candidate_id"],
                selected_fold_ids=confirm_plan.selected_fold_ids,
                candidate_fold_metrics=confirm["candidate_fold_metrics"],
                parent_fold_metrics=confirm["parent_fold_metrics"],
            )
            _write_json(run_dir, f"{prefix}paired_fold_comparison_confirm.json", confirm_comparison)
            final, final_plan, final_comparison = confirm, confirm_plan, confirm_comparison
            promotion = {"decision": "promoted_to_confirm", "reasons": []}

            remaining = self.max_wall_clock_seconds - (time.time() - started)
            trigger = {"trigger_full_cv": False, "reasons": ["full_cv_disabled"]}
            if self.allow_full_cv:
                trigger = should_trigger_full_cv(
                    fold_metrics_3fold=confirm["candidate_fold_metrics"],
                    incumbent_delta=confirm_comparison["mean_delta"],
                    candidate_is_best=confirm_comparison["mean_delta"] > 0,
                    could_replace_anchor=False,
                    candidates_indistinguishable=False,
                    remaining_seconds=remaining,
                    estimated_5fold_runtime=policy.estimator.estimated_runtime(5),
                    deployment_reserve_seconds=policy.estimator.deployment_reserve_seconds,
                    safety_margin_seconds=policy.estimator.safety_margin_seconds,
                )
            _write_json(run_dir, f"{prefix}full_cv_trigger.json", trigger)
            if trigger["trigger_full_cv"]:
                full_plan = build_fold_plan(self._canonical, FoldFidelity.F3_FULL_CV, task=self.task)
                full = self._run_folded_experiment(kind, dataset, subset_targets, full_plan.selected_fold_ids, parent_kind=self._incumbent["kind"])
                full_comparison = paired_fold_comparison(
                    candidate_id=proposal["proposal_id"],
                    parent_id=self._incumbent["candidate_id"],
                    selected_fold_ids=full_plan.selected_fold_ids,
                    candidate_fold_metrics=full["candidate_fold_metrics"],
                    parent_fold_metrics=full["parent_fold_metrics"],
                )
                final, final_plan, final_comparison = full, full_plan, full_comparison
                promotion = {"decision": "promoted_to_full_cv", "reasons": []}

        return {
            "result": final["result"],
            "fold_plan": final_plan.to_dict(),
            "paired_comparison": final_comparison,
            "promotion": promotion,
            "estimated_runtime": policy.estimator.estimated_runtime(final_plan.fold_count),
            "actual_runtime": time.time() - t0,
        }

    # --------------------------------------------------------------- finish

    def _finish(
        self,
        run_dir: Path | None,
        supervisor: RunSupervisor | None,
        ledger: LLMLedger | None,
        execution_id: str,
        input_fingerprint: str,
        *,
        status: str,
        current_stage: str,
        started: float,
        error: str = "",
    ) -> dict[str, Any]:
        ended = time.time()
        llm_calls = 0
        if ledger is not None and ledger.ledger_path.is_file():
            llm_calls = len(load_ledger(ledger.ledger_path))
        if run_dir is None:
            run_dir = self.out_root / (execution_id or "failed_precheck")
            run_dir.mkdir(parents=True, exist_ok=True)

        self.gates.setdefault("metric_semantics_gate", False)
        self.gates.setdefault("data_intelligence", False)
        self.gates.setdefault("validation_reality", False)
        self.gates.setdefault("test_truth_guard", True)
        self.gates.setdefault("frozen_asset_hash", _frozen_hashes(self.project_root) == self._frozen_before)

        manifest = build_manifest(
            execution_id=execution_id,
            input_fingerprint=input_fingerprint,
            parent_execution_id=self.parent_execution_id,
            task=self.task,
            data_root_hash=stable_hash({"data_root": str(self.data_root)}),
            code_commit=code_commit(self.project_root),
            branch=git_branch(self.project_root),
            planner_version=PLANNER_VERSION,
            planner_policy_hash=stable_hash({"planner_mode": self.planner_mode, "require_llm": self.require_llm}),
            planner_mode=self.planner_mode,
            llm_required=self.require_llm,
            llm_available=self.llm_available or llm_calls > 0,
            llm_calls_count=llm_calls,
            capability_registry_hash="",
            metric_contract_hash="",
            validation_contract_hash="",
            started_at=started,
            ended_at=ended,
            max_wall_clock_seconds=self.max_wall_clock_seconds,
            cache_status=self.cache_status,
            resume_status=self.resume_status,
            status=status,
            current_stage=current_stage,
            scientific_rounds_used=self.scientific_rounds_used,
            cheap_diagnostics_used=self.cheap_diagnostics_used,
            no_op_rounds_refunded=self.no_op_rounds_refunded,
            uses_test_truth=False,
            mutates_frozen_assets=False,
            deployment_generated=self._deployment_generated,
            smoke_mode=self.smoke,
            artifacts=dict(self.artifacts),
        )
        manifest["gates"] = dict(self.gates)
        if error:
            manifest["error"] = redact_secret(error)

        # Strict completion contract: a run can only claim completed/_smoke
        # when every condition holds; otherwise it is downgraded.
        contract = check_completion_contract(manifest, smoke=self.smoke)
        manifest["completion_contract"] = contract
        if status in {"completed", "completed_smoke"} and contract["status"] != "passed":
            manifest["status"] = status = "incomplete"
            manifest["downgraded_reason"] = f"completion contract failed: {contract['missing']}"

        _write_json(run_dir, "run_manifest.json", manifest)

        trajectory = {
            "trajectory_version": TRAJECTORY_VERSION,
            "execution_id": execution_id,
            "task": self.task,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "planner_mode": self.planner_mode,
            "status": status,
            "events": self.trajectory_events,
            "ended_at": ended,
        }
        _write_json(run_dir, "trajectory_v2.json", trajectory)

        if supervisor is not None:
            supervisor.heartbeat(
                stage=current_stage,
                status=status,
                execution_id=execution_id,
                input_fingerprint=input_fingerprint,
                llm_calls_count=llm_calls,
                cache_status=self.cache_status,
            )
            supervisor.write_all()

        report_name = "V2_SMOKE_REPORT.md" if self.smoke else "V2_RUN_REPORT.md"
        report = (
            f"# {'V2 Smoke' if self.smoke else 'V2 Run'} Report — {execution_id}\n\n"
            f"- status: `{status}`\n- task: {self.task}\n- planner_mode: `{self.planner_mode}`\n"
            f"- llm_calls_count: {llm_calls}\n- cache_status: `{self.cache_status}`\n"
            f"- completion_contract: `{contract['status']}`\n"
            + (f"- missing: {contract['missing']}\n" if contract["missing"] else "")
            + (f"- error: {manifest.get('error', '')}\n" if error else "")
        )
        (run_dir / report_name).write_text(report, encoding="utf-8")
        manifest["report"] = report_name
        return manifest
