# CHANGELOG

## 2026-07-24 - B2 v2.1 Scientific Execution, Data Contract, Budget and Deployment Permission Repair

Root cause: the v2.0 orchestrator had no experiment-kind permission model, so
`candidate_recall_diagnostic` (0-fold, no training) was deployed as if it were
a confirmed scientific candidate. The fault run
(`74ba8db66a4b8f50d9196865a611e9d78058e14d35fd8ac99c6e2426279dcd41`) ran 3
diagnostic-only rounds, exceeded the 7200s hard budget (8340s actual), and
generated `candidate_B2.csv` with `deployment_generated=true` while
`scientific_attempts_used=0` and `all_rounds_diagnostic=true`.

- Added `afac_agent/v2/data_contract.py`: `B2CanonicalDataContract` with
  `ScaleValue` provenance for every count field. Cross-stage reconcile
  (`reconcile_data_contract()`) blocks execution when Input Discovery,
  Data Intelligence, Experiment Executor, and Deployment universes disagree.
  Canonical B2: n_items=14065 (item.csv::iid), n_train=40000, n_test=10000,
  n_interactions=1797067.
- Added `afac_agent/v2/experiment_kind.py`: `ExperimentKind` enum
  (DETERMINISTIC_DIAGNOSTIC, CACHED_REPLAY, SCREEN_EXPERIMENT,
  CONFIRM_EXPERIMENT, FULL_CV_EXPERIMENT, DEPLOYMENT_MODEL) with static
  `ExperimentPermission` contract. Diagnostics cannot deploy or be incumbent;
  Screens can enter portfolio but cannot deploy; Confirms (3-fold) can deploy
  and be incumbent.
- Added `afac_agent/v2/proposal_compiler.py`: `compile_proposal()` maps LLM
  `diagnostic_type` → canonical `operator_id`, rejects diagnostics in
  formal mode, and records `experiment_kind`. `semantic_revision_delta()`
  detects duplicates and reports which fields changed.
- Added `afac_agent/v2/anchor_registry.py`: `B2AnchorRegistry` builds
  popularity and history baselines on canonical 5-fold with same evaluator.
  Used as safe fallback when no scientific candidate has deployment permission.
- Added `afac_agent/v2/target_metric_contract.py`: same-fold, same-panel,
  same-metric, same-K, same-evaluator comparison contract.
- Extended `afac_agent/v2/capability_registry.py`: per-operator flags
  (supports_diagnostic/screen/confirm/full_cv/full_train/checkpoint/warm_start),
  deployment_ready override, new B2 v2.1 operators.
- Repaired `afac_agent/v2/orchestrator.py`: data contract build & reconcile,
  compiled operator pipeline replacing raw diagnostic_type, experiment
  permission checks at every gate, monotonic hard deadline (`time.monotonic()`)
  with `_hard_deadline` / `_research_deadline`, multi-round smoke loop
  (runs until ≥1 scientific experiment or budget exhausted), anchor fallback
  deployment, heartbeat transparency fields, completion contract v2.1.
- Extended `afac_agent/v2/completion_contract.py`: data_contract_status,
  budget_contract_status, deployment_permission_status, scientific_attempts,
  effective_scientific_rounds, all_rounds_diagnostic, no_op_in_portfolio,
  incumbent_can_deploy, anchor_fallback, wall_clock vs budget check.
- Repaired `afac_agent/v2/data_intelligence.py`: n_train_total/n_test_total
  separated from n_train_profiled/n_test_profiled; profile_scope recorded;
  raw/dedup/unique/repeat length buckets with full metadata.
- Extended `afac_agent/supervisor/heartbeat.py`: v2.1 transparency fields
  (operator_id, experiment_kind, semantic_genome_hash, fidelity, target_panel,
  permissions, deadlines, data contract fields).
- Added `tests/test_b2_science_repair.py`: 21 tests covering data contract,
  experiment permission, proposal compiler, no-op isolation, hard budget,
  orchestrator integration, capability registry, and fault manifest regression.
- Real 900s B2 science smoke: execution_id
  `0dd92953ffdcc53f9907aa1ebada1b9912f27022604adbf9783e09cef580dda1`,
  status `completed_smoke`, 599.5s wall clock, 12 LLM calls, 3 rounds
  (1 diagnostic + 2 scientific), scientific_attempts_used=2,
  all_rounds_diagnostic=false, data contract passed, completion contract
  passed, no deployment generated.
- Synthetic Smoke A-D: all covered by 21 pytest tests — all PASS.
- Fault run marked `INVALID_SCIENTIFIC_DEPLOYMENT`; artifact directory
  preserved as evidence under `artifacts/v2_runs/b2/`.
- Frozen assets: A1 v53Q-1 (0.7800), A2 0.5093, B1 V1/V2, B2 V1 — all
  verified unchanged. Frozen hashes match.
- Formal B2 two-hour run: **NOT started.**

## 2026-07-23 - Adaptive Fold Validation and Budget-aware Promotion

- Added `afac_agent/v2/adaptive_fold.py`: fidelity ladder F0_DETERMINISTIC
  (0 folds) → F1_SCREEN (B2: 2 folds, B1: 1-2 folds, screening only) →
  F2_CONFIRM (3 folds, incumbent promotions allowed) → F3_FULL_CV (5 folds,
  trigger-only, never default) → F4_DEPLOYMENT (full train + test inference).
- Fixed canonical folds: one hash-stable 5-fold master assignment per run;
  every fidelity selects a fixed prefix subset ([0,1]/[0,1,2]/[0..4]); no
  re-randomization, so candidates and parents stay comparable across rounds.
- Paired parent comparison on identical folds only
  (`paired_fold_comparison.json`); mismatched fold means are rejected as not
  comparable.
- Configurable promotion rules (mean delta, positive folds, worst-fold delta,
  target-bucket gain, rescue/damage/net, no-op veto, remaining budget);
  screen results never promote an incumbent — confirm-level evidence only.
- 5-fold trigger requires BOTH budget headroom (est. 5-fold + deployment
  reserve + safety margin) AND an uncertainty reason (3-fold variance,
  near-boundary delta, anchor replacement, indistinguishable candidates).
- FoldRuntimeEstimator: per-fold runtime history, dynamic 2/3/5-fold
  estimates, deployment reserve + safety margin enforcement.
- Stagnation semantics fixed: 1 no-improvement round → revise/switch
  operator family; 2 → switch problem/global explore; stopping requires
  global explore attempted + no viable routes, or deployment reserve entered
  (stop_decision records all fields).
- Formal multi-round loop wired to the ladder: each round runs F1 screen,
  promotes to F2 confirm when rules pass, triggers F3 only when justified;
  round records carry iteration/fidelity/fold ids/canonical hash/paired
  delta/promotion/runtime/next-decision fields.
- Deployment: full-train retrain (default) or 3-fold ensemble, recorded in
  `deployment_decision.json`; TO_UPLOAD with strict submission audit.
- Added `--limit-users` CLI flag for bounded control-flow smokes on the
  formal path.
- Added `docs/FORMAL_EXPERIMENT_CAPABILITY_AUDIT.md` and
  `tests/test_v2_adaptive_fold.py` (22 tests): fold ladder, canonical
  integrity, paired comparison, promotion, full-CV trigger, budget/reserve,
  stagnation semantics, formal loop + deployment end-to-end.

## 2026-07-23 - v2 Runtime Wiring, LLM Orchestration and False-Completion Repair

Root cause repaired: the only CLI entry ever wired was the legacy v1
deterministic runner, so a "v2 formal run" was actually a byte-identical v1
replay (`fcf5ad3dbcdf9700dd644eff`, candidate sha256 equal to the v1 run)
returning `status=completed` with zero LLM calls.  The directory
`artifacts/v2_formal_runs/b2/fcf5ad3dbcdf9700dd644eff` is preserved as
evidence and marked `invalid_v2_replay` (INVALID_V2_REPLAY_AUDIT.json /
INVALID_V2_REPLAY_REPORT.md).  CLI wiring audit:
`artifacts/v2_runtime_repair/v2_runtime_repair_d41fee9/`.

- Added `afac_agent/v2/execution_identity.py`: `input_fingerprint`
  (data/fold/config/metric-contract hashes, cache-compatible) vs
  `execution_id` (fingerprint + code commit + branch + orchestrator/planner
  versions + contract hashes + execution nonce + start time; unique per
  execution; secrets never enter any hash).
- Added `afac_agent/v2/llm_ledger.py`: LLM call contract — every call
  appended to `llm_calls.jsonl` with prompt/response hashes, latency, token
  counts, retry count and sanitized errors; secrets are redacted.
- Added `afac_agent/v2/legacy_replay_detector.py`: flags v1 manifests,
  missing orchestrator/planner/LLM fields, missing Problem/M6B/M6C
  artifacts and v1 run-id collisions as `invalid_replay`.
- Added `afac_agent/v2/completion_contract.py`: v2 run manifest spec and
  the strict completion contract (27 formal conditions, reduced smoke set);
  failed contracts downgrade `completed` to `incomplete`.
- Added `afac_agent/v2/orchestrator.py`: `V2AutonomousResearchOrchestrator`
  with the full PRECHECK → … → COMPLETED state machine.  Deterministic code
  computes stats, runs M5 gates and executes whitelisted cheap diagnostics;
  the LLM only synthesizes problems, writes M6B proposals, performs M6C
  counterfactual review and explains postmortems.  `--require-llm` blocks
  (`blocked_missing_llm` / `blocked_llm_error`) instead of silently
  completing; deterministic fallback only with an explicit flag and ends in
  `degraded_deterministic_fallback`.
- CLI separation (`afac_agent/main.py`): new `v2-run` (real v2 entrypoint)
  and `legacy-b2-closed-loop`; `b2-closed-loop` is now a loud legacy alias.
  Legacy commands refuse out-roots containing `v2_formal_runs` unless
  `--allow-legacy-output` is passed.
- Supervisor extended with v2 identity fields (execution_id, planner_mode,
  llm_calls_count, cache_status, problem/proposal/critic) and the dashboard
  shows a green V2 badge or a red LEGACY EXECUTION banner.
- Added `tests/test_v2_runtime.py` (28 tests): CLI separation, identity,
  ledger contract + secret redaction, replay detection, M5 non-bypass,
  completion contract, end-to-end smoke with a fake provider.
- Real-LLM B2 orchestration smoke passed: execution_id
  `9f668c3fe1986001d2744b16a0b807c9c628e88c083f193aef06fd4e655a3661`,
  status `completed_smoke`, 4 real LLM calls (problem synthesis, M6B, M6C,
  postmortem — all mode=llm), wall clock 187s, completion contract passed,
  no deployment and no formal submission generated.

## 2026-07-23 - AFAC Self-Evolving Research Agent v2.0 (Competition Full Edition)

Built on the frozen v1.6 baseline (`30efa2c`, branch `feat/afac-v2-full`).
No A1/A2/B1/B2 frozen asset, champion, anchor, fold, history, or trajectory
was modified.

- Added `knowledge/v1_6_baseline/`: baseline manifest, verified online results
  (B1 V1 `0.37908`, B1 V2 `0.37974`, B2 V1 `0.06838` with sha256-verified
  submission identity), B1/B2 postmortems, capability/validation/metric
  semantics gaps, no-op round records, memory failures, and the v2 initial
  priority queue.
- Added `knowledge/recommendation/champions/A2_05093/`: A2 champion
  architecture package (pipeline DAG, data view / candidate set / model
  permission / evaluation / safety contracts, lineage, closed routes).
  Only portable principles transfer (anchor_first, bucket_specialist,
  protected_residual, novel_only_rerank, boundary_admission,
  source_consensus_gate); A2 ids/scores/weights are forbidden in B2.
- Added `afac_agent/supervisor/`: Run Supervisor with heartbeat.json,
  STATUS.md, run_events.jsonl, unbuffered.log, self-contained dashboard.html,
  stall detection, and resume manager.
- Added `afac_agent/v2/metric_semantics.py`: pool recall @20/50/100/200
  separated from top-10 metrics, mutually exclusive error decomposition
  (top10_success + in_pool_outside_top10 + missing_from_candidate_pool = 1),
  panel membership-hash audit with duplicate-panel detection.
- Added `afac_agent/v2/noop_detector.py`: prediction/argmax hashes, changed
  fraction, score diffs, source contribution; no-op experiments refund the
  scientific round and are excluded from the portfolio.
- Added `afac_agent/v2/validation_reality.py`: offline/online anchors,
  submission identity, offline-online gap, panel calibration, deployment
  confidence, required B1/B2 panel sets.
- Added `afac_agent/v2/budget_scheduler.py`: fidelity ladder
  (static_audit/cached_replay/cheap_diagnostic/single_fold/full_oof/
  deployment), ROI scheduling, continuation past 3 rounds while budget and
  positive-ROI candidates remain, premature-stop detection.
- Added `afac_agent/v2/problem_hierarchy.py`,
  `afac_agent/v2/model_genome.py` (L0-L9 genome, Parent + One Primary Change +
  Optional Safety Adjustment), `afac_agent/v2/capability_registry.py`
  (scientific best vs available best vs selected, availability bias,
  waiting_for_adapter), `afac_agent/v2/exploration_controller.py`
  (exploit/adjacent/global explore with stagnation, local-optimum,
  availability/anchor/scope/validation bias and premature-stop detectors),
  `afac_agent/v2/competition_intelligence.py` (solution/method/failure/
  validation cards + MLE-STAR loop).
- Added `afac_agent/v2/memory_safe.py`: SparseCandidateTable, chunked
  scoring, TopK streaming, Memory Preflight (dense float64 user-item
  matrices above budget are never allowed).
- Added `afac_agent/v2/operators/classification.py`: feature ops (linear,
  GBDT, residual MLP, PCA low-rank, prototype), graph ops (LP, SGC, APPNP,
  degraded GraphSAGE/GCN, heterophily GNN registered unavailable), seven
  graph views with content hashes, strict cross-fit fusion ops, operator
  catalog.
- Added `afac_agent/v2/operators/recommendation.py`: R0-R8 pipeline with
  greenfield and champion-preserving modes, 15 retrievers, candidate union
  with source_count/RRF, true candidate-table ranker (sklearn GBDT backend;
  lightgbm LambdaRank gap honestly recorded), stable top-10 anchor, bucket
  expert router, TopK/history-slot/novel-only protection, position-10
  admission (at most one external item), fallback-keep-parent.
- Added `afac_agent/v2/portfolio.py` (incumbent-default parent rule with
  M5-style rejection of unjustified non-incumbent parents) and
  `afac_agent/v2/online_feedback.py` (online feedback guardrails).
- Added `afac_agent/v2/data_intelligence.py`: unified deterministic
  classification/recommendation data intelligence with LLM-mutation guard.
- Added `afac_agent/v2/smokes.py`: four task smokes (A1/A2/B1/B2), each
  bounded to minutes, all passing with frozen hashes unchanged.
- Added 106 synthetic tests (313 passed total, doctor PASS).

## 2026-07-23 - B1 Repair + B2 First Autonomous Dual-Task Closed Loop

- Added `afac_agent.b1.repair`: scientific validity repair module with raw graph
  direction audit, prediction asset audit, homophily repair, fold label leakage
  audit, propensity semantics audit, effective rank audit, and offline-to-online
  gap audit.
- Fixed `afac_agent/b1/models.py` graph view construction bug: `directed_out`
  and `directed_in` now correctly use `adj` and `adj.T` instead of both being
  `adj + adj.T`.
- Re-audited `B1_EVAL_ANCHOR_V1` and materialized `B1_EVAL_ANCHOR_V2` from the
  repaired `undirected_union` label propagation candidate.
- B1 V2 closed loop completed: run id `c5e33663d37c6e49004e8213`, 3 rounds,
  wall clock `75.60s`, best candidate `B1_LP_UNDIRECTED_ALPHA7`
  (standard accuracy `0.4964`, macro `0.3941`), submission
  `candidate_B1_v2.csv` generated and audited, not uploaded.
- Added `afac_agent.b2` package: generic `B2TaskAdapter`, Data Intelligence,
  `AFAC_B2_FOLD_V1` with nine validation panels, `B2Evaluator` with
  retrieval/ranking failure split, and CPU-only retrieval/ranking models
  (popularity, history recall, item-item co-occurrence, last-item transition,
  score blend, candidate ranker).
- Added `B2ClosedLoopRunner` enforcing max 3 scientific rounds, 2-hour wall
  clock, single-process/single-GPU budget, automatic best scientific/deployment
  selection, full-train retrain, Test inference, `candidate_B2.csv` generation,
  `TO_UPLOAD` packaging, and `trajectory_B2.json`.
- Added CLI `python -m afac_agent.main b2-closed-loop`.
- Added `afac_agent.dual_task_master` to create master manifest, task sequence,
  resource budget, cross-task isolation audit, dual-task report, and combined
  TO_UPLOAD bundles for B1 and B2.
- Added synthetic tests: `tests/test_b1_repair.py` and
  `tests/test_b2_integration.py` covering B1 repair semantics and B2 adapter,
  fold, evaluator, models, closed loop, namespace isolation, and A1/A2/B1
  no-regression.
- B2 first closed loop completed: run id `fcf5ad3dbcdf9700dd644eff`, 3 rounds,
  wall clock `1807.57s`, best candidate `B2_HISTORY_RECALL`
  (NDCG@10 `0.1548`, HitRate@10 `0.2515`, MRR@10 `0.1234`), submission
  `candidate_B2.csv` generated and audited, not uploaded.
- Dual-task master run: `86f75dbc66d25382eb0c6a24` with TO_UPLOAD/B1 and
  TO_UPLOAD/B2.
- Full test suite: 207 passed; Doctor: PASS.

No A1/A2/B1 Champion, Anchor, Fold, history, scientific-round counters, or
Project State was modified. No automatic platform upload.

## 2026-07-22 - B1 Data-First Autonomous Classification Closed Loop

- Added `afac_agent.b1` package: generic `NodeClassificationTaskAdapter`
  parameterized over node/feature/class counts; reads B1.npz, validates
  train/test isolation, hidden Test truth, and official sample-submission order.
- Added B1 Data Intelligence orchestrator with integrity audit, feature geometry
  (density, PCA variance, effective rank, feature shift), graph regime
  (directed-out/in/undirected views, PageRank, degree assortativity,
  connectivity), label-graph reliability (homophily, class-conditioned
  homophily, hop reliability, degree buckets), train/test propensity shift
  audit, and A1 transferability matrix.
- Added `AFAC_B1_FOLD_V1` stratified 5-fold with degree-matched,
  propensity-matched, low-degree and test-like validation panels.
- Added B1 CPU-only models: feature LR (with StandardScaler), small MLP,
  iterative label propagation, APPNP-like smoothing + LR, and neighbor-mean
  feature augmentation + LR.
- Added `B1Evaluator` with standard/degree/propensity/low-degree/test-like
  panels, per-class accuracy, fold stability, and rescue/damage/change metrics;
  fixed per-class accuracy computation to report recall per class rather than
  class frequency.
- Added cross-fit fusion operators (probability blend, logit blend,
  class-weighted blend) and cross-fit node-level gate with single-class-target
  fallback.
- Added `B1ClosedLoopRunner` enforcing max 3 scientific rounds, 2-hour wall
  clock, single-process/single-GPU budget, automatic best scientific/deployment
  selection, full-train retrain, Test inference, `candidate_B1.csv` generation,
  `TO_UPLOAD` packaging, and `trajectory_B1.json`.
- Added CLI `python -m afac_agent.main b1-closed-loop`.
- Added 11 B1 synthetic tests covering adapter parameterization, namespace
  isolation, transfer gate, Data Intelligence gate, fold integrity, OOF/Test
  isolation, fusion operators, gate fallback, submission format, and A1/A2
  no-regression.
- Real B1 run completed: run id `e95368a24e0780e65e92aceb`, 3 rounds,
  wall clock `43.66s`, best candidate `B1_LP_DIRECTED_OUT`
  (standard accuracy `0.4907`, macro `0.3838`), submission `candidate_B1.csv`
  generated and audited, not uploaded.

No A1/A2 Champion, Anchor, Fold, history, scientific-round counters, or
Project State was modified. B2 (B榜推荐) path was not read.

## 2026-07-22 - A2 Task Integration and Evaluation Foundation

- Added `afac_agent.a2` package: read-only A2 TaskAdapter (uid/iid validation,
  history sequence parsing, Test user order, Top10 legality, Test truth
  isolation), A2 Data Profiler with len0/1/2/exact_len3/4+, history/novel and
  retrieval-vs-ranking bucket registry, and A2 Evaluator (NDCG@10, HitRate@10,
  MRR@10, Candidate Recall, Target Rank, bucket breakdowns, Rescue/Damage/Net,
  Changed User Count, Change Precision).
- Added AFAC_A2_FOLD_V1 validation for the explicit `train_folds.csv`
  assignment (each train user exactly once, no Test users, legal folds,
  explicit user order, stable hash, auditable per-fold distributions). Folds
  are never re-split.
- Added A2 dual anchors: `A2_ONLINE_CHAMPION_05093` (deployment identity only,
  never an OOF artifact) and `A2_EVAL_ANCHOR_V1` (materialized from the
  verified v42c-DIN OOF route; deployment_equivalent=false).
- Added A2 asset portfolio registration with strict OOF/Test scope isolation;
  assets without explicit local artifacts are `declared_unmaterialized`,
  never fabricated.
- Added recommendation fusion operators (score_blend, rank_fusion,
  candidate_union, bucket_route, topk_protected_rerank,
  slot_protected_rerank, retriever_ranker_composition) with strict outer-fold
  cross-fit for learned parameters; oracle union recall is diagnostic only.
- Added A2 complementarity audit (retriever coverage A-only/B-only, rank
  rescue/damage/net, Len and history/novel buckets, topK ranges,
  candidate-set-change vs fixed-set-rerank separation).
- Added `python -m afac_agent.main a2-integration` dry-run chaining
  Profile → Scientific Queue → Portfolio → Complementarity → Fusion Planner
  → M6B → M6C → M5 → Execution/Evaluation/Trajectory Previews. No training,
  no GPU, no Test prediction, no submission, no LLM, no network, and
  A2 scientific_rounds_used stays 0.
- Added 21 A2 tests covering validation, metrics, failure separation, buckets,
  fold integrity, OOF/Test isolation, operators, cross-fit, anchor identity,
  and dry-run safety.

No A1 champion CSV, online score, Fold definition, Gate definition, OOF
anchor, Project State, closed branch, confirmed history, or A2 scientific
round was changed.

## 2026-07-20 - M3A Tool Adapter Foundation

- Added the minimal M3A Tool Adapter protocol and AdapterRunner.
- Added standard Adapter Execution Result Schema.
- Extended Tool Registry metadata with optional Adapter binding fields.
- Added `A1_V53Q1_PATCH_AUDIT`, the first real read-only Adapter.
- Added CLI support for `python -m afac_agent.main run-adapter`.
- Integrated Adapter execution into Orchestrator without removing legacy
  `command_template` execution.
- Added Doctor checks for Adapter schema, registry bindings and output root.
- Added tests for missing inputs, failed validation, duplicate identity,
  read-only frozen-file hashes, stdout/stderr logs, CLI, Orchestrator and
  Windows Chinese/space paths.

No champion CSV, online score, Fold definition, Gate definition, OOF anchor,
Project State, closed branch, or historical record was changed.

## 2026-07-20 - M2 A1 Data Profiler

- Added a read-only, CPU-only, deterministic A1 Data Profiler.
- Added dataset-only, fold-aware-structure, and full-anchor-OOF analysis tiers.
- Added directed graph auditing with self-loop and duplicate directed-edge counts.
- Added in/out/incident/either degree fields and either-neighbor degree buckets.
- Added exact 1/2/3/4-hop visible-train counts with fold-aware visibility when a canonical fold file is provided.
- Added OOF global `train_idx` alignment and prediction Sink/Source based on the OOF confusion matrix.
- Added deterministic core artifact hashing with manifest timestamps excluded from the core hash.
- Added M2 schemas, tests, CLI entrypoint, legacy tool wrapper, doctor integration, and Tool Registry metadata.
- Added generic tool metadata for read-only behavior, experiment-round consumption, prediction mutation, Project State mutation, and GPU requirements.
- Ensured `PROFILE_A1_DATASET` does not consume successful experiment rounds and does not mutate Project State.
- Added Champion Test predicted-label distribution to deterministic M2 artifacts without treating it as Test truth.
- Replaced empty `train_test_shift` placeholders with structured observed/unavailable/not-applicable shift status.
- Explicitly marked directed exact-hop breakdown as `not_generated` for M2 v1 while keeping the primary exact-hop view as either-direction.
- Added `legacy_data_profile_flag_stale` doctor warning for stale legacy data-profile state flags.

No champion CSV, online score, Fold definition, Gate definition, OOF anchor, closed branch, or historical record was changed.

## 2026-07-20 - M0 + M1 Stabilization

- Added M0/M1 tests for path handling, missing inputs, idempotent history import, idempotent champion registration, validation, doctor, and Windows Chinese/space paths.
- Added project-root-relative path resolver with packaged A1 champion default.
- Added `python -m afac_agent.doctor` preflight.
- Added lightweight contract validation for ProjectState, Tool Registry, Memory records, Trajectory, Anchor Manifest, and A1 champion CSV.
- Added structured `waiting_for_input` handling for missing files and unbound registered tools.
- Ensured failed tool executions do not consume successful experiment rounds.
- Added `pyproject.toml` and `.gitignore`.
- Updated bootstrap scripts to remove hard-coded Python interpreter paths.

No champion CSV, online score, Fold definition, Gate definition, OOF anchor, closed branch, or historical record was changed.
