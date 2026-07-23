# CHANGELOG

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
