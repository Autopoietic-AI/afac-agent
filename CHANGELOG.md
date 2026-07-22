# CHANGELOG

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
