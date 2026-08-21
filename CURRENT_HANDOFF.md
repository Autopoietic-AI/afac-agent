# AFAC Agent Current Handoff

## Current state (2026-07-24 wrap-up)

- Branch `feat/afac-v2-full`, HEAD `00840d7` (post-merge of `fix/b2-v2-scientific-execution`).
- All four tasks have a verified, budget-safe scientific loop:
  - A1: frozen champion v53Q-1 (0.7800); closed loop complete.
  - A2: integration dry-run complete (`ready_for_experiment_design`); real closed loop waits for explicit user asset paths (V23 Top10 etc.).
  - B1: v2.2 wired into v2 orchestrator with hard-deadline circuit breaker; earlier V1/V2 submissions preserved under `artifacts/b1_runs/`.
  - B2: v2.1 repaired (data contract, experiment permissions, budget, deployment permission, anchor fallback); 900s real-LLM smoke passed; fault run marked invalid.
- Known limits (recorded honestly):
  - candidate_ranker at real B2 scale: ~3885s/fold, retrieval-bound; 2h formal run needs retrieval optimization first (see `artifacts/b2_science_repair/ranker_benchmark/B2_OPERATOR_PERFORMANCE_REPORT.md`).
  - lightgbm/torch absent: LambdaRank/DIN/SASRec registered as unavailable.
- Full suite: 407 passed; Doctor: PASS.

## Next steps (need explicit user input)

1. B2 two-hour formal run — only after retrieval optimization; use `run_b2_formal_safe.sh` preflight.
2. A2 real closed loop — only after user supplies explicit asset paths (V23 Top10 candidate set first).
3. Platform upload of existing TO_UPLOAD candidates — manual user decision only.

## Previous module: B2 v2.1 repair (merged)

# AFAC Agent Current Handoff

## Current module

- Module: Adaptive Fold Validation and Budget-aware Promotion
- Branch: `feat/afac-v2-full`
- Previous baseline: `ba226bd31aa61c3e75ca11740655595fa10d1aba` (v2 runtime wiring + false-completion repair)
- Scope: AdaptiveFoldPolicy (F0→F1→F2→F3→F4 ladder), fixed canonical folds with prefix subsets, paired same-fold parent comparison, configurable promotion rules, full-CV trigger, runtime estimation with deployment reserve, stagnation semantics fix, formal multi-round loop integration, deployment decision (full retrain / 3-fold ensemble), capability audit doc, bounded 600s control-flow smoke. No two-hour formal run started.

## Adaptive fold key facts

- Ladder: F0_DETERMINISTIC (0 folds) → F1_SCREEN (B2: 2 folds, B1: 1 fold) → F2_CONFIRM (3 folds, incumbent promotion allowed) → F3_FULL_CV (5 folds, trigger-only) → F4_DEPLOYMENT (full train + test inference).
- Canonical folds: single hash-stable 5-fold master per run; fixed prefix subsets; never re-randomized.
- Parent comparison: identical folds only; mismatched means are rejected as not comparable; screen evidence alone never promotes an incumbent.
- 5-fold trigger: budget headroom AND uncertainty (variance / near-boundary / anchor replacement / indistinguishable); disabled via `--no-full-cv`.
- M5 budget handling: clean proposals with over-estimated budgets are clamped (recorded `budget_clamped`), not killed; BUDGET_DECISION consumes the clamped estimate.
- Stagnation: round 1 no-improvement → revise/switch family; round 2 → switch problem/global explore; stop only after global explore with no viable routes or deployment reserve entered.
- Capability audit: `docs/FORMAL_EXPERIMENT_CAPABILITY_AUDIT.md`.
- Tests: `tests/test_v2_adaptive_fold.py` (22), full suite 363 passed, Doctor PASS.

## Previous module (P0 runtime repair, committed ba226bd)
- Branch: `feat/afac-v2-full`
- Previous frozen baseline: `d41fee95abce61aa314f1c08f69db8a31d472252` (v2.0 architecture build)
- Scope: invalid-replay marking, CLI wiring audit, v2 orchestrator (real state machine), execution identity, LLM call ledger, legacy replay detector, strict completion contract, CLI separation, runtime tests.

## Invalid v2 replay (root cause and disposition)

- Directory: `artifacts/v2_formal_runs/b2/fcf5ad3dbcdf9700dd644eff` — preserved as evidence, marked `invalid_v2_replay` (INVALID_V2_REPLAY_AUDIT.json / INVALID_V2_REPLAY_REPORT.md).
- Root cause: the only CLI entry was the legacy v1 deterministic runner; its `run_id` is a data-only hash, so the "v2 formal run" reproduced the v1 run byte-for-byte (candidate sha256 `b705d15f…` identical) and falsely returned `status=completed` with zero LLM calls.
- CLI wiring audit: `artifacts/v2_runtime_repair/v2_runtime_repair_d41fee9/` (legacy/expected call graphs, gap analysis, CLI_WIRING_AUDIT.md).

## New v2 runtime contract

- Real v2 entry: `python -m afac_agent.main v2-run --task B2 --data-root <path> --out-root <path> --require-llm --force-new-execution [--smoke ...]`.
- Legacy reproduction: `python -m afac_agent.main legacy-b2-closed-loop ...` (refuses `v2_formal_runs` out-roots without `--allow-legacy-output`).
- `b2-closed-loop` is a loud legacy alias only — never a v2 run.
- `input_fingerprint` (cache identity) is strictly separated from `execution_id` (unique per execution, includes code commit + nonce).
- LLM contract: M6B/M6C (+ problem synthesis, postmortem) are real provider calls recorded in `llm_calls.jsonl`; `--require-llm` blocks instead of silently completing.
- Strict completion contract: missing any of Problem Node / M6B / M6C / M5 / Genome / Budget / No-op audit / Postmortem / LLM ledger downgrades `completed` to `incomplete`.
- Runtime smoke with real LLM: COMPLETED — execution_id `9f668c3fe1986001d2744b16a0b807c9c628e88c083f193aef06fd4e655a3661`, status `completed_smoke`, 4 real LLM calls (problem synthesis, M6B, M6C, postmortem; all mode=llm), wall clock 187s, union pool recall@100 0.466 vs popularity parent 0.233, no deployment, no formal submission.
- Formal two-hour B2 v2 run has NOT started.

## Previous module (v2.0 architecture, committed d8775fd + d41fee9)
- Branch: `feat/afac-v2-full`
- Commit status: this file is part of the module commit; use `git rev-parse HEAD` after commit for the immutable commit id.
- Previous frozen baseline: `30efa2cbb4264630e0e1e984d55e3fffeba354e1` (B1 Repair + B Dual-Task Autonomous Loops, v1.6)
- Scope: v1.6 baseline/postmortem knowledge package, Run Supervisor, Metric Semantics Gate, No-op Detector, Dynamic Budget Scheduler, Unified Data Intelligence, Validation Reality Manager, Problem Hierarchy, Competition Intelligence + MLE-STAR, Global Exploration Controller, Hierarchical Model Constructor, Capability Registry, Memory-safe Executor, full classification and recommendation operator spaces, A2 Champion Architecture Package, portfolio/parent rules, online feedback registration, four task smokes. No full two-hour loop was started.

## v2.0 key facts

- pytest: `313 passed` (207 v1.6 baseline + 106 new v2.0 tests); Doctor: PASS.
- Four smokes passed: A1 classification, A2 recommendation, B1 classification (real B分类 data, LP held-out accuracy `0.2549`), B2 recommendation (real B推荐 data, pool recall@100 `0.4233`, sparse candidate table only).
- v1.6 online results registered with verified submission identity: B1 V1 `0.37908`, B1 V2 `0.37974`, B2 V1 `0.06838`.
- v1.6 postmortem package: `knowledge/v1_6_baseline/`; A2 champion package: `knowledge/recommendation/champions/A2_05093/`.
- Smoke outputs: `artifacts/v2_smokes/` (gitignored).
- Frozen hashes verified unchanged before/after smokes (`config/project_state.json`, `history/confirmed_experiments_a1.json`, `artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv`).

## Previous module (v1.6, frozen)

- Module: A2 Task Integration + Evaluation Foundation + Existing Asset Integration + A2 Dry-run
- Previous frozen baseline: `e82c01842138502be70b6d2ea8ed1d7d09ead3fe` (A1 Autonomous Scientific Closed Loop v1)
- Scope: A2 TaskAdapter, Data Profiler, Evaluator, AFAC_A2_FOLD_V1 validation, dual anchors, asset portfolio, ranking fusion operators, complementarity audit, full dry-run. No training, no Test prediction, no submission.

## Frozen facts

- Online Champion: `v53Q-1`
- Online score: `0.7800`
- Canonical OOF baseline remains frozen by policy.
- Fold, Gate, confirmed history, closed branches, Champion CSV, Test predictions, and submissions must not be modified by this module.
- Scientific rounds used by this module: `0`

## Evaluation anchor

- Anchor id: `A1_EVAL_ANCHOR_V1`
- Anchor directory: `artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1`
- Anchor OOF: `artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1/A1_EVAL_ANCHOR_V1_oof.npz`
- Anchor role: offline evaluation anchor only; not an online Champion replacement.

## Registered portfolio assets

- `A1_EVAL_ANCHOR_V1`: offline evaluation anchor OOF.
- `V43C_ASSOCIATED_REPRODUCIBLE_BASE_OOF`: v43C associated reproducible base OOF, using `base_proba`.
- `V46A_ISOLATED_EXPERT_OOF`: v46A isolated expert OOF, using `expert_proba` plus `isolated_mask`.
- `V46A_ISOLATED_COMPOSED_OOF`: v46A isolated composed OOF, using full `proba`.

All registered OOF assets must be verified against the explicit train index, labels, and fold assignment before use.

## Fusion controller boundaries

- No model training.
- No GPU use.
- No Test prediction generation.
- No submission generation.
- No Champion mutation.
- No Project State or confirmed history mutation.
- No LLM call, network call, Adapter execution, or scientific round consumption.
- Oracle upper bound is diagnostic only and cannot be used as accepted gain.

## Real smoke command

```powershell
python -m afac_agent.main fusion-controller --project_root . --anchor-dir artifacts/evaluation_anchor/A1_EVAL_ANCHOR_V1 --v43c-oof "C:\Users\李天皓\agent比赛\model_pro\a1\new-a1\a1_openroute_cs_v1_v43c\correct_smooth_v1_oof_proba.npz" --v46a-oof "C:\Users\李天皓\agent比赛\model_pro\a1\new-a1\a1_v46a_tabm_two_seed_decision\two_seed_balanced_candidate_oof.npz" --out-root artifacts/fusion_runs --force-rebuild
```

## Latest real smoke result

- Status: `completed`
- Run id: `b053e405ff9f6dcf9d7df95b287e16d16c5d3669f2c17a076c7ce00efda98244`
- Candidate count: `11`
- Accepted count: `2`
- Best candidate: `class_weighted_blend_base_composed`
- Handoff status: `ready_for_real_single_round_experiment_design`

## Next recommended step

Use the Fusion Controller report package as offline evidence for the next strictly controlled single-round experiment design. Do not start training or Test prediction until the next module explicitly approves the experiment boundary and output path.

## A1 Closed Loop v1 status

- Module: A1 Autonomous Scientific Closed Loop v1
- Scope: bounded single-run closed loop with multi-round control and safe stop.
- Execution path: real offline OOF fusion execution only.
- Max scientific rounds: `3`
- Test truth: forbidden.
- Candidate CSV / submission: not generated by default.
- Online Champion, Project State JSON, confirmed history, Fold and Gate remain frozen.
- Runtime artifacts: `artifacts/a1_closed_loop_runs/`
- Commit status: this section is part of the module commit; use `git rev-parse HEAD` after commit for the immutable commit id.
- Latest real run id: `a3f0c67c984e0fafc7c97ca9`
- Actual scientific rounds used: `1`
- Best runtime portfolio candidate: `class_weighted_blend_base_composed`
- Fusion candidate final status: `accepted_portfolio`
- Stop reason: `macro_protection_prevents_direct_promotion_and_repeating_same_information_source_is_low_value`

## A2 Integration status

- Module: A2 Task Integration + Evaluation Foundation (dry-run only).
- A2 data root and runs root: external, supplied via CLI (`--a2-data-dir`, `--a2-runs-root`), never hard-coded.
- A2 fold protocol: `AFAC_A2_FOLD_V1`, validated from `runs/stageA/train_folds.csv` (each train user exactly once, no Test users, folds 0-4, explicit order, stable hash).
- A2 Online Deployment Anchor: `A2_ONLINE_CHAMPION_05093` (identity only; online score `0.5093`; never an OOF artifact).
- A2 Offline Evaluation Anchor: `A2_EVAL_ANCHOR_V1`, materialized from verified v42c-DIN OOF (`runs/C2_v42_merged/oof_scores_full.npz`); `deployment_equivalent=false`.
- Anchor offline metrics (v42c OOF): NDCG@10 `0.5924893069425867`, HitRate@10 `0.819125`, MRR@10 `0.520807996031746`, Candidate Recall `1.0`.
- Registered verified OOF assets: `A2_V42C_DIN_OOF`, `A2_V48A_SASREC_OOF`.
- Declared unmaterialized assets (explicit artifact required before OOF use): V23 Top10, C_all Novel, DCN-Mix, LambdaRank, Len0 experts, Len3 experts, 0.5093 composition.
- Fusion planner best offline candidate: `score_blend_v42_v48` (accepted; NDCG@10 gain `+0.007551445340384433`, rescue/damage/net `815/522/293`).
- A2 scientific_rounds_used: `0`.
- Dry-run status: `ready_for_experiment_design`.
- Runtime artifacts: `artifacts/a2_integration/` (ignored by Git).

## Unique blocking item

- None for the integration dry-run. The final A2 real closed loop additionally requires explicit materialized artifacts for the declared assets above (V23 Top10 candidate set first) plus explicit next-module approval.

## Next main version

- Final A2 real closed loop (experiment execution). Do not start without explicit approval and explicit asset paths.

## Key run commands

```powershell
python -m afac_agent.main a2-integration --project_root . --a2-data-dir "C:\Users\李天皓\agent比赛\A推荐\A推荐" --a2-runs-root "C:\Users\李天皓\agent比赛\model_pro\versions\v49_featurebank" --force-rebuild
python -m pytest -p no:cacheprovider -q --basetemp "C:/tmp/afac_a2_integration"
python -m afac_agent.doctor --project_root .
```

## B1 Repair / V2 Autonomous Classification Closed Loop Status

- Module: B1 node classification (B榜节点分类) repair and V2 loop; independent namespace from A1/A2/B2.
- Data root: `C:\Users\李天皓\agent比赛\B分类`; no B2 data entered B1 pipeline.
- Closed loop status: `completed`; run id `c5e33663d37c6e49004e8213`; wall clock within 2-hour budget; `scientific_rounds_used=3`.
- Best scientific candidate: `B1_LP_UNDIRECTED_ALPHA7`.
- Best deployment candidate: `B1_LP_UNDIRECTED_ALPHA7` (standard accuracy `0.4964`, macro accuracy `0.3941`).
- Submission: `artifacts/b1_runs/c5e33663d37c6e49004e8213/TO_UPLOAD/candidate_B1_v2.csv` (audit passed).
- Evaluation anchor: `B1_EVAL_ANCHOR_V2` materialized from `B1_LP_UNDIRECTED_ALPHA7`.
- Reference canonical first B1 run: `e95368a24e0780e65e92aceb` (online score `0.37908`, offline standard `0.49069`, gap `0.11161`).
- No A1/A2/B2 Champion, Anchor, Fold, history, or scientific-round records modified.
- No Test truth used; no external public B1 labels/edges/checkpoints used; no automatic platform upload.

### Key run commands (B1 V2)

```powershell
python -m afac_agent.main b1-closed-loop --project_root . --data-root "C:\Users\李天皓\agent比赛\B分类" --out-root artifacts/b1_runs --max-wall-clock-seconds 7200 --max-rounds 3 --force-rebuild
```

## B2 Data-First Autonomous Recommendation Closed Loop Status

- Module: B2 sequence recommendation (B榜推荐); independent namespace from A1/A2/B1.
- Data root: `C:\Users\李天皓\agent比赛\B推荐`; actual files discovered under `C:\Users\李天皓\agent比赛\B推荐\B推荐\`.
- Data Intelligence status: `verified`; run id `8e96fac363943d65f1c1947d`.
- Fold protocol: `AFAC_B2_FOLD_V1` (stratified user-group-aware 5-fold, seed 2026, stable hash).
- Validation panels: `B2_STANDARD_PANEL`, `B2_SHORT_HISTORY_PANEL`, `B2_EXACT_LEN3_PANEL`, `B2_LEN4_PLUS_PANEL`, `B2_HISTORY_TARGET_PANEL`, `B2_NOVEL_TARGET_PANEL`, `B2_LONG_TAIL_PANEL`, `B2_TEST_LIKE_PANEL`, `B2_TOP10_BOUNDARY_PANEL`.
- Closed loop status: `completed`; run id `fcf5ad3dbcdf9700dd644eff`; wall clock `1807.57s` (within 2-hour budget); `scientific_rounds_used=3`.
- Best scientific candidate: `B2_HISTORY_RECALL` (NDCG@10 `0.15482521769229524`, HitRate@10 `0.25146856642919635`, MRR@10 `0.12341284720362335`).
- Best deployment candidate: `B2_HISTORY_RECALL`.
- Round 1 (`retrieval_foundation`): popularity, history recall, item-item co-occurrence, last-item transition, score blends.
- Round 2 (`ranking_explore`): candidate ranker (logistic) with batched cooc features.
- Round 3 (`bucket_and_rerank_explore`): bucket rerank blending ranker and retrieval by sequence length.
- Submission: `artifacts/b2_runs/fcf5ad3dbcdf9700dd644eff/TO_UPLOAD/candidate_B2.csv` (10000 rows, Top10, legal iids, test user order preserved, audit passed).
- Evaluation anchor: `B2_EVAL_ANCHOR_V1` materialized from `B2_HISTORY_RECALL` OOF.
- Online anchor: `B2_ONLINE_ANCHOR` (status `no_submission_yet`).
- No A1/A2/B1 Champion, Anchor, Fold, history, or scientific-round records modified.
- No Test truth used; no A2 model weights/OOF/Fold/candidates used for B2; no automatic platform upload.

### Key run commands (B2)

```powershell
python -m afac_agent.main b2-closed-loop --project_root . --data-root "C:\Users\李天皓\agent比赛\B推荐" --out-root artifacts/b2_runs --max-wall-clock-seconds 7200 --max-rounds 3
python -m pytest -p no:cacheprovider -q --basetemp "C:/tmp/afac_b2_post"
python -m afac_agent.doctor --project_root .
```

### Next main version

- B2 follow-up rounds or dual-task packaging only after explicit approval.

## B1 + B2 Dual-Task Master Run Status

- Master run id: `86f75dbc66d25382eb0c6a24`.
- Canonical first B1 run: `e95368a24e0780e65e92aceb` (online score `0.37908`, offline standard `0.49069`, gap `0.11161`).
- B1 V2 run: `c5e33663d37c6e49004e8213` (best `B1_LP_UNDIRECTED_ALPHA7`, standard `0.4964`, macro `0.3941`).
- B2 first run: `fcf5ad3dbcdf9700dd644eff` (best `B2_HISTORY_RECALL`, NDCG@10 `0.1548`).
- Master artifacts: `artifacts/b_dual_task_runs/86f75dbc66d25382eb0c6a24/`.
- TO_UPLOAD:
  - `artifacts/b_dual_task_runs/86f75dbc66d25382eb0c6a24/TO_UPLOAD/B1/candidate_B1_v2.csv`
  - `artifacts/b_dual_task_runs/86f75dbc66d25382eb0c6a24/TO_UPLOAD/B2/candidate_B2.csv`
- Cross-task isolation: A1/A2/B1 frozen assets not modified; B1 and B2 data/anchors/folds/submissions isolated; no Test truth used.
- No automatic platform upload.

## B1 Data-First Autonomous Classification Closed Loop Status (Canonical First Run)

- Module: B1 node classification (B榜节点分类); independent namespace from A1/A2.
- Data root: `C:\Users\李天皓\agent比赛\B分类`.
- B2 (B榜推荐) path registered but not read; no B2 data entered B1 pipeline.
- Data Intelligence status: `verified`; run id `657a4bfbf22d525350d28642`.
- Fold protocol: `AFAC_B1_FOLD_V1` (stratified 5-fold, seed 2026, stable hash).
- Validation panels: `B1_STANDARD_PANEL`, `B1_DEGREE_MATCHED_PANEL`, `B1_PROPENSITY_MATCHED_PANEL`, `B1_LOW_DEGREE_PANEL`, `B1_TEST_LIKE_PANEL`.
- Closed loop status: `completed`; run id `e95368a24e0780e65e92aceb`; wall clock `43.66s`; `scientific_rounds_used=3`.
- Evaluation anchor: `B1_EVAL_ANCHOR_V1` materialized from `B1_LP_DIRECTED_OUT`.
- Best scientific candidate: `B1_LP_DIRECTED_OUT` (macro accuracy `0.3838`).
- Best deployment candidate: `B1_LP_DIRECTED_OUT` (standard accuracy `0.4907`).
- Submission: `artifacts/b1_runs/e95368a24e0780e65e92aceb/TO_UPLOAD/candidate_B1.csv` (1530 rows, label range 0-7, audit passed).
- No A1/A2 Champion, Anchor, Fold, history, or scientific-round records modified.
- No Test truth used; no external public B1 labels/edges/checkpoints used; no automatic platform upload.

### Key run commands (B1 canonical first run)

```powershell
python -m afac_agent.main b1-closed-loop --project_root . --data-root "C:\Users\李天皓\agent比赛\B分类" --out-root artifacts/b1_runs --force-rebuild
python -m pytest -p no:cacheprovider -q --basetemp "C:/tmp/afac_b1_post"
python -m afac_agent.doctor --project_root .
```

## B2 v2.1 Scientific Execution, Data Contract, Budget and Deployment Permission Repair

- Module: B2 scientific execution repair; branch: `fix/b2-v2-scientific-execution` (worktree `afac_agent_v1_b2_science_repair`).
- Fault run: execution_id `74ba8db66a4b8f50d9196865a611e9d78058e14d35fd8ac99c6e2426279dcd41` — all diagnostic, budget exceeded, falsely deployed, completion contract failed. Marked `INVALID_SCIENTIFIC_DEPLOYMENT`.
- Scope: Data Contract (canonical n_items=14065 with provenance), Experiment Permission (Diagnostic/Screen/Confirm/Deployment), Proposal-to-Operator Compiler, Target Metric Contract, Anchor Registry, Budget Safety (monotonic hard deadline), Deployment Permission Contract, Capability Registry v2.1, Heartbeat transparency fields.

### Key new files

- `afac_agent/v2/data_contract.py`: `B2CanonicalDataContract`, `ScaleValue`, `build_b2_data_contract()`, `reconcile_data_contract()`.
- `afac_agent/v2/experiment_kind.py`: `ExperimentKind` enum, `ExperimentPermission`, `permission_for_kind()`, `kind_from_operator_and_folds()`.
- `afac_agent/v2/proposal_compiler.py`: `CompiledOperator`, `compile_proposal()`, `semantic_revision_delta()`.
- `afac_agent/v2/anchor_registry.py`: `B2AnchorRegistry`, `AnchorRecord`, 5-fold popularity/history anchor materialization.
- `afac_agent/v2/target_metric_contract.py`: `TargetMetricContract`, `evaluate_target_metric_contract()`.
- `tests/test_b2_science_repair.py`: 21 tests (data contract, permission, compiler, no-op, budget, orchestrator integration, capability registry, fault manifest regression).

### Key modified files

- `afac_agent/v2/orchestrator.py`: data contract build & reconcile, compiled operator pipeline, experiment permission checks, monotonic hard deadline, multi-round smoke loop, heartbeat transparency, anchor fallback deployment, completion contract v2.1.
- `afac_agent/v2/capability_registry.py`: per-operator capability flags (diagnostic/screen/confirm/full_cv/checkpoint/deployment_ready), B2 v2.1 operators.
- `afac_agent/v2/completion_contract.py`: data/budget/deployment permission contract checks, scientific execution reality check.
- `afac_agent/v2/data_intelligence.py`: n_train_total/n_test_total vs n_train_profiled/n_test_profiled, profile_scope, raw/dedup/unique/repeat buckets.
- `afac_agent/supervisor/heartbeat.py`: v2.1 transparency fields (operator_id, experiment_kind, semantic_genome_hash, fidelity, target_panel, permissions, deadlines, data contract fields).

### Real 900s B2 Science Smoke

- Execution ID: `0dd92953ffdcc53f9907aa1ebada1b9912f27022604adbf9783e09cef580dda1`
- Status: `completed_smoke`; wall clock: `599.5s` (budget: 900s).
- 12 real LLM calls; 3 rounds: 1 diagnostic + 2 scientific (retrieval_union_experiment as SCREEN_EXPERIMENT).
- `scientific_attempts_used=2`, `all_rounds_diagnostic=false`.
- Data contract: n_items=14065 consistent across all stages.
- Completion contract: passed. No deployment generated.

### Synthetic Smoke A-D

All 4 smokes covered by 21 pytest tests in `tests/test_b2_science_repair.py` — all PASS.

### Formal B2 two-hour run

**NOT started.** Do not start without explicit approval. The fix branch must be committed and merged first.

### Frozen assets

A1 `v53Q-1` (0.7800), A2 0.5093 champion, B1 V1/V2, B2 V1 results — all verified unchanged. Frozen hashes match.

### Key run commands (v2.1 smoke)

```powershell
python -u -m afac_agent.main v2-run --task B2 --data-root "C:/Users/李天皓/agent比赛/B推荐/B推荐" --out-root artifacts/v2_science_smoke/b2 --max-wall-clock-seconds 900 --deployment-reserve-seconds 180 --require-llm --force-new-execution --smoke --no-deployment --no-full-cv --smoke-max-seconds 900
```
