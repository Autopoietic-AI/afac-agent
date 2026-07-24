# Real B2 Science Smoke Audit

- **Execution ID**: `0dd92953ffdcc53f9907aa1ebada1b9912f27022604adbf9783e09cef580dda1`
- **Data Root**: `C:\Users\李天皓\agent比赛\B推荐\B推荐`
- **Smoke Command**: 900s real LLM, no-deployment, no-full-cv
- **Code Commit**: `8c7339594311459150a3e5a326a5ee20188f2919`
- **Branch**: `fix/b2-v2-scientific-execution`

## 1. Real Training Executed?

**YES.** Round 2 executed `retrieval_union_experiment` as a `SCREEN_EXPERIMENT` (2-fold), consuming a scientific round. Round 3 executed a second scientific experiment.

## 2. Operator ID

- Last round: `retrieval_union_experiment`
- Round 1: `candidate_recall_diagnostic` (diagnostic)

## 3. Experiment Kind

- Round 1: `DETERMINISTIC_DIAGNOSTIC` (F0, 0 folds)
- Round 2: `SCREEN_EXPERIMENT` (F1_SCREEN, 2 folds)
- Round 3: `SCREEN_EXPERIMENT` (F1_SCREEN, 2 folds)

## 4. Fold Count

- Round 1: 0 folds (diagnostic)
- Round 2: 2 folds (screen)
- Round 3: 2 folds (screen)

## 5. Scientific Attempts

- `scientific_attempts_used`: 2
- `scientific_rounds_used`: 2.0
- `effective_scientific_rounds`: 2.0

## 6. Effective Scientific Rounds

`effective_scientific_rounds` = 2.0 (no-op refunds = 0)

## 7. Checkpoint Count

0 (no full-CV triggered; smoke mode)

## 8. Genome Hash Count

Multiple unique hashes:
- Round 1: `57279eb65b2056f9d724b6bb986f33404552ec195022f6f95dd78c5428c09a25`
- Round 2: `18c3d6389e4ca4a9a685b977ca098e7426e930a04886d6923f9d29f9fd97ae5e`
- Round 3: distinct hash (different operator)

## 9. No-op Isolation

- `no_op_in_portfolio`: **false**
- No no-op rounds refunded
- No-op audit for each round showed changed_fraction > 0

## 10. Diagnostic Deployment Path?

**BLOCKED.** `deployment_permission_status`: `failed`. No `candidate_B2.csv` generated. `deployment_generated`: false. `all_rounds_diagnostic`: false. `incumbent_can_deploy`: false.

## 11. Actual Wall Clock

`wall_clock_seconds`: 599.5s (within 900s hard deadline + 5s)

## 12. Budget Overshoot?

**NO.** `budget_contract_status`: `passed`. 599.5s < 905s (900s + 5s margin).

## 13. Data Contract Consistency

| Field | Value | Source |
|-------|-------|--------|
| `n_items_total` | 14065 | item.csv::iid |
| `n_train_users_total` | 40000 | train.csv::uid |
| `n_test_users_total` | 10000 | test.csv::uid |
| `n_interactions_total` | 1797067 | train.csv (raw seq + targets) |
| `profiler_sample_train_users` | 512 | sampled (smoke mode) |
| `profiler_sample_test_users` | 512 | sampled (smoke mode) |
| `profile_scope` | sampled_head | smoke mode |

- Data contract reconcile: **passed** (Input Discovery n_items = Data Intelligence n_items = Experiment Executor universe = Deployment candidate universe = 14065)
- No mismatch between Input Discovery (14065) and Data Intelligence (14065)

## 14. LLM Calls

- Total: 12 (3 rounds × 4 calls each: problem synthesis, M6B, M6C, postmortem)
- All `mode=llm`
- Provider: aliyun_bailian_openai

## 15. Completion Contract

- Status: **passed** (smoke mode)
- 0 missing fields

## 16. Key Findings

1. **Data Contract fix works**: `n_items=14065` has `source_file: item.csv`, `source_column: iid`. The previous issue of mixing 14065 and 40011 is permanently resolved.
2. **Multi-round smoke works**: The smoke now loops until at least one scientific experiment is executed, not just one diagnostic round.
3. **Experiment permission model works**: Diagnostics → diagnostic_registry; Screens → scientific_portfolio (screen_only); No deployment without Confirm.
4. **Hard budget enforced**: 599.5s elapsed with 900s budget — no overshoot.
5. **Anchors built**: popularity and history anchors materialized on canonical folds.
6. **Pool recall improved**: From popularity parent recall@100=0.233 to retrieval_union@100=0.431.

## 17. Known Gaps

1. `candidate_ranker_experiment` was not selected by the LLM — it chose `retrieval_union_experiment` instead. This is a valid scientific operator.
2. `target_metric_contract` shows `parent_target_metric missing` — panel-level metrics not fully populated for the B2_NOVEL_TARGET_PANEL in the first scientific round.
3. Heartbeat transparency fields show 0 for n_items_total/n_test_total/n_test_profiled — these fields weren't being updated by the heartbeat writer at the time of this smoke.
