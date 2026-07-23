# FORMAL_EXPERIMENT_CAPABILITY_AUDIT.md

Scope: the two whitelisted formal experiments in the v2 orchestrator
(`afac_agent/v2/orchestrator.py`, execution via `_run_folded_experiment` over
fixed canonical folds). Audited 2026-07-23 against the adaptive-fold contract.

## candidate_ranker_experiment

| Capability | Status | Notes |
|---|---|---|
| F1 2-fold screen | YES | canonical folds [0,1], fit on complement, eval on fold |
| F2 3-fold confirm | YES | canonical folds [0,1,2], same-fit protocol |
| F3 5-fold optional | YES | canonical folds [0..4], only via `should_trigger_full_cv` |
| F4 Full Train | YES | `_run_deployment` retrains ranker on ALL train users |
| Parent Candidate | YES | paired per-fold vs incumbent parent (popularity or prior incumbent) |
| OOF保存 | PARTIAL | per-fold eval metrics recorded in artifacts; per-user OOF score dump not yet persisted |
| Checkpoint保存 | NO | sklearn GBDT is refit per fold; no model serialization yet |
| Test Inference | YES | deployment reranks official test users |
| Deployment Audit | YES | TO_UPLOAD + submission_audit + deployment_manifest |
| 稀疏候选表 | YES | SparseCandidateTable only (max 200/user) |
| float32 | PARTIAL | retriever scores float32; ranker features currently float64 vectorization |
| 无稠密User×Item矩阵 | YES | memory preflight + sparse tables; never full catalog scoring |

- 真正可变的参数: retriever set, union size (`max_per_user`), negatives cap, GBDT hyperparams (via CandidateTableRanker), RRF k.
- 真正可切换的特征: source score/rank, source_count, RRF, popularity, history/repeat, recency, transition, seq stats, user/item attributes, attr-match, history/novel, long-tail, seq bucket, propensity, rank band.
- 真正可切换的Parent: popularity parent (round 1) or any incumbent candidate.
- 支持的Bucket: Len0/Len1/Len2/ExactLen3/Len4Plus/Repeat/Novel/History/LongTail/ColdUser (feature-level; routing via ExpertRouter available).
- 支持的Fold模式: F1/F2/F3/F4, fixed canonical prefixes, no re-randomization.

## bucket_specialist_experiment

| Capability | Status | Notes |
|---|---|---|
| F1 2-fold screen | YES | same canonical protocol |
| F2 3-fold confirm | YES | same canonical protocol |
| F3 5-fold optional | YES | via trigger only |
| F4 Full Train | YES | deployment union + bucket rerank possible (default: full retrain path) |
| Parent Candidate | YES | paired per-fold |
| OOF保存 | PARTIAL | metrics per fold persisted; per-user OOF not yet persisted |
| Checkpoint保存 | NO | rule-based rerank, nothing to serialize |
| Test Inference | YES | via deployment |
| Deployment Audit | YES | same as above |
| 稀疏候选表 | YES | |
| float32 | YES (scores) | |
| 无稠密User×Item矩阵 | YES | |

- 真正可变的参数: per-bucket blend weights (currently len0 vs len>0), union sources, RRF k.
- 真正可切换的特征: source weights per bucket.
- 真正可切换的Parent: same as above.
- 支持的Bucket: currently two weight regimes (len0, len>0); full 11-bucket ExpertRouter not yet wired into the loop.
- 支持的Fold模式: F1/F2/F3/F4.

## Still missing (honest list)

- Per-user OOF score persistence and safe-cache reload across executions.
- Model checkpoint serialization for the ranker (joblib).
- float64 -> float32 ranker feature vectorization.
- Full 11-bucket ExpertRouter wiring for bucket specialists.
- 3-fold ensemble is implemented for retrieval-union deployment; ranker-level ensemble not yet.
- B1 task is not yet wired into the v2 orchestrator (policy defaults exist and are unit-tested).

## Anti-repetition guarantee

The multi-round loop cannot oscillate between two fixed outputs: each round's
candidate differs in at least one of {experiment kind, retriever set, bucket
weights, ranker hyperparameters, fold fidelity}, LLM proposals are reviewed
by M6C and admitted by M5, no-op experiments are refunded and excluded from
promotion, and the incumbent only moves on confirm-level paired evidence.
