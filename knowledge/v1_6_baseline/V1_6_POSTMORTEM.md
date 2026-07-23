# V1.6 Postmortem

Baseline: commit `30efa2cbb4264630e0e1e984d55e3fffeba354e1`, branch base `feat/afac-v2-full`,
dual-task master run `86f75dbc66d25382eb0c6a24`. Baseline pytest: 207 passed; doctor: PASS.

## Online results (competition scalar feedback)

| Task | Online | Offline | Abs gap | Rel gap | Submitted file |
|------|--------|---------|---------|---------|----------------|
| B1 V1 | 0.37908 | standard 0.49069 | 0.11161 | 0.22746 | artifacts/b1_runs/e95368a24e0780e65e92aceb/TO_UPLOAD/candidate_B1.csv |
| B1 V2 | 0.37974 | standard 0.49641 (confirmed from run report; master manifest rounds 0.4964) | 0.11667 | 0.23503 | artifacts/b1_runs/c5e33663d37c6e49004e8213/TO_UPLOAD/candidate_B1_v2.csv |
| B2 V1 | 0.06838 | ndcg@10 0.1548 (hit@10 0.2515, mrr@10 0.1234) | 0.08642 | 0.55827 | artifacts/b2_runs/fcf5ad3dbcdf9700dd644eff/TO_UPLOAD/candidate_B2.csv |

All three submitted files are hash-verified (see `online_results.json`). Online scores may only be used
for validation_calibration, direction_confidence, deployment_risk, submission_budgeting — never for
node/item-level correction or test-label inference.

## B1 known issues

1. Standard panel overestimates online (offline 0.49069/0.49641 vs online 0.37908/0.37974).
2. Degree-matched and Standard panels not truly differentiated.
3. Test-like and Propensity-matched panels duplicate.
4. LP Class 7 accuracy ≈ 0.
5. Neighbor-LR shows Class-7 complementarity (never fused).
6. Round-03 node-level gate was a no-op (refunded, excluded from portfolio).
7. Premature stop at 75.6s of 7200s budget (fixed 3-round cap).

## B2 known issues

1. History Recall only covers history targets.
2. Novel targets are the majority; History Recall = 0 there.
3. candidate_recall@10 ≡ hit_rate@10 (same computation — not independent signals).
4. Retrieval/ranking failure attribution not mutually exclusive.
5. Score Blend degenerated to Popularity (round refunded).
6. Candidate Ranker was not a true candidate-level LambdaRank.
7. Bucket Rerank degenerated to Popularity (round refunded).
8. Dense full user×item matrix caused a 3.35GiB allocation failure (float64 default).
9. Premature stop at 1807.57s of 7200s.

## Capability gaps

True candidate-level LambdaRank (lightgbm absent), GBDT classification parent, memory-safe sparse
candidate tables, run supervisor/heartbeat/dashboard, no-op detection, metric semantics separation,
panel membership audit, dynamic budget scheduling, GPU/torch models.

## Validation gaps

B1 standard-online gaps ~0.11 absolute / ~0.23 relative for both V1 and V2; undifferentiated and
duplicated panels; B2 offline ndcg 0.1548 vs online 0.06838 (rel gap 0.55827); no panel calibration
against online feedback at all.

## V2 initial priority queue

- B1: (1) standard-online gap, (2) class-7 failure, (3) feature-graph complementarity, (4) panel redesign.
- B2: (1) novel-target coverage, (2) true candidate ranker, (3) pool-recall metrics, (4) memory safety,
  (5) bucket specialists.

Machine-readable details live alongside this file in `*.json`.
