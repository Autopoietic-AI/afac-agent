# B2 Scientific Execution — Root Cause Report

**Repair ID:** `b2_science_repair_8c73395_20260723`  
**Fault Execution:** `74ba8db66a4b8f50d9196865a611e9d78058e14d35fd8ac99c6e2426279dcd41`  
**Fault Commit:** `8c7339594311459150a3e5a326a5ee20188f2919`

## TL;DR

The run produced a `completed` manifest and a `candidate_B2.csv`, but it was not a
scientific experiment. Three rounds of diagnostics ran through the fold ladder
without ever being classified as scientific experiments, without ever producing a
3-fold Confirm candidate, and without ever checking whether the selected
candidate had deployment permission. The hard wall-clock budget was exceeded by
~19 minutes, and the data contract was internally inconsistent.

## 1. Data Contract Breakdown

### 1.1 Item-count inconsistency

| Source | Value | Origin |
|--------|-------|--------|
| `B2TaskAdapter` / `item.csv` | **14,065** | Unique `iid` values in the official item table. |
| `Data Intelligence` | **40,011** | `coverage_map.item_popularity.n_items` in `data_intelligence.json`. |

The Data Intelligence module (`analyze_recommendation`) declares its inputs as
`list[list[Any]]`, but the orchestrator called it with dictionaries:

```python
# afac_agent/v2/orchestrator.py:496
analyze_recommendation(
    subset_seq,        # dict[str, list[str]]
    subset_targets,    # dict[str, str]
    {u: dataset.test_seq.get(u, []) for u in list(dataset.test_seq)[:512]},
)
```

`zip(train_seq, train_targets)` over two dicts yields **uid strings**, so the
popularity counter treated train `uid`s (and the characters inside them) as
items. The resulting 40,011 is approximately `40,000` train users plus the
~11 distinct characters appearing in uid strings.

This also explains:
- `history_recall_coverage = 0.0` (uid strings do not equal each other across users),
- `test_length_buckets` all falling into `len4_plus` (uid string lengths are 6+),
- `n_test = 512` being reported as the full test size because the profiler was
  hard-limited to `smoke_max_users=512`.

### 1.2 Full vs sampled scale confusion

The Data Intelligence call sampled test users to 512 but wrote the result into
the canonical `dataset_fingerprint.n_test` field. There was no separate
`n_test_total` / `n_test_profiled` split.

## 2. Experiment Type / Permission Breakdown

`budget_scheduler.py` defines `Fidelity.CHEAP_DIAGNOSTIC` with `round_cost=0.0`.
In the orchestrator:

```python
# afac_agent/v2/orchestrator.py:714
fidelity = Fidelity.CHEAP_DIAGNOSTIC if diagnostic_type in ALLOWED_DIAGNOSTICS else Fidelity.SINGLE_FOLD
```

All three rounds used `diagnostic_type` values that are members of
`ALLOWED_DIAGNOSTICS` (`candidate_recall_diagnostic`, `small_retrieval_compare`,
`cached_replay`). Even though they ran 2-fold canonical evaluations, they were
classified as cheap diagnostics and therefore:

- consumed `cheap_diagnostics_used += 1`,
- did **not** consume `scientific_rounds_used`,
- never produced Confirm-level evidence.

## 3. Proposal / Operator Mapping Breakdown

The M5 allowed-list for round > 1 is:

```python
ALLOWED_DIAGNOSTICS + ALLOWED_FORMAL_EXPERIMENTS
```

There was no `ProposalOperatorCompiler`. The LLM could keep choosing diagnostic
names and the deterministic gate admitted them. `small_retrieval_compare` and
`cached_replay` are just different names for the same underlying computation,
so no real operator family switch happened despite three M6C `revise` verdicts.

## 4. Promotion Gate Breakdown

The promotion gate in `adaptive_fold.py` uses `target_bucket_gain` computed from
the candidate-vs-parent hit-rate difference on the `B2_NOVEL_TARGET_PANEL`. It
never saw a Confirm-level candidate because no experiment reached fidelity
`F2_CONFIRM`. The recorded `target_bucket_metric` in the round record was the
overall `pool_recall@100`, not the panel-specific metric.

## 5. Portfolio / No-op Breakdown

`Portfolio.register_candidate` assigns role `"diagnostic"` when `no_op=True`.
That still inserts the candidate into the portfolio registry. The orchestrator
later selects the best round record by `hit_rate@10` regardless of role or
`no_op` status.

Round 3 (`cached_replay`) was a true no-op (`changed_fraction=0.0`) and still
appears in `portfolio_update.json`.

## 6. Deployment Permission Breakdown

`_run_deployment` selected:

```python
best = max(round_records, key=lambda r: r["metrics"].get("hit_rate@10", 0.0))
```

It then wrote `candidate_B2.csv` using the union retrieval path. It did not:
- check `candidate.can_deploy`,
- require `fold_count >= 3`,
- require Confirm / Full-CV / validated-anchor kind,
- reject a no-op candidate,
- separate `format_audit_passed` from `scientific_permission_passed`.

## 7. Budget / Wall-clock Breakdown

`started_at = time.time()` and elapsed was computed with `time.time() - started`.
No `time.monotonic()` hard deadline was maintained. The budget decision only
checked that the candidate's estimated cost fit into `remaining_wall_clock_seconds`;
it did not require `remaining_seconds > estimated_runtime + deployment_reserve +
safety_margin`. Round 3 started with ~1848 s remaining, estimated 450 s, but
took ~2300 s, pushing the total to 8340 s.

## 8. Completion Contract Breakdown

`completion_contract.py` checked artifact presence and gates, but it did not
require:
- `scientific_rounds_used >= 1`,
- a deployable (non-diagnostic) candidate,
- budget compliance (`wall_clock_seconds <= max_wall_clock_seconds + epsilon`).

Therefore the run was allowed to self-certify as `completed`.

## 9. Fixes Required

1. **B2CanonicalDataContract** reconciles item/user/interaction counts across
   Input Discovery, Data Intelligence, Experiment Executor, and Deployment.
2. **Data Intelligence** must accept dict inputs and explicitly report
   `n_train_total`, `n_train_profiled`, `n_test_total`, `n_test_profiled`,
   plus raw/dedup/unique-item length buckets.
3. **ExperimentKind / permission model** fixes diagnostic/screen/confirm/deploy
   permissions and round consumption.
4. **ProposalOperatorCompiler** maps LLM proposals to real operators; unavailable
   operators become `blocked_missing_adapter`, not silent diagnostics.
5. **Semantic genome hash + revise delta** prevents renaming-only revisions.
6. **Target metric contract + promotion gate** enforces same-fold same-panel
   comparisons.
7. **No-op / portfolio isolation** keeps no-ops out of the scientific and
   deployment portfolios.
8. **Hard wall-clock deadline** using `time.monotonic()` with research deadline
   and deployment reserve.
9. **Deployment Permission Contract** with four separate audit flags.
10. **Completion Contract** checks scientific rounds, deployable incumbent,
    budget, and data contract.
11. **B2AnchorRegistry** with validated popularity/history anchors for safe
    fallback when no scientific candidate is confirmed.
