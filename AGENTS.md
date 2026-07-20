# AGENTS.md — AFAC2026 Automated Research Agent

## Mission

Build a bounded, auditable, serial automated-research system for AFAC2026 Task 3.

There are two distinct agents:

1. **Build-time Codex**
   - edits this repository;
   - creates adapters, tests, schemas, documentation and CI;
   - may run local smoke tests;
   - must not invent competition results.

2. **Run-time AFAC Agent**
   - profiles a supplied dataset;
   - selects only registered tools and configurations;
   - runs experiments serially under the time budget;
   - parses real results;
   - updates append-only memory and trajectory;
   - stops, pivots or finalizes based on frozen gates;
   - must not rewrite source code during competition execution.

Do not merge these two roles.

## Current confirmed state

- A1 online champion: `v53Q-1`, score `0.7800`.
- Stack:
  - Graph-visible: `v43C`;
  - Isolated: `v46A-1 Balanced Seed Consensus`;
  - Micro patch: transition-stable Edge-H2.
- Current OOF anchor: `0.77538405599`.
- Existing-signal Isolated repair routes are closed.
- Current build target is an agent platform, not another manually selected model.

## Non-negotiable safety rules

- No parallel experiment processes.
- No Test-label use.
- No threshold selection on Test or leaderboard deltas.
- No A-list node IDs, Frozen lists or class-specific Test rules in B-list general code.
- No unregistered command execution.
- No LLM-generated arbitrary shell command.
- No reopening a closed branch unless a new independent signal and an explicit reopening audit exist.
- Trajectory is written during execution, never reconstructed afterward.
- Every model action must emit a standard result artifact.
- Every expensive action must be preceded by a low-cost upper-bound or information-gain audit.

## Development protocol

Before editing:
1. inspect repository state and current tests;
2. identify the single requested capability;
3. write or update tests;
4. implement the smallest change;
5. run syntax, unit, integration and smoke tests;
6. update `PROJECT_STATE.md`, `CHANGELOG.md` and relevant schemas.

Never silently change:
- validation folds;
- pass/fail gates;
- model defaults;
- data definitions;
- budget;
- champion identity.

## Standard experiment contract

Every tool must declare:

```json
{
  "name": "",
  "task": "A1 or A2",
  "layer": "",
  "only_change": "",
  "kept_fixed": [],
  "forbidden": [],
  "expected_runtime_seconds": 0,
  "prediction_changing": false,
  "submission_creating": false,
  "required_inputs": {},
  "output_schema": "schemas/experiment_result.schema.json",
  "pass_conditions": {},
  "stop_conditions": {}
}
```

## Standard A1 feedback

- overall OOF;
- per-fold metrics;
- Graph-visible;
- Isolated;
- degree/train-neighbor buckets;
- per-class accuracy and errors;
- rescue/damage/neutral;
- unique rescue and oracle;
- OOF/Test shift;
- runtime and resource use;
- leakage/safety checks.

## Standard A2 feedback

- overall NDCG@10;
- Len0/Len1/Len2/Len3/Len4+;
- Seen/Novel;
- Repeat/Explore;
- candidate coverage;
- slot-level results;
- history-item preservation;
- OOF/Test shift;
- runtime and resource use;
- leakage/safety checks.

## Done criteria

A feature is not complete until:
- unit tests pass;
- one local integration smoke test passes;
- failure paths are tested;
- outputs validate against schemas;
- state transition is verified;
- no prediction/submission is changed unless declared;
- documentation is updated.
